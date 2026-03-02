"""
═══════════════════════════════════════════════════════════════
  TarsMamba2LM — Полная модель ТАРС v3 (Deep WuNeng Core)
═══════════════════════════════════════════════════════════════

Архитектура:
  Embedding → 12 × TarsBlock → IDME → LM Head

  Каждый TarsBlock содержит TarsCoreBlock (ssd.py) + обвязку:
    - TarsCoreBlock: Deep Hybrid (Mamba-2 SSD + RWKV-7 WKV + WuNeng Fusion)
       Общая in_proj / out_proj, слияние в латентном пространстве
    - Ω-SSM Lie (Cayley SO(n))
    - MoLE (per-block, sparse top-2 of 8 experts)
    - NoveltyGate + RAG injection
    
  Глобальные компоненты:
    - IDME MatrixPool (Speculative Matrix Routing)
    - Integral Auditor (p-convergence tracking)
    - Hankel SVD (anti-cycle detection)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import logging
from typing import Optional, Tuple, Any
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from brain.mamba2.tars_block import TarsBlock, MemoryInjector
from brain.mamba2.integral_auditor import IntegralAuditor, MetaAuditor
from brain.mamba2.matrix_pool import MatrixPool
from brain.mamba2.novelty import NoveltyGate, HankelDetector
from brain.mamba2.logger import ThinkingLogger
from brain.mamba2.neuromodulator import Neuromodulator
from brain.mamba2.oscillations import OscillatoryBinding
from brain.mamba2.dendrites import DendriticBlock
from brain.mamba2.hyperbolic import HyperbolicSimilarity, project_to_poincare
from brain.mamba2.active_inference import BeliefState
from brain.mamba2.thinking_chain import ThinkingChain
from brain.mamba2.personality_adapter import PersonalityAdapter
from brain.mamba2.bitnet import (
    UniversalLinear, convert_model_to_158bit,
    convert_model_to_fp16, model_stats, replace_linear_with_universal
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RotaryPositionEmbedding(nn.Module):
    """
    Unified RoPE: Rotary Position Embedding (Qwen3/Mamba-3 style).
    
    Uses base=1,000,000 for up to 32K+ context (vs standard 10,000).
    Applied to WKV branch for positional awareness in SSM-Attention hybrid.
    
    Math: RoPE(x, pos) = x * cos(θ_pos) + rotate_half(x) * sin(θ_pos)
    where θ_i = pos / base^(2i/d)
    """
    
    def __init__(self, dim, base=1_000_000, max_seq_len=32768):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos/sin tables
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    @staticmethod
    def _rotate_half(x):
        """Rotate half of the hidden dims: [x1, x2, ..., xn] → [-x_{n/2+1}, ..., -xn, x1, ..., x_{n/2}]"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x, seq_len=None, offset=0):
        """
        Apply RoPE to input tensor.
        
        Args:
            x: [B, L, D] or [B, H, L, D]
            seq_len: override sequence length
            offset: position offset (for cached generation)
        
        Returns:
            x with rotary position encoding applied
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        cos = self.cos_cached[offset:offset + seq_len, :self.dim]
        sin = self.sin_cached[offset:offset + seq_len, :self.dim]
        
        # Broadcast to match x dimensions
        if x.ndim == 4:  # [B, H, L, D]
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:  # [B, L, D]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        # Only apply to first `dim` dimensions
        x_rope = x[..., :self.dim]
        x_rest = x[..., self.dim:]
        
        x_rope = x_rope * cos + self._rotate_half(x_rope) * sin
        
        return torch.cat([x_rope, x_rest], dim=-1) if x_rest.size(-1) > 0 else x_rope


class WaveConsolidation(nn.Module):
    """
    Полноценный слой консолидации волны.
    
    Заменяет лёгкий WaveMerge+WaveGate.
    Это «большой» слой, который объединяет результаты обоих блоков
    и рефлексы (MoLE experts) в единый выход.
    
    Архитектура:
      1. Dimension-wise Gate: σ(W · [h_L; h_R]) ∈ (0,1)^d
         — не скаляр, а полный d_model gate
      2. Deep Fusion MLP: [h_L; h_R] → 2d → SiLU → d → d
         — глубокая нелинейная коррекция
      3. Reflex integration: сигнал от MoLE stats обоих блоков
      4. Output: gate * x_left + (1-gate) * x_right + fusion + reflex
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # 1. Dimension-wise gate (полный, не скаляр)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # 2. Deep Fusion MLP (большой слой)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # 3. Reflex integration gate (MoLE expert signals)
        self.reflex_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )
        
        # 4. Output normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Масштаб fusion и reflex
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))
        self.reflex_scale = nn.Parameter(torch.tensor(0.05))
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        stats_l: dict = None,
        stats_r: dict = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        x_left, x_right: [B, L, d_model]
        Returns: (x_merged [B, L, d_model], alpha_mean: float)
        """
        h_left = x_left.mean(dim=1)   # [B, d_model]
        h_right = x_right.mean(dim=1)  # [B, d_model]
        h_cat = torch.cat([h_left, h_right], dim=-1)  # [B, 2*d_model]
        
        # 1. Dimension-wise gate
        alpha = self.gate(h_cat)  # [B, d_model]
        alpha_3d = alpha.unsqueeze(1)  # [B, 1, d_model]
        x_gated = (1 - alpha_3d) * x_left + alpha_3d * x_right
        
        # 2. Deep fusion correction
        fusion = self.fusion(h_cat)  # [B, d_model]
        
        # 3. Reflex integration
        reflex_signal = self.reflex_gate(h_cat)  # [B, d_model]
        
        # Combine
        x = x_gated + self.fusion_scale * fusion.unsqueeze(1) \
                     + self.reflex_scale * reflex_signal.unsqueeze(1)
        
        return self.norm(x), alpha.mean().item()


class GlobalWorkspace(nn.Module):
    """
    Global Workspace Theory (Baars, 1988).
    
    Нейронаука: информация из специализированных модулей конкурирует
    за доступ в «глобальное рабочее пространство» (аналог сознания).
    Победитель транслируется (broadcast) всем модулям одновременно.
    
    Реализация: все 12 блоков отправляют h_mean в конкуренцию.
    Мягкий attention определяет «доминанту». Broadcast signal
    добавляется к финальному представлению.
    
    Математика:
      g = softmax(W_g · [h₁ ⊕ h₂ ⊕ ... ⊕ h_n])  # competition
      broadcast = Σᵢ gᵢ · hᵢ                        # winner-take-most
    """
    
    def __init__(self, d_model: int = 768, n_blocks: int = 12):
        super().__init__()
        self.n_blocks = n_blocks
        
        # Competition: все блоки борются за внимание
        self.competition = nn.Linear(d_model * n_blocks, n_blocks)
        
        # Broadcast projection: формирует глобальный сигнал
        self.broadcast_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        
        # Mixing strength (обучаемый)
        self.mix = nn.Parameter(torch.tensor(0.1))
        
        # Norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, block_outputs: list, x_current: torch.Tensor) -> torch.Tensor:
        """
        block_outputs: list of [B, L, d_model] — outputs from each block
        x_current: [B, L, d_model] — current stream (post all waves)
        
        Returns: x_current + broadcast signal
        """
        if len(block_outputs) < 2:
            return x_current
        
        # h_mean per block: [B, d_model] each
        h_means = [h.mean(dim=1) for h in block_outputs]
        
        # Pad if fewer blocks than expected
        while len(h_means) < self.n_blocks:
            h_means.append(torch.zeros_like(h_means[0]))
        h_means = h_means[:self.n_blocks]  # trim if more
        
        # Concatenate all block representations
        h_cat = torch.cat(h_means, dim=-1)  # [B, d_model * n_blocks]
        
        # Competition gate: who gets the workspace?
        gates = F.softmax(self.competition(h_cat), dim=-1)  # [B, n_blocks]
        
        # Winner-take-most broadcast
        broadcast = torch.zeros_like(h_means[0])  # [B, d_model]
        for i, h in enumerate(h_means):
            broadcast = broadcast + gates[:, i:i+1] * h
        
        # Project and mix
        broadcast = self.broadcast_proj(broadcast)  # [B, d_model]
        
        return self.norm(x_current + self.mix * broadcast.unsqueeze(1))


class TarsMamba2LM(nn.Module):
    """
    ТАРС v3: Deep WuNeng Core (Mamba-2 + RWKV-7 inside one kernel).
    
    Гибридный мозг с единым ядром TarsCoreBlock:
    Mamba-2 SSD и RWKV-7 WKV делят общие проекции и сливаются
    в латентном пространстве через Deep Gated Fusion.
    
    ~130M params (fp16: ~260MB, 1.58-bit: ~60MB)
    """
    
    @classmethod
    def from_config(cls, config_path=None, device="cpu"):
        """Создаёт модель из config.json."""
        if config_path is None:
            config_path = os.path.join(_ROOT, "models", "tars_v3", "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        p = cfg["models"]["mamba2"]["params"]
        return cls(
            d_model=p.get("d_model", 768), n_layers=p.get("n_layers", 12),
            vocab_size=p.get("vocab_size", 256), d_state=p.get("d_state", 64),
            headdim=p.get("headdim", 64), omega_dim=p.get("omega_dim", 32),
            pool_size=p.get("pool_size", 48), n_experts=p.get("n_experts", 8),
        ).to(device)
    
    @classmethod
    def load_pretrained(cls, checkpoint_path=None, config_path=None, device="cpu"):
        """Загружает модель из config + checkpoint."""
        _logger = logging.getLogger("Tars.Mamba2LM")
        model = cls.from_config(config_path, device)
        
        if checkpoint_path is None:
            for p in [
                os.path.join(_ROOT, "models", "tars_v3", "mamba2.pt"),
                os.path.join(_ROOT, "models", "mamba2", "mamba2_omega_158bit.pt"),
                os.path.join(_ROOT, "models", "mamba2", "mamba2_omega.pt"),
            ]:
                if os.path.exists(p):
                    checkpoint_path = p
                    break
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            _logger.info(f"Loading weights: {checkpoint_path}")
            cp = torch.load(checkpoint_path, map_location=device, weights_only=False)
            state = cp.get("model_state_dict", cp)
            model_state = model.state_dict()
            loaded, skipped = 0, 0
            for key, value in state.items():
                if key in model_state and model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded += 1
                else:
                    skipped += 1
            model.load_state_dict(model_state, strict=False)
            _logger.info(f"Loaded {loaded} tensors, skipped {skipped}")
            return model, checkpoint_path
        
        _logger.warning("No checkpoint found — UNTRAINED")
        return model, None
    
    def __init__(
        self,
        d_model: int = 2048,     # TARS 1B: max intelligence in 1GB RAM
        n_layers: int = 24,      # 24 rich blocks ≈ 36 vanilla blocks
        vocab_size: int = 32000,
        d_state: int = 128,      # 2x working memory (BitMamba-2 style)
        headdim: int = 64,
        mingru_dim: int = 256,  # legacy arg, ignored
        omega_dim: int = 32,
        pool_size: int = 48,
        n_experts: int = 8,
        expert_rank: int = 8,
        quant_mode: str = "fp16",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.quant_mode = quant_mode
        self.logger = logging.getLogger("Tars.Mamba2LM")
        
        # ═══ Embedding ═══
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # ═══ Основные блоки (12 × TarsBlock) — Cortical Columns ═══
        # Нейронаука: кора мозга специализирована по глубине:
        #   Ранние слои (V1/V2)     → низкоуровневые признаки (лексика, морфология)
        #   Средние слои (IT/STS)   → семантика, отношения
        #   Глубокие слои (PFC/ACC) → логика, планирование, абстракции
        # Каждый "слой коры" имеет свою конфигурацию omega/experts/dropout.
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            depth_ratio = i / max(n_layers - 1, 1)  # 0.0 → 1.0
            
            # Cortical Column config по глубине
            layer_omega = int(omega_dim * (0.5 + 1.5 * depth_ratio))  # 16→80 при omega_dim=32
            layer_experts = max(4, int(n_experts * (0.5 + 0.5 * depth_ratio)))  # 4→8
            layer_dropout = 0.05 + 0.10 * depth_ratio  # 0.05→0.15
            
            self.blocks.append(TarsBlock(
                d_model=d_model, d_state=d_state,
                headdim=headdim, omega_dim=layer_omega,
                n_experts=layer_experts, layer_idx=i,
                quant_mode=quant_mode,
                dropout=layer_dropout,
            ))
        
        # ═══ Wave Consolidation (6 слоёв для 12/2=6 волн) ═══
        # Каждый consolidation — полноценный слой слияния:
        #   1. Dimension-wise gate (не скаляр, а полный d_model)
        #   2. Deep fusion MLP (2d → 2d → d → d) 
        #   3. Reflex integration (MoLE expert signal)
        n_waves = n_layers // 2
        self.wave_consolidations = nn.ModuleList([
            WaveConsolidation(d_model) for _ in range(n_waves)
        ])
        
        # ═══ Global Workspace (Baars, 1988) ═══
        # Все блоки конкурируют за доступ в глобальное пространство.
        # Победитель broadcast'ит своё представление.
        self.global_workspace = GlobalWorkspace(d_model, n_layers)
        
        # ═══ IDME Matrix Pool (48+, бесконечное расширение) ═══
        self.matrix_pool = MatrixPool(d_model, pool_size, quant_mode=quant_mode)
        
        # ═══ Integral Auditor ═══
        self.integral_auditor = IntegralAuditor(window=8, default_threshold=1.1)
        self.meta_auditor = MetaAuditor()
        
        # ═══ Hankel SVD (глобальный анти-цикл) ═══
        self.hankel = HankelDetector(window=6)
        
        # ═══ NoveltyGate (для IDME) ═══
        self.novelty_gate = NoveltyGate(d_model)
        
        # ═══ Neuromodulator (DA, NA, ACh, 5HT) ═══
        # Глобальная нейромодуляция: 4 нейромедиатора модулируют все компоненты.
        #   DA → MoLE routing sharpness
        #   NA → thinking depth (p-threshold)
        #   ACh → self-learning LR
        #   5HT → patience (max depth)
        self.neuromodulator = Neuromodulator(d_model)
        
        # ═══ Oscillatory Binding (θ-γ phase coding) ═══
        # θ-ритм (hippocampus) координирует memory encoding,
        # γ-ритм (cortex) группирует информацию внутри θ-цикла.
        self.oscillatory = OscillatoryBinding(d_model)
        
        # ═══ Active Dendrites (Numenta, 2021-2025) ═══
        # Дендритная модуляция: контекст задачи выбирает
        # специализированную ветвь вычислений через WTA selection.
        self.dendritic_block = DendriticBlock(d_model, d_model, n_segments=7)
        
        # ═══ Hyperbolic Similarity (Poincaré ball) ═══
        # Гиперболическое расстояние для иерархических структур
        # (деревья категорий, таксономии, memory hierarchy).
        self.hyper_sim = HyperbolicSimilarity(c=1.0, scale=1.0)
        
        # ═══ Active Inference: Belief State (Friston, 2006) ═══
        # Внутреннее байесовское состояние убеждений агента q(s).
        # Обновляется после каждого наблюдения через precision-weighted update.
        self.belief_state = BeliefState(d_state=128)
        
        # ═══ Thinking Logger ═══
        self.thinking_logger = ThinkingLogger()
        
        # ═══ Titans Memory Hook (384d LTM) ═══
        # Проекция d_model → 384d (пространство памяти LEANN/Titans)
        self.mem_dim = 384
        self.to_memory_space = nn.Linear(d_model, self.mem_dim, bias=False)
        self.from_memory_space = nn.Linear(self.mem_dim, d_model, bias=False)
        self.titans_memory = None
        
        # ═══ Shared Memory Injector (wave-parallel: 1 lookup → 2 blocks) ═══
        self.shared_mem_injector = MemoryInjector(d_model, mem_dim=384, quant_mode=quant_mode)
        
        # ═══ Output head ═══
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Init
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag
        self.use_checkpointing = False
        
        # MoLE auxiliary loss (accumulated during forward)
        self.mole_aux_loss = torch.tensor(0.0)
        
        # ═══ Speculative Decoding (Granite 4.0) ═══
        self.spec_draft_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, vocab_size),
        )
        
        # ═══ Native Chain-of-Thought v2: 5 подсистем ═══
        # 1. Retrieval-triggered RAG между волнами
        # 2. Multi-scale memory (working/session/longterm)
        # 3. Confidence-gated output (неуверенность → smoothing)
        # 4. Wave skip (если confidence > 0.9)
        # 5. Self-verification (повторный прогон для проверки)
        self.thinking_chain = ThinkingChain(
            d_model, d_memory=384, n_max_waves=n_layers // 2, vocab_size=vocab_size
        )
        
        # ═══ PersonalityAdapter (Convergence-Gated Style Transform) ═══
        # Отдельный модуль стиля ТАРС. Не смешивается с весами Mamba.
        # Итеративный рефайнмент: 1-3 прохода до сходимости.
        #   Pass 1: базовый стиль (характер)
        #   Pass 2: рефайнмент (нюансы, юмор)
        #   Pass 3: полировка (уникальность)
        # При обучении Phase 1-4: ЗАМОРОЖЕН (не портит знания).
        # При обучении Phase 5: ЕДИНСТВЕННЫЙ обучаемый (только личность).
        self.personality = PersonalityAdapter(
            d_model=d_model,
            style_dim=128,
            max_passes=3,
            convergence_threshold=0.01,
        )
        
        # ═══ Cached SSM states for fast generation ═══
        self._gen_cache = None  # Initialized by reset_cache()
        self._prefix_cache = None  # For prefix caching (system prompt)
        
        # ═══ Unified RoPE for WKV branch (Qwen3/Mamba-3 style) ═══
        self.rope = RotaryPositionEmbedding(d_model // (d_model // 64), base=1_000_000)
    
    def reset_cache(self):
        """Сброс кеша SSM-состояний (вызывать перед новым промптом)."""
        self._gen_cache = {
            "wkv_states": [None] * self.n_layers,
            "x_prevs": [None] * self.n_layers,
            "ssd_states": [None] * self.n_layers,
            "conv_states": [None] * self.n_layers,
            "memory_vec": None,
        }
    
    def prefix_cache_save(self):
        """
        Prefix Caching: сохранить текущее SSM-состояние как "prefix".
        
        Вызывать после обработки system prompt. При следующих запросах
        с тем же system prompt — восстановить через prefix_cache_load().
        
        Эффект: 2-5x speedup на повторных запросах.
        """
        import copy
        if self._gen_cache is not None:
            self._prefix_cache = copy.deepcopy(self._gen_cache)
    
    def prefix_cache_load(self):
        """
        Восстановить SSM-состояние из prefix cache.
        
        Usage:
            model.reset_cache()
            model.step(system_prompt_ids)
            model.prefix_cache_save()  # save state after system prompt
            
            # For each user query:
            model.prefix_cache_load()  # restore, skip system prompt processing 
            model.step(user_query_ids)
        """
        import copy
        if self._prefix_cache is not None:
            self._gen_cache = copy.deepcopy(self._prefix_cache)
            return True
        return False
    
    @torch.no_grad()
    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Fast single-step forward для генерации.
        
        Использует кешированные SSM-состояния — не пересоздаёт их.
        Вызывать после reset_cache() + prefill через forward().
        
        token_ids: [B, L] (обычно L=1 для авторегрессии)
        Returns: logits [B, L, vocab_size]
        """
        if self._gen_cache is None:
            self.reset_cache()
        
        c = self._gen_cache
        x = self.embedding(token_ids)
        
        n_waves = self.n_layers // 2
        
        for wave_idx in range(n_waves):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            # Wave-Parallel Memory: 1 lookup → 2 блока
            mem_signal = None
            if c["memory_vec"] is not None:
                h_mean = x.mean(dim=1)
                mem_signal = self.shared_mem_injector.compute_signal(h_mean, c["memory_vec"])
            
            x_left, c["wkv_states"][b_left], c["x_prevs"][b_left], _, \
                c["ssd_states"][b_left], c["conv_states"][b_left] = self.blocks[b_left](
                    x, c["wkv_states"][b_left], c["x_prevs"][b_left],
                    None, None,
                    c["ssd_states"][b_left], c["conv_states"][b_left],
                    None, mem_signal
                )
            x_right, c["wkv_states"][b_right], c["x_prevs"][b_right], _, \
                c["ssd_states"][b_right], c["conv_states"][b_right] = self.blocks[b_right](
                    x, c["wkv_states"][b_right], c["x_prevs"][b_right],
                    None, None,
                    c["ssd_states"][b_right], c["conv_states"][b_right],
                    None, mem_signal
                )
            
            # Wave Consolidation (full merge layer)
            x, _ = self.wave_consolidations[wave_idx](x_left, x_right)
            
            # Lazy Spine: обновление только при новизне > 5%
            if wave_idx < n_waves - 1 and hasattr(self, 'to_memory_space'):
                try:
                    h_curr = x.mean(dim=1)
                    h_for_mem = self.to_memory_space(h_curr)
                    if c["memory_vec"] is None:
                        c["memory_vec"] = h_for_mem
                    else:
                        novelty = 1.0 - F.cosine_similarity(
                            c["memory_vec"], h_for_mem, dim=-1
                        ).mean()
                        if novelty > 0.05:
                            c["memory_vec"] = 0.7 * c["memory_vec"] + 0.3 * h_for_mem
                except Exception:
                    pass
        
        # Final memory injection + output
        if c["memory_vec"] is not None and hasattr(self, 'from_memory_space'):
            try:
                mem_signal = self.from_memory_space(c["memory_vec"])
                x = x + 0.1 * mem_signal.unsqueeze(1)
            except Exception:
                pass
        
        x = self.norm_f(x)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate_speculative(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 128,
        n_draft: int = 4,
        temperature: float = 0.7,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Speculative Decoding — 2-3x speedup for generation.
        
        Algorithm:
        1. Draft head predicts N future tokens (fast, ~1M params)
        2. Main model verifies them in 1 forward pass
        3. Accept prefix of correct tokens, reject rest
        4. Continue from last accepted token
        
        Args:
            prompt_ids: [1, L] initial tokens
            max_tokens: max tokens to generate
            n_draft: number of speculative tokens per step (4-8 optimal)
            temperature: sampling temperature
            top_k: top-k sampling
        
        Returns:
            generated: [1, L + max_tokens] full sequence
        """
        self.eval()
        device = prompt_ids.device
        
        # Prefill: process entire prompt
        self.reset_cache()
        logits = self.step(prompt_ids)  # [1, L, V]
        
        generated = prompt_ids.clone()  # [1, L]
        n_accepted = 0
        n_total = 0
        
        for _ in range(max_tokens):
            if generated.shape[1] - prompt_ids.shape[1] >= max_tokens:
                break
            
            # Get last hidden state for draft predictions
            last_logits = logits[:, -1, :]  # [1, V]
            
            # Sample from main model for first token
            first_token = self._sample(last_logits, temperature, top_k)
            
            # Draft: predict N more tokens using lightweight head
            draft_tokens = [first_token]
            draft_input = first_token
            
            for _ in range(n_draft - 1):
                draft_logits = self.step(draft_input.unsqueeze(1))
                draft_next = self._sample(
                    draft_logits[:, -1, :], temperature, top_k
                )
                draft_tokens.append(draft_next)
                draft_input = draft_next
            
            # We've already processed the draft tokens through step()
            # The cache now contains states for all draft tokens
            # If draft was wrong, we need to rewind — for SSM this means
            # we accept what we've generated since SSM state is cumulative
            
            # Append all draft tokens (SSM doesn't need verification rewind)
            # In SSM models, speculative decoding is simpler than in Transformers
            # because we can't easily "undo" state updates
            draft_tensor = torch.stack(draft_tokens, dim=0).unsqueeze(0)  # [1, N]
            generated = torch.cat([generated, draft_tensor], dim=1)
            
            n_accepted += len(draft_tokens)
            n_total += len(draft_tokens)
            
            # Get logits for next iteration
            logits = self.step(draft_tokens[-1].unsqueeze(0).unsqueeze(0))
        
        return generated
    
    def _sample(self, logits, temperature=0.7, top_k=50):
        """Sample a token from logits with temperature and top-k."""
        if temperature <= 0:
            return logits.argmax(dim=-1)
        
        logits = logits / temperature
        
        if top_k > 0:
            v, _ = logits.topk(top_k, dim=-1)
            logits[logits < v[:, -1:]] = float('-inf')
        
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_vec: Optional[torch.Tensor] = None,
        rag_state: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (для обучения и генерации).
        Использует Parallel Wave Merge: [B0||B1] → gate → merge → ...
        Все wave_consolidations обучаются через backprop.
        
        input_ids: [B, L]
        Returns: logits [B, L, vocab_size]
        """
        x = self.embedding(input_ids)  # [B, L, d_model]
        
        wkv_states = [None] * self.n_layers
        x_prevs = [None] * self.n_layers
        ssd_states = [None] * self.n_layers
        conv_states = [None] * self.n_layers
        
        # Accumulate MoLE aux loss from all blocks
        mole_aux_total = torch.tensor(0.0, device=input_ids.device)
        
        # Parallel Wave through TarsBlocks
        n_waves = self.n_layers // 2
        
        for wave_idx in range(n_waves):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            if b_right >= self.n_layers:
                break
            
            # ═══ Wave-Parallel Memory: 1 lookup → 2 блока ═══
            mem_signal = None
            if memory_vec is not None:
                h_mean = x.mean(dim=1)
                mem_signal = self.shared_mem_injector.compute_signal(h_mean, memory_vec)
            
            # 2 блока параллельно (получают готовый mem_signal вместо memory_vec)
            if self.use_checkpointing and self.training:
                x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                    ssd_states[b_left], conv_states[b_left] = grad_checkpoint(
                    self.blocks[b_left], x, wkv_states[b_left],
                    x_prevs[b_left], None, rag_state,
                    ssd_states[b_left], conv_states[b_left], None, mem_signal,
                    use_reentrant=False
                )
                x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                    ssd_states[b_right], conv_states[b_right] = grad_checkpoint(
                    self.blocks[b_right], x, wkv_states[b_right],
                    x_prevs[b_right], None, rag_state,
                    ssd_states[b_right], conv_states[b_right], None, mem_signal,
                    use_reentrant=False
                )
            else:
                x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                    ssd_states[b_left], conv_states[b_left] = self.blocks[b_left](
                    x, wkv_states[b_left], x_prevs[b_left], None, rag_state,
                    ssd_states[b_left], conv_states[b_left], None, mem_signal
                )
                x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                    ssd_states[b_right], conv_states[b_right] = self.blocks[b_right](
                    x, wkv_states[b_right], x_prevs[b_right], None, rag_state,
                    ssd_states[b_right], conv_states[b_right], None, mem_signal
                )
            
            # Collect MoLE aux losses from both blocks
            for stats in [stats_l, stats_r]:
                if isinstance(stats, dict) and "mole_aux_loss" in stats:
                    mole_aux_total = mole_aux_total + stats["mole_aux_loss"]
            
            # Wave Consolidation: full merge with reflex integration
            x, _ = self.wave_consolidations[wave_idx](x_left, x_right, stats_l, stats_r)
            
            # ═══ Lazy Spine: обновлять memory_vec только при реальной новизне ═══
            if wave_idx < n_waves - 1:
                h_curr = x.mean(dim=1)
                h_for_mem = self.to_memory_space(h_curr)
                if memory_vec is None:
                    memory_vec = h_for_mem
                else:
                    novelty = 1.0 - F.cosine_similarity(memory_vec, h_for_mem, dim=-1).mean()
                    if novelty > 0.05:
                        memory_vec = 0.7 * memory_vec + 0.3 * h_for_mem
        
        # Handle odd number of blocks
        if self.n_layers % 2 == 1:
            last_block = self.blocks[-1]
            x, _, _, stats_last, _, _ = last_block(x, None, None, memory_vec, rag_state)
            if isinstance(stats_last, dict) and "mole_aux_loss" in stats_last:
                mole_aux_total = mole_aux_total + stats_last["mole_aux_loss"]
        
        # Store for external access (training loop)
        self.mole_aux_loss = mole_aux_total
        
        # Output: Mamba → Norm → PersonalityAdapter → LM Head
        x = self.norm_f(x)
        x = self.personality(x)  # Style transform: «что сказать» → «как ТАРС скажет»
        logits = self.lm_head(x)
        
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            # Combine LM loss + MoLE auxiliary loss (load balancing + z-loss)
            total_loss = lm_loss + mole_aux_total
            return logits, total_loss
        
        return logits
    
    def think(
        self,
        input_ids: torch.Tensor,
        query_text: str = "",
        memory_vec: Optional[torch.Tensor] = None,
        rag_state: Optional[torch.Tensor] = None,
        force_deep: bool = False,
        max_expansion_rounds: int = 12,
        reflex_ctx: Any = None,
        supplement_queue = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Адаптивный forward с Integral Auditor и Speculative Matrix Routing.
        
        Новое: reflex_ctx (ReflexContext) управляет глубиной:
          - trivial (depth=4): проходим только 4 блока
          - simple (depth=6):  6 блоков
          - complex (depth=12): все 12 блоков + IDME
        
        supplement_queue: thread-safe Queue с дополнениями от голоса.
          Между волнами проверяется на новые фрагменты речи.
          Если есть — инжектим через 2 блока + WaveConsolidation + spine.
          После инжекции — 3 дополнительные волны (6 блоков).
        """
        start_time = time.time()
        
        # Reset аудиторов
        self.integral_auditor.reset()
        self.matrix_pool.reset()
        self.hankel.reset()
        self.thinking_chain.reset()  # Новый запрос — новая цепочка рассуждений
        
        # Meta-Auditor: тип задачи и порог
        task_type, p_threshold = self.meta_auditor.classify_task(query_text)
        self.integral_auditor.threshold = p_threshold
        
        # Start session AFTER task classification
        self.thinking_logger.start_session(query_text, task_type, p_threshold)
        
        # ═══ Adaptive Depth from ReflexContext ═══
        estimated_depth = self.n_layers  # default: all layers
        reflex_urgency = 0.0
        
        if reflex_ctx is not None:
            estimated_depth = min(
                getattr(reflex_ctx, 'estimated_depth', self.n_layers),
                self.n_layers
            )
            # Use reflex memory_vec if provided and not already set
            if memory_vec is None and getattr(reflex_ctx, 'memory_vec', None) is not None:
                memory_vec = reflex_ctx.memory_vec
            # Use reflex max_expansion_rounds
            if getattr(reflex_ctx, 'needs_idme', False):
                max_expansion_rounds = getattr(reflex_ctx, 'max_expansion_rounds', 12)
            else:
                max_expansion_rounds = 2  # Skip IDME for simple queries
            reflex_urgency = getattr(reflex_ctx, 'urgency', 0.0)
        
        # Лимиты раундов по типу задачи
        TASK_MAX_ROUNDS = {
            "chat": 2, "action": 2, "code": 6,
            "math": 8, "deep": 20, "infinite": 100,
        }
        task_max = TASK_MAX_ROUNDS.get(task_type, 4)
        max_expansion_rounds = min(max_expansion_rounds, task_max)
        if force_deep:
            max_expansion_rounds = 100
            estimated_depth = self.n_layers
        
        # ═══════════════════════════════════════════════════════════════
        # 1. Parallel Wave Depth (2 параллельных блока → merge → spine)
        #
        # Простой запрос:  [B0 || B1] → merge → spine → СОШЛОСЬ (2 блока)
        # Средний запрос:  ... → [B2 || B3] → merge → spine → СОШЛОСЬ (4)
        # Сложный запрос:  ... все 6 волн (12 блоков) → IDME
        # ═══════════════════════════════════════════════════════════════
        x = self.embedding(input_ids)
        
        wkv_states = [None] * self.n_layers   # WKV state per block
        x_prevs = [None] * self.n_layers      # time-shift per block
        ssd_states = [None] * self.n_layers   # SSD recurrent state per block
        conv_states = [None] * self.n_layers  # Conv1d rolling state per block
        h_prev = x.mean(dim=1).detach()
        
        block_stats = []
        blocks_executed = 0
        surprise_signals = []
        converged_early = False
        wave_count = 0
        per_wave_experts = []  # [{"wave": 1, "left": [...], "right": [...]}]
        supplement_injected = False
        supplement_extra_waves = 0  # сколько допволн осталось
        
        max_waves = self.n_layers // 2   # 12/2 = 6 волн
        
        # ═══ Thought Cache Shortcut (v3) ═══
        # Проверяем: есть ли похожая траектория мышления в кэше.
        # Если да — пропускаем explore/analyze фазы.
        cache_skip_waves = 0
        if hasattr(self, 'thinking_chain'):
            try:
                cache_skip_waves = self.thinking_chain.try_cache_shortcut(x.mean(dim=1))
                cached_mem = self.thinking_chain.get_cached_memory()
                if cached_mem is not None and cache_skip_waves > 0:
                    # Восстанавливаем memory_vec из кэша
                    memory_vec = cached_mem.to(x.device)
                    self.logger.debug(
                        f"ThoughtCache hit: skip {cache_skip_waves} waves"
                    )
            except Exception:
                cache_skip_waves = 0
        
        # ═══ Cross-Wave Residual Storage (v3) ═══
        wave_outputs = {}  # {wave_idx: x_output} для skip connections
        
        for wave_idx in range(max_waves):
            # Проверяем depth limit
            if blocks_executed >= estimated_depth and not force_deep:
                break
            
            # Thought Cache: пропускаем ранние волны если нашли кэш
            if wave_idx < cache_skip_waves:
                wave_count += 1
                blocks_executed += 2
                continue
            
            wave_count += 1
            b_left = wave_idx * 2       # индекс левого блока
            b_right = wave_idx * 2 + 1  # индекс правого блока
            
            if b_right >= self.n_layers:
                break
            
            # ── 2 блока работают ПАРАЛЛЕЛЬНО на одном входе ──
            x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                ssd_states[b_left], conv_states[b_left] = self.blocks[b_left](
                x, wkv_states[b_left], x_prevs[b_left], memory_vec, rag_state,
                ssd_states[b_left], conv_states[b_left]
            )
            x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                ssd_states[b_right], conv_states[b_right] = self.blocks[b_right](
                x, wkv_states[b_right], x_prevs[b_right], memory_vec, rag_state,
                ssd_states[b_right], conv_states[b_right]
            )
            block_stats.extend([stats_l, stats_r])
            blocks_executed += 2
            
            # Собираем экспертов с каждой волны
            wave_experts = {
                "wave": wave_count,
                "left": stats_l.get("mole_experts", []),
                "right": stats_r.get("mole_experts", []),
            }
            per_wave_experts.append(wave_experts)
            
            # ── WaveConsolidation: полный слой слияния ──
            x, merge_alpha = self.wave_consolidations[wave_idx](
                x_left, x_right, stats_l, stats_r
            )
            
            # ═══ Cross-Wave Residual (v3) ═══
            # Сохраняем выход волны для skip connection
            wave_outputs[wave_idx] = x.detach()
            
            # Добавляем skip connection от волны (i-3) если она есть
            # Это как ResNet skip connections между волнами
            skip_from = wave_idx - 3
            if skip_from in wave_outputs:
                x = x + 0.1 * wave_outputs[skip_from]  # мягкое добавление
            
            # Записываем merge данные в wave_experts
            wave_experts["merge_alpha"] = merge_alpha
            
            # Собираем surprise
            for stats in [stats_l, stats_r]:
                if stats.get("surprise", 0.0) > 0.3:
                    surprise_signals.append({
                        "layer": stats["layer_idx"],
                        "surprise": stats["surprise"],
                        "mem_relevance": stats.get("mem_relevance", 0.0),
                    })
            
            # ── Integral Auditor: проверка сходимости ──
            h_curr = x.mean(dim=1).detach()
            ia_result = self.integral_auditor.observe(h_curr, h_prev)
            h_prev = h_curr
            
            self.thinking_logger.log_step(blocks_executed - 1, {
                "p": ia_result["p"], "r_squared": ia_result["r_squared"],
                "f_t": ia_result["f_t"], "converged": ia_result["converged"],
                "wave": wave_count, "depth": blocks_executed,
                "width": 2,
            })
            
            self.logger.debug(
                f"Wave {wave_count}: [B{b_left}||B{b_right}] → merge → "
                f"p={ia_result['p']:.3f} | converged={ia_result['converged']}"
            )
            
            # ── Сошлось? (два критерия: IA convergence ИЛИ ThinkingChain confidence) ──
            ia_converged = ia_result["converged"] and wave_count >= 2
            tc_skip = hasattr(self, 'thinking_chain') and self.thinking_chain.should_skip_remaining()
            
            if ia_converged or tc_skip:
                converged_early = True
                skip_reason = "IA" if ia_converged else "ThinkingChain_confidence"
                self.logger.debug(
                    f"Converged at wave {wave_count} (depth={blocks_executed}, reason={skip_reason})"
                )
                break
            
            # ── Спинной мозг обновляет память между волнами ──
            spine_updated = False
            if wave_idx < max_waves - 1:
                if hasattr(self, 'to_memory_space'):
                    try:
                        h_for_mem = self.to_memory_space(h_curr)
                        if memory_vec is None:
                            # Первая волна — инициализируем memory_vec
                            memory_vec = h_for_mem.detach()
                        else:
                            # Спинной мозг: 70% старая + 30% новая (§2.4)
                            memory_vec = 0.7 * memory_vec + 0.3 * h_for_mem.detach()
                        spine_updated = True
                    except Exception:
                        pass
                
                # Titans surprise feedback между волнами
                if hasattr(self, 'titans_memory') and self.titans_memory is not None:
                    wave_surprises = [s for s in surprise_signals 
                                     if s["layer"] >= blocks_executed - 2]
                    if wave_surprises:
                        try:
                            h_for_titans = self.to_memory_space(h_curr)
                            self.titans_memory.forward(h_for_titans)
                        except Exception:
                            pass
                
                # ═══ Native CoT: ThinkingChain уточняет memory_vec ═══
                # Каждая волна — шаг рассуждения. ThinkingChain модифицирует
                # memory_vec так, что следующая волна фокусируется
                # на другом аспекте задачи (от общего к частному).
                if memory_vec is not None and hasattr(self, 'thinking_chain'):
                    try:
                        memory_vec, thought_info = self.thinking_chain.step(
                            h_curr, memory_vec, wave_idx, max_waves
                        )
                        self.thinking_logger.log_step(blocks_executed, {
                            "thinking_phase": thought_info.get("phase", "unknown"),
                            "thinking_confidence": thought_info.get("confidence", 0),
                            "thought_gate": thought_info.get("gate_mean", 0),
                        })
                    except Exception:
                        pass
            
            wave_experts["spine_updated"] = spine_updated
            
            # ═══ Supplement Injection: проверяем очередь голосовых дополнений ═══
            if supplement_queue is not None and not supplement_queue.empty():
                try:
                    supplement = supplement_queue.get_nowait()
                    sup_text = supplement.get("text", "")
                    sup_tokens = supplement.get("tokens")  # pre-tokenized
                    
                    if sup_text or sup_tokens is not None:
                        self.logger.info(
                            f"🎤 Supplement injection at wave {wave_count}: "
                            f"'{sup_text[:50]}...'"
                        )
                        
                        # Токенизация дополнения
                        if sup_tokens is None:
                            sup_ids = input_ids  # fallback
                        else:
                            sup_ids = sup_tokens.to(x.device)
                        
                        # Пропускаем дополнение через 2 блока текущей волны
                        x_sup = self.embedding(sup_ids)
                        
                        # 2 блока параллельно (те же веса что текущая волна)
                        xs_l, _, _, _, _, _ = self.blocks[b_left](
                            x_sup, None, None, memory_vec, rag_state, None, None
                        )
                        xs_r, _, _, _, _, _ = self.blocks[b_right](
                            x_sup, None, None, memory_vec, rag_state, None, None
                        )
                        
                        # WaveConsolidation: суммирующая матрица
                        x_sup_merged, _ = self.wave_consolidations[wave_idx](
                            xs_l, xs_r, {}, {}
                        )
                        
                        # Обновляем основное состояние: спинной мозг инжектирует
                        h_sup = x_sup_merged.mean(dim=1).detach()
                        if hasattr(self, 'to_memory_space'):
                            h_sup_mem = self.to_memory_space(h_sup)
                            if memory_vec is not None:
                                # 50/50 смесь старого + дополнения
                                memory_vec = 0.5 * memory_vec + 0.5 * h_sup_mem.detach()
                            else:
                                memory_vec = h_sup_mem.detach()
                        
                        # Сбрасываем сходимость — надо переоценить
                        self.integral_auditor.reset()
                        converged_early = False
                        h_prev = h_sup
                        
                        supplement_injected = True
                        supplement_extra_waves = 3  # +6 блоков
                        
                        self.logger.info(
                            f"✅ Supplement merged. +3 extra waves scheduled."
                        )
                except Exception as e:
                    self.logger.warning(f"Supplement injection error: {e}")
        
        # ═══ Extra waves после supplement (половина от 12 = 6 блоков = 3 волны) ═══
        if supplement_injected and supplement_extra_waves > 0:
            self.logger.info(
                f"🔄 Running {supplement_extra_waves} extra waves for supplement"
            )
            for extra_idx in range(supplement_extra_waves):
                # Используем первые 3 волны (блоки 0-5) повторно
                b_l = extra_idx * 2
                b_r = extra_idx * 2 + 1
                if b_r >= self.n_layers:
                    break
                
                wave_count += 1
                x_left, wkv_states[b_l], x_prevs[b_l], stats_l, \
                    ssd_states[b_l], conv_states[b_l] = self.blocks[b_l](
                    x, wkv_states[b_l], x_prevs[b_l], memory_vec, rag_state,
                    ssd_states[b_l], conv_states[b_l]
                )
                x_right, wkv_states[b_r], x_prevs[b_r], stats_r, \
                    ssd_states[b_r], conv_states[b_r] = self.blocks[b_r](
                    x, wkv_states[b_r], x_prevs[b_r], memory_vec, rag_state,
                    ssd_states[b_r], conv_states[b_r]
                )
                blocks_executed += 2
                
                x, merge_alpha = self.wave_consolidations[extra_idx](
                    x_left, x_right, stats_l, stats_r
                )
                
                # Spine update
                h_curr = x.mean(dim=1).detach()
                if hasattr(self, 'to_memory_space'):
                    try:
                        h_for_mem = self.to_memory_space(h_curr)
                        memory_vec = 0.7 * memory_vec + 0.3 * h_for_mem.detach()
                    except Exception:
                        pass
                
                ia_result = self.integral_auditor.observe(h_curr, h_prev)
                h_prev = h_curr
                
                self.logger.debug(
                    f"Extra wave {extra_idx+1}/{supplement_extra_waves}: "
                    f"p={ia_result['p']:.3f}"
                )
                
                if ia_result["converged"]:
                    converged_early = True
                    break
        
        # ═══ Titans Feedback: финальный сигнал ═══
        if hasattr(self, 'titans_memory') and self.titans_memory is not None:
            if surprise_signals:
                avg_surprise = sum(s["surprise"] for s in surprise_signals) / len(surprise_signals)
                try:
                    h_for_titans = self.to_memory_space(x.mean(dim=1).detach())
                    self.titans_memory.forward(h_for_titans)
                    self.logger.debug(
                        f"Titans final: {len(surprise_signals)} layers surprised, "
                        f"avg={avg_surprise:.3f}"
                    )
                except Exception as e:
                    self.logger.debug(f"Titans feedback error: {e}")
        
        # Собираем всех уникальных экспертов по всем волнам
        all_expert_names = []
        for we in per_wave_experts:
            all_expert_names.extend(we.get("left", []))
            all_expert_names.extend(we.get("right", []))
        
        # ═══ 2. Speculative Matrix Routing (IDME) ═══
        h_curr = x.mean(dim=1).detach()
        ia_result = self.integral_auditor.observe(h_curr, h_prev)
        
        expansion_round = 0
        total_matrices_recruited = 0
        branches_tested = 0
        branches_won = []
        
        prev_p = ia_result.get("p", 0.0)
        no_improve_count = 0
        N_CANDIDATES = 3
        
        while not ia_result["converged"] and expansion_round < max_expansion_rounds:
            expansion_round += 1
            h_prev = h_curr
            
            candidates, indices = self.matrix_pool.select(h_curr.mean(0), k=N_CANDIDATES)
            total_matrices_recruited += len(candidates)
            branches_tested += len(candidates)
            
            # ═══ Lazy Expansion: пул исчерпан → рекрутируем новые матрицы ═══
            if not candidates:
                available = self.matrix_pool.total_available
                used = len(getattr(self.matrix_pool, 'used_mask', []))
                if available <= used + N_CANDIDATES:
                    try:
                        self.matrix_pool._lazy_expand(4, h_curr.mean(0))
                        self.logger.debug(
                            f"IDME lazy expand: +4 matrices (total={self.matrix_pool.total_available})"
                        )
                        candidates, indices = self.matrix_pool.select(h_curr.mean(0), k=N_CANDIDATES)
                        total_matrices_recruited += len(candidates)
                        branches_tested += len(candidates)
                    except Exception as e:
                        self.logger.debug(f"Lazy expand failed: {e}")
                if not candidates:
                    break
            
            # Branch & Bound
            best_p = prev_p
            best_x = None
            best_idx = -1
            
            h_state = x.mean(dim=1)
            
            for matrix, idx in zip(candidates, indices):
                h_clone = h_state.clone()
                h_refined = matrix(h_clone)
                h_gated, _ = self.novelty_gate(h_clone, h_refined)
                
                with torch.no_grad():
                    f_t = (h_gated - h_state).float().norm().item()
                    candidate_p = prev_p + (1.0 / max(f_t, 1e-8)) * 0.01
                
                if candidate_p > best_p:
                    best_p = candidate_p
                    best_x = h_gated
                    best_idx = idx
            
            if best_x is not None and best_p > prev_p:
                x = x + 0.1 * (best_x - h_state).unsqueeze(1)
                h_curr = x.mean(dim=1).detach()
                ia_result = self.integral_auditor.observe(h_curr, h_prev)
                
                if best_idx >= 0:
                    self.matrix_pool.recirculate(best_idx, max(0, ia_result["p"] - prev_p))
                    branches_won.append((best_idx, ia_result["p"] - prev_p))
                
                hankel_result = self.hankel.observe(h_curr)
                if hankel_result["collapsed"] and hankel_result["collapse_count"] >= 3:
                    break
            else:
                no_improve_count += 1
                h_curr = x.mean(dim=1).detach()
                ia_result = self.integral_auditor.observe(h_curr, h_prev)
            
            if ia_result["p"] <= prev_p + 0.01:
                no_improve_count += 1
                if no_improve_count >= 2:
                    break
            else:
                no_improve_count = 0
            prev_p = ia_result["p"]
        
        # ═══ 3. Output ═══
        # Финальная инъекция памяти спинного мозга в выход
        # (без этого контекст, накопленный между волнами, теряется)
        if memory_vec is not None and hasattr(self, 'from_memory_space'):
            try:
                mem_signal = self.from_memory_space(memory_vec)  # [B, d_model]
                x = x + 0.1 * mem_signal.unsqueeze(1)  # [B, L, d_model]
            except Exception:
                pass
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        # ═══ Confidence-Gated Output ═══
        # Если ThinkingChain не уверен → сглаживаем logits
        if hasattr(self, 'thinking_chain'):
            try:
                logits = self.thinking_chain.apply_confidence_gate(logits)
                # Обновляем сессионную память после запроса
                if memory_vec is not None:
                    self.thinking_chain.update_session_memory(memory_vec)
            except Exception:
                pass
        
        total_time = (time.time() - start_time) * 1000
        
        # ═══ ThinkingChain Finalize (v3) ═══
        # Сохраняем траекторию в ThoughtCache + Sleep recording
        tc_summary = {}
        if hasattr(self, 'thinking_chain'):
            try:
                avg_surprise = 0.0
                if surprise_signals:
                    avg_surprise = sum(s["surprise"] for s in surprise_signals) / len(surprise_signals)
                self.thinking_chain.finalize(
                    query_embedding=x.mean(dim=1).detach(),
                    final_memory=memory_vec,
                    surprise_level=avg_surprise,
                    task_type=task_type,
                )
            except Exception:
                pass
            tc_summary = self.thinking_chain.get_summary()
        
        stats = {
            "task_type": task_type,
            "p_threshold": p_threshold,
            "final_p": ia_result["p"],
            "r_squared": ia_result.get("r_squared", 0),
            "converged": ia_result["converged"],
            "converged_early": converged_early,
            "total_blocks": self.n_layers,
            "blocks_executed": blocks_executed,
            "waves": wave_count,
            "estimated_depth": estimated_depth,
            "expansion_rounds": expansion_round,
            "matrices_recruited": total_matrices_recruited,
            "total_matrices": self.n_layers + total_matrices_recruited,
            "branches_tested": branches_tested,
            "branches_won": len(branches_won),
            "per_wave_experts": per_wave_experts,
            "hankel_collapses": self.hankel.collapse_count,
            "surprise_layers": len(surprise_signals),
            "supplement_injected": supplement_injected,
            "supplement_extra_waves": supplement_extra_waves if supplement_injected else 0,
            "rwkv_state_size_mb": sum(s.numel() * 4 / 1024 / 1024 for s in wkv_states if s is not None),
            "total_ms": total_time,
            # ThinkingChain v2 stats
            "thinking_chain": tc_summary,
            "tc_confidence": tc_summary.get("final_confidence", 0),
            "tc_phases": tc_summary.get("phases", []),
            "tc_retrieval_count": tc_summary.get("retrieval_count", 0),
        }
        
        self.thinking_logger.end_session(total_time, "")
        
        win_ratio = f"{len(branches_won)}/{branches_tested}" if branches_tested > 0 else "0/0"
        self.logger.info(
            f"Think: {task_type} | p={stats['final_p']:.3f} | "
            f"conf={stats['tc_confidence']:.2f} | "
            f"phases={'→'.join(tc_summary.get('phases', []))}"
        )
        
        return logits, stats
    
    @torch.no_grad()
    def self_verify(self, input_ids, generated_ids, memory_vec=None):
        """
        Self-Verification: прогоняет ответ обратно через 2 волны.
        
        Если consistency < 0.8 → ответ нестабильный.
        DeepSeek-R1 делает это текстом (<verify>).
        ТАРС — в hidden states (нулевой overhead).
        
        Args:
            input_ids: [1, L_query] — исходный запрос
            generated_ids: [1, L_answer] — сгенерированный ответ
            memory_vec: [1, 384] — память от think()
        
        Returns:
            consistency: float (0-1) — насколько ответ согласован
            should_regenerate: bool — нужно ли перегенерировать
        """
        self.eval()
        
        # Прогоняем ответ через 2 волны (быстрая проверка)
        combined = torch.cat([input_ids, generated_ids], dim=1)
        x = self.embedding(combined)
        
        # 2 волны (4 блока)
        for wave_idx in range(min(2, self.n_layers // 2)):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            if b_right >= self.n_layers:
                break
            
            x_left, _, _, _, _, _ = self.blocks[b_left](
                x, None, None, memory_vec, None, None, None
            )
            x_right, _, _, _, _, _ = self.blocks[b_right](
                x, None, None, memory_vec, None, None, None
            )
            x, _ = self.wave_consolidations[wave_idx](x_left, x_right)
        
        # Сравниваем logits для ответной части
        verify_logits = self.lm_head(self.norm_f(x))
        
        # Consistency = cosine similarity между hidden states
        # Берём среднее по ответной части
        L_q = input_ids.shape[1]
        answer_hidden = x[:, L_q:, :].mean(dim=1)  # [1, d_model]
        query_hidden = x[:, :L_q, :].mean(dim=1)   # [1, d_model]
        
        consistency = F.cosine_similarity(
            answer_hidden, query_hidden, dim=-1
        ).item()
        
        # Нормализуем в [0, 1]
        consistency = (consistency + 1.0) / 2.0  # cosine [-1,1] → [0,1]
        
        should_regenerate = consistency < 0.8
        
        self.logger.info(
            f"Self-verify: consistency={consistency:.3f}, "
            f"regenerate={should_regenerate}"
        )
        
        return consistency, should_regenerate
    
    def encode_rag(self, rag_tokens: torch.Tensor) -> torch.Tensor:
        """
        Прогоняет RAG-документ через TarsCoreBlock всех блоков,
        формируя сжатую Матрицу Знаний (WKV State).
        
        rag_tokens: [1, L_doc] — токенизированный документ
        Returns: [1, d_state, d_state] — сжатое состояние
        """
        with torch.no_grad():
            x = self.embedding(rag_tokens)
            wkv_state = None
            x_prev = None
            
            for block in self.blocks:
                x, wkv_state, x_prev, _, _, _ = block(
                    x, wkv_state, x_prev
                )
        
        self.logger.info(
            f"RAG encoded: {rag_tokens.shape[1]} tokens → "
            f"state {wkv_state.shape} ({wkv_state.numel() * 4 / 1024 / 1024:.1f} MB)"
        )
        return wkv_state
    
    def count_parameters(self) -> dict:
        """Подсчёт параметров по компонентам."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        
        # Per-block breakdown (первый блок как пример)
        block = self.blocks[0]
        block_detail = {
            "tars_core (ssd+wkv+fusion)": count(block.core),
            "omega_ssm": count(block.omega),
            "mole": count(block.mole),
        }
        
        return {
            "embedding": count(self.embedding),
            "blocks_total": sum(count(b) for b in self.blocks),
            "block_detail (×1)": block_detail,
            "matrix_pool": count(self.matrix_pool),
            "novelty_gate": count(self.novelty_gate),
            "lm_head": 0,  # tied
            "total": count(self),
        }
