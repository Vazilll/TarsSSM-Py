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
from brain.doubt_engine import DoubtEngine, load_doubt_engine, DoubtVerdict
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from brain.mamba2.tars_block import TarsBlock, MemoryInjector
from brain.mamba2.integral_auditor import (
    IntegralAuditor, MetaAuditor,
    TemporalEmbedding, MetaCortex,
)
from brain.mamba2.critic import CriticHead, WaveCritic, TaskSpec
from brain.mamba2.matrix_pool import MatrixPool
from brain.mamba2.novelty import NoveltyGate, HankelDetector
from brain.mamba2.logger import ThinkingLogger

from brain.mamba2.thinking_chain import ThinkingChain
from brain.mamba2.personality_adapter import PersonalityAdapter
from brain.mamba2.query_router import QueryRouter, ProgressiveMemoryManager
from brain.mamba2.bitnet import (
    UniversalLinear, convert_model_to_158bit,
    convert_model_to_fp16, model_stats, replace_linear_with_universal,
    RMSNorm,
)

# ══ T07 Decomposition: new composable modules ══
# These provide clean, testable APIs for the pipeline.
# TarsMamba2LM remains the backward-compatible wrapper.
from brain.mamba2.brain_core import (
    BrainCore,
    RotaryPositionEmbedding as _BrainCoreRoPE,
    WaveConsolidation as _BrainCoreWC,
    GlobalWorkspace as _BrainCoreGW,
    SharedGlobalAttention as _BrainCoreSGA,
    WaveScratchpad as _BrainCoreWS,
    TTTLoRA as _BrainCoreTTT,
)
from brain.mamba2.verification_suite import VerificationSuite
from brain.mamba2.inference_engine import InferenceEngine

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RotaryPositionEmbedding(nn.Module):
    """
    Unified RoPE: Rotary Position Embedding.
    
    Uses base=500,000 (TARS default) for up to 32K+ context.
    Applied to WKV branch for positional awareness in SSM-Attention hybrid.
    
    Math: RoPE(x, pos) = x * cos(θ_pos) + rotate_half(x) * sin(θ_pos)
    where θ_i = pos / base^(2i/d)
    
    Lazy compute: starts with small buffer (512), auto-extends on demand.
    """
    
    def __init__(self, dim, base=500_000, max_seq_len=32768):
        super().__init__()
        self.dim = dim
        self.base = base
        self._max_seq_len_config = max_seq_len  # absolute cap
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Lazy: start small, extend on demand
        initial_len = min(512, max_seq_len)
        self._build_cache(initial_len)
    
    def _build_cache(self, seq_len: int):
        """Build cos/sin cache up to seq_len."""
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def _extend(self, needed_len: int):
        """Extend cache to cover needed_len (2x growth factor)."""
        new_len = max(needed_len, self.max_seq_len * 2)
        new_len = min(new_len, self._max_seq_len_config)
        if new_len <= self.max_seq_len:
            return  # Already big enough or at max
        self._build_cache(new_len)
    
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
        
        # Lazy extend if needed
        needed = offset + seq_len
        if needed > self.max_seq_len:
            self._extend(needed)
        
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
    WuNeng Fusion — информационное горлышко (HELIX v6).
    
    Архитектура:
      1. Dimension-wise Gate: σ(W · [h_L; h_R]) ∈ (0,1)^d
      2. WuNeng Bottleneck: [h_L; h_R] → 2d → 192 → GELU → d
      3. Reflex integration: сигнал от MoLE stats обоих блоков
      4. Output: (1-gate) * x_left + gate * x_right + fusion + reflex
    """
    
    BOTTLENECK_DIM = 192  # HELIX v6: информационное горлышко
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # 1. Dimension-wise gate (полный, не скаляр)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # 2. WuNeng bottleneck: 2d → 192 → GELU → d (per HELIX v6 spec)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, self.BOTTLENECK_DIM),
            nn.GELU(),
            nn.Linear(self.BOTTLENECK_DIM, d_model),
        )
        
        # 3. Reflex integration gate (MoLE expert signals)
        self.reflex_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )
        
        # 4. Output normalization — RMSNorm: 15-20% faster than LayerNorm
        self.norm = RMSNorm(d_model)
        
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
        h_left = x_left.clone().mean(dim=1)   # [B, d_model] — clone for CUDAGraph safety
        h_right = x_right.clone().mean(dim=1)  # [B, d_model]
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
        
        return self.norm(x), alpha.mean().detach()


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
        
        # Norm — RMSNorm: faster than LayerNorm
        self.norm = RMSNorm(d_model)
    
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
        
        # Winner-take-most broadcast (vectorized)
        h_stack = torch.stack(h_means, dim=1)  # [B, n_blocks, d_model]
        broadcast = torch.einsum('bn,bnd->bd', gates, h_stack)  # [B, d_model]
        
        # Project and mix
        broadcast = self.broadcast_proj(broadcast)  # [B, d_model]
        
        return self.norm(x_current + self.mix * broadcast.unsqueeze(1))


class SharedGlobalAttention(nn.Module):
    """
    Zamba-pattern Shared Global Attention.
    
    Одна shared attention на все слои (Zamba: 1 attention на 6 Mamba).
    Экономия ~90% параметров vs per-layer attention.
    
    GQA (Grouped Query Attention) с 4 головами.
    Применяется после каждой N-ой волны.
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, n_kv_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # GQA repetition
        
        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # Gated residual: starts small
        self.gate = nn.Parameter(torch.tensor(0.1))
        
        # BUG-4 fix: learnable temperature for QK-norm (Qwen3-style)
        # After L2-normalization, dot products ∈ [-1, 1]. Static 1/√d kills attention.
        # Learnable log-temperature lets the model control softmax sharpness.
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.0)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        Returns: x + gate * attention_output
        """
        B, L, D = x.shape
        h = self.norm(x)
        
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # GQA expand: repeat KV heads
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        # QK-Norm: L2 normalize Q and K for training stability
        # Prevents attention entropy collapse in deep models (20+ blocks)
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # Scaled dot-product attention with learnable temperature (BUG-4 fix)
        # QK-norm makes dot products ∈ [-1,1], so we use learned temperature
        # instead of fixed 1/√d which would compress values to ≈[-0.125, 0.125]
        scale = torch.exp(self.log_temperature)
        attn = (q @ k.transpose(-2, -1)) * scale
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)
        
        return x + self.gate * out


class WaveScratchpad(nn.Module):
    """
    WaveScratchpad — последовательное мышление между волнами.
    
    Идея: каждая волна — один «шаг мысли». Между волнами модель
    передаёт сжатый итог (scratchpad) что было сделано,
    и следующая волна получает этот контекст.
    
    Это предотвращает попытку «сделать всё сразу в одном блоке».
    Вместо этого:
      Wave 0: «понять задачу» → записать итог
      Wave 1: получить итог → «собрать данные» → записать
      Wave 2: получить итог → «выполнить» → записать
      ...
      Wave N: получить итог → «подвести итоги» → финал
    
    Также поддерживает action_slots: волна может запросить
    выполнение действия (tool call), результат которого
    передаётся следующей волне через scratchpad.
    """
    
    def __init__(self, d_model: int, compress_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.compress_dim = compress_dim or d_model // 4
        
        # Сжатие итога волны → компактный вектор
        self.compress = nn.Linear(d_model, self.compress_dim, bias=False)
        # Расширение для инъекции в следующую волну
        self.expand = nn.Linear(self.compress_dim, d_model, bias=False)
        # Learnable gate: насколько сильно предыдущий итог влияет
        self.inject_gate = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ≈ 0.27
        # Norm для стабильности
        self.norm = nn.LayerNorm(self.compress_dim)
    
    def summarize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Сжать результат волны в компактную «записку».
        
        Args:
            x: [B, L, d_model] — выход волны после consolidation
        
        Returns:
            summary: [B, compress_dim] — сжатый итог
        """
        h_mean = x.mean(dim=1)  # [B, d_model]
        summary = self.norm(self.compress(h_mean))  # [B, compress_dim]
        return summary
    
    def inject(self, x: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        """
        Вставить «записку» от предыдущей волны в текущий поток.
        
        Args:
            x: [B, L, d_model] — вход текущей волны
            summary: [B, compress_dim] — итог предыдущей волны
        
        Returns:
            x_enriched: [B, L, d_model] — вход + контекст предыдущей волны
        """
        hint = self.expand(summary)  # [B, d_model]
        gate = torch.sigmoid(self.inject_gate)
        return x + gate * hint.unsqueeze(1)  # broadcast по L


class TTTLoRA(nn.Module):
    """
    Test-Time Training with LoRA (MIT, 2025).
    
    При inference: делаем 3-5 шагов SGD с LoRA-адаптером на входном тексте,
    затем генерируем ответ с обновлёнными весами, потом откат.
    
    Эффект: до 6x улучшение на reasoning tasks.
    LoRA rank=4, alpha=8 — минимальные вычисления.
    """
    
    def __init__(self, d_model: int, rank: int = 4, alpha: float = 8.0,
                 n_steps: int = 3, lr: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.n_steps = n_steps
        self.lr = lr
        
        # LoRA adapters: A down-projects, B up-projects
        self.lora_A = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_model))
    
    @torch.no_grad()
    def adapt(self, x: torch.Tensor, model_forward_fn):
        """
        Test-Time Training: adapt LoRA weights to input.
        
        Args:
            x: [B, L, d_model] — embedded input
            model_forward_fn: callable that returns logits given x
            
        Returns:
            adapted_delta: [d_model, d_model] — weight delta to apply
        """
        # Save originals
        orig_A = self.lora_A.data.clone()
        orig_B = self.lora_B.data.clone()
        
        # Enable gradients temporarily
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        
        for step in range(self.n_steps):
            # Forward with LoRA delta
            delta = (self.lora_A @ self.lora_B) * self.scale  # [D, D]
            x_adapted = x + F.linear(x, delta)  # apply delta
            
            # Self-supervised loss: predict next token (perplexity reduction)
            with torch.enable_grad():
                logits = model_forward_fn(x_adapted)
                # Shift for next-token prediction
                shift_logits = logits[:, :-1].contiguous()
                shift_targets = logits[:, 1:].argmax(dim=-1)  # use own predictions as targets
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1)
                )
            
            # Manual SGD step on LoRA params
            loss.backward()
            if self.lora_A.grad is not None:
                self.lora_A.data -= self.lr * self.lora_A.grad
                self.lora_A.grad = None
            if self.lora_B.grad is not None:
                self.lora_B.data -= self.lr * self.lora_B.grad
                self.lora_B.grad = None
        
        # Compute final delta
        adapted_delta = (self.lora_A @ self.lora_B) * self.scale
        
        # Rollback LoRA weights
        self.lora_A.data.copy_(orig_A)
        self.lora_B.data.copy_(orig_B)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)
        
        return adapted_delta
    
    def apply_delta(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """
        Apply TTT-LoRA delta to hidden states.
        
        Args:
            x: [B, L, d_model]
            delta: [d_model, d_model]
        Returns:
            x + LoRA(x)
        """
        return x + F.linear(x, delta)


class TarsMamba2LM(nn.Module):
    """
    ТАРС v3: Deep WuNeng Core (Mamba-2 + RWKV-7 inside one kernel).
    
    Гибридный мозг с единым ядром TarsCoreBlock:
    Mamba-2 SSD и RWKV-7 WKV делят общие проекции и сливаются
    в латентном пространстве через Deep Gated Fusion.
    
    ~130M params (fp16: ~260MB, 1.58-bit: ~60MB)
    
    ═══════════════════════════════════════════════════
    АРХИТЕКТУРА (21 шаг forward pass):
    ═══════════════════════════════════════════════════
    
    1.  Embedding(input_ids)           — токены → d_model-вектора
    2.  TemporalEmbedding(x)           — 4 частоты sin/cos (чувство времени)
        ┌─── ВОЛНА × n_layers//2 ───────────────────────┐
    3.  │ WaveScratchpad.inject         — итог прошлой волны
    4.  │ SharedMemInjector             — 1 запрос к памяти → 2 блока
    5.  │ proj_left / proj_right        — разные проекции (разные ракурсы)
        │ ┌── TarsBlock (×2 параллельно) ──────────────┐
    6.  │ │ RMSNorm                     — нормализация
    7.  │ │ TarsCoreBlock:              — ЯДРО (~5.2M/блок)
        │ │   A. time_mix(u, u_shifted)  — контекст соседа
        │ │   B. in_proj(shared)         — split → Mamba + RWKV
        │ │   C. Mamba SSD Path:         — conv1d → ssd_scan + MetaTokens
        │ │   D. RWKV-7 Path:            — GatedDeltaNet β → wkv_scan
        │ │   E. WuNeng Fusion:          — gate·SSD + (1-gate)·RWKV
        │ │   F. SwiGLU + out_proj       — y*(SiLU(z)⊙W(z)) → d_model
    8.  │ │ + Residual                   — x = residual + core_out
    9.  │ │ RAG Injection                — bmm(rag_state, query) × 0.1
    10. │ │ MoLE (top-2/8 experts)       — sparse LoRA routing
    11. │ │ NoveltyGate                  — skip если бесполезен (inference)
        │ └────────────────────────────────────────────┘
    12. │ WaveConsolidation             — merge left + right (gate+MLP)
    13. │ SharedGlobalAttention         — GQA attention (каждые N волн)
    14. │ WaveScratchpad.summarize      — сжать итог → d/4
    15. │ Lazy Spine                    — обновить память (если novelty>5%)
        └───────────────────────────────────────────────┘
    16. RMSNorm                         — финальная нормализация
    17. PersonalityAdapter              — «как ТАРС скажет» (стиль)
    18. LM Head                         — d_model → vocab_size logits
    19. CrossEntropy + MoLE aux         — loss
    20. backward()                      — градиенты
    21. optimizer.step()                — обновление весов
    
    Ключевые файлы:
      ssd.py          — TarsCoreBlock (SSD+RWKV+GatedDeltaNet+SwiGLU)
      tars_block.py   — TarsBlock (Core+RAG+MoLE+Novelty+Memory)
      model.py        — TarsMamba2LM (wave loop, forward, think)
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
            vocab_size=p.get("vocab_size", 4096), d_state=p.get("d_state", 64),
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
            # ═══ Security: weights_only=True to prevent pickle RCE attacks ═══
            try:
                cp = torch.load(checkpoint_path, map_location=device, weights_only=True)
            except Exception:
                _logger.warning("weights_only=True failed, trying safe legacy load...")
                import numpy as np
                from collections import OrderedDict
                safe_globals = [OrderedDict, np.ndarray, np.dtype]
                # PyTorch 2.6+ needs numpy internals whitelisted for legacy checkpoints
                for attr_path in ['numpy._core.multiarray.scalar',
                                  'numpy._core.multiarray._reconstruct',
                                  'numpy.core.multiarray.scalar',
                                  'numpy.core.multiarray._reconstruct']:
                    parts = attr_path.rsplit('.', 1)
                    try:
                        mod = __import__(parts[0], fromlist=[parts[1]])
                        obj = getattr(mod, parts[1], None)
                        if obj is not None:
                            safe_globals.append(obj)
                    except Exception:
                        pass
                torch.serialization.add_safe_globals(safe_globals)
                try:
                    cp = torch.load(checkpoint_path, map_location=device, weights_only=True)
                except Exception:
                    _logger.warning("Safe globals load failed, using weights_only=False (trusted source)")
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
            
            # ═══ CPU-OPT: INT8 Dynamic Quantization ═══
            # Converts all nn.Linear to INT8 → 2-3x CPU speedup, ~50% RAM savings
            # Uses AVX-512 VNNI instructions on Intel CPUs
            if str(device) == 'cpu' and not os.environ.get('TARS_NO_QUANT'):
                try:
                    model = torch.ao.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    _logger.info("INT8 dynamic quantization applied (CPU mode)")
                except Exception as e:
                    _logger.debug(f"INT8 quantization skipped: {e}")
            
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
        #   2. WuNeng Bottleneck (2d → 192 → GELU → d, HELIX v6)
        #   3. Reflex integration (MoLE expert signal)
        n_waves = n_layers // 2
        self.wave_consolidations = nn.ModuleList([
            WaveConsolidation(d_model) for _ in range(n_waves)
        ])
        
        # ═══ Global Workspace (Baars, 1988) ═══
        # Lazy-initialized: only used during inference/think(), not training
        self._global_workspace = None
        
        # ═══ Zamba-style Shared Global Attention ═══
        # One shared attention layer applied every N waves
        # Saves ~90% params vs per-layer attention
        self.shared_attn_interval = max(1, n_layers // 4)  # every 5 waves for 20 layers
        self.shared_global_attn = SharedGlobalAttention(d_model, n_heads=4, n_kv_heads=2)
        
        # ═══ WaveScratchpad: Sequential Thinking ═══
        # One shared scratchpad — each wave writes a summary, next wave reads it
        # Forces step-by-step reasoning instead of trying everything at once
        self.wave_scratchpad = WaveScratchpad(d_model)
        
        # ═══ TTT-LoRA (Test-Time Training) ═══
        # Adapts model at inference via 3-5 SGD steps with LoRA
        # Lazy-initialized: only needed during think()
        self._ttt_lora = None
        
        # ═══ IDME Matrix Pool (48+, бесконечное расширение) ═══
        self.matrix_pool = MatrixPool(d_model, pool_size, quant_mode=quant_mode)
        
        # ═══ Integral Auditor ═══
        self.integral_auditor = IntegralAuditor(window=8, default_threshold=1.1)
        self.meta_auditor = MetaAuditor()
        
        # ═══ Temporal Embedding (нейронные часы) ═══
        # Мульти-частотные осцилляции: fast (100ms), medium (1s), slow (1h), circadian (24h)
        self.temporal_embedding = TemporalEmbedding(d_model, n_frequencies=4)
        
        # ═══ MetaCortex (метакогниция — думать о думании) ═══
        # Предсказывает P(error) и адаптирует глубину мышления
        self.meta_cortex = MetaCortex(d_model=d_model, history_size=100)
        

        
        # ═══ Hankel SVD (глобальный анти-цикл) ═══
        # Lazy-initialized: only used during think(), not training
        self._hankel = None
        
        # ═══ NoveltyGate (для IDME) ═══
        self.novelty_gate = NoveltyGate(d_model)
        

        
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
        
        # ═══ Cerebellum: CriticHead (inter-wave verification) ═══
        # Проверяет соответствие текущего hidden state задаче (ТЗ).
        # Вставляется МЕЖДУ WaveConsolidation и ThinkingChain (RAG).
        self.critic_head = CriticHead(d_model, n_criteria=8)
        self.wave_critic = WaveCritic(self.critic_head, d_model)
        
        # ═══ Output head — RMSNorm everywhere for consistency ═══
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Init
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag
        self.use_checkpointing = False
        
        # MoLE auxiliary loss (accumulated during forward)
        self.mole_aux_loss = torch.tensor(0.0)
        
        # ═══ Performance: CPU Threading + torch.compile ═══
        n_threads = int(os.environ.get('TARS_THREADS', os.cpu_count() or 4))
        try:
            torch.set_num_threads(n_threads)
            torch.set_num_interop_threads(max(1, n_threads // 2))
            self.logger.info(f"CPU threads: {n_threads} intra, {max(1, n_threads // 2)} inter")
        except RuntimeError:
            pass  # Already set or parallel work started
        
        # torch.compile для 1.5-2x ускорения (PyTorch 2.0+)
        # NOTE: НЕ компилируем автоматически — модель содержит sub-modules,
        # несовместимые с torch.compile (integral_auditor: time.time(),
        # personality_adapter: iterative loop + .item()). Тренировочные скрипты
        # применяют torch.compile самостоятельно с нужным mode.
        # Включить вручную: TARS_FORCE_COMPILE=1
        self._compiled = False
        if hasattr(torch, 'compile') and os.environ.get('TARS_FORCE_COMPILE'):
            try:
                import platform
                if platform.system() != 'Windows':
                    self.forward = torch.compile(
                        self.forward,
                        mode="default",
                        fullgraph=False,
                    )
                    self._compiled = True
                    self.logger.info("torch.compile activated (default mode, forced)")
            except Exception as e:
                self.logger.debug(f"torch.compile failed: {e}")
        
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
        
        # ═══ QueryRouter — нейронный маршрутизатор памяти ═══
        # Определяет: нужен ли RAG, какой источник, уточняет запрос.
        # Решает проблему sync: если RAG не нужен → блоки не ждут.
        self.query_router = QueryRouter(d_model, mem_dim=384, quant_mode=quant_mode)
        
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
        
        # ═══ RoPE (base=500K per TARS spec, matches config.py) ═══
        self.rope = RotaryPositionEmbedding(d_model // (d_model // 64), base=500_000)
        
        # ═══ Wave-Parallel Input Diversification ═══
        # Prevents both blocks from learning identical features
        self.proj_left = nn.Linear(d_model, d_model, bias=False)
        self.proj_right = nn.Linear(d_model, d_model, bias=False)
        # Init as near-identity to avoid breaking pretrained weights
        nn.init.eye_(self.proj_left.weight)
        nn.init.eye_(self.proj_right.weight)
        self.proj_left.weight.data += torch.randn_like(self.proj_left.weight) * 0.01
        self.proj_right.weight.data += torch.randn_like(self.proj_right.weight) * 0.01
    
    def _ensure_inference_modules(self):
        """Lazy-initialize inference-only modules (saves ~15% VRAM during training)."""
        device = next(self.parameters()).device
        if self._hankel is None:
            self._hankel = HankelDetector(window=6)
        if self._dream_engine is None:
            self._dream_engine = DreamEngine(noise_scale=0.1, recombine_k=20)
        if self._global_workspace is None:
            self._global_workspace = GlobalWorkspace(self.d_model, self.n_layers)
        if self._neuromodulator is None:
            self._neuromodulator = Neuromodulator(self.d_model)
        if self._oscillatory is None:
            self._oscillatory = OscillatoryBinding(self.d_model)
        if self._dendritic_block is None:
            self._dendritic_block = DendriticBlock(self.d_model, self.d_model, n_segments=7).to(device)
        if self._hyper_sim is None:
            self._hyper_sim = HyperbolicSimilarity(c=1.0, scale=1.0)
        if self._belief_state is None:
            self._belief_state = BeliefState(d_state=128).to(device)
        # ═══ DoubtEngine: adversarial verifier (System 0) ═══
        if not hasattr(self, '_doubt_engine') or self._doubt_engine is None:
            self._doubt_engine = load_doubt_engine(
                d_model=self.d_model, device=str(device)
            )
    
    @property
    def hankel(self):
        if self._hankel is None:
            self._hankel = HankelDetector(window=6)
        return self._hankel
    
    @property
    def dream_engine(self):
        if self._dream_engine is None:
            self._dream_engine = DreamEngine(noise_scale=0.1, recombine_k=20)
        return self._dream_engine
    
    @property
    def global_workspace(self):
        if self._global_workspace is None:
            self._global_workspace = GlobalWorkspace(self.d_model, self.n_layers)
        return self._global_workspace
    
    @property
    def belief_state(self):
        if self._belief_state is None:
            device = next(self.parameters()).device
            self._belief_state = BeliefState(d_state=128).to(device)
        return self._belief_state
    
    @property
    def ttt_lora(self):
        """TTT-LoRA: lazy-init for test-time training."""
        if self._ttt_lora is None:
            device = next(self.parameters()).device
            self._ttt_lora = TTTLoRA(self.d_model, rank=4, alpha=8.0, n_steps=3).to(device)
        return self._ttt_lora
    
    # ═══════════════════════════════════════════════════════════════
    # Performance Optimization Methods  
    # ═══════════════════════════════════════════════════════════════
    
    def optimize_model(self, compile_mode: str = "default", gpu_name: str = "auto"):
        """
        Apply all optimizations for maximum performance.
        
        Automatically detects hardware and applies optimal settings:
          - RTX 4090: max-autotune compile, bf16, chunk_size=128
          - A100: max-autotune compile, bf16, chunk_size=256
          - L4: default compile, bf16, chunk_size=64, grad checkpointing
          - T4: default compile, fp16, chunk_size=64, grad checkpointing
        
        Args:
            compile_mode: "default", "max-autotune", "reduce-overhead", or None
            gpu_name: "auto", "4090", "A100", "L4", "T4", "cpu"
        
        Returns:
            dict with applied settings
        """
        try:
            from brain.mamba2.optimizations import optimize_for_training
            return optimize_for_training(self, gpu_name)
        except ImportError:
            self.logger.warning("optimizations.py not found, skipping")
            return {}
    
    def optimize_for_inference(self, gpu_name: str = "auto"):
        """
        Apply all inference optimizations (compile + CUDA graph + eval mode).
        """
        try:
            from brain.mamba2.optimizations import optimize_for_inference
            return optimize_for_inference(self, gpu_name)
        except ImportError:
            self.eval()
            return {}
    
    def prune(self, sparsity: float = 0.5):
        """
        SparseSSM-style pruning: zero out low-importance weights.
        
        No fine-tuning needed for sparsity <= 0.5 (per SparseSSM paper).
        50% sparsity → ~2x smaller, ~1.3x faster, ~0% accuracy loss.
        
        Args:
            sparsity: fraction of weights to prune (0.0-1.0)
        """
        try:
            from brain.mamba2.optimizations import prune_ssm_weights
            return prune_ssm_weights(self, sparsity=sparsity)
        except ImportError:
            self.logger.warning("optimizations.py not found, skipping prune")
            return {}
    
    def count_parameters(self):
        """Count model parameters by component."""
        result = {}
        total = sum(p.numel() for p in self.parameters())
        result["total"] = total
        
        blocks_total = sum(p.numel() for b in self.blocks for p in b.parameters())
        result["blocks"] = blocks_total
        
        if len(self.blocks) > 0:
            b0_params = sum(p.numel() for p in self.blocks[0].parameters())
            result["block_detail (×1)"] = {
                "total": b0_params,
                "core": sum(p.numel() for p in self.blocks[0].core.parameters()),
                "omega": sum(p.numel() for p in self.blocks[0].omega.parameters()),
                "mole": sum(p.numel() for p in self.blocks[0].mole.parameters()),
            }
        
        result["embedding"] = self.embedding.weight.numel()
        result["wave_consolidations"] = sum(
            p.numel() for wc in self.wave_consolidations for p in wc.parameters()
        )
        result["personality"] = sum(
            p.numel() for p in self.personality.parameters()
        )
        
        return result
    
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
        x = self.embedding(token_ids) * math.sqrt(self.d_model)  # Vaswani scaling
        
        n_waves = self.n_layers // 2
        wave_summary = None  # scratchpad: итог предыдущей волны
        
        for wave_idx in range(n_waves):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            # ═══ WaveScratchpad: inject previous wave's summary ═══
            if wave_summary is not None:
                x = self.wave_scratchpad.inject(x, wave_summary)
            
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
            
            # Zamba: Shared Global Attention every N waves
            if (wave_idx + 1) % self.shared_attn_interval == 0:
                x = self.shared_global_attn(x)
            
            # ═══ WaveScratchpad: summarize this wave for next ═══
            wave_summary = self.wave_scratchpad.summarize(x)
            
            # Lazy Spine: обновление только при новизне > 5%
            if wave_idx < n_waves - 1:
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
        
        # Final memory injection + output
        if c["memory_vec"] is not None:
            mem_signal = self.from_memory_space(c["memory_vec"])
            x = x + 0.1 * mem_signal.unsqueeze(1)
        
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
        Speculative Decoding с честной верификацией draft-токенов.
        
        Алгоритм:
        1. Сохраняем SSM state (snapshot)
        2. Draft head генерирует N кандидатов (быстро, ~1M params)
        3. Full model верифицирует ВСЮ последовательность за 1 forward
        4. Accept корректный prefix, reject остальные
        5. Rewind cache при отклонении
        
        Для SSM: поскольку state кумулятивен, при reject мы rewind к
        последнему принятому snapshot и пересчитываем.
        
        Args:
            prompt_ids: [1, L] initial tokens
            max_tokens: max tokens to generate
            n_draft: number of speculative tokens per step (4-8 optimal)
            temperature: sampling temperature
            top_k: top-k sampling
        
        Returns:
            generated: [1, L + generated] full sequence
        """
        import copy
        
        self.eval()
        device = prompt_ids.device
        
        # Prefill: process entire prompt
        self.reset_cache()
        logits = self.step(prompt_ids)  # [1, L, V]
        
        generated = prompt_ids.clone()  # [1, L]
        n_accepted_total = 0
        n_drafted_total = 0
        
        for _ in range(max_tokens):
            if generated.shape[1] - prompt_ids.shape[1] >= max_tokens:
                break
            
            # ═══ Step 1: Snapshot SSM state (shallow copy, not deepcopy) ═══
            # Shallow dict copy + .clone() on tensors is 10x cheaper than deepcopy
            cache_snapshot = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in self._gen_cache.items()
            } if isinstance(self._gen_cache, dict) else copy.deepcopy(self._gen_cache)
            
            # ═══ Step 2: Sample first token from main model ═══
            last_logits = logits[:, -1, :]  # [1, V]
            first_token = self._sample(last_logits, temperature, top_k)
            
            # ═══ Step 3: Draft N-1 more tokens using main model ═══
            draft_tokens = [first_token]
            draft_logits_list = [last_logits]
            draft_input = first_token
            
            for _ in range(n_draft - 1):
                step_logits = self.step(draft_input.unsqueeze(1))
                draft_next = self._sample(
                    step_logits[:, -1, :], temperature, top_k
                )
                draft_tokens.append(draft_next)
                draft_logits_list.append(step_logits[:, -1, :])
                draft_input = draft_next
            
            # ═══ Step 4: Verify draft tokens ═══
            # Восстанавливаем cache к snapshot и прогоняем все draft-токены
            # через основную модель за один batch
            self._gen_cache = cache_snapshot
            
            verify_input = torch.stack(draft_tokens).unsqueeze(0)  # [1, N]
            verify_logits = self.step(verify_input)  # [1, N, V]
            
            # ═══ Step 5: Accept/Reject ═══
            # Сравниваем: argmax(verify_logits[i]) == draft_tokens[i+1]?
            n_accepted = 1  # Первый токен всегда принимается (sampled from main)
            
            for i in range(len(draft_tokens) - 1):
                # Verify: детерминистичная проверка через argmax (не _sample!)
                verify_token = verify_logits[:, i, :].argmax(dim=-1)
                if verify_token.item() == draft_tokens[i + 1].item():
                    n_accepted += 1
                else:
                    # Mismatch: принимаем verified token вместо draft
                    draft_tokens[i + 1] = verify_token
                    n_accepted += 1
                    break  # Остальные draft-токены после первого reject невалидны
            
            n_drafted_total += len(draft_tokens)
            n_accepted_total += n_accepted
            
            # Добавляем принятые токены
            accepted = torch.stack(draft_tokens[:n_accepted]).unsqueeze(0)
            generated = torch.cat([generated, accepted], dim=1)
            
            # ═══ Step 6: Подготовка к следующей итерации ═══
            # Если приняли меньше чем draft → нужно rewind cache
            if n_accepted < len(draft_tokens):
                # Rewind: прогоняем только принятые через cache
                self._gen_cache = cache_snapshot
                logits = self.step(accepted)
            else:
                # Все приняты → последние logits из verify
                logits = verify_logits[:, n_accepted - 1:n_accepted, :]
        
        accept_rate = n_accepted_total / max(n_drafted_total, 1)
        self.logger.debug(
            f"Speculative: accepted {n_accepted_total}/{n_drafted_total} "
            f"({accept_rate:.0%}), generated {generated.shape[1] - prompt_ids.shape[1]} tokens"
        )
        
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
        
        Pipeline: Embedding → TemporalEmbedding → Wave Loop × n_waves:
          [Scratchpad.inject → MemInjector → proj_left/right →
           TarsBlock×2(RMSNorm→TarsCoreBlock[SSD∥RWKV→Fusion→SwiGLU]→
           RAG→MoLE→NoveltyGate→MemInject) →
           WaveConsolidation → SharedGlobalAttention → Scratchpad.summarize →
           Lazy Spine] → RMSNorm → PersonalityAdapter → LM Head → Loss
        
        Используется Parallel Wave Merge: [B0||B1] → gate → merge
        Все wave_consolidations обучаются через backprop.
        
        Args:
            input_ids: [B, L] — token indices
            memory_vec: [B, 384] — LEANN/Titans memory vector (optional)
            rag_state: [B, d_state, d_state] — compressed RAG document (optional)
            labels: [B, L] — target tokens for loss computation (optional)
        
        Returns:
            logits [B, L, vocab_size] if labels is None
            (logits, loss) if labels is provided
        """
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # Vaswani scaling
        
        # ═══ Temporal Embedding: inject time sense ═══
        x = self.temporal_embedding(x)
        
        wkv_states = [None] * self.n_layers
        x_prevs = [None] * self.n_layers
        ssd_states = [None] * self.n_layers
        conv_states = [None] * self.n_layers
        
        # Accumulate MoLE aux loss from all blocks
        mole_aux_total = torch.tensor(0.0, device=input_ids.device)
        
        # Parallel Wave through TarsBlocks
        n_waves = self.n_layers // 2
        wave_summary = None  # scratchpad: итог предыдущей волны
        
        for wave_idx in range(n_waves):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            if b_right >= self.n_layers:
                break
            
            # ═══ WaveScratchpad: inject previous wave's summary ═══
            if wave_summary is not None:
                x = self.wave_scratchpad.inject(x, wave_summary)
            
            # ═══ Wave-Parallel Memory: 1 lookup → 2 блока ═══
            mem_signal = None
            if memory_vec is not None:
                h_mean = x.mean(dim=1)
                mem_signal = self.shared_mem_injector.compute_signal(h_mean, memory_vec)
            
            # 2 блока параллельно с diversified input (получают готовый mem_signal)
            x_in_left = self.proj_left(x)
            x_in_right = self.proj_right(x)
            
            if self.use_checkpointing and self.training:
                # use_reentrant=True is required here because:
                # 1. SSM blocks have stateful operations (triu_indices cache, JIT-compiled WKV)
                # 2. AMP autocast context must be preserved during recomputation
                # 3. use_reentrant=False requires perfectly deterministic shapes/dtypes
                #    which is violated by torch.empty, triu_indices cache, and AMP dtype casting
                x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                    ssd_states[b_left], conv_states[b_left] = grad_checkpoint(
                    self.blocks[b_left], x_in_left, wkv_states[b_left],
                    x_prevs[b_left], None, rag_state,
                    ssd_states[b_left], conv_states[b_left], None, mem_signal,
                    use_reentrant=True
                )
                x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                    ssd_states[b_right], conv_states[b_right] = grad_checkpoint(
                    self.blocks[b_right], x_in_right, wkv_states[b_right],
                    x_prevs[b_right], None, rag_state,
                    ssd_states[b_right], conv_states[b_right], None, mem_signal,
                    use_reentrant=True
                )
            else:
                x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                    ssd_states[b_left], conv_states[b_left] = self.blocks[b_left](
                    x_in_left, wkv_states[b_left], x_prevs[b_left], None, rag_state,
                    ssd_states[b_left], conv_states[b_left], None, mem_signal
                )
                x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                    ssd_states[b_right], conv_states[b_right] = self.blocks[b_right](
                    x_in_right, wkv_states[b_right], x_prevs[b_right], None, rag_state,
                    ssd_states[b_right], conv_states[b_right], None, mem_signal
                )
            
            # Collect MoLE aux losses from both blocks
            for stats in [stats_l, stats_r]:
                if isinstance(stats, dict) and "mole_aux_loss" in stats:
                    mole_aux_total = mole_aux_total + stats["mole_aux_loss"]
            
            # Wave Consolidation: full merge with reflex integration
            x, _ = self.wave_consolidations[wave_idx](x_left, x_right, stats_l, stats_r)
            
            # Zamba: Shared Global Attention every N waves
            if (wave_idx + 1) % self.shared_attn_interval == 0:
                x = self.shared_global_attn(x)
            
            # ═══ WaveScratchpad: summarize this wave for next ═══
            wave_summary = self.wave_scratchpad.summarize(x)
            
            # ═══ Lazy Spine: обновлять memory_vec только при реальной новизне ═══
            # Optimized: use torch.where to avoid CPU-GPU sync on scalar comparison
            if wave_idx < n_waves - 1:
                h_curr = x.mean(dim=1)
                h_for_mem = self.to_memory_space(h_curr)
                if memory_vec is None:
                    memory_vec = h_for_mem
                else:
                    novelty = 1.0 - F.cosine_similarity(memory_vec, h_for_mem, dim=-1).mean()
                    # GPU-side conditional: avoids CPU sync from if novelty > 0.05
                    update_mask = (novelty > 0.05).float()
                    new_mem = 0.7 * memory_vec + 0.3 * h_for_mem
                    memory_vec = update_mask * new_mem + (1.0 - update_mask) * memory_vec
        
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
        task_spec: Any = None,
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
        
        # ═══ PHASE 0: Init inference modules ═══
        self._ensure_inference_modules()
        
        # Reset аудиторов
        self.integral_auditor.reset()
        self.matrix_pool.reset()
        self.hankel.reset()
        self.thinking_chain.reset()  # Новый запрос — новая цепочка рассуждений
        
        # ═══ Cerebellum: установить ТЗ ═══
        if task_spec is not None:
            self.wave_critic.set_task_spec(task_spec)
        else:
            self.wave_critic.set_task_spec(TaskSpec(query=query_text))
        
        # Meta-Auditor: тип задачи и порог
        task_type, p_threshold = self.meta_auditor.classify_task(query_text)
        
        # MetaCortex: метакогнитивная адаптация порога
        # (если модель часто ошибалась на этом типе → повысить порог)
        try:
            with torch.no_grad():
                # Используем текущий embedding запроса для предсказания ошибки
                if not hasattr(self, '_tokenizer'):
                    from brain.tokenizer import TarsTokenizer
                    self._tokenizer = TarsTokenizer(mode="auto")
                query_token_ids = self._tokenizer.encode(query_text)[:256]
                query_ids = torch.tensor([query_token_ids], dtype=torch.long).to(input_ids.device)
                query_emb = self.embedding(query_ids).mean(dim=1)  # [1, d_model]
                error_pred = self.meta_cortex.predict_error(query_emb).item()
                p_threshold = self.meta_cortex.adapt_threshold(p_threshold, task_type, error_pred)
        except Exception:
            error_pred = 0.5
        
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
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # Vaswani scaling
        
        # ═══ QueryRouter: определяем стратегию поиска ПЕРЕД волнами ═══
        # Вместо слепого RAG → нейросеть решает: искать / не искать / где.
        query_emb = x.mean(dim=1).detach()  # [B, d_model]
        router_decision = self.query_router(query_emb)
        
        # ═══ Progressive Memory Manager: non-blocking RAG ═══
        pmm = ProgressiveMemoryManager()
        pmm.set_initial_memory(memory_vec=memory_vec, rag_state=rag_state)
        
        # Сохраняем router decision для статистики
        rag_skipped = False
        if not router_decision["needs_rag"].any():
            # Роутер решил: RAG не нужен (простой запрос)
            rag_skipped = True
            self.logger.debug(
                f"QueryRouter: RAG skipped (score={router_decision['needs_rag_score'].mean():.3f})"
            )
        else:
            source = self.query_router.get_primary_source(router_decision["source_weights"])
            urgency_val = router_decision["urgency"].mean().item()
            self.logger.debug(
                f"QueryRouter: RAG needed, source={source}, urgency={urgency_val:.2f}"
            )
            # ═══ PMM: регистрируем запрошенные источники для tracking ═══
            source_names = ["leann", "titans", "web", "history"]
            weights = router_decision["source_weights"][0].tolist()
            # Отмечаем источники с весом > 0.1 как запрошенные
            requested = [n for n, w in zip(source_names, weights) if w > 0.1]
            pmm.register_request(requested, urgency=urgency_val)
        
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
        doubt_per_wave = []  # DoubtEngine scores per wave
        failed_waves = 0     # T08: count failed waves for partial return
        per_wave_experts = []  # [{"wave": 1, "left": [...], "right": [...]}]
        supplement_injected = False
        supplement_extra_waves = 0  # сколько допволн осталось
        rag_injection_wave = -1  # волна, на которой RAG был инжектирован
        
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
        critic_result = None  # initialized before loop — used in convergence check
        
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
            
            # ═══ Progressive Memory Injection: check RAG readiness ═══
            # Non-blocking: если RAG ещё не готов, блоки работают с тем что есть
            if pmm.check_and_inject(wave_idx):
                rag_injection_wave = wave_idx
                memory_vec = pmm.get_memory_vec()
                rag_state = pmm.get_rag_state()
                # Сбрасываем аудитор — новые данные меняют картину
                self.integral_auditor.reset()
                h_prev = x.mean(dim=1).detach()
                self.logger.debug(
                    f"Progressive RAG injection at wave {wave_count} "
                    f"(memory_vec={'ready' if memory_vec is not None else 'none'})"
                )
            else:
                memory_vec = pmm.get_memory_vec()
                rag_state = pmm.get_rag_state()
            
            # ── 2 блока работают ПАРАЛЛЕЛЬНО на одном входе ──
            # T08: wrapped in try/except for per-wave error recovery
            try:
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
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # T08: OOM fallback — skip MoLE/RAG, core-only path
                    self.logger.warning(
                        f"OOM at wave {wave_count}! Falling back to core-only path."
                    )
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    try:
                        x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                            ssd_states[b_left], conv_states[b_left] = self.blocks[b_left](
                            x, wkv_states[b_left], x_prevs[b_left], None, None,
                            ssd_states[b_left], conv_states[b_left]
                        )
                        x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                            ssd_states[b_right], conv_states[b_right] = self.blocks[b_right](
                            x, wkv_states[b_right], x_prevs[b_right], None, None,
                            ssd_states[b_right], conv_states[b_right]
                        )
                    except Exception as e2:
                        self.logger.error(f"Core-only fallback also failed: {e2}")
                        failed_waves += 1
                        continue
                else:
                    self.logger.error(f"RuntimeError at wave {wave_count}: {e}")
                    failed_waves += 1
                    continue
            except Exception as e:
                self.logger.error(f"Wave {wave_count} failed: {e}")
                failed_waves += 1
                # T08: >50% waves failed → stop and return partial
                if failed_waves > max_waves // 2:
                    self.logger.error(
                        f">{max_waves//2} waves failed! Returning partial result."
                    )
                    break
                continue

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
            
            # ═══ NaN/Inf Guard: protect hidden state ═══
            if torch.isnan(x).any() or torch.isinf(x).any():
                self.logger.warning(
                    f"NaN/Inf detected at wave {wave_count}! Applying nan_to_num."
                )
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                # If still bad after nan_to_num, revert
                if torch.isnan(x).any():
                    x = wave_outputs.get(wave_count - 1, self.embedding(input_ids))
            
            # ═══ T08: Per-wave health check — norm clamping ═══
            x_norm = x.norm().item()
            if x_norm < 0.01:
                self.logger.warning(f"Wave {wave_count}: x.norm={x_norm:.4f} too low, scaling up")
                x = x * (0.1 / max(x_norm, 1e-8))
            elif x_norm > 100.0:
                self.logger.warning(f"Wave {wave_count}: x.norm={x_norm:.1f} too high, scaling down")
                x = x * (10.0 / x_norm)
            
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
            
            # ── Сошлось? (КОНСЕНСУС: максимальная защита от ложного выхода) ──
            ia_converged = ia_result["converged"] and wave_count >= 3  # минимум 3 волны (6 блоков)
            tc_skip = hasattr(self, 'thinking_chain') and self.thinking_chain.should_skip_remaining()
            
            # Для code/math/deep: КОНСЕНСУС (IA + TC/R² + Critic)
            # Для chat/action: любой достаточен (OR) — скорость важнее
            strict_tasks = {"code", "math", "deep"}
            if task_type in strict_tasks:
                ia_quality = ia_converged and ia_result["r_squared"] > 0.92
                tc_quality = tc_skip
                # Critic голос (если есть предыдущий результат)
                critic_ok = critic_result is not None and critic_result.get("score", 0) > 0.6
                # Нужно минимум 2 из 3: (IA_quality, TC_quality, Critic_ok)
                votes = sum([ia_quality, tc_quality, critic_ok])
                should_stop = ia_converged and votes >= 2
            else:
                should_stop = ia_converged or tc_skip
            
            if should_stop:
                converged_early = True
                skip_reason = f"CONSENSUS({sum([ia_converged, tc_skip, critic_result is not None and critic_result.get('score',0)>0.6])}/3)"
                self.logger.debug(
                    f"Converged at wave {wave_count} (depth={blocks_executed}, reason={skip_reason})"
                )
                break
            
            # ═══ Cerebellum: CriticHead проверяет ТЗ между волнами ═══
            critic_result = None
            if wave_idx < max_waves - 1:
                try:
                    h_query_mean = self.embedding(input_ids).mean(dim=1).detach()
                    critic_result = self.wave_critic.evaluate_wave(
                        h_curr, h_query_mean, wave_idx, memory_vec
                    )
                    
                    self.thinking_logger.log_step(blocks_executed, {
                        "critic_score": critic_result["score"],
                        "critic_needs_data": critic_result["needs_data"],
                    })
                    
                    # Inject feedback into memory_vec
                    if critic_result["feedback_vec"] is not None and memory_vec is not None:
                        fb = critic_result["feedback_vec"]
                        # Project feedback to memory space if sizes differ
                        if fb.shape[-1] != memory_vec.shape[-1]:
                            if hasattr(self, 'to_memory_space'):
                                fb = self.to_memory_space(fb)
                        if fb.shape == memory_vec.shape:
                            memory_vec = memory_vec + 0.3 * fb  # additive correction
                    
                    # If critic says "all good" AND IA is well-converging → early stop
                    # Ужесточённый критерий: p > 0.8 (was 0.5) + R² > 0.7 + wave >= 3
                    ia_strong = ia_result["p"] > 0.8 and ia_result["r_squared"] > 0.7
                    if critic_result["should_stop"] and ia_strong and wave_count >= 3:
                        converged_early = True
                        self.logger.debug(
                            f"Critic+IA approved at wave {wave_count} "
                            f"(score={critic_result['score']:.0%}, p={ia_result['p']:.2f})"
                        )
                        break
                except Exception as e:
                    self.logger.debug(f"Critic eval error: {e}")
            
            # ═══ DoubtEngine: inter-wave adversarial verification ═══
            if hasattr(self, '_doubt_engine') and self._doubt_engine is not None:
                try:
                    with torch.no_grad():
                        q_emb = self.embedding(input_ids).mean(dim=1).detach()
                        r_emb = x.mean(dim=1).detach()
                        doubt_scores = self._doubt_engine(q_emb, r_emb)
                        
                        d_coherence = doubt_scores["coherence"].mean().item()
                        d_safety = doubt_scores["safety"].mean().item()
                        d_repetition = doubt_scores["repetition"].mean().item()
                        
                        doubt_per_wave.append({
                            "wave": wave_count,
                            "coherence": d_coherence,
                            "safety": d_safety,
                            "repetition": d_repetition,
                        })
                        
                        self.thinking_logger.log_step(blocks_executed, {
                            "doubt_coherence": d_coherence,
                            "doubt_safety": d_safety,
                            "doubt_repetition": d_repetition,
                        })
                        
                        # Coherence < 0.2 → rewind to previous wave output
                        if d_coherence < DoubtEngine.COHERENCE_BLOCK and wave_count > 1:
                            prev_wave = wave_idx - 1
                            if prev_wave in wave_outputs:
                                x = wave_outputs[prev_wave]
                                self.logger.warning(
                                    f"DoubtEngine: coherence={d_coherence:.2f} < {DoubtEngine.COHERENCE_BLOCK} "
                                    f"at wave {wave_count}. Rewinding to wave {prev_wave+1}."
                                )
                                self.integral_auditor.reset()
                                h_prev = x.mean(dim=1).detach()
                        
                        # Repetition > 0.9 → forced early stop (loop detected)
                        if d_repetition > DoubtEngine.REPEAT_BLOCK:
                            converged_early = True
                            self.logger.warning(
                                f"DoubtEngine: repetition={d_repetition:.2f} > {DoubtEngine.REPEAT_BLOCK} "
                                f"at wave {wave_count}. Forced early stop (loop detected)."
                            )
                            break
                except Exception as e:
                    self.logger.debug(f"DoubtEngine inter-wave error: {e}")
            
            # ── Спинной мозг обновляет память между волнами (через PMM) ──
            spine_updated = False
            if wave_idx < max_waves - 1:
                if hasattr(self, 'to_memory_space'):
                    try:
                        pmm.update_spine(h_curr, self.to_memory_space, alpha=0.3)
                        memory_vec = pmm.get_memory_vec()
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
            # Fix: используем try/except вместо .empty() + get_nowait() для атомарности
            if supplement_queue is not None:
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
                        
                        # Обновляем через PMM: спинной мозг инжектирует
                        h_sup = x_sup_merged.mean(dim=1).detach()
                        if hasattr(self, 'to_memory_space'):
                            pmm.update_spine(h_sup, self.to_memory_space, alpha=0.5)
                            memory_vec = pmm.get_memory_vec()
                        
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
                    if "Empty" not in str(type(e).__name__):
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
                
                if ia_result["converged"] and ia_result["r_squared"] > 0.88:
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
                # NaN guard на IDME результат
                if torch.isnan(best_x).any() or torch.isinf(best_x).any():
                    self.logger.warning(f"IDME round {expansion_round}: NaN in matrix output, skipping")
                    no_improve_count += 1
                    continue
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
        
        # ═══ 2.5. Cerebellum: RAG Completion Verification ═══
        # Мозжечок проверяет, все ли RAG-данные загрузились.
        # Если нет — рекрутирует СВЕЖИЕ IDME-матрицы для догрузки.
        rag_verification = {}
        try:
            delivery_report = pmm.get_delivery_report()
            rag_verification = self.wave_critic.verify_rag_completion(
                delivery_report=delivery_report,
                memory_vec=memory_vec,
                pmm=pmm,
                matrix_pool=self.matrix_pool,
                x=x,
                h_prev=h_prev,
                integral_auditor=self.integral_auditor,
                novelty_gate=self.novelty_gate,
                from_memory_space=self.from_memory_space,
                to_memory_space=self.to_memory_space,
            )
            # Обновляем memory_vec и x из мозжечка
            if rag_verification.get("updated_memory_vec") is not None:
                memory_vec = rag_verification["updated_memory_vec"]
            if rag_verification.get("updated_x") is not None:
                x = rag_verification["updated_x"]
            total_matrices_recruited += rag_verification.get("matrices_recruited", 0)
        except Exception as e:
            self.logger.debug(f"Cerebellum RAG verification error: {e}")
        
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
        
        # ═══ Output Sanity Check (Zero-Glitch Guard) ═══
        # Проверяем что logits не содержат NaN/Inf и не дегенерированы
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            self.logger.warning("CRITICAL: NaN/Inf in output logits! Replacing with uniform.")
            logits = torch.zeros_like(logits)  # uniform → безопасный fallback
        else:
            # Проверка дегенерации: все logits одинаковые (мёртвая модель)
            logit_std = logits.float().std().item()
            if logit_std < 1e-6:
                self.logger.warning(f"WARNING: Degenerate logits (std={logit_std:.8f}). Adding noise.")
                logits = logits + torch.randn_like(logits) * 0.01
        
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
            # QueryRouter & Progressive Memory stats
            "rag_skipped": rag_skipped,
            "rag_injection_wave": rag_injection_wave,
            "query_router_needs_rag": not rag_skipped,
            "query_router_urgency": router_decision["urgency"].mean().item(),
            "query_router_source": (
                self.query_router.get_primary_source(router_decision["source_weights"])
                if not rag_skipped else "none"
            ),
            # Cerebellum RAG Verification
            "rag_verification": rag_verification,
            "rag_all_loaded": rag_verification.get("rag_complete", True),
            "rag_continuation_rounds": rag_verification.get("rounds_used", 0),
            "rag_missing_sources": rag_verification.get("missing_sources", []),
            # DoubtEngine v4 stats
            "doubt_per_wave": doubt_per_wave,
            "doubt_final": doubt_per_wave[-1] if doubt_per_wave else {},
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
            "mole": count(block.mole),
        }
        if hasattr(block, 'omega'):
            block_detail["omega_ssm"] = count(block.omega)
        
        return {
            "embedding": count(self.embedding),
            "blocks_total": sum(count(b) for b in self.blocks),
            "block_detail (×1)": block_detail,
            "matrix_pool": count(self.matrix_pool),
            "novelty_gate": count(self.novelty_gate),
            "lm_head": 0,  # tied
            "total": count(self),
        }
    
    # ═══ T07: Composition Accessors ═══
    # Allow downstream code to use the decomposed, testable modules.
    
    @property
    def inference_engine(self):
        """Lazy InferenceEngine wrapping this model's internals via BrainCore."""
        if not hasattr(self, '_inference_engine') or self._inference_engine is None:
            # Create a BrainCore-compatible facade that reuses our own weights
            self._verification_suite = VerificationSuite(self.d_model)
            # Share state with our own auditors
            self._verification_suite.integral_auditor = self.integral_auditor
            self._verification_suite.meta_auditor = self.meta_auditor
            self._verification_suite.meta_cortex = self.meta_cortex
            self._verification_suite.temporal_embedding = self.temporal_embedding
            self._verification_suite.critic_head = self.critic_head
            self._verification_suite.wave_critic = self.wave_critic
            self._inference_engine = InferenceEngine(self, self._verification_suite)
        return self._inference_engine
