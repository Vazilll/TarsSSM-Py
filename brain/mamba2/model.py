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

from brain.mamba2.tars_block import TarsBlock
from brain.mamba2.integral_auditor import IntegralAuditor, MetaAuditor
from brain.mamba2.matrix_pool import MatrixPool
from brain.mamba2.novelty import NoveltyGate, HankelDetector
from brain.mamba2.logger import ThinkingLogger
from brain.mamba2.bitnet import (
    UniversalLinear, convert_model_to_158bit,
    convert_model_to_fp16, model_stats, replace_linear_with_universal
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
        d_model: int = 768,
        n_layers: int = 12,
        vocab_size: int = 32000,
        d_state: int = 64,
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
        
        # ═══ Основные блоки (12 × TarsBlock) ═══
        # Используются парами: [B0||B1] → merge → spine → [B2||B3] → ...
        self.blocks = nn.ModuleList([
            TarsBlock(
                d_model=d_model, d_state=d_state,
                headdim=headdim, omega_dim=omega_dim,
                n_experts=n_experts, layer_idx=i,
                quant_mode=quant_mode,
            )
            for i in range(n_layers)
        ])
        
        # ═══ Wave Merge Gates (6 gates для 12/2=6 волн) ═══
        # Каждый merge сливает 2 параллельных блока в один выход
        n_waves = n_layers // 2
        self.wave_merges = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(n_waves)
        ])
        # Обучаемый гейт: сколько от каждого из 2 блоков взять
        self.wave_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(n_waves)
        ])
        
        # ═══ IDME Matrix Pool (48+, бесконечное расширение) ═══
        self.matrix_pool = MatrixPool(d_model, pool_size)
        
        # ═══ Integral Auditor ═══
        self.integral_auditor = IntegralAuditor(window=8, default_threshold=1.1)
        self.meta_auditor = MetaAuditor()
        
        # ═══ Hankel SVD (глобальный анти-цикл) ═══
        self.hankel = HankelDetector(window=6)
        
        # ═══ NoveltyGate (для IDME) ═══
        self.novelty_gate = NoveltyGate(d_model)
        
        # ═══ Thinking Logger ═══
        self.thinking_logger = ThinkingLogger()
        
        # ═══ Titans Memory Hook (384d LTM) ═══
        # Проекция d_model → 384d (пространство памяти LEANN/Titans)
        self.mem_dim = 384
        self.to_memory_space = nn.Linear(d_model, self.mem_dim, bias=False)
        self.from_memory_space = nn.Linear(self.mem_dim, d_model, bias=False)
        self.titans_memory = None  # Set externally: model.titans_memory = TitansMemory(384)
        
        # ═══ Output head ═══
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Init
        self.apply(self._init_weights)
        
        # Gradient checkpointing flag
        self.use_checkpointing = False
    
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
        Все wave_merges и wave_gates обучаются через backprop.
        
        input_ids: [B, L]
        Returns: logits [B, L, vocab_size]
        """
        x = self.embedding(input_ids)  # [B, L, d_model]
        
        wkv_states = [None] * self.n_layers
        x_prevs = [None] * self.n_layers
        
        # Parallel Wave through TarsBlocks
        n_waves = self.n_layers // 2
        
        for wave_idx in range(n_waves):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            if b_right >= self.n_layers:
                break
            
            # 2 блока параллельно
            if self.use_checkpointing and self.training:
                x_left, wkv_states[b_left], x_prevs[b_left], _ = grad_checkpoint(
                    self.blocks[b_left], x, wkv_states[b_left],
                    x_prevs[b_left], memory_vec, rag_state,
                    use_reentrant=False
                )
                x_right, wkv_states[b_right], x_prevs[b_right], _ = grad_checkpoint(
                    self.blocks[b_right], x, wkv_states[b_right],
                    x_prevs[b_right], memory_vec, rag_state,
                    use_reentrant=False
                )
            else:
                x_left, wkv_states[b_left], x_prevs[b_left], _ = self.blocks[b_left](
                    x, wkv_states[b_left], x_prevs[b_left], memory_vec, rag_state
                )
                x_right, wkv_states[b_right], x_prevs[b_right], _ = self.blocks[b_right](
                    x, wkv_states[b_right], x_prevs[b_right], memory_vec, rag_state
                )
            
            # Wave Merge: gate + correction
            h_left = x_left.mean(dim=1)
            h_right = x_right.mean(dim=1)
            gate_input = torch.cat([h_left, h_right], dim=-1)
            
            alpha = self.wave_gates[wave_idx](gate_input).unsqueeze(1)
            x_merged = (1 - alpha) * x_left + alpha * x_right
            
            correction = self.wave_merges[wave_idx](gate_input).unsqueeze(1)
            x = x_merged + 0.1 * correction
        
        # Handle odd number of blocks
        if self.n_layers % 2 == 1:
            last_block = self.blocks[-1]
            x, _, _, _ = last_block(x, None, None, memory_vec, rag_state)
        
        # Output
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            return logits, loss
        
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
    ) -> Tuple[torch.Tensor, dict]:
        """
        Адаптивный forward с Integral Auditor и Speculative Matrix Routing.
        
        Новое: reflex_ctx (ReflexContext) управляет глубиной:
          - trivial (depth=4): проходим только 4 блока
          - simple (depth=6):  6 блоков
          - complex (depth=12): все 12 блоков + IDME
        """
        start_time = time.time()
        
        # Reset аудиторов
        self.integral_auditor.reset()
        self.matrix_pool.reset()
        self.hankel.reset()
        
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
        h_prev = x.mean(dim=1).detach()
        
        block_stats = []
        blocks_executed = 0
        surprise_signals = []
        converged_early = False
        wave_count = 0
        
        max_waves = self.n_layers // 2   # 12/2 = 6 волн
        
        for wave_idx in range(max_waves):
            # Проверяем depth limit
            if blocks_executed >= estimated_depth and not force_deep:
                break
            
            wave_count += 1
            b_left = wave_idx * 2       # индекс левого блока
            b_right = wave_idx * 2 + 1  # индекс правого блока
            
            if b_right >= self.n_layers:
                break
            
            # ── 2 блока работают ПАРАЛЛЕЛЬНО на одном входе ──
            x_left, wkv_states[b_left], x_prevs[b_left], stats_l = self.blocks[b_left](
                x, wkv_states[b_left], x_prevs[b_left], memory_vec, rag_state
            )
            x_right, wkv_states[b_right], x_prevs[b_right], stats_r = self.blocks[b_right](
                x, wkv_states[b_right], x_prevs[b_right], memory_vec, rag_state
            )
            block_stats.extend([stats_l, stats_r])
            blocks_executed += 2
            
            # ── Merge: обучаемый гейт сливает два выхода ──
            h_left = x_left.mean(dim=1)   # [B, d_model]
            h_right = x_right.mean(dim=1)
            gate_input = torch.cat([h_left, h_right], dim=-1)  # [B, 2*d_model]
            
            # Гейт: 0.0=только левый, 1.0=только правый
            alpha = self.wave_gates[wave_idx](gate_input)  # [B, 1]
            alpha = alpha.unsqueeze(1)                       # [B, 1, 1]
            x_merged = (1 - alpha) * x_left + alpha * x_right
            
            # Residual: добавляем обучаемую нелинейную коррекцию
            merge_input = torch.cat([h_left, h_right], dim=-1)
            correction = self.wave_merges[wave_idx](merge_input)  # [B, d_model]
            x = x_merged + 0.1 * correction.unsqueeze(1)
            
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
            
            # ── Сошлось? ──
            if ia_result["converged"] and wave_count >= 2:
                converged_early = True
                self.logger.debug(
                    f"Converged at wave {wave_count} (depth={blocks_executed})"
                )
                break
            
            # ── Спинной мозг обновляет память между волнами ──
            if wave_idx < max_waves - 1:
                if hasattr(self, 'to_memory_space') and memory_vec is not None:
                    try:
                        h_for_mem = self.to_memory_space(h_curr)
                        memory_vec = 0.7 * memory_vec + 0.3 * h_for_mem.detach()
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
        
        # Собираем экспертов из последнего выполненного блока
        last_idx = min(blocks_executed - 1, len(self.blocks) - 1)
        try:
            active_experts = self.blocks[last_idx].mole.get_active_experts(x)
        except Exception:
            active_experts = []
        
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
                available = self.matrix_pool.total_available()
                used = len(getattr(self.matrix_pool, 'used_mask', []))
                if available <= used + N_CANDIDATES:
                    try:
                        self.matrix_pool._lazy_expand(4, h_curr.mean(0))
                        self.logger.debug(
                            f"IDME lazy expand: +4 matrices (total={self.matrix_pool.total_available()})"
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
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        total_time = (time.time() - start_time) * 1000
        
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
            "active_experts": active_experts,
            "hankel_collapses": self.hankel.collapse_count,
            "surprise_layers": len(surprise_signals),
            "rwkv_state_size_mb": (wkv_state.numel() * 4 / 1024 / 1024) if wkv_state is not None else 0,
            "total_ms": total_time,
        }
        
        self.thinking_logger.end_session(total_time, "")
        
        win_ratio = f"{len(branches_won)}/{branches_tested}" if branches_tested > 0 else "0/0"
        self.logger.info(
            f"Think: {task_type} | p={stats['final_p']:.3f} | "
            f"branches={win_ratio} | experts={active_experts} | {total_time:.0f}ms"
        )
        
        return logits, stats
    
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
                x, wkv_state, x_prev, _ = block(
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
