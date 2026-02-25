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


class TarsMamba2LM(nn.Module):
    """
    ТАРС v3: Deep WuNeng Core (Mamba-2 + RWKV-7 inside one kernel).
    
    Гибридный мозг с единым ядром TarsCoreBlock:
    Mamba-2 SSD и RWKV-7 WKV делят общие проекции и сливаются
    в латентном пространстве через Deep Gated Fusion.
    
    ~130M params (fp16: ~260MB, 1.58-bit: ~60MB)
    """
    
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
        self.blocks = nn.ModuleList([
            TarsBlock(
                d_model=d_model, d_state=d_state,
                headdim=headdim, omega_dim=omega_dim,
                n_experts=n_experts, layer_idx=i,
                quant_mode=quant_mode,
            )
            for i in range(n_layers)
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
        Обычный forward pass (для обучения и быстрой генерации).
        Без IDME — только 12 блоков TarsBlock.
        
        input_ids: [B, L]
        Returns: logits [B, L, vocab_size]
        """
        x = self.embedding(input_ids)  # [B, L, d_model]
        
        # WKV state (переносится между блоками внутри TarsCoreBlock)
        wkv_state = None
        x_prev = None
        
        # Forward through TarsBlocks
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x, wkv_state, x_prev, _ = grad_checkpoint(
                    block, x, wkv_state, x_prev, memory_vec, rag_state,
                    use_reentrant=False
                )
            else:
                x, wkv_state, x_prev, _ = block(
                    x, wkv_state, x_prev, memory_vec, rag_state
                )
        
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
        
        # ═══ 1. Forward через TarsBlocks (adaptive depth) ═══
        x = self.embedding(input_ids)
        
        wkv_state = None
        x_prev = None
        h_prev = x.mean(dim=1).detach()
        
        block_stats = []
        blocks_executed = 0
        
        for i, block in enumerate(self.blocks):
            # ═══ Adaptive depth: skip blocks beyond estimated_depth ═══
            if i >= estimated_depth and not force_deep:
                self.logger.debug(
                    f"Adaptive skip: block {i}+ (depth={estimated_depth})"
                )
                break
            
            x, wkv_state, x_prev, stats = block(
                x, wkv_state, x_prev, memory_vec, rag_state
            )
            block_stats.append(stats)
            blocks_executed = i + 1
            
            # IA check mid-stream (каждые 3 блока)
            if (i + 1) % 3 == 0:
                h_curr = x.mean(dim=1).detach()
                ia_result = self.integral_auditor.observe(h_curr, h_prev)
                h_prev = h_curr
                
                self.thinking_logger.log_step(i, {
                    "p": ia_result["p"], "r_squared": ia_result["r_squared"],
                    "f_t": ia_result["f_t"], "converged": ia_result["converged"],
                    "block": i,
                })
                
                # Early exit: если уже сошлось раньше estimated_depth
                if ia_result["converged"] and i >= 5 and task_type in ("chat", "action"):
                    self.logger.debug(
                        f"Early exit at block {i}: p={ia_result['p']:.3f}"
                    )
                    break
        
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
            "total_blocks": self.n_layers,
            "blocks_executed": blocks_executed,
            "estimated_depth": estimated_depth,
            "expansion_rounds": expansion_round,
            "matrices_recruited": total_matrices_recruited,
            "total_matrices": self.n_layers + total_matrices_recruited,
            "branches_tested": branches_tested,
            "branches_won": len(branches_won),
            "active_experts": active_experts,
            "hankel_collapses": self.hankel.collapse_count,
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
