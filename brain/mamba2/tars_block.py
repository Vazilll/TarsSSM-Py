"""
═══════════════════════════════════════════════════════════════
  TarsBlock — Единый гибридный блок TARS v3
═══════════════════════════════════════════════════════════════

Один слой нейронной сети TARS с ВСЕМИ технологиями внутри.
TarsCoreBlock (ssd.py) содержит: Mamba-2 SSD + RWKV-7 WKV + WuNeng Fusion.
TarsBlock добавляет:   Ω-SSM + MoLE + NoveltyGate + RAG injection.

Архитектура:
  x → TarsCoreBlock → Ω-SSM → MoLE → NoveltyGate → out
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from brain.mamba2.ssd import TarsCoreBlock
from brain.mamba2.omega_layer import OmegaSSMLayer
from brain.mamba2.mole_router import MoLELayer
from brain.mamba2.novelty import NoveltyGate
from brain.mamba2.bitnet import UniversalLinear, ActivationQuantizer
from brain.mamba2.neuromodulator import PredictiveCodingLayer


class MemoryInjector(nn.Module):
    """
    Fused Memory Injection: запрос + гейт + проекция в одном модуле.
    
    Заменяет 4 отдельных операции (mem_query_proj, cosine_similarity,
    mem_gate, mem_proj) одним проходом.
    
    Два режима:
      - Per-block mode: каждый блок вызывает inject() самостоятельно
      - Shared mode: model.py вызывает compute_signal() один раз,
        оба блока в wave-паре получают готовый mem_signal
    """
    def __init__(self, d_model: int, mem_dim: int = 384, quant_mode: str = "fp16"):
        super().__init__()
        self.mem_dim = mem_dim
        # Fused: [h_mean, memory_vec] → gate + signal за один проход
        self.fused_proj = UniversalLinear(d_model + mem_dim, d_model, bias=False, mode=quant_mode)
        self.gate = nn.Sequential(
            UniversalLinear(d_model + mem_dim, 1, bias=True, mode=quant_mode),
            nn.Sigmoid(),
        )
        self.quant = ActivationQuantizer(enabled=(quant_mode == "158bit"))
    
    def compute_signal(self, h_mean: torch.Tensor, memory_vec: torch.Tensor) -> torch.Tensor:
        """
        Вычислить memory signal (для shared mode в wave-parallel).
        
        Returns: mem_signal [B, d_model] — готовый сигнал для injection
        """
        memory_vec_q = self.quant(memory_vec)
        combined = torch.cat([h_mean, memory_vec_q], dim=-1)  # [B, d_model + 384]
        gate = self.gate(combined)                              # [B, 1]
        signal = self.fused_proj(combined)                      # [B, d_model]
        return gate * signal  # [B, d_model]
    
    def inject(self, x: torch.Tensor, memory_vec: torch.Tensor,
               drop: nn.Dropout = None) -> Tuple[torch.Tensor, float]:
        """
        Per-block injection (fallback для случаев без wave-parallel).
        
        Returns: (x + mem_signal, gate_value)
        """
        h_mean = x.mean(dim=1)
        signal = self.compute_signal(h_mean, memory_vec)
        if drop is not None:
            signal = drop(signal)
        x = x + signal.unsqueeze(1)
        gate_val = signal.abs().mean().item()
        return x, gate_val


class TarsBlock(nn.Module):
    """
    Единый блок TARS v3.
    
    Внутри:
      1. TarsCoreBlock — deep hybrid scan (Mamba-2 + RWKV-7 + WuNeng fusion)
      2. Ω-SSM (Cayley SO(n)) — стабилизация на Lie manifold
      3. MoLE (per-block, sparse top-2) — экспертная маршрутизация
      4. NoveltyGate — пропуск бесполезных обновлений
      5. RAG injection — впрыск знаний из сжатого RWKV-состояния
      6. Fused MemoryInjector — оптимизированный впрыск из спинного мозга
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 64,
        headdim: int = 64,
        omega_dim: int = 32,
        n_experts: int = 8,
        layer_idx: int = 0,
        quant_mode: str = "fp16",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.d_state = d_state
        
        # ═══ 1. Deep Hybrid Core (SSD + WKV + WuNeng) ═══
        self.norm = nn.LayerNorm(d_model)
        self.core = TarsCoreBlock(
            d_model=d_model, d_state=d_state,
            headdim=headdim, chunk_size=64,
            quant_mode=quant_mode,
        )
        
        # ═══ 2. Ω-SSM Lie (Cayley SO(n)) ═══
        self.omega = OmegaSSMLayer(d_model, omega_dim, quant_mode=quant_mode)
        
        # ═══ 3. MoLE (per-block, sparse top-2) ═══
        self.mole = MoLELayer(d_model, n_experts=n_experts, rank=8, top_k=2, quant_mode=quant_mode)
        
        # ═══ 4. NoveltyGate ═══
        self.novelty_gate = NoveltyGate(d_model)
        
        # ═══ 5. RAG injection ═══
        self.rag_query = UniversalLinear(d_model, d_state, bias=False, mode=quant_mode)
        self.rag_out = UniversalLinear(d_state, d_model, bias=False, mode=quant_mode)
        
        # ═══ 6. Fused Memory Injection (оптимизировано: 4 ops → 1) ═══
        self.mem_injector = MemoryInjector(d_model, mem_dim=384, quant_mode=quant_mode)
        
        # ═══ 7. Dropout ═══
        self.drop = nn.Dropout(dropout)
        
        # ═══ 8. Predictive Coding (Rao & Ballard, 1999) ═══
        self.predictive_coding = PredictiveCodingLayer(d_model)
        
        # ═══ 9. BitNative int8 — квантизация активаций между блоками ═══
        self.output_quant = ActivationQuantizer(enabled=(quant_mode == "158bit"))
        
        # Stats
        self.last_stats = {}
        self.last_surprise = 0.0
    
    def forward(
        self,
        x: torch.Tensor,
        wkv_state: Optional[torch.Tensor] = None,
        x_prev: Optional[torch.Tensor] = None,
        memory_vec: Optional[torch.Tensor] = None,
        rag_state: Optional[torch.Tensor] = None,
        ssd_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
        x_prev_layer: Optional[torch.Tensor] = None,
        mem_signal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: [B, L, d_model]
            wkv_state: [B, d_state, d_state] — RWKV carry state
            x_prev: [B, 1, d_model] — last token for time-shift
            memory_vec: [B, 384] — LEANN/RAG vector (per-block mode)
            rag_state: [B, d_state, d_state] — compressed knowledge
            ssd_state, conv_state: recurrent states
            x_prev_layer: [B, L, d_model] — for predictive coding
            mem_signal: [B, d_model] — pre-computed memory signal (wave-parallel mode)
        
        Returns:
            output, wkv_state, x_prev, stats, ssd_state, conv_state
        """
        residual = x
        
        # ═══ 0. Predictive Coding — process only prediction error ═══
        x, pred_error = self.predictive_coding(x, x_prev_layer)
        
        # ═══ 1. Deep Hybrid Core ═══
        core_out, wkv_state, x_prev, ssd_state, conv_state = self.core(
            self.norm(x), wkv_state, x_prev, ssd_state, conv_state
        )
        x = residual + self.drop(core_out)
        
        # ═══ Cache h_mean once (used by RAG, NoveltyGate) ═══
        h_mean = x.mean(dim=1)  # [B, d_model]
        
        # ═══ RAG injection ═══
        if rag_state is not None:
            q = self.rag_query(h_mean)
            info = torch.bmm(rag_state, q.unsqueeze(-1)).squeeze(-1)
            x = x + 0.1 * self.rag_out(info).unsqueeze(1)
        
        # ═══ 2. Ω-SSM ═══
        x = self.omega(x)
        
        # ═══ 3. MoLE (returns aux_loss for load balancing) ═══
        x, mole_aux_loss = self.mole(x)
        x = self.drop(x)
        
        # ═══ 4. NoveltyGate — adaptive residual ═══
        h_old = residual.mean(dim=1)
        h_new = x.mean(dim=1)
        _, novelty = self.novelty_gate(h_old, h_new)
        n = novelty.unsqueeze(-1).unsqueeze(-1)
        x = n * x + (1 - n) * residual
        
        # ═══ 5. Memory Injection (2 режима) ═══
        mem_relevance = 0.0
        if mem_signal is not None:
            # Wave-parallel mode: сигнал уже вычислен model.py
            x = x + self.drop(mem_signal.unsqueeze(1))
            mem_relevance = mem_signal.abs().mean().item()
        elif memory_vec is not None:
            # Per-block mode: fallback
            x, self.last_surprise = self.mem_injector.inject(x, memory_vec, self.drop)
            mem_relevance = self.last_surprise
        
        # ═══ 6. BitNative Int8 — квантизация выхода блока ═══
        x = self.output_quant(x)
        
        # Stats
        self.last_stats = {
            "layer_idx": self.layer_idx,
            "novelty": novelty.mean().item() if isinstance(novelty, torch.Tensor) else novelty,
            "has_rag": rag_state is not None,
            "mem_relevance": mem_relevance,
            "surprise": self.last_surprise,
            "mole_aux_loss": mole_aux_loss,
            "mole_experts": getattr(self.mole, '_last_expert_names', []),
            "pred_error": pred_error.mean().item() if isinstance(pred_error, torch.Tensor) else 0.0,
        }
        
        return x, wkv_state, x_prev, self.last_stats, ssd_state, conv_state
