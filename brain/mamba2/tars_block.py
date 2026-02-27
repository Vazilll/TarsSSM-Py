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
from brain.mamba2.bitnet import UniversalLinear


class TarsBlock(nn.Module):
    """
    Единый блок TARS v3.
    
    Внутри:
      1. TarsCoreBlock — deep hybrid scan (Mamba-2 + RWKV-7 + WuNeng fusion)
      2. Ω-SSM (Cayley SO(n)) — стабилизация на Lie manifold
      3. MoLE (per-block, sparse top-2) — экспертная маршрутизация
      4. NoveltyGate — пропуск бесполезных обновлений
      5. RAG injection — впрыск знаний из сжатого RWKV-состояния
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
        
        # ═══ 6. Dynamic Memory Injection (вместо статичного 0.05) ═══
        # Проекция h → пространство памяти для динамического запроса
        self.mem_query_proj = UniversalLinear(d_model, 384, bias=False, mode=quant_mode)
        self.mem_proj = UniversalLinear(384, d_model, bias=False, mode=quant_mode)
        # Обучаемый гейт: насколько сильно слой использует память
        self.mem_gate = nn.Sequential(
            nn.Linear(d_model + 384, 1),
            nn.Sigmoid(),
        )
        
        # Stats
        self.last_stats = {}
        self.last_surprise = 0.0  # Для Titans feedback
    
    def forward(
        self,
        x: torch.Tensor,
        wkv_state: Optional[torch.Tensor] = None,
        x_prev: Optional[torch.Tensor] = None,
        memory_vec: Optional[torch.Tensor] = None,
        rag_state: Optional[torch.Tensor] = None,
        ssd_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            x: [B, L, d_model]
            wkv_state: [B, d_state, d_state] — RWKV carry state
            x_prev: [B, 1, d_model] — last token for time-shift
            memory_vec: [B, 384] — LEANN/RAG vector
            rag_state: [B, d_state, d_state] — compressed knowledge
            ssd_state: [B, nheads, headdim, d_state] — SSD recurrent state
            conv_state: [B, conv_dim, d_conv] — conv1d rolling state
        
        Returns:
            output, wkv_state, x_prev, stats, ssd_state, conv_state
        """
        residual = x
        
        # ═══ 1. Deep Hybrid Core ═══
        core_out, wkv_state, x_prev, ssd_state, conv_state = self.core(
            self.norm(x), wkv_state, x_prev, ssd_state, conv_state
        )
        x = residual + core_out
        
        # ═══ Cache h_mean once (used by RAG, NoveltyGate, Memory) ═══
        h_mean = x.mean(dim=1)  # [B, d_model]
        
        # ═══ RAG injection ═══
        if rag_state is not None:
            q = self.rag_query(h_mean)                                # [B, d_state]
            info = torch.bmm(rag_state, q.unsqueeze(-1)).squeeze(-1)  # [B, d_state]
            x = x + 0.1 * self.rag_out(info).unsqueeze(1)
        
        # ═══ 2. Ω-SSM ═══
        x = self.omega(x)
        
        # ═══ 3. MoLE (returns aux_loss for load balancing) ═══
        x, mole_aux_loss = self.mole(x)
        
        # ═══ 4. NoveltyGate — adaptive residual ═══
        h_old = residual.mean(dim=1)                             # [B, d_model]
        h_new = x.mean(dim=1)                                    # [B, d_model]
        _, novelty = self.novelty_gate(h_old, h_new)             # novelty: [B]
        n = novelty.unsqueeze(-1).unsqueeze(-1)                  # [B, 1, 1]
        x = n * x + (1 - n) * residual
        
        # ═══ 5. Dynamic Memory Injection (спинной мозг → кора) ═══
        mem_relevance = 0.0
        if memory_vec is not None:
            # Re-compute h_mean after novelty gate changed x
            h_mean_post = x.mean(dim=1)                               # [B, d_model]
            h_query = self.mem_query_proj(h_mean_post)                # [B, 384]
            similarity = F.cosine_similarity(h_query, memory_vec, dim=-1)  # [B]
            gate_input = torch.cat([h_mean_post, memory_vec], dim=-1) # [B, d_model+384]
            gate = self.mem_gate(gate_input).squeeze(-1)              # [B]
            mem_strength = (similarity * gate).unsqueeze(-1).unsqueeze(-1)
            mem_signal = self.mem_proj(memory_vec).unsqueeze(1)       # [B, 1, d_model]
            x = x + mem_strength * mem_signal
            mem_relevance = similarity.mean().item()
            self.last_surprise = gate.mean().item()
        
        # Stats
        self.last_stats = {
            "layer_idx": self.layer_idx,
            "novelty": novelty.mean().item() if isinstance(novelty, torch.Tensor) else novelty,
            "has_rag": rag_state is not None,
            "mem_relevance": mem_relevance,
            "surprise": self.last_surprise,
            "mole_aux_loss": mole_aux_loss,
            "mole_experts": getattr(self.mole, '_last_expert_names', []),
        }
        
        return x, wkv_state, x_prev, self.last_stats, ssd_state, conv_state
