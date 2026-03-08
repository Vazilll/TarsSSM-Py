"""
═══════════════════════════════════════════════════════════════
  WuNeng Fusion — Deep Gated Fusion Gate (Agent 4)
═══════════════════════════════════════════════════════════════

Standalone WuNeng fusion module.
gate = σ(W · [y_ssd; y_wkv])
y = gate · y_ssd + (1 - gate) · y_wkv

This is a NEURAL module: the gate is a learned matrix.
No if/else — the model learns to balance SSD vs WKV per-token.

Agent 1 (C++) will mirror this as:
    gate = sigmoid(bitnet_matmul(W_gate, concat(y_ssd, y_wkv)))

Usage:
    from brain.omega_core.wuneng_fusion import WuNengFusion

    fusion = WuNengFusion(d_inner=2048)
    y_fused = fusion(y_ssd, y_wkv)  # [B, L, d_inner]
"""

import torch
import torch.nn as nn
from typing import Tuple

# Use UniversalLinear for 1.58-bit compatibility
try:
    from brain.mamba2.bitnet import UniversalLinear
except ImportError:
    from torch.nn import Linear as UniversalLinear  # type: ignore


class WuNengFusion(nn.Module):
    """
    Deep Gated Fusion (WuNeng style).

    gate = σ(MLP([y_ssd; y_wkv]))
    y = gate · y_ssd + (1 - gate) · y_wkv

    Architecture:
        concat(y_ssd, y_wkv) → Linear(2d, d) → SiLU → Linear(d, d) → Sigmoid
        → gate ∈ (0, 1)^d

    This is NOT a simple scalar gate. It's a dimension-wise gate:
    each feature dimension independently decides SSD vs WKV ratio.

    Parameters:
        For d_inner=2048: ~8M params in gate MLP
        Trainable end-to-end (backprop through sigmoid)
    """

    def __init__(
        self,
        d_inner: int,
        *,
        quant_mode: str = "fp16",
        init_bias: float = 0.0,  # sigmoid(0) = 0.5 → equal split
    ):
        super().__init__()
        self.d_inner = d_inner

        self.gate = nn.Sequential(
            UniversalLinear(d_inner * 2, d_inner, bias=True, mode=quant_mode),
            nn.SiLU(),
            UniversalLinear(d_inner, d_inner, bias=True, mode=quant_mode),
            nn.Sigmoid(),
        )

        # Initialize final bias for desired initial split
        if init_bias != 0.0 and hasattr(self.gate[-2], 'bias') and self.gate[-2].bias is not None:
            nn.init.constant_(self.gate[-2].bias, init_bias)

    def forward(
        self,
        y_ssd: torch.Tensor,   # [B, L, d_inner]
        y_wkv: torch.Tensor,   # [B, L, d_inner]
    ) -> torch.Tensor:
        """
        Fuse SSD and WKV outputs via learned gate.

        Returns:
            y_fused: [B, L, d_inner]
        """
        gate = self.gate(torch.cat([y_ssd, y_wkv], dim=-1))  # [B, L, d_inner]
        return gate * y_ssd + (1 - gate) * y_wkv

    def get_gate_stats(
        self,
        y_ssd: torch.Tensor,
        y_wkv: torch.Tensor,
    ) -> dict:
        """Get gate statistics for logging/debugging."""
        with torch.no_grad():
            gate = self.gate(torch.cat([y_ssd, y_wkv], dim=-1))
            return {
                "gate_mean": gate.mean().item(),
                "gate_std": gate.std().item(),
                "gate_min": gate.min().item(),
                "gate_max": gate.max().item(),
                "ssd_weight": gate.mean().item(),
                "wkv_weight": 1.0 - gate.mean().item(),
            }

    def extra_repr(self) -> str:
        return f"d_inner={self.d_inner}"


class FlashSigmoidFusion(nn.Module):
    """
    Lightweight fusion: single matrix + sigmoid.

    gate = σ(W · x)     ← 1 matmul only (no MLP)
    y = gate · y_ssd + (1 - gate) · y_wkv

    ~4x fewer params than WuNengFusion.
    Suitable for when d_inner is large and parameter budget is tight.
    """

    def __init__(self, d_inner: int, *, quant_mode: str = "fp16"):
        super().__init__()
        self.d_inner = d_inner
        self.gate_proj = UniversalLinear(
            d_inner * 2, d_inner, bias=True, mode=quant_mode
        )

    def forward(self, y_ssd: torch.Tensor, y_wkv: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(
            torch.cat([y_ssd, y_wkv], dim=-1)
        ))
        return gate * y_ssd + (1 - gate) * y_wkv

    def extra_repr(self) -> str:
        return f"d_inner={self.d_inner}, mode=flash"
