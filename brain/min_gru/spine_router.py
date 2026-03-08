"""
═══════════════════════════════════════════════════════════════
  SpineRouter — Block Skip Predictor (Agent 4)
═══════════════════════════════════════════════════════════════

MinGRU-based predictor that learns which blocks to skip.
This is a NEURAL module: entirely learned, no rules.

Input:  hidden state after each block
Output: p_skip[i] per block — probability to skip block i

Agent 1 mirrors: spine_router weights are just small matrices
in the C++ engine, executed as bitnet_matmul.

Usage:
    from brain.min_gru.spine_router import SpineRouter

    router = SpineRouter(d_model=2048, n_blocks=24)

    # During inference, predict which blocks to skip:
    skip_probs = router(hidden_state)  # [B, n_blocks]

    for i, block in enumerate(blocks):
        if skip_probs[0, i] > threshold:
            continue  # skip this block
        x = block(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import MinGRU from sibling module
try:
    from brain.min_gru.mingru import MinGRU
except ImportError:
    MinGRU = None  # type: ignore


class SpineRouter(nn.Module):
    """
    Spine Router: learns to skip blocks adaptively.

    Architecture:
        x → project(d_model → spine_dim)
        → MinGRU(spine_dim)   ← recurrent: accumulates block context
        → Linear(spine_dim → n_blocks) → sigmoid
        → p_skip ∈ [0, 1]^n_blocks

    The MinGRU state carries context from previous blocks,
    so the router can make informed decisions about which
    future blocks to skip based on accumulated information.

    Parameters:
        ~1M params for d_model=2048, spine_dim=256, n_blocks=24
    """

    def __init__(
        self,
        d_model: int = 2048,
        n_blocks: int = 24,
        spine_dim: int = 256,
        skip_threshold: float = 0.6,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.spine_dim = spine_dim
        self.skip_threshold = skip_threshold

        # Project from d_model to smaller spine space
        self.project_in = nn.Linear(d_model, spine_dim, bias=False)

        # MinGRU accumulates block-level context
        if MinGRU is not None:
            self.rnn = MinGRU(spine_dim, expansion_factor=1.0)
        else:
            # Fallback: simple GRU
            self.rnn = nn.GRU(spine_dim, spine_dim, batch_first=True)

        # Predict skip probability per block
        self.skip_head = nn.Sequential(
            nn.Linear(spine_dim, spine_dim),
            nn.SiLU(),
            nn.Linear(spine_dim, n_blocks),
        )

        # Initialize to NOT skip (conservative start)
        nn.init.constant_(self.skip_head[-1].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(
        self,
        hidden_state: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict block skip probabilities.

        Args:
            hidden_state: [B, d_model] — current hidden state (mean-pooled)
            prev_hidden:  [B, 1, spine_dim] — previous RNN hidden (optional)

        Returns:
            skip_probs: [B, n_blocks] — probability to skip each block
        """
        # Project to spine space
        x = self.project_in(hidden_state)  # [B, spine_dim]
        x = x.unsqueeze(1)  # [B, 1, spine_dim] for RNN

        # Run through MinGRU
        if isinstance(self.rnn, nn.GRU):
            h0 = prev_hidden.squeeze(1).unsqueeze(0) if prev_hidden is not None else None
            rnn_out, _ = self.rnn(x, h0)
        else:
            # MinGRU
            rnn_out = self.rnn(x, prev_hidden=prev_hidden)

        # Predict skip probs
        rnn_feat = rnn_out.squeeze(1)  # [B, spine_dim]
        skip_logits = self.skip_head(rnn_feat)  # [B, n_blocks]
        skip_probs = torch.sigmoid(skip_logits)

        return skip_probs

    def get_skip_mask(
        self,
        hidden_state: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Get binary skip mask.

        Args:
            hidden_state: [B, d_model]
            threshold: override default skip_threshold

        Returns:
            skip_mask: [B, n_blocks] — True where block should be skipped
        """
        if threshold is None:
            threshold = self.skip_threshold

        skip_probs = self.forward(hidden_state, prev_hidden)
        return skip_probs > threshold

    def compute_efficiency(self, skip_mask: torch.Tensor) -> float:
        """Fraction of blocks skipped (0=none, 1=all)."""
        return skip_mask.float().mean().item()

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, n_blocks={self.n_blocks}, "
                f"spine_dim={self.spine_dim}, threshold={self.skip_threshold}")
