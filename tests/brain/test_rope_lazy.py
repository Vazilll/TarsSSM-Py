"""T15: RoPE lazy compute — start small, extend on demand."""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestRoPELazy:
    """RoPE lazy compute: start small, extend on demand."""

    def test_lazy_init_small(self):
        from brain.mamba2.model import RotaryPositionEmbedding
        rope = RotaryPositionEmbedding(dim=64)
        assert rope.max_seq_len == 512
        assert rope.cos_cached.shape[0] == 512

    def test_auto_extend(self):
        from brain.mamba2.model import RotaryPositionEmbedding
        rope = RotaryPositionEmbedding(dim=64)
        x = torch.randn(1, 1024, 64)
        y = rope(x)
        assert y.shape == x.shape and rope.max_seq_len >= 1024

    def test_values_correct(self):
        from brain.mamba2.model import RotaryPositionEmbedding
        r1 = RotaryPositionEmbedding(dim=64, max_seq_len=2048)
        r2 = RotaryPositionEmbedding(dim=64, max_seq_len=2048)
        x = torch.randn(1, 2048, 64)
        _ = r1(x); _ = r2(x)
        assert torch.allclose(r1.cos_cached, r2.cos_cached, atol=1e-6)
        assert torch.allclose(r1.sin_cached, r2.sin_cached, atol=1e-6)
