"""
Tests for TarsBlock — core neural block of TARS v3.

Validates:
  - Output shape preservation
  - Gradient flow through the block
  - Memory injection compatibility
  - State passing (WKV, SSD, conv)
"""
import pytest
import torch


class TestTarsBlock:
    """TarsBlock shape and gradient tests."""

    @pytest.fixture
    def block(self, small_d_model):
        from brain.mamba2.tars_block import TarsBlock
        return TarsBlock(
            d_model=small_d_model,
            d_state=16,
            headdim=16,
            omega_dim=8,
            n_experts=4,
            layer_idx=0,
            quant_mode="fp16",
            dropout=0.0,
        )

    def test_output_shape(self, block, small_d_model):
        """TarsBlock(x) should return output with same shape as input."""
        B, L = 2, 16
        x = torch.randn(B, L, small_d_model)
        output, wkv_state, x_prev, stats, ssd_state, conv_state = block(x)

        assert output.shape == (B, L, small_d_model), \
            f"Expected {(B, L, small_d_model)}, got {output.shape}"

    def test_single_token(self, block, small_d_model):
        """Should work with L=1 (single token for autoregressive generation)."""
        x = torch.randn(1, 1, small_d_model)
        output, *_ = block(x)
        assert output.shape == (1, 1, small_d_model)

    def test_gradient_flow(self, block, small_d_model):
        """loss.backward() should not crash and gradients should reach input."""
        x = torch.randn(2, 8, small_d_model, requires_grad=True)
        output, *_ = block(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradient did not flow back to input"
        assert not torch.isnan(x.grad).any(), "NaN in gradients"

    def test_with_states(self, block, small_d_model):
        """Forward should accept and return carry states."""
        B, L = 1, 4
        x = torch.randn(B, L, small_d_model)

        # First pass: get states
        out1, wkv1, xprev1, _, ssd1, conv1 = block(x)

        # Second pass: use returned states
        x2 = torch.randn(B, L, small_d_model)
        out2, wkv2, xprev2, _, ssd2, conv2 = block(
            x2, wkv_state=wkv1, x_prev=xprev1,
            ssd_state=ssd1, conv_state=conv1,
        )

        assert out2.shape == (B, L, small_d_model)

    def test_with_memory_signal(self, block, small_d_model):
        """Forward with mem_signal should not crash."""
        B, L = 1, 8
        x = torch.randn(B, L, small_d_model)
        mem_signal = torch.randn(B, small_d_model)

        output, *_ = block(x, mem_signal=mem_signal)
        assert output.shape == (B, L, small_d_model)

    def test_deterministic(self, block, small_d_model):
        """Same input should produce same output in eval mode."""
        block.eval()
        x = torch.randn(1, 4, small_d_model)

        with torch.no_grad():
            out1, *_ = block(x)
            out2, *_ = block(x)

        assert torch.allclose(out1, out2, atol=1e-6), "Non-deterministic in eval mode"

    def test_batch_independence(self, block, small_d_model):
        """Different batch elements should produce different outputs."""
        B, L = 3, 4
        x = torch.randn(B, L, small_d_model)
        # Make sure batch items are different
        x[1] = x[0] + 1.0
        x[2] = x[0] - 1.0

        with torch.no_grad():
            block.eval()
            output, *_ = block(x)

        # Outputs should differ between batches
        assert not torch.allclose(output[0], output[1], atol=1e-4)
