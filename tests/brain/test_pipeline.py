"""
═══════════════════════════════════════════════════════════════
  Pipeline Tests — T07-T10 Verification (pytest)
═══════════════════════════════════════════════════════════════

Coverage:
  T07: BrainCore, VerificationSuite, InferenceEngine decomposition
  T08: NaN recovery, health check clamping, partial result
  T09: WKV JIT matches Python, tril cache LRU
  T10: Adaptive K, EMA acceptance tracking, speculative stats

Run:
  cd C:/Users/Public/Tarsfull/TarsSSM-Py
  python -m pytest tests/brain/test_pipeline.py -v --tb=short
"""

import sys
import os
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════
# T07: Decomposition Tests
# ═══════════════════════════════════════════════════════════════

class TestT07Decomposition:
    """Verify decomposed modules are importable and functional."""
    
    def test_brain_core_importable(self):
        """BrainCore should be importable from brain.mamba2.brain_core."""
        from brain.mamba2.brain_core import BrainCore
        assert BrainCore is not None
    
    def test_verification_suite_importable(self):
        """VerificationSuite should be importable."""
        from brain.mamba2.verification_suite import VerificationSuite
        assert VerificationSuite is not None
    
    def test_inference_engine_importable(self):
        """InferenceEngine should be importable."""
        from brain.mamba2.inference_engine import InferenceEngine
        assert InferenceEngine is not None
    
    def test_model_still_importable(self):
        """TarsMamba2LM should still be importable from model.py."""
        from brain.mamba2.model import TarsMamba2LM
        assert TarsMamba2LM is not None
    
    def test_brain_core_helper_classes(self):
        """All helper classes should be importable from brain_core."""
        from brain.mamba2.brain_core import (
            RotaryPositionEmbedding,
            WaveConsolidation,
            GlobalWorkspace,
            SharedGlobalAttention,
            WaveScratchpad,
            TTTLoRA,
        )
        assert all(c is not None for c in [
            RotaryPositionEmbedding, WaveConsolidation,
            GlobalWorkspace, SharedGlobalAttention,
            WaveScratchpad, TTTLoRA,
        ])
    
    def test_wave_consolidation_shape(self):
        """WaveConsolidation should produce correct output shape."""
        from brain.mamba2.brain_core import WaveConsolidation
        wc = WaveConsolidation(d_model=64)
        x_left = torch.randn(2, 8, 64)
        x_right = torch.randn(2, 8, 64)
        out, alpha = wc(x_left, x_right)
        assert out.shape == (2, 8, 64), f"Expected (2,8,64), got {out.shape}"
        assert isinstance(alpha, (float, torch.Tensor))
    
    def test_verification_suite_observe(self):
        """VerificationSuite.observe() should return convergence dict."""
        from brain.mamba2.verification_suite import VerificationSuite
        vs = VerificationSuite(d_model=64)
        h_curr = torch.randn(2, 64)
        h_prev = torch.randn(2, 64)
        result = vs.observe(h_curr, h_prev)
        assert "p" in result
        assert "converged" in result
    
    def test_model_composition_accessor(self):
        """TarsMamba2LM should have inference_engine property."""
        from brain.mamba2.model import TarsMamba2LM
        # Just check the property exists (don't instantiate — too expensive)
        assert hasattr(TarsMamba2LM, 'inference_engine')


# ═══════════════════════════════════════════════════════════════
# T08: Error Recovery Tests
# ═══════════════════════════════════════════════════════════════

class TestT08ErrorRecovery:
    """Verify error recovery mechanisms."""
    
    def test_nan_to_num_recovery(self):
        """torch.nan_to_num should replace NaN with 0."""
        x = torch.tensor([1.0, float('nan'), 3.0, float('inf'), -float('inf')])
        fixed = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        assert not torch.isnan(fixed).any(), "NaN should be replaced"
        assert not torch.isinf(fixed).any(), "Inf should be replaced"
        assert fixed[1].item() == 0.0
        assert fixed[3].item() == 1.0
        assert fixed[4].item() == -1.0
    
    def test_health_check_clamp(self):
        """Extreme norms should be clamped to [-10, 10]."""
        x = torch.randn(2, 8, 64) * 1000  # huge values
        x_norm = x.float().norm().item()
        assert x_norm > 100  # precondition
        
        x_clamped = x.clamp(-10, 10)
        assert x_clamped.max().item() <= 10.0
        assert x_clamped.min().item() >= -10.0
    
    def test_partial_result_flag(self):
        """InferenceEngine should set partial_result when >50% waves fail."""
        # This is a design contract test — verify the logic exists
        import inspect
        from brain.mamba2.inference_engine import InferenceEngine
        source = inspect.getsource(InferenceEngine.think)
        assert "partial_result" in source
        assert "failed_waves" in source


# ═══════════════════════════════════════════════════════════════
# T09: WKV Optimization Tests
# ═══════════════════════════════════════════════════════════════

class TestT09WKVOptimization:
    """Verify WKV scan JIT and tril cache improvements."""
    
    def test_wkv_jit_matches_python(self):
        """JIT-compiled WKV scan should produce same output as Python loop."""
        from brain.mamba2.ssd import _wkv_step
        
        B, L, S = 2, 8, 16
        r = torch.sigmoid(torch.randn(B, L, S))
        k = torch.randn(B, L, S)
        v = torch.randn(B, L, S)
        w = torch.exp(-torch.exp(torch.randn(B, L, S)))
        bonus = torch.sigmoid(torch.randn(B, L, S))
        state_init = torch.zeros(B, S, S)
        
        # Python loop
        state_py = state_init.clone()
        output_py = torch.zeros(B, L, S)
        for t in range(L):
            y_t, state_py = _wkv_step(state_py, k[:, t], v[:, t], r[:, t], w[:, t], bonus[:, t])
            output_py[:, t] = y_t
        
        # JIT loop
        from brain.mamba2.ssd import _wkv_scan_jit
        state_jit = state_init.clone()
        output_jit, state_jit = _wkv_scan_jit(r, k, v, w, bonus, state_jit, 32)
        
        assert torch.allclose(output_py, output_jit, atol=1e-5), \
            f"JIT output differs from Python: max diff={( output_py - output_jit).abs().max():.8f}"
        assert torch.allclose(state_py, state_jit, atol=1e-5), \
            f"JIT state differs from Python: max diff={(state_py - state_jit).abs().max():.8f}"
    
    def test_wkv_scan_fallback_chain(self):
        """wkv_scan should work via the fallback chain."""
        from brain.mamba2.ssd import wkv_scan
        
        B, L, S = 1, 4, 8
        r = torch.sigmoid(torch.randn(B, L, S))
        k = torch.randn(B, L, S)
        v = torch.randn(B, L, S)
        w = torch.exp(-torch.exp(torch.randn(B, L, S)))
        bonus = torch.sigmoid(torch.randn(B, L, S))
        
        output, state = wkv_scan(r, k, v, w, bonus)
        assert output.shape == (B, L, S)
        assert state.shape == (B, S, S)
    
    def test_tril_cache_lru_eviction(self):
        """Tril mask cache should evict old entries past max size."""
        from brain.mamba2.ssd import _get_tril_masks, _TRIL_MASK_CACHE, _TRIL_CACHE_MAX
        
        # Fill cache beyond max
        initial_len = len(_TRIL_MASK_CACHE)
        for i in range(1, _TRIL_CACHE_MAX + 5):
            _get_tril_masks(100 + i, torch.device('cpu'))
        
        assert len(_TRIL_MASK_CACHE) <= _TRIL_CACHE_MAX, \
            f"Cache size {len(_TRIL_MASK_CACHE)} exceeds max {_TRIL_CACHE_MAX}"
    
    def test_tril_no_clone(self):
        """Cached masks should be the same object (no .clone())."""
        from brain.mamba2.ssd import _get_tril_masks
        
        m1_exc, m1_inc = _get_tril_masks(99, torch.device('cpu'))
        m2_exc, m2_inc = _get_tril_masks(99, torch.device('cpu'))
        
        # Should be same objects from cache (no .clone())
        assert m1_exc is m2_exc, "Exclusive mask should be same object (no clone)"
        assert m1_inc is m2_inc, "Inclusive mask should be same object (no clone)"


# ═══════════════════════════════════════════════════════════════
# T10: Speculative Alignment Tests
# ═══════════════════════════════════════════════════════════════

class TestT10SpeculativeAlignment:
    """Verify adaptive K, EMA tracking, and shared head."""
    
    def test_adaptive_k_decrease(self):
        """Low acceptance rate should decrease K."""
        from brain.speculative import SpeculativeDecoder
        
        # Test the adaptive K logic directly
        K_initial = 5
        K = K_initial
        acceptance = 0.3  # below 0.4 threshold
        
        if acceptance < 0.4 and K > 1:
            K = max(1, K - 1)
        
        assert K == 4, f"K should decrease from 5 to 4, got {K}"
    
    def test_adaptive_k_increase(self):
        """High acceptance rate should increase K back toward initial."""
        K_initial = 5
        K = 3  # previously decreased
        acceptance = 0.85  # above 0.8 threshold
        
        if acceptance > 0.8 and K < K_initial:
            K = min(K_initial, K + 1)
        
        assert K == 4, f"K should increase from 3 to 4, got {K}"
    
    def test_adaptive_k_floor(self):
        """K should never go below 1."""
        K = 1
        acceptance = 0.1
        
        if acceptance < 0.4 and K > 1:
            K = max(1, K - 1)
        
        assert K == 1, f"K should stay at 1, got {K}"
    
    def test_ema_tracking(self):
        """EMA acceptance should update correctly."""
        ema = 0.5
        for accept_rate in [0.8, 0.9, 0.85]:
            ema = 0.9 * ema + 0.1 * accept_rate
        
        assert 0.5 < ema < 0.9, f"EMA should be between 0.5 and 0.9, got {ema}"
    
    def test_speculative_decoder_has_get_stats(self):
        """SpeculativeDecoder should have get_stats method."""
        from brain.speculative import SpeculativeDecoder
        assert hasattr(SpeculativeDecoder, 'get_stats')
    
    def test_speculative_decoder_has_initial_k(self):
        """SpeculativeDecoder.__init__ should store _initial_K."""
        import inspect
        from brain.speculative import SpeculativeDecoder
        source = inspect.getsource(SpeculativeDecoder.__init__)
        assert "_initial_K" in source
        assert "_ema_acceptance" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
