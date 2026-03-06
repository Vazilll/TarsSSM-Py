"""T16: _TRIL_MASK_CACHE LRU eviction + thread safety."""
import pytest
import threading
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTrilMaskCache:
    """LRU eviction + thread safety + no clone."""

    def test_lru_eviction(self):
        from brain.mamba2.ssd import _TRIL_MASK_CACHE, _get_tril_masks, _TRIL_MASK_MAX
        _TRIL_MASK_CACHE.clear()
        for t in range(12):
            _get_tril_masks(t + 1, torch.device('cpu'))
        assert len(_TRIL_MASK_CACHE) <= _TRIL_MASK_MAX

    def test_no_clone(self):
        from brain.mamba2.ssd import _TRIL_MASK_CACHE, _get_tril_masks
        _TRIL_MASK_CACHE.clear()
        m1_exc, m1_inc = _get_tril_masks(4, torch.device('cpu'))
        m2_exc, m2_inc = _get_tril_masks(4, torch.device('cpu'))
        assert m1_exc is m2_exc and m1_inc is m2_inc

    def test_thread_safety(self):
        from brain.mamba2.ssd import _TRIL_MASK_CACHE, _get_tril_masks
        _TRIL_MASK_CACHE.clear()
        errors = []
        def worker(tid):
            try:
                for i in range(10):
                    _get_tril_masks(tid * 10 + i + 1, torch.device('cpu'))
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert not errors, f"Thread errors: {errors}"
