"""
═══════════════════════════════════════════════════════════════
  tensor_pool.py — Pre-allocated Tensor Cache (Agent 4)
═══════════════════════════════════════════════════════════════

Reduces allocation pressure during inference by reusing tensors.
Thread-safe, shape-keyed cache with LRU eviction.

Usage:
    from utils.tensor_pool import TensorPool

    pool = TensorPool(max_tensors=64)
    t = pool.get((4, 2048), dtype=torch.float32)   # from cache or new
    pool.release(t)                                  # back to pool

    # Context manager:
    with pool.borrow((4, 2048)) as t:
        result = model(t)    # t returned to pool on exit
"""

import threading
from collections import defaultdict
from typing import Tuple, Optional
import logging

logger = logging.getLogger("Tars.TensorPool")

try:
    import torch
except ImportError:
    torch = None  # type: ignore


class TensorPool:
    """
    Shape-keyed tensor cache with thread-safe access.

    Pre-allocates and reuses tensors to avoid repeated malloc/free.
    Especially useful for inference hot path where shapes are predictable.
    """

    def __init__(self, max_tensors: int = 64, device: str = "cpu"):
        """
        Args:
            max_tensors: max total tensors in pool across all shapes
            device: default device for new tensors
        """
        self.max_tensors = max_tensors
        self.device = device
        self._lock = threading.Lock()
        # Key: (shape, dtype_str) → list of available tensors
        self._pool: dict = defaultdict(list)
        self._total = 0
        self._hits = 0
        self._misses = 0

    def _key(self, shape: Tuple[int, ...], dtype) -> tuple:
        """Create cache key from shape and dtype."""
        dtype_str = str(dtype) if dtype is not None else "float32"
        return (tuple(shape), dtype_str)

    def get(self, shape: Tuple[int, ...], dtype=None,
            zero: bool = True) -> "torch.Tensor":
        """
        Get a tensor from pool or allocate new one.

        Args:
            shape: desired tensor shape
            dtype: torch dtype (default: float32)
            zero: zero-fill the tensor before returning

        Returns:
            tensor of requested shape and dtype
        """
        if torch is None:
            raise RuntimeError("PyTorch not available")

        if dtype is None:
            dtype = torch.float32

        key = self._key(shape, dtype)

        with self._lock:
            if self._pool[key]:
                tensor = self._pool[key].pop()
                self._total -= 1
                self._hits += 1
                if zero:
                    tensor.zero_()
                return tensor

        # Cache miss: allocate new
        self._misses += 1
        if zero:
            return torch.zeros(shape, dtype=dtype, device=self.device)
        return torch.empty(shape, dtype=dtype, device=self.device)

    def release(self, tensor: "torch.Tensor") -> None:
        """Return a tensor to the pool for reuse."""
        if torch is None or tensor is None:
            return

        key = self._key(tuple(tensor.shape), tensor.dtype)

        with self._lock:
            if self._total < self.max_tensors:
                # Detach from any computation graph
                t = tensor.detach()
                self._pool[key].append(t)
                self._total += 1
            # else: let GC collect it

    def borrow(self, shape: Tuple[int, ...], dtype=None,
               zero: bool = True):
        """Context manager: get tensor, auto-release on exit."""
        return _BorrowedTensor(self, shape, dtype, zero)

    def clear(self) -> None:
        """Clear all cached tensors."""
        with self._lock:
            self._pool.clear()
            self._total = 0

    def stats(self) -> dict:
        """Return pool statistics."""
        with self._lock:
            total_cached = sum(len(v) for v in self._pool.values())
            return {
                "cached_tensors": total_cached,
                "unique_shapes": len(self._pool),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(self._hits + self._misses, 1),
            }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"TensorPool(cached={s['cached_tensors']}, "
                f"shapes={s['unique_shapes']}, "
                f"hit_rate={s['hit_rate']:.1%})")


class _BorrowedTensor:
    """Context manager for auto-releasing tensors."""

    def __init__(self, pool: TensorPool, shape, dtype, zero):
        self.pool = pool
        self.shape = shape
        self.dtype = dtype
        self.zero = zero
        self.tensor = None

    def __enter__(self):
        self.tensor = self.pool.get(self.shape, self.dtype, self.zero)
        return self.tensor

    def __exit__(self, *args):
        if self.tensor is not None:
            self.pool.release(self.tensor)
            self.tensor = None


# ═══════════════════════════════════════
# Global pool (lazy singleton)
# ═══════════════════════════════════════
_global_pool: Optional[TensorPool] = None


def get_tensor_pool(max_tensors: int = 64) -> TensorPool:
    """Get global tensor pool singleton."""
    global _global_pool
    if _global_pool is None:
        _global_pool = TensorPool(max_tensors=max_tensors)
    return _global_pool
