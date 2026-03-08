"""
tars_core — C++/Rust Inference Engine (Type Stubs)

Auto-generated type stubs for the tars_core native module.
Provides IDE autocompletion and type checking.

Install: copy to site-packages/tars_core.pyi or use py.typed
"""

from typing import List, Tuple, Optional


class TarsEngine:
    """TARS inference engine (C++/Rust backend)."""

    def __init__(self) -> None:
        """Create new engine with default config (d=2048, L=24, vocab=32K)."""
        ...

    def load(self, path: str) -> None:
        """Load model weights from .safetensors file.

        Args:
            path: Path to model.safetensors

        Raises:
            IOError: If file not found or invalid format
        """
        ...

    def generate(
        self,
        prompt_ids: List[int],
        max_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[int]:
        """Generate tokens from prompt.

        Args:
            prompt_ids: Input token IDs
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling

        Returns:
            List of generated token IDs

        Raises:
            RuntimeError: If model not loaded
        """
        ...

    def reset_state(self) -> None:
        """Reset all internal states (SSM, caches). Call between conversations."""
        ...

    def memory_stats(self) -> Tuple[int, int]:
        """Get (current_bytes, peak_bytes) arena memory usage."""
        ...

    def is_loaded(self) -> bool:
        """Check if model weights are loaded."""
        ...


# ═══════════════════════════════
# Standalone kernel functions
# ═══════════════════════════════

def bitnet_matmul(
    w_ternary: List[int],
    x: List[float],
    alpha: float,
) -> List[float]:
    """Ternary matmul: y = α · W{-1,0,+1} @ x (AVX2 accelerated)."""
    ...


def rmsnorm(
    x: List[float],
    gamma: List[float],
    eps: float = 1e-6,
) -> List[float]:
    """RMSNorm: y = γ·x/√(mean(x²)+ε) (AVX2 fused 2-pass)."""
    ...


def ssd_scan(
    state: List[float],
    gamma: List[float],
    b: List[float],
    x: List[float],
) -> Tuple[List[float], List[float]]:
    """SSD scan step: s' = γ·s + B·x (AVX2 FMA)."""
    ...


def wkv7_update(
    s: List[float],
    w: List[float],
    a: List[float],
    b: List[float],
    v: List[float],
    k: List[float],
) -> List[float]:
    """WKV-7 state update: S' = S·(diag(w)+aᵀb)+vᵀk."""
    ...


def swiglu_fused(
    w1: List[float],
    w2: List[float],
    x: List[float],
    sparsity_threshold: float = 0.0,
) -> List[float]:
    """SwiGLU: y = SiLU(W₁x)⊙W₂x with optional sparsity."""
    ...
