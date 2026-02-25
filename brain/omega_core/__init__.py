# brain/omega_core/__init__.py
"""
OmegaCore — High-Performance C++ Kernel for TARS v3.
Provides: BitLinear 1.58-bit, SSM Step, Cayley SO(n), Advanced Sampler,
          Integral Auditor (p-convergence), Hankel Rank Detector.
Falls back to pure Python/NumPy if DLL not compiled.
"""
from .omega_core_py import get_omega_core, OmegaCore, OmegaCoreFallback

__all__ = ["get_omega_core", "OmegaCore", "OmegaCoreFallback"]
