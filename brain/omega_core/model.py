"""
═══════════════════════════════════════════════════════════════
  brain/omega_core/model.py — TarsModel Re-export (Agent 4)
═══════════════════════════════════════════════════════════════

Bridge module: re-exports TarsMamba2LM as TarsModel from
the canonical implementation in brain/mamba2/core/model.py.

Usage:
    from brain.omega_core.model import TarsModel
"""

from brain.mamba2.core.model import TarsMamba2LM as TarsModel

__all__ = ["TarsModel"]
