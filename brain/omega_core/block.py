"""
═══════════════════════════════════════════════════════════════
  brain/omega_core/block.py — TarsCoreBlock Re-export (Agent 4)
═══════════════════════════════════════════════════════════════

Bridge module: re-exports TarsCoreBlock from the canonical
implementation in brain/mamba2/core/ssd.py.

This matches the plan's directory structure without breaking
backward compatibility or duplicating code.

Usage:
    from brain.omega_core.block import TarsCoreBlock
"""

from brain.mamba2.core.ssd import (
    TarsCoreBlock,
    ssd_scan,
    ssd_step,
    wkv_scan,
    CausalConv1d,
)

__all__ = [
    "TarsCoreBlock",
    "ssd_scan",
    "ssd_step",
    "wkv_scan",
    "CausalConv1d",
]
