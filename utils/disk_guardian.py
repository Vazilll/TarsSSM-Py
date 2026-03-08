"""
═══════════════════════════════════════════════════════════════
  disk_guardian.py — Disk Space Monitor (Agent 4)
═══════════════════════════════════════════════════════════════

Monitors disk space. Prevents writes when critically low.
Thread-safe singleton for use across all TARS modules.

Usage:
    from utils.disk_guardian import DiskGuardian

    guardian = DiskGuardian()
    guardian.check()             # raises if critically low
    guardian.can_write(1e8)      # can we write 100MB?

    with guardian.guard():       # context: checks before & after
        save_model(path)
"""

import os
import shutil
import logging
import threading
from typing import Optional

logger = logging.getLogger("Tars.DiskGuardian")


class DiskGuardian:
    """
    Disk space monitoring and protection.

    Thresholds (configurable):
        warn_gb:  1.0 GB — log warning
        block_gb: 0.5 GB — prevent writes, raise DiskFullError
    """

    _instance: Optional["DiskGuardian"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        path: str = ".",
        warn_gb: float = 1.0,
        block_gb: float = 0.5,
    ):
        if self._initialized:
            return
        self.path = os.path.abspath(path)
        self.warn_gb = warn_gb
        self.block_gb = block_gb
        self._last_free_gb: float = -1
        self._initialized = True

    def free_space_gb(self) -> float:
        """Get free disk space in GB."""
        try:
            usage = shutil.disk_usage(self.path)
            free_gb = usage.free / (1024 ** 3)
            self._last_free_gb = free_gb
            return free_gb
        except OSError as e:
            logger.warning(f"DiskGuardian: cannot check disk: {e}")
            return float("inf")  # fail-open

    def check(self) -> float:
        """
        Check disk space. Warns or raises as needed.

        Returns:
            free space in GB

        Raises:
            DiskFullError: if free space < block_gb
        """
        free_gb = self.free_space_gb()

        if free_gb < self.block_gb:
            msg = (
                f"CRITICAL: Disk nearly full! "
                f"{free_gb:.2f} GB free < {self.block_gb} GB threshold. "
                f"Path: {self.path}"
            )
            logger.critical(msg)
            raise DiskFullError(msg)

        if free_gb < self.warn_gb:
            logger.warning(
                f"Disk space low: {free_gb:.2f} GB free "
                f"(warn threshold: {self.warn_gb} GB). Path: {self.path}"
            )

        return free_gb

    def can_write(self, size_bytes: int = 0) -> bool:
        """
        Check if we can safely write `size_bytes` to disk.

        Args:
            size_bytes: planned write size (0 = just check threshold)

        Returns:
            True if write is safe
        """
        free_gb = self.free_space_gb()
        needed_gb = size_bytes / (1024 ** 3)
        remaining = free_gb - needed_gb
        return remaining >= self.block_gb

    def guard(self):
        """Context manager: check disk before and after operation."""
        return _DiskGuardContext(self)

    def __repr__(self) -> str:
        free = self._last_free_gb
        status = "OK" if free > self.warn_gb else ("WARN" if free > self.block_gb else "CRITICAL")
        return f"DiskGuardian(free={free:.1f}GB, status={status})"


class _DiskGuardContext:
    """Context manager for disk-guarded operations."""

    def __init__(self, guardian: DiskGuardian):
        self.guardian = guardian

    def __enter__(self):
        self.guardian.check()
        return self.guardian

    def __exit__(self, *args):
        # Post-check (informational only, don't suppress exceptions)
        try:
            self.guardian.check()
        except DiskFullError:
            logger.error("DiskGuardian: disk became critically full during operation!")


class DiskFullError(IOError):
    """Raised when disk space is critically low."""
    pass
