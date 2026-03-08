"""
═══════════════════════════════════════════════════════════════
  power_manager.py — CPU Power Management (Agent 4)
═══════════════════════════════════════════════════════════════

Platform-aware CPU power management for 24/7 inference.
Detects thermal throttling, manages process priority,
and optionally pins CPU affinity.

Usage:
    from utils.power_manager import PowerManager

    pm = PowerManager()
    pm.set_high_priority()       # boost process priority
    pm.check_thermal()           # warn if CPU is hot
    print(pm.cpu_info())         # CPU name, cores, features
"""

import os
import sys
import logging
import platform
from typing import Optional, Dict, Any

logger = logging.getLogger("Tars.PowerManager")

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


class PowerManager:
    """
    CPU power management for TARS inference.

    Features:
        - Process priority control (Windows/Linux)
        - Thermal throttling detection
        - CPU feature detection (AVX2, AVX-512, VNNI)
        - Affinity pinning (optional)
    """

    # Thermal thresholds (°C)
    TEMP_WARN = 80
    TEMP_CRITICAL = 95

    def __init__(self):
        self._system = platform.system()
        self._cpu_info_cache: Optional[Dict] = None

    def set_high_priority(self) -> bool:
        """
        Set process to high (but not realtime) priority.

        Windows: ABOVE_NORMAL_PRIORITY_CLASS
        Linux: nice -5

        Returns:
            True if successful
        """
        if psutil is None:
            logger.debug("psutil not available for priority control")
            return False

        try:
            p = psutil.Process()
            if self._system == "Windows":
                p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            else:
                # Unix: lower nice = higher priority (range -20 to 19)
                current = p.nice()
                if current > -5:
                    try:
                        p.nice(-5)
                    except PermissionError:
                        p.nice(0)  # at least normal
            logger.info(f"Process priority set: nice={p.nice()}")
            return True
        except Exception as e:
            logger.debug(f"Could not set priority: {e}")
            return False

    def set_normal_priority(self) -> bool:
        """Reset process priority to normal."""
        if psutil is None:
            return False
        try:
            p = psutil.Process()
            if self._system == "Windows":
                p.nice(psutil.NORMAL_PRIORITY_CLASS)
            else:
                p.nice(0)
            return True
        except Exception:
            return False

    def check_thermal(self) -> Optional[float]:
        """
        Check CPU temperature.

        Returns:
            temperature in °C, or None if unavailable
        """
        if psutil is None:
            return None

        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return None

            # Find CPU temperature (varies by platform)
            for name in ["coretemp", "k10temp", "cpu_thermal", "cpu-thermal"]:
                if name in temps:
                    entries = temps[name]
                    if entries:
                        current = max(e.current for e in entries)
                        if current >= self.TEMP_CRITICAL:
                            logger.critical(
                                f"CPU CRITICAL: {current}°C >= {self.TEMP_CRITICAL}°C!"
                            )
                        elif current >= self.TEMP_WARN:
                            logger.warning(
                                f"CPU warm: {current}°C >= {self.TEMP_WARN}°C"
                            )
                        return current
        except Exception:
            pass
        return None

    def pin_affinity(self, cores: Optional[list] = None) -> bool:
        """
        Pin process to specific CPU cores.

        Args:
            cores: list of core indices (None = use physical cores only,
                   no hyperthreads)

        Returns:
            True if successful
        """
        if psutil is None:
            return False

        try:
            p = psutil.Process()
            if cores is None:
                # Use physical cores only (no hyperthreads)
                physical = psutil.cpu_count(logical=False) or 1
                cores = list(range(physical))

            p.cpu_affinity(cores)
            logger.info(f"CPU affinity set to cores: {cores}")
            return True
        except Exception as e:
            logger.debug(f"Could not set CPU affinity: {e}")
            return False

    def cpu_info(self) -> Dict[str, Any]:
        """
        Get CPU information and feature flags.

        Returns:
            dict with keys: name, cores_physical, cores_logical,
                           freq_mhz, features (list)
        """
        if self._cpu_info_cache is not None:
            return self._cpu_info_cache

        info: Dict[str, Any] = {
            "name": platform.processor() or "unknown",
            "arch": platform.machine(),
            "cores_physical": 1,
            "cores_logical": 1,
            "freq_mhz": 0,
            "features": [],
        }

        if psutil is not None:
            info["cores_physical"] = psutil.cpu_count(logical=False) or 1
            info["cores_logical"] = psutil.cpu_count(logical=True) or 1
            try:
                freq = psutil.cpu_freq()
                if freq:
                    info["freq_mhz"] = int(freq.current)
            except Exception:
                pass

        # Detect SIMD features (best-effort)
        features = []
        try:
            # Try to detect via cpuinfo package
            import cpuinfo
            cpu_flags = cpuinfo.get_cpu_info().get("flags", [])
            for flag in ["avx2", "avx512f", "avx512_vnni", "avx512bw",
                        "amx_int8", "amx_bf16", "sse4_2", "fma"]:
                if flag in cpu_flags:
                    features.append(flag)
        except ImportError:
            # Fallback: check via platform
            if platform.machine() in ("x86_64", "AMD64"):
                features.append("x86_64 (AVX2 likely)")

        info["features"] = features
        self._cpu_info_cache = info
        return info

    def memory_status(self) -> Dict[str, float]:
        """Get RAM usage status."""
        if psutil is None:
            return {"available_gb": -1, "used_pct": -1}

        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024 ** 3), 1),
            "available_gb": round(mem.available / (1024 ** 3), 1),
            "used_pct": mem.percent,
        }

    def summary(self) -> str:
        """One-line status summary."""
        info = self.cpu_info()
        mem = self.memory_status()
        temp = self.check_thermal()
        temp_str = f"{temp:.0f}°C" if temp else "N/A"
        return (
            f"CPU: {info['name']} "
            f"({info['cores_physical']}P/{info['cores_logical']}L cores, "
            f"{info['freq_mhz']}MHz) | "
            f"RAM: {mem.get('available_gb', '?')}GB free "
            f"({mem.get('used_pct', '?')}% used) | "
            f"Temp: {temp_str}"
        )
