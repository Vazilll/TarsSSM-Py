"""
═══════════════════════════════════════════════════════════════
  system_monitor.py — Мониторинг системы TARS v3
═══════════════════════════════════════════════════════════════

"Почему комп тормозит?" → CPU 95%, Chrome жрёт 4GB
"Сколько места на диске?" → C: 92% (осталось 18GB)
Проактивно: "⚠️ Диск почти полон — почистить временные файлы?"
"""

import os
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("Tars.SystemMonitor")


class SystemMonitor:
    """
    Мониторинг ресурсов Windows.
    Проактивно предупреждает о проблемах.
    """
    
    def __init__(self):
        self._alerts: List[str] = []
        self._last_check = {}
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def get_status(self) -> str:
        """Полный статус системы."""
        lines = ["💻 Статус системы:\n"]
        
        cpu = self._get_cpu()
        ram = self._get_ram()
        disk = self._get_disk()
        battery = self._get_battery()
        top_proc = self._get_top_processes()
        
        # CPU
        cpu_bar = self._bar(cpu["percent"])
        lines.append(f"  CPU: [{cpu_bar}] {cpu['percent']:.0f}%")
        
        # RAM
        ram_bar = self._bar(ram["percent"])
        lines.append(f"  RAM: [{ram_bar}] {ram['used_gb']:.1f}/{ram['total_gb']:.1f} GB ({ram['percent']:.0f}%)")
        
        # Disk
        for d in disk:
            disk_bar = self._bar(d["percent"])
            lines.append(f"  {d['drive']}: [{disk_bar}] {d['used_gb']:.0f}/{d['total_gb']:.0f} GB ({d['percent']:.0f}%)")
        
        # Battery
        if battery:
            bat_icon = "🔌" if battery["plugged"] else "🔋"
            lines.append(f"  {bat_icon} Батарея: {battery['percent']}%")
        
        # Top processes
        if top_proc:
            lines.append("\n  📊 Топ процессы по RAM:")
            for p in top_proc[:5]:
                lines.append(f"    {p['name']}: {p['ram_mb']:.0f} MB")
        
        return "\n".join(lines)
    
    def get_alerts(self) -> List[str]:
        """Получить и очистить накопившиеся алерты."""
        alerts = list(self._alerts)
        self._alerts.clear()
        return alerts
    
    def why_slow(self) -> str:
        """Диагностика: почему тормозит?"""
        lines = ["🔍 Диагностика производительности:\n"]
        issues = []
        
        cpu = self._get_cpu()
        ram = self._get_ram()
        disk = self._get_disk()
        top_proc = self._get_top_processes()
        
        if cpu["percent"] > 80:
            issues.append(f"  ⚠️ CPU загружен на {cpu['percent']:.0f}%")
        
        if ram["percent"] > 85:
            issues.append(f"  ⚠️ RAM используется на {ram['percent']:.0f}% ({ram['available_gb']:.1f} GB свободно)")
            if top_proc:
                biggest = top_proc[0]
                issues.append(f"     Больше всего RAM ест: {biggest['name']} ({biggest['ram_mb']:.0f} MB)")
        
        for d in disk:
            if d["percent"] > 90:
                issues.append(f"  ⚠️ Диск {d['drive']} заполнен на {d['percent']:.0f}%")
        
        if not issues:
            lines.append("  ✅ Всё в норме — серьёзных проблем не обнаружено")
        else:
            lines.extend(issues)
            lines.append("\n  💡 Рекомендации:")
            if cpu["percent"] > 80:
                lines.append("  • Закрой неиспользуемые приложения")
            if ram["percent"] > 85:
                lines.append("  • Закрой тяжёлые вкладки браузера")
                lines.append("  • Или перезагрузи компьютер для очистки RAM")
            for d in disk:
                if d["percent"] > 90:
                    lines.append(f"  • Очисти диск {d['drive']}: Temp файлы, Корзина, Downloads")
        
        return "\n".join(lines)
    
    def cleanup_suggestion(self) -> str:
        """Предложения по очистке."""
        lines = ["🧹 Что можно почистить:\n"]
        
        temp_size = self._get_folder_size(os.environ.get("TEMP", "C:\\Temp"))
        lines.append(f"  Temp файлы: ~{temp_size} MB")
        
        downloads = Path.home() / "Downloads"
        if downloads.exists():
            dl_size = self._get_folder_size(str(downloads))
            lines.append(f"  Downloads: ~{dl_size} MB")
        
        recycle = "C:\\$Recycle.Bin"
        lines.append(f"  Корзина: проверь вручную")
        
        lines.append("\n  Скажи «почисти temp» если хочешь удалить временные файлы")
        return "\n".join(lines)
    
    def _get_cpu(self) -> Dict:
        """CPU usage."""
        try:
            import psutil
            return {"percent": psutil.cpu_percent(interval=0.5)}
        except ImportError:
            # Fallback без psutil
            try:
                import subprocess
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'loadpercentage', '/value'],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    if 'LoadPercentage' in line:
                        return {"percent": float(line.split('=')[1].strip())}
            except Exception:
                pass
        return {"percent": 0.0}
    
    def _get_ram(self) -> Dict:
        """RAM usage."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "used_gb": mem.used / (1024**3),
                "available_gb": mem.available / (1024**3),
                "percent": mem.percent,
            }
        except ImportError:
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(stat)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total = stat.ullTotalPhys / (1024**3)
                avail = stat.ullAvailPhys / (1024**3)
                return {
                    "total_gb": total, "used_gb": total - avail,
                    "available_gb": avail, "percent": (1 - avail/total) * 100,
                }
            except Exception:
                pass
        return {"total_gb": 0, "used_gb": 0, "available_gb": 0, "percent": 0}
    
    def _get_disk(self) -> List[Dict]:
        """Disk usage для всех дисков."""
        disks = []
        try:
            import psutil
            for part in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                    disks.append({
                        "drive": part.device[:2],
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": usage.percent,
                    })
                except Exception:
                    pass
        except ImportError:
            # Fallback
            import shutil
            for letter in "CDEFGH":
                drive = f"{letter}:\\"
                if os.path.exists(drive):
                    try:
                        total, used, free = shutil.disk_usage(drive)
                        disks.append({
                            "drive": f"{letter}:",
                            "total_gb": total / (1024**3),
                            "used_gb": used / (1024**3),
                            "free_gb": free / (1024**3),
                            "percent": used / total * 100,
                        })
                    except Exception:
                        pass
        return disks
    
    def _get_battery(self) -> Optional[Dict]:
        """Battery status (если ноутбук)."""
        try:
            import psutil
            bat = psutil.sensors_battery()
            if bat:
                return {"percent": bat.percent, "plugged": bat.power_plugged}
        except (ImportError, AttributeError):
            pass
        return None
    
    def _get_top_processes(self, n: int = 5) -> List[Dict]:
        """Топ N процессов по использованию RAM."""
        try:
            import psutil
            procs = []
            for p in psutil.process_iter(['name', 'memory_info']):
                try:
                    info = p.info
                    ram_mb = info['memory_info'].rss / (1024**2)
                    procs.append({"name": info['name'], "ram_mb": ram_mb})
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            procs.sort(key=lambda x: x['ram_mb'], reverse=True)
            return procs[:n]
        except ImportError:
            return []
    
    def _get_folder_size(self, path: str) -> int:
        """Размер папки в MB."""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames[:100]:  # Лимит для скорости
                    try:
                        total += os.path.getsize(os.path.join(dirpath, f))
                    except (OSError, PermissionError):
                        pass
        except Exception:
            pass
        return total // (1024 * 1024)
    
    def _bar(self, percent: float, length: int = 10) -> str:
        filled = int(percent / 100 * length)
        return "█" * filled + "░" * (length - filled)
    
    def _monitor_loop(self):
        """Фоновый поток: проверяет ресурсы каждые 60 секунд."""
        while self._running:
            try:
                ram = self._get_ram()
                disk = self._get_disk()
                battery = self._get_battery()
                
                # Алерт: RAM > 90%
                if ram["percent"] > 90 and "ram" not in self._last_check:
                    self._alerts.append(
                        f"⚠️ RAM загружена на {ram['percent']:.0f}%! "
                        f"Свободно: {ram['available_gb']:.1f} GB. Закрой лишнее?"
                    )
                    self._last_check["ram"] = time.time()
                elif ram["percent"] <= 85:
                    self._last_check.pop("ram", None)
                
                # Алерт: Диск > 92%
                for d in disk:
                    key = f"disk_{d['drive']}"
                    if d["percent"] > 92 and key not in self._last_check:
                        self._alerts.append(
                            f"⚠️ Диск {d['drive']} заполнен на {d['percent']:.0f}%! "
                            f"Осталось {d['free_gb']:.1f} GB. Почистить?"
                        )
                        self._last_check[key] = time.time()
                    elif d["percent"] <= 88:
                        self._last_check.pop(key, None)
                
                # Алерт: Батарея < 15%
                if battery and not battery["plugged"]:
                    if battery["percent"] < 15 and "battery" not in self._last_check:
                        self._alerts.append(
                            f"🔋 Батарея {battery['percent']}%! Подключи зарядку!"
                        )
                        self._last_check["battery"] = time.time()
                    elif battery["percent"] >= 20:
                        self._last_check.pop("battery", None)
                
            except Exception as e:
                logger.debug(f"Monitor error: {e}")
            time.sleep(60)
    
    def stop(self):
        self._running = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mon = SystemMonitor()
    print(mon.get_status())
    print()
    print(mon.why_slow())
    mon.stop()
