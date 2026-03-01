"""
═══════════════════════════════════════════════════════════════════════
  ТАРС v3 — Colab Training (Medium, 103M params)
═══════════════════════════════════════════════════════════════════════

Обучение на Google Colab с авто-оптимизацией под GPU.

  Модель:       512d × 8 слоёв (~103M params)
  Данные:       Wikipedia + HuggingFace + Personality
  Квантизация:  1.58-bit BitNet
  
  A100 (40GB) — batch=32, bf16, ~30-45 мин    🔥 Рекомендуется
  L4   (24GB) — batch=24, bf16, ~45-60 мин    ⚡ Лучший баланс
  T4   (15GB) — batch=16, fp16, ~1-2 часа     ✅ Бесплатный

ИНСТРУКЦИЯ:
  1. Runtime → Change runtime type → L4 (рекомендуется)
  2. Загрузите проект (ZIP / Git / Drive)
  3. !python colab_train.py

ОПЦИИ:
  !python colab_train.py --resume           # Продолжить с чекпоинта
  !python colab_train.py --skip-download    # Данные уже есть

═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

# Fix encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")
PYTHON = sys.executable
DATA = ROOT / "data"
MODELS = ROOT / "models"

# ═══════════════════════════════════════════
# 1. Google Drive
# ═══════════════════════════════════════════

DRIVE_DATA = None
DRIVE_MODELS = None

if IS_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_DATA = Path("/content/drive/MyDrive/TarsData")
        DRIVE_MODELS = Path("/content/drive/MyDrive/TarsModels")
        DRIVE_DATA.mkdir(parents=True, exist_ok=True)
        DRIVE_MODELS.mkdir(parents=True, exist_ok=True)
        print(f"  ☁️  Google Drive подключён")
        print(f"     Данные:  {DRIVE_DATA}")
        print(f"     Модели:  {DRIVE_MODELS}")
    except Exception as e:
        print(f"  ⚠️  Drive не подключён: {e}")


def restore_cached_data():
    """Восстановить данные с Drive (если есть)."""
    if not DRIVE_DATA or not DRIVE_DATA.exists():
        return 0
    
    restored = 0
    DATA.mkdir(parents=True, exist_ok=True)
    
    for f in DRIVE_DATA.glob("*"):
        dest = DATA / f.name
        if not dest.exists():
            if f.is_file():
                shutil.copy2(str(f), str(dest))
                restored += 1
    
    if restored > 0:
        print(f"  📂 Восстановлено {restored} файлов с Drive")
    return restored


def save_data_to_drive():
    """Сохранить скачанные данные на Drive."""
    if not DRIVE_DATA:
        return
    
    saved = 0
    for f in DATA.glob("*.txt"):
        dest = DRIVE_DATA / f.name
        if not dest.exists() or f.stat().st_size != dest.stat().st_size:
            shutil.copy2(str(f), str(dest))
            saved += 1
    
    for f in DATA.glob("*.json"):
        dest = DRIVE_DATA / f.name
        if not dest.exists():
            shutil.copy2(str(f), str(dest))
            saved += 1
    
    if saved > 0:
        print(f"  💾 Сохранено {saved} файлов на Drive (не будут скачиваться повторно)")


def save_models_to_drive():
    """Сохранить модели на Drive."""
    if not DRIVE_MODELS:
        return
    
    tars_v3 = MODELS / "tars_v3"
    if tars_v3.exists():
        for f in tars_v3.glob("*.pt"):
            dest = DRIVE_MODELS / f.name
            shutil.copy2(str(f), str(dest))
            mb = f.stat().st_size / 1024 / 1024
            print(f"  💾 {f.name}: {mb:.1f} MB → Drive")


# ═══════════════════════════════════════════
# 2. GPU Detection + Auto-Optimization
# ═══════════════════════════════════════════

print()
print("═" * 65)
print("  🤖 ТАРС v3 — MEDIUM TRAINING (Colab)")
print("═" * 65)
print()

gpu_tier = "t4"
bf16_ok = False
vram = 0

try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        bf16_ok = torch.cuda.get_device_capability(0) >= (8, 0)
        print(f"  🎮 GPU:    {gpu}")
        print(f"  💾 VRAM:   {vram:.1f} GB")
        print(f"  ⚡ bf16:   {'Yes' if bf16_ok else 'No (fp16)'}")
        
        if vram >= 35:
            gpu_tier = "a100"
            print(f"  🔥 A100/H100 → batch=32, bf16, ~30-45 мин")
        elif vram >= 20:
            gpu_tier = "l4"
            print(f"  ⚡ L4/RTX → batch=24, bf16, ~45-60 мин")
        elif vram >= 14:
            gpu_tier = "t4"
            print(f"  ✅ T4 → batch=16, fp16, ~1-2 часа")
        else:
            gpu_tier = "small"
            print(f"  ⚠️  Маленький VRAM — batch=8")
    else:
        print("  ⚠️  GPU не найден!")
        print("  🔧 Runtime → Change runtime type → L4")
        sys.exit(1)
except ImportError:
    print("  📦 PyTorch не установлен (будет установлен)")


# ═══════════════════════════════════════════
# 3. Restore cached data
# ═══════════════════════════════════════════

restore_cached_data()

# ═══════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════

configs = {
    "a100": {"batch": 32, "accum": 1, "amp": "bf16",  "time": "30-45 мин"},
    "l4":   {"batch": 24, "accum": 1, "amp": "bf16",  "time": "45-60 мин"},
    "t4":   {"batch": 16, "accum": 2, "amp": "fp16",  "time": "1-2 часа"},
    "small":{"batch": 8,  "accum": 4, "amp": "fp16",  "time": "2-4 часа"},
}
cfg = configs[gpu_tier]

print()
print(f"  Конфигурация (авто-{gpu_tier.upper()}):")
print(f"    Модель:        512d × 8 слоёв (~103M params)")
print(f"    Batch:         {cfg['batch']} × {cfg['accum']} = {cfg['batch']*cfg['accum']} effective")
print(f"    AMP:           {cfg['amp']}")
print(f"    Mamba-2:       10+5+3+3 = 21 эпоха × 4 фазы + Phase 5")
print(f"    Квантизация:   1.58-bit BitNet")
print(f"    Время:         ~{cfg['time']}")
print()
print("─" * 65)

t0 = time.time()

# Parse extra args
extra_args = []
for arg in sys.argv[1:]:
    if arg in ("--skip-download", "--resume", "--skip-quantize"):
        extra_args.append(arg)

# mega_train.py сам определит GPU и выберет batch/bf16
cmd = [PYTHON, "mega_train.py", "--skip-voice", "--drive"] + extra_args
result = subprocess.run(cmd, cwd=str(ROOT))

# ═══════════════════════════════════════════
# 5. Save + Report
# ═══════════════════════════════════════════

save_data_to_drive()
if result.returncode == 0:
    save_models_to_drive()

elapsed = time.time() - t0
hours = elapsed / 3600
minutes = elapsed / 60

print()
print("═" * 65)
if result.returncode == 0:
    print(f"  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО за {minutes:.0f} мин ({hours:.1f} ч)!")
    print()
    print(f"  Модель: 512d × 8L (~103M params)")
    print()
    
    tars_v3 = ROOT / "models" / "tars_v3"
    if tars_v3.exists():
        total_mb = 0
        for f in tars_v3.glob("*.pt"):
            mb = f.stat().st_size / 1024 / 1024
            total_mb += mb
            print(f"    {f.name}: {mb:.1f} MB")
        print(f"    {'─' * 30}")
        print(f"    Итого: {total_mb:.0f} MB")
    
    print()
    if DRIVE_MODELS:
        print(f"  💾 Модели на Drive: {DRIVE_MODELS}")
        print(f"  💾 Данные на Drive: {DRIVE_DATA}")
    print()
    print("  🚀 Запуск: python launch_tars.py")
else:
    print(f"  ⚠️  Ошибка (код {result.returncode})")
    print(f"     Время: {minutes:.0f} мин")
    print()
    print("  Логи: !cat mega_train.log | tail -50")
    print("  Продолжить: !python colab_train.py --resume")
print("═" * 65)
