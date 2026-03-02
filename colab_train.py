"""
═══════════════════════════════════════════════════════════════════════
  ТАРС v3 — Colab Training (Medium, 103M params)
═══════════════════════════════════════════════════════════════════════

Обучение на Google Colab с авто-оптимизацией под GPU.
ВСЕ ДАННЫЕ сохраняются на Google Drive — не теряются при Disconnect.

  A100 (40GB) — batch=авто(~1024), bf16, ~15-25 мин  🔥 Рекомендуется
  L4   (24GB) — batch=авто(~512),  bf16, ~25-40 мин  ⚡ Лучший баланс
  T4   (15GB) — batch=авто(~256),  fp16, ~40-60 мин  ✅ Бесплатный

ИНСТРУКЦИЯ:
  1. Runtime → Change runtime type → L4
  2. Запустить ячейки блокнота по порядку
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

# ═══════════════════════════════════════════
# 1. Google Drive — ВСЁ хранится здесь
# ═══════════════════════════════════════════

DRIVE_BASE = None
DRIVE_DATA = None
DRIVE_MODELS = None

if IS_COLAB:
    if Path("/content/drive/MyDrive").exists():
        DRIVE_BASE = Path("/content/drive/MyDrive")
        DRIVE_DATA = DRIVE_BASE / "TarsData"
        DRIVE_MODELS = DRIVE_BASE / "TarsModels"
        DRIVE_DATA.mkdir(parents=True, exist_ok=True)
        DRIVE_MODELS.mkdir(parents=True, exist_ok=True)
        
        # === Symlink: data/ → Drive/TarsData ===
        # Датасеты скачиваются СРАЗУ на Drive!
        local_data = ROOT / "data"
        if local_data.is_symlink():
            pass  # уже симлинк
        else:
            if local_data.exists():
                # Перенести существующие файлы на Drive
                for f in local_data.glob("*.txt"):
                    dest = DRIVE_DATA / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
                for f in local_data.glob("*.json"):
                    dest = DRIVE_DATA / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
                shutil.rmtree(str(local_data))
            local_data.symlink_to(DRIVE_DATA)
        
        # === Symlink: models/ → Drive/TarsModels ===
        local_models = ROOT / "models"
        if local_models.is_symlink():
            pass
        else:
            if local_models.exists():
                for f in local_models.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(local_models)
                        dest = DRIVE_MODELS / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not dest.exists():
                            shutil.move(str(f), str(dest))
                shutil.rmtree(str(local_models))
            local_models.symlink_to(DRIVE_MODELS)
        
        # === LEANN int8 index → Drive/TarsMemory ===
        # LEANN и ingest_to_leann.py автоматически определяют Drive путь
        # Достаточно создать директорию
        DRIVE_MEMORY = DRIVE_BASE / "TarsMemory"
        DRIVE_MEMORY.mkdir(parents=True, exist_ok=True)
        
        # Перенести старые локальные данные на Drive (миграция)
        local_memory = ROOT / "memory"
        if local_memory.exists():
            for ext in ("*.npz", "*.json"):
                for f in local_memory.glob(ext):
                    if not f.is_symlink():
                        dest = DRIVE_MEMORY / f.name
                        if not dest.exists() and f.stat().st_size > 0:
                            shutil.move(str(f), str(dest))
        
        print(f"  ☁️  Google Drive подключён")
        print(f"     data/   → {DRIVE_DATA}")
        print(f"     models/ → {DRIVE_MODELS}")
        print(f"     LEANN   → {DRIVE_MEMORY}")
        
        # Показать что уже есть на Drive
        existing_data = list(DRIVE_DATA.glob("hf_*.txt"))
        if existing_data:
            total_mb = sum(f.stat().st_size for f in existing_data) / (1024*1024)
            print(f"     📂 На Drive уже {len(existing_data)} датасетов ({total_mb:.0f} MB)")
    else:
        print("  ⚠️  Drive не смонтирован!")
        print("     drive.mount('/content/drive')")
        print("     Без Drive данные ПОТЕРЯЮТСЯ при Disconnect!")


# ═══════════════════════════════════════════
# 2. GPU Detection
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
    print("  📦 PyTorch не установлен")

# ═══════════════════════════════════════════
# 3. LEANN — пропуск если нет индекса
# ═══════════════════════════════════════════

leann_index = ROOT / "memory" / "leann.index"
if not leann_index.exists():
    leann_index.parent.mkdir(parents=True, exist_ok=True)
    leann_index.touch()
    print("  🧠 LEANN: создан пустой индекс (пропуск)")

# ═══════════════════════════════════════════
# 4. Training
# ═══════════════════════════════════════════

configs = {
    "a100": {"batch": 32, "accum": 1, "amp": "bf16",  "time": "30-45 мин"},
    "l4":   {"batch": 48, "accum": 2, "amp": "bf16",  "time": "30-50 мин"},
    "t4":   {"batch": 16, "accum": 2, "amp": "fp16",  "time": "1-2 часа"},
    "small":{"batch": 8,  "accum": 4, "amp": "fp16",  "time": "2-4 часа"},
}
cfg = configs[gpu_tier]

print()
print(f"  Конфигурация (авто-{gpu_tier.upper()}):")
print(f"    Модель:        512d × 8 слоёв (~103M params)")
print(f"    Batch:         {cfg['batch']} × {cfg['accum']} = {cfg['batch']*cfg['accum']} effective")
print(f"    AMP:           {cfg['amp']}")
print(f"    Время:         ~{cfg['time']}")
print()
print("─" * 65)

t0 = time.time()

# Parse extra args
extra_args = []
for arg in sys.argv[1:]:
    if arg == "--resume":
        extra_args.append("--skip-download")  # resume = пропуск скачивания
    elif arg in ("--skip-download", "--skip-quantize"):
        extra_args.append(arg)

# mega_train.py сам определит GPU
cmd = [PYTHON, "mega_train.py", "--skip-voice", "--drive"] + extra_args
result = subprocess.run(cmd, cwd=str(ROOT))

# ═══════════════════════════════════════════
# 5. Report + Resource Monitoring
# ═══════════════════════════════════════════

elapsed = time.time() - t0
hours = elapsed / 3600
minutes = elapsed / 60

print()
print("═" * 65)
if result.returncode == 0:
    print(f"  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО за {minutes:.0f} мин ({hours:.1f} ч)!")
    print()
    
    if DRIVE_MODELS:
        total_mb = 0
        for f in DRIVE_MODELS.rglob("*.pt"):
            mb = f.stat().st_size / 1024 / 1024
            total_mb += mb
            print(f"    💾 {f.name}: {mb:.1f} MB (на Drive)")
        if total_mb > 0:
            print(f"    {'─' * 30}")
            print(f"    Итого модели: {total_mb:.0f} MB")
    
    print()
    
    # Resource monitoring
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  📊 RAM: {ram.used / 1024**3:.1f}/{ram.total / 1024**3:.1f} GB ({ram.percent}%)")
    except Exception:
        pass
    
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  🎮 VRAM: {alloc:.1f}/{total:.1f} GB")
    except Exception:
        pass
    
    # LEANN stats
    leann_npz = ROOT / "memory" / "leann.npz"
    if leann_npz.exists():
        mb = leann_npz.stat().st_size / 1024 / 1024
        print(f"  🧠 LEANN: {mb:.0f} MB (int8 embeddings)")
    
    print()
    print(f"  Данные на Drive:  MyDrive/TarsData/")
    print(f"  Модели на Drive:  MyDrive/TarsModels/")
    print()
    print("  🚀 Запуск: python launch_tars.py")
else:
    print(f"  ⚠️  Ошибка (код {result.returncode})")
    print(f"     Время: {minutes:.0f} мин")
    print()
    
    # RAM при ошибке — помогает диагностировать OOM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  📊 RAM: {ram.used / 1024**3:.1f}/{ram.total / 1024**3:.1f} GB ({ram.percent}%)")
    except Exception:
        pass
    
    print()
    print("  Логи: !cat mega_train.log | tail -50")
    print("  Продолжить: !python colab_train.py --resume")
print("═" * 65)

