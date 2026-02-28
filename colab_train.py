"""
═══════════════════════════════════════════════════════════════════════
  ТАРС v3 — Colab ПОЛНОЕ ОБУЧЕНИЕ
═══════════════════════════════════════════════════════════════════════

Полный пайплайн обучения TARS v3 на Colab/Kaggle GPU.
Время: ~8-15 часов (A100) / ~15-24 часа (T4/L4)

ИНСТРУКЦИЯ:
  1. Colab: Runtime → Change runtime type → A100 (или T4/L4)
  2. Загрузите проект:
       !git clone https://github.com/<ваш-репо>/TarsSSM-Py.git
       %cd TarsSSM-Py
  3. ⚠️ РЕКОМЕНДУЕТСЯ сначала запустить тестовый прогон:
       !python colab_test.py
  4. Запуск полного обучения:
       !python colab_train.py

  Опции:
    !python colab_train.py --skip-voice    # Без голоса (~5-8ч)
    !python colab_train.py --resume        # Продолжить обучение
    !python colab_train.py --skip-download # Данные уже скачаны

═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

# ═══════════════════════════════════════════
# 1. Определение окружения
# ═══════════════════════════════════════════

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

PYTHON = sys.executable

# ═══════════════════════════════════════════
# 2. Аргументы
# ═══════════════════════════════════════════

parser = argparse.ArgumentParser(description="ТАРС v3 — Colab Full Training")
parser.add_argument("--skip-voice", action="store_true",
                    help="Пропустить голосовые фазы (Whisper + Piper), экономит ~8ч")
parser.add_argument("--skip-download", action="store_true",
                    help="Пропустить скачивание данных (если уже скачаны)")
parser.add_argument("--skip-quantize", action="store_true",
                    help="Пропустить квантизацию 1.58-bit")
parser.add_argument("--resume", action="store_true",
                    help="Продолжить обучение с последнего чекпоинта")
parser.add_argument("--phase", type=int, default=None,
                    help="Запустить только конкретную фазу (0-10)")
args = parser.parse_args()


# ═══════════════════════════════════════════
# 3. Настройка Colab/Kaggle
# ═══════════════════════════════════════════

IS_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")
IS_KAGGLE = "KAGGLE_DATA_DIR" in os.environ

# Подключение Google Drive для сохранения моделей (Colab)
if IS_COLAB:
    try:
        from google.colab import drive
        drive_path = Path("/content/drive/MyDrive/TarsModels")
        if not drive_path.exists():
            print("  📁 Подключение Google Drive для сохранения моделей...")
            drive.mount("/content/drive")
            drive_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Drive подключён: {drive_path}")
        SAVE_TO_DRIVE = True
    except Exception:
        SAVE_TO_DRIVE = False
        print("  ℹ️  Google Drive не подключён — модели останутся в /content/")
else:
    SAVE_TO_DRIVE = False


# ═══════════════════════════════════════════
# 4. Информация о системе
# ═══════════════════════════════════════════

print()
print("═" * 65)
print("  ТАРС v3 — ПОЛНОЕ ОБУЧЕНИЕ")
print("═" * 65)
print()

# GPU info
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  🎮 GPU:    {gpu}")
        print(f"  💾 VRAM:   {vram:.1f} GB")
        
        # Рекомендации по батчу
        if vram >= 40:
            print(f"  ⚡ A100 — максимальная производительность")
        elif vram >= 15:
            print(f"  ✅ T4/L4 — хорошая производительность")
        else:
            print(f"  ⚠️  Маленький VRAM — возможно OOM")
    else:
        print("  ⚠️  GPU не обнаружен!")
        print("  🔧 Включите GPU: Runtime → Change runtime type → A100/T4")
except ImportError:
    print("  📦 PyTorch не установлен — будет установлен автоматически")

print()
print("  Параметры обучения:")
print("    Модель:       2048d × 24 слоя (~1B params)")
print("    Vocabulary:    256 (cp1251 byte-level)")
print("    Фазы Mamba-2:  4 (full → WKV → MoLE → RAG)")
print("    Квантизация:   1.58-bit BitNet")
if args.skip_voice:
    print("    Голос:         ⏭ Пропущен (--skip-voice)")
else:
    print("    Голос:         Whisper Tiny (RU) + Piper TTS (RU)")
print()


# ═══════════════════════════════════════════
# 5. Запуск обучения
# ═══════════════════════════════════════════

print("  Фазы:")
print("    0. Установка зависимостей")
print("    1. Скачивание данных (Wiki 100K + HuggingFace)")
print("    2. Рефлексы (ReflexClassifier, 100 эпох)")
print("    3. MinGRU LM (dim=512, 6 слоёв, 25 эпох, +HF augment)")
print("    4. Mamba-2 Brain (2048d, 24 слоя, 4 фазы)")
print("       4.1 Full pretrain  (5 эпох, lr=3e-4)")
print("       4.2 WKV + Fusion   (3 эпохи, lr=1e-4, SSD frozen)")
print("       4.3 MoLE + Pool    (2 эпохи, lr=3e-5)")
print("       4.4 RAG + Memory   (2 эпохи, lr=1.5e-5)")
print("    5. Квантизация 1.58-bit (3 эпохи)")
print("    6. Финальная сборка → models/tars_v3/")
print("    7. Валидация")
if not args.skip_voice:
    print("    8. Whisper STT (LoRA, 3 эпохи, 5000 samples)")
    print("    9. Piper TTS (1000 эпох, 3000 samples)")
    print("   10. Квантизация голосовых ONNX (INT8)")
print("   11. Instruction Tuning (3 эпохи)")
print()
print("─" * 65)

t0 = time.time()

# Собираем аргументы для mega_train.py
cmd = [PYTHON, "mega_train.py"]
if args.skip_voice:
    cmd.append("--skip-voice")
if args.skip_download:
    cmd.append("--skip-download")
if args.skip_quantize:
    cmd.append("--skip-quantize")
if args.phase is not None:
    cmd += ["--phase", str(args.phase)]

result = subprocess.run(cmd, cwd=str(ROOT))

elapsed = time.time() - t0
hours = elapsed / 3600


# ═══════════════════════════════════════════
# 6. Итоги
# ═══════════════════════════════════════════

print()
print("═" * 65)
if result.returncode == 0:
    print(f"  ✅ ОБУЧЕНИЕ ЗАВЕРШЕНО за {hours:.1f} часов!")
    print()
    
    # Размеры моделей
    tars_v3 = ROOT / "models" / "tars_v3"
    if tars_v3.exists():
        total_mb = 0
        for f in tars_v3.glob("*.pt"):
            size_mb = f.stat().st_size / 1024 / 1024
            total_mb += size_mb
            print(f"    {f.name}: {size_mb:.1f} MB")
        print(f"    ──────────────────────")
        print(f"    Итого: {total_mb:.0f} MB")
    
    print()
    print("  Запуск ТАРС:")
    print("    python launch_tars.py")
    
    # Копирование на Google Drive
    if SAVE_TO_DRIVE and IS_COLAB:
        print()
        print(f"  💾 Копирование моделей на Google Drive...")
        import shutil
        try:
            drive_dest = Path("/content/drive/MyDrive/TarsModels")
            drive_dest.mkdir(parents=True, exist_ok=True)
            if tars_v3.exists():
                for f in tars_v3.glob("*.pt"):
                    shutil.copy2(str(f), str(drive_dest / f.name))
                print(f"  ✅ Скопировано в: {drive_dest}")
                print()
                print("  ⚠️  Colab сессия может завершиться!")
                print("  Ваши модели сохранены на Google Drive.")
        except Exception as e:
            print(f"  ⚠️  Копирование не удалось: {e}")
else:
    print(f"  ⚠️  Обучение завершилось с ошибками (код {result.returncode})")
    print(f"     Время: {hours:.1f} часов")
    print()
    print("  Проверьте логи:")
    print("    !cat mega_train.log | tail -100")
    print()
    print("  Можно продолжить с последнего чекпоинта:")
    print("    !python colab_train.py --resume --skip-download")
print("═" * 65)
