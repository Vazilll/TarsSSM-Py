"""
═══════════════════════════════════════════════════════════════
  ТАРС — Полная подготовка системы на новом ПК
═══════════════════════════════════════════════════════════════

Один скрипт делает ВСЁ:
  1. Создаёт виртуальное окружение (venv)
  2. Устанавливает все Python-зависимости
  3. Скачивает модели (Whisper, Silero VAD, SentenceTransformer)
  4. Скачивает базы знаний (Wikipedia 100K, HuggingFace датасеты)
  5. Загружает знания в LEANN (векторную память)
  6. Обучает все модели (Reflex → MinGRU → Mamba-2 Brain)
  7. Проверяет готовность системы

Использование:
  python setup.py              # Полная установка + обучение
  python setup.py --skip-data  # Без скачивания баз данных
  python setup.py --skip-train # Без обучения
  python setup.py --gpu        # С поддержкой CUDA GPU
"""

import os
import sys
import subprocess
import shutil
import time
import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
VENV_DIR = ROOT / "venv"
PYTHON = str(VENV_DIR / "Scripts" / "python.exe") if sys.platform == "win32" else str(VENV_DIR / "bin" / "python")
PIP = str(VENV_DIR / "Scripts" / "pip.exe") if sys.platform == "win32" else str(VENV_DIR / "bin" / "pip")

# ═══════════════════════════════════════════════════════════════
#  Цвета и форматирование
# ═══════════════════════════════════════════════════════════════

def banner(text):
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}\n")

def phase(num, total, text):
    print(f"\n{'━' * 60}")
    print(f"  [{num}/{total}] {text}")
    print(f"{'━' * 60}")

def ok(text):
    print(f"  ✅ {text}")

def warn(text):
    print(f"  ⚠️  {text}")

def fail(text):
    print(f"  ❌ {text}")

def info(text):
    print(f"  ℹ  {text}")

def run_cmd(cmd, desc="", check=True, capture=False):
    """Запускает команду с описанием."""
    if desc:
        info(desc)
    try:
        result = subprocess.run(
            cmd, shell=isinstance(cmd, str),
            capture_output=capture, text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        if capture:
            warn(f"Команда вернула ошибку: {e.stderr[:200] if e.stderr else ''}")
        return e
    except FileNotFoundError:
        warn(f"Команда не найдена: {cmd[0] if isinstance(cmd, list) else cmd}")
        return None


# ═══════════════════════════════════════════════════════════════
#  Фаза 1: Виртуальное окружение
# ═══════════════════════════════════════════════════════════════

def setup_venv():
    phase(1, 7, "🐍 Виртуальное окружение Python")
    
    if VENV_DIR.exists() and Path(PYTHON).exists():
        ok(f"venv уже существует: {VENV_DIR}")
        return True
    
    info("Создаю виртуальное окружение...")
    result = run_cmd([sys.executable, "-m", "venv", str(VENV_DIR)], check=False)
    if result and not isinstance(result, Exception):
        ok("venv создан")
        return True
    else:
        fail("Не удалось создать venv")
        return False


# ═══════════════════════════════════════════════════════════════
#  Фаза 2: Python-зависимости
# ═══════════════════════════════════════════════════════════════

def install_dependencies(use_gpu=False):
    phase(2, 7, "📦 Python-зависимости")
    
    python = PYTHON if Path(PYTHON).exists() else sys.executable
    pip = PIP if Path(PIP).exists() else f"{python} -m pip"
    
    # Обновляем pip
    run_cmd(f"{python} -m pip install --upgrade pip", "Обновление pip...", check=False)
    
    # Основные зависимости
    core_deps = [
        "numpy", "einops", "tqdm", "sentencepiece", "tokenizers",
    ]
    
    # PyTorch (с CUDA если нужен)
    if use_gpu:
        info("Установка PyTorch с CUDA...")
        run_cmd(
            f"{python} -m pip install torch --index-url https://download.pytorch.org/whl/cu121",
            check=False
        )
    else:
        core_deps.append("torch")
    
    # Устанавливаем core
    for dep in core_deps:
        run_cmd(f"{python} -m pip install {dep}", f"Установка {dep}...", check=False)
    
    # Голос (STT/TTS)
    voice_deps = ["faster-whisper", "sounddevice"]
    for dep in voice_deps:
        run_cmd(f"{python} -m pip install {dep}", f"Установка {dep}...", check=False)
    
    # RAG и обучение
    ml_deps = ["sentence-transformers", "datasets"]
    for dep in ml_deps:
        run_cmd(f"{python} -m pip install {dep}", f"Установка {dep}...", check=False)
    
    # Опционально
    optional = ["onnxruntime", "pyttsx3"]
    for dep in optional:
        run_cmd(f"{python} -m pip install {dep}", f"Установка {dep} (опционально)...", check=False)
    
    ok("Зависимости установлены")
    return True


# ═══════════════════════════════════════════════════════════════
#  Фаза 3: Создание директорий
# ═══════════════════════════════════════════════════════════════

def create_directories():
    phase(3, 7, "📁 Структура директорий")
    
    dirs = [
        ROOT / "data",
        ROOT / "data" / "thinking_logs",
        ROOT / "models",
        ROOT / "models" / "voice",
        ROOT / "models" / "embeddings",
        ROOT / "models" / "brain",
        ROOT / "models" / "mamba2",
        ROOT / "models" / "reflex",
        ROOT / "memory",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    ok(f"Создано {len(dirs)} директорий")
    return True


# ═══════════════════════════════════════════════════════════════
#  Фаза 4: Модели
# ═══════════════════════════════════════════════════════════════

def download_models():
    phase(4, 7, "🤖 Скачивание моделей")
    
    python = PYTHON if Path(PYTHON).exists() else sys.executable
    
    # ── 1. SentenceTransformer (для LEANN) ──
    emb_dir = ROOT / "models" / "embeddings"
    if not (emb_dir / "config.json").exists():
        info("Скачивание SentenceTransformer (all-MiniLM-L6-v2)...")
        try:
            result = run_cmd(
                f'{python} -c "from sentence_transformers import SentenceTransformer; '
                f"m = SentenceTransformer('all-MiniLM-L6-v2'); "
                f"m.save('{str(emb_dir)}')" + '"',
                check=False
            )
            if (emb_dir / "config.json").exists():
                ok("SentenceTransformer скачан и сохранён")
            else:
                warn("SentenceTransformer: не удалось сохранить (будет скачан при первом запуске)")
        except Exception as e:
            warn(f"SentenceTransformer: {e}")
    else:
        ok("SentenceTransformer уже на месте")
    
    # ── 2. Whisper Tiny (STT) ──
    whisper_dir = ROOT / "models" / "voice" / "whisper_tiny"
    if not (whisper_dir / "model.bin").exists():
        info("Whisper Tiny будет скачан при первом запуске (faster-whisper)")
        info("  Или: python -c \"from faster_whisper import WhisperModel; WhisperModel('tiny')\"")
    else:
        ok("Whisper Tiny уже на месте")
    
    # ── 3. Silero VAD (опционально) ──
    vad_path = ROOT / "models" / "voice" / "silero_vad.onnx"
    if not vad_path.exists():
        info("Скачивание Silero VAD (ONNX, ~2MB)...")
        vad_url = "https://models.silero.ai/models/en/vad_v5.onnx"
        try:
            req = urllib.request.Request(vad_url, headers={"User-Agent": "TARS-Setup/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                with open(str(vad_path), 'wb') as f:
                    f.write(resp.read())
            ok(f"Silero VAD скачан ({vad_path.stat().st_size / 1024:.0f} KB)")
        except Exception as e:
            warn(f"Silero VAD: {e} (система будет работать с energy-based VAD)")
    else:
        ok("Silero VAD уже на месте")
    
    return True


# ═══════════════════════════════════════════════════════════════
#  Фаза 5: Базы знаний
# ═══════════════════════════════════════════════════════════════

def download_knowledge():
    phase(5, 7, "📚 Базы знаний (Wikipedia + HuggingFace + LEANN)")
    
    python = PYTHON if Path(PYTHON).exists() else sys.executable
    
    # Используем наш download_all.py
    download_script = ROOT / "training" / "download_all.py"
    if download_script.exists():
        info("Запуск training/download_all.py...")
        run_cmd(f"{python} {download_script}", check=False)
    else:
        warn("training/download_all.py не найден!")
    
    return True


# ═══════════════════════════════════════════════════════════════
#  Фаза 6: Обучение моделей
# ═══════════════════════════════════════════════════════════════

def train_models(use_gpu=False):
    phase(6, 7, "Training: FP16 init -> 1.58-bit -> train on data")
    
    python = PYTHON if Path(PYTHON).exists() else sys.executable
    train_script = ROOT / "training" / "train_all.py"
    
    if not train_script.exists():
        warn("training/train_all.py не найден!")
        return False
    
    # Авто-определение CUDA
    device = "auto"  # train_all.py сам определит cuda/cpu
    
    # Пайплайн (сразу 1.58-bit, без FP16 стадии):
    #   Phase 1: Reflex Classifier (~30 сек)
    #   Phase 2: MinGRU (~5 мин)
    #   Phase 3: Mamba-2 + RWKV-7 1.58-bit (~30 мин+ GPU)
    #   Phase 4: Whisper Vocabulary Boost (~1 мин)
    info("CUDA определится автоматически")
    info("  Phase 1: Reflex Classifier (~30 сек)")
    info("  Phase 2: MinGRU Language Model (~5 мин)")
    info("  Phase 3: Mamba-2 1.58-bit на WIKI+HF данных (~30 мин+ GPU)")
    info("  Phase 4: Whisper Vocabulary Boost (~1 мин)")
    print()
    
    args = ["--device", device]
    run_cmd(
        [python, str(train_script)] + args,
        check=False
    )
    
    # Проверяем что модели появились (пути совпадают с train_*.py)
    brain_model = ROOT / "models" / "mamba2" / "mamba2_omega.pt"
    brain_158 = ROOT / "models" / "mamba2" / "mamba2_omega_158bit.pt"
    reflex_model = ROOT / "models" / "reflex" / "reflex_classifier.pt"
    mingru_model = ROOT / "models" / "mingru_weights.pt"
    whisper_ctx = ROOT / "models" / "voice" / "whisper_context.json"
    
    trained = []
    if brain_model.exists():
        ok(f"Mamba-2 Brain: {brain_model.stat().st_size / (1024*1024):.1f} MB")
        trained.append("brain")
    if brain_158.exists():
        ok(f"Mamba-2 Brain 1.58-bit: {brain_158.stat().st_size / (1024*1024):.1f} MB")
        trained.append("brain-158bit")
    if reflex_model.exists():
        ok(f"Reflex Classifier: {reflex_model.stat().st_size / 1024:.0f} KB")
        trained.append("reflex")
    if mingru_model.exists():
        ok(f"MinGRU: {mingru_model.stat().st_size / (1024*1024):.1f} MB")
        trained.append("mingru")
    if whisper_ctx.exists():
        ok("Whisper Vocabulary Boost")
        trained.append("whisper")
    
    if not trained:
        warn("Модели не найдены после обучения. Проверьте вывод выше.")
        return False
    
    ok(f"Обучено {len(trained)} моделей: {', '.join(trained)}")
    return True


# ═══════════════════════════════════════════════════════════════
#  Фаза 7: Финальная проверка
# ═══════════════════════════════════════════════════════════════

def verify_system():
    phase(7, 7, "🔍 Финальная проверка")
    
    python = PYTHON if Path(PYTHON).exists() else sys.executable
    checks = []
    
    # 1. Python и torch
    result = run_cmd(
        f'{python} -c "import torch; print(f\'PyTorch {{torch.__version__}}, CUDA: {{torch.cuda.is_available()}}\')"',
        capture=True, check=False
    )
    if result and hasattr(result, 'stdout') and result.stdout:
        ok(f"PyTorch: {result.stdout.strip()}")
        checks.append(True)
    else:
        fail("PyTorch не установлен")
        checks.append(False)
    
    # 2. einops
    result = run_cmd(f'{python} -c "import einops; print(\'OK\')"', capture=True, check=False)
    checks.append(result and hasattr(result, 'stdout') and 'OK' in (result.stdout or ''))
    if checks[-1]:
        ok("einops")
    else:
        fail("einops не установлен")
    
    # 3. Brain modules
    result = run_cmd(
        f'{python} -c "from brain.mamba2.model import TarsMamba2LM; print(\'OK\')"',
        capture=True, check=False
    )
    if result and hasattr(result, 'stdout') and 'OK' in (result.stdout or ''):
        ok("Brain (TarsMamba2LM)")
        checks.append(True)
    else:
        warn("Brain: ошибка импорта (возможно нужны другие зависимости)")
        checks.append(False)
    
    # 4. Данные
    wiki_path = ROOT / "data" / "wiki_ru.txt"
    if wiki_path.exists():
        size_mb = wiki_path.stat().st_size / (1024 * 1024)
        ok(f"Wikipedia: {size_mb:.1f} MB")
        checks.append(True)
    else:
        warn("Wikipedia: не скачана (запустите python training/download_all.py)")
        checks.append(False)
    
    # 5. LEANN индекс
    leann_path = ROOT / "memory" / "leann.index"
    if leann_path.exists():
        size_mb = leann_path.stat().st_size / (1024 * 1024)
        try:
            with open(leann_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            docs = len(data.get("texts", []))
            ok(f"LEANN: {docs} документов, {size_mb:.1f} MB")
        except Exception:
            ok(f"LEANN: {size_mb:.1f} MB")
        checks.append(True)
    else:
        warn("LEANN: пуст (будет заполнен при скачивании данных)")
        checks.append(False)
    
    # 6. HF данные
    hf_files = list((ROOT / "data").glob("hf_*.txt")) if (ROOT / "data").exists() else []
    if hf_files:
        total_mb = sum(f.stat().st_size for f in hf_files) / (1024 * 1024)
        ok(f"HuggingFace: {len(hf_files)} датасетов, {total_mb:.1f} MB")
        checks.append(True)
    else:
        warn("HuggingFace: не скачаны")
        checks.append(False)
    
    # 7. Модели
    emb_ok = (ROOT / "models" / "embeddings" / "config.json").exists()
    vad_ok = (ROOT / "models" / "voice" / "silero_vad.onnx").exists()
    if emb_ok:
        ok("SentenceTransformer (embeddings)")
    else:
        warn("SentenceTransformer: не скачан")
    if vad_ok:
        ok("Silero VAD (ONNX)")
    else:
        warn("Silero VAD: не скачан")
    
    # Итог
    passed = sum(1 for c in checks if c)
    total = len(checks)
    print()
    if passed == total:
        banner("✅ ТАРС ПОЛНОСТЬЮ ГОТОВ К РАБОТЕ")
    else:
        banner(f"⚠️  ТАРС частично готов ({passed}/{total} проверок)")
    
    print(f"  Запуск:      python launch_tars.py")
    print(f"  Обучение:    python training/train_mamba2.py --phase 1")
    print(f"  Данные:      python training/download_all.py")
    print()
    
    return all(checks)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ТАРС — Полная подготовка системы",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python setup.py              # Полная установка + обучение
  python setup.py --skip-data  # Без скачивания баз (быстро)
  python setup.py --skip-train # Без обучения моделей
  python setup.py --gpu        # С поддержкой CUDA GPU
  python setup.py --check      # Только проверка
        """
    )
    parser.add_argument("--skip-data", action="store_true",
                        help="Пропустить скачивание баз знаний")
    parser.add_argument("--skip-train", action="store_true",
                        help="Пропустить обучение моделей")
    parser.add_argument("--gpu", action="store_true",
                        help="Установить PyTorch с CUDA")
    parser.add_argument("--check", action="store_true",
                        help="Только проверить готовность")
    args = parser.parse_args()
    
    banner("🤖 ТАРС — Автоматическая подготовка системы")
    
    start = time.time()
    
    if args.check:
        verify_system()
        return
    
    # 1. Виртуальное окружение
    setup_venv()
    
    # 2. Зависимости
    install_dependencies(use_gpu=args.gpu)
    
    # 3. Директории
    create_directories()
    
    # 4. Модели
    download_models()
    
    # 5. Базы знаний
    if not args.skip_data:
        download_knowledge()
    else:
        info("Скачивание баз пропущено (--skip-data)")
    
    # 6. Обучение
    if not args.skip_train:
        train_models(use_gpu=args.gpu)
    else:
        info("Обучение пропущено (--skip-train)")
    
    # 7. Проверка
    verify_system()
    
    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"  ⏱  Время установки: {minutes} мин {seconds} сек")
    print()


if __name__ == "__main__":
    main()
