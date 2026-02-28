"""
═══════════════════════════════════════════════════════════════════════════
  mega_train.py — ТАРС v3: Полностью автономный пайплайн обучения
═══════════════════════════════════════════════════════════════════════════

Один скрипт делает ВСЁ:

  Фаза 0:  Установка зависимостей (pip install)
  Фаза 1:  Скачивание данных (Wikipedia + HuggingFace + LEANN)
  Фаза 2:  Обучение рефлексов (MinGRU classifier, ~1 мин)
  Фаза 3:  Обучение MinGRU LM (System 1, ~30 мин GPU)
  Фаза 4:  Обучение Mamba-2 (12 слоёв × 768d, 4 фазы, ~2-4ч GPU)
  Фаза 5:  Квантизация 1.58-bit + дообучение (~30 мин)
  Фаза 6:  Финальная сборка в models/tars_v3/
  Фаза 7:  Валидация (тестовая генерация)
  Фаза 8:  Whisper Tiny LoRA — дообучение STT для русского (~2ч GPU)
  Фаза 9:  Piper TTS — дообучение голоса для русского (~5ч GPU)
  Фаза 10: Квантизация голосовых ONNX-моделей (INT8, ~5 мин)

Оптимизировано для: Kaggle P100 (16GB) / Colab A100 / RTX 4090

Использование:
  python mega_train.py              # Полный пайплайн (~15ч)
  python mega_train.py --skip-download  # Без скачивания (данные есть)
  python mega_train.py --phase 4    # Только Mamba-2
  python mega_train.py --phase 8    # Только Whisper fine-tune
  python mega_train.py --quick      # Быстрый тест (маленькая модель)
  python mega_train.py --skip-voice # Без голосовых фаз (8-10)

═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import re
import time
import json
import shutil
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ═══ Пути ═══
ROOT = Path(__file__).resolve().parent
TRAINING = ROOT / "training"
DATA = ROOT / "data"
MODELS = ROOT / "models"
TARS_V3 = MODELS / "tars_v3"
PYTHON = sys.executable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(ROOT / "mega_train.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("MegaTrain")


# ═════════════════════════════════════════════════════════════════════════
#  УТИЛИТЫ
# ═════════════════════════════════════════════════════════════════════════

def banner(phase: int, title: str, total: int = 10):
    """Печатает баннер фазы."""
    logger.info("")
    logger.info("╔" + "═" * 62 + "╗")
    logger.info(f"║  Фаза {phase}/{total}: {title:<50s}  ║")
    logger.info("╚" + "═" * 62 + "╝")
    logger.info("")


def run(cmd: list, cwd=None, retries=3, check=True) -> bool:
    """Запускает команду с retry логикой (Windows Defender / launcher bugs)."""
    cmd_str = " ".join(str(c) for c in cmd)
    logger.info(f"▶ {cmd_str}")
    
    for attempt in range(retries):
        try:
            result = subprocess.run(
                [str(c) for c in cmd],
                cwd=str(cwd or ROOT),
                timeout=7200,  # 2 часа макс
            )
            if result.returncode == 0:
                return True
            if result.returncode == 101 and attempt < retries - 1:
                logger.warning(f"  Ошибка лаунчера (код 101), повтор {attempt+2}/{retries}...")
                time.sleep(3)
                continue
            if check:
                logger.error(f"  ❌ Код возврата: {result.returncode}")
            return False
        except PermissionError:
            if attempt < retries - 1:
                logger.warning(f"  PermissionError, повтор {attempt+2}/{retries}...")
                time.sleep(5)
            else:
                logger.error("  ❌ PermissionError после всех попыток")
                return False
        except subprocess.TimeoutExpired:
            logger.error("  ❌ Таймаут (2 часа)")
            return False
    return False


def gpu_info():
    """Возвращает информацию о GPU."""
    # Попытка 1: import torch в текущем процессе
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return name, vram
    except Exception as e:
        logger.debug(f"  gpu_info torch attempt failed: {e}")
    
    # Попытка 2: через subprocess (PYTHON может быть venv)
    try:
        code = (
            "import torch; "
            "print(torch.cuda.is_available()); "
            "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); "
            "print(torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0)"
        )
        r = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            lines = r.stdout.strip().split('\n')
            if len(lines) >= 3 and lines[0].strip() == "True":
                name = lines[1].strip()
                vram = float(lines[2].strip())
                return name, vram
        elif r.stderr:
            logger.debug(f"  gpu_info subprocess stderr: {r.stderr[:200]}")
    except Exception as e:
        logger.debug(f"  gpu_info subprocess failed: {e}")
    
    # Попытка 3: напрямую через nvidia-smi (не зависит от PyTorch)
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = r.stdout.strip().split(", ")
            if len(parts) >= 2:
                name = parts[0].strip()
                vram = float(parts[1].strip()) / 1024  # MiB → GiB
                return name, vram
    except Exception as e:
        logger.debug(f"  gpu_info nvidia-smi failed: {e}")
    
    return None, 0


def get_ram_gb():
    """Возвращает объём RAM в GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        # Fallback для Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                           ("dwMemoryLoad", ctypes.c_ulong),
                           ("ullTotalPhys", ctypes.c_ulonglong),
                           ("ullAvailPhys", ctypes.c_ulonglong),
                           ("ullTotalPageFile", ctypes.c_ulonglong),
                           ("ullAvailPageFile", ctypes.c_ulonglong),
                           ("ullTotalVirtual", ctypes.c_ulonglong),
                           ("ullAvailVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024**3
        except Exception:
            return 0


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 0: УСТАНОВКА ЗАВИСИМОСТЕЙ
# ═════════════════════════════════════════════════════════════════════════

def _detect_cuda_version():
    """Определяет версию CUDA через nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            result2 = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, timeout=10
            )
            output = result2.stdout

            match = re.search(r'CUDA Version:\s*(\d+)\.(\d+)', output)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2))
                # PyTorch wheels: cu124 (latest), cu121, cu118
                # CUDA 13.x обратно совместима с cu124
                if major >= 13 or (major == 12 and minor >= 4):
                    return "cu124"
                elif major == 12:
                    return "cu121"
                elif major >= 11:
                    return "cu118"
            return "cu124"  # default для новых драйверов
    except Exception:
        pass
    return None


def _install_torch_cuda(pip_cmd, cuda_tag, extra_flags=None):
    """Устанавливает PyTorch с CUDA поддержкой."""
    extra = extra_flags or []
    
    # Проверяем — может torch уже с CUDA?
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"  ✅ torch — уже установлен (CUDA: True)")
            return
    except ImportError:
        pass
    
    if cuda_tag:
        logger.info(f"  📦 Установка PyTorch с CUDA ({cuda_tag})...")
        torch_url = f"https://download.pytorch.org/whl/{cuda_tag}"
        run(pip_cmd + ["install", "torch", "--index-url", torch_url, "--quiet", "--force-reinstall"] + extra, check=False)
    else:
        logger.info("  📦 Установка PyTorch (CPU)...")
        run(pip_cmd + ["install", "torch", "--quiet"] + extra, check=False)


def phase_0_install():
    """Устанавливает все необходимые пакеты."""
    banner(0, "Установка зависимостей")
    
    global PYTHON
    
    cuda_tag = _detect_cuda_version()
    if cuda_tag:
        logger.info(f"  🎮 NVIDIA GPU обнаружен, CUDA tag: {cuda_tag}")
    
    # ═══ Попытка создания venv (PEP 668 fix) ═══
    venv_dir = ROOT / "venv"
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"
    
    use_venv = False
    if not venv_python.exists():
        logger.info("  🔧 Создание виртуального окружения (venv)...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            logger.info(f"  ✅ venv создан: {venv_dir}")
            use_venv = True
        except (subprocess.CalledProcessError, Exception) as e:
            logger.warning(f"  ⚠ venv не удалось создать: {e}")
            # Удаляем сломанный venv если он частично создался
            if venv_dir.exists():
                shutil.rmtree(str(venv_dir), ignore_errors=True)
                logger.info("  🗑 Сломанный venv удалён")
            logger.info("  Используем системный pip + --break-system-packages")
    elif venv_pip.exists():
        use_venv = True
    else:
        # venv_python есть, но pip нет — сломанный venv
        logger.warning("  ⚠ venv сломан (нет pip), удаляем...")
        shutil.rmtree(str(venv_dir), ignore_errors=True)
        logger.info("  Используем системный pip + --break-system-packages")
    
    if use_venv:
        PYTHON = str(venv_python)
        pip_cmd = [str(venv_pip)]
        extra_flags = []
    else:
        PYTHON = sys.executable
        pip_cmd = [PYTHON, "-m", "pip"]
        extra_flags = ["--break-system-packages"]
    
    logger.info(f"  🐍 Python: {PYTHON}")
    
    # ═══ Установка PyTorch (отдельно, с CUDA) ═══
    _install_torch_cuda(pip_cmd, cuda_tag, extra_flags)
    
    # Остальные пакеты
    packages = [
        # ═══ Core (обучение мозга) ═══
        "numpy", "einops", "tqdm",
        "sentencepiece", "tokenizers",
        "sentence-transformers",
        "datasets",
        "psutil",
        # ═══ Voice (фазы 8-10) ═══
        "transformers",          # Whisper model
        "peft",                  # LoRA adapters
        "jiwer",                 # WER метрика для Whisper
        "onnxruntime",           # INT8 квантизация ONNX
        "faster-whisper",        # STT runtime
        "sounddevice",           # Захват микрофона
        # ═══ Hub (API сервер) ═══
        "fastapi",               # REST API
        "uvicorn",               # ASGI сервер
        "websockets",            # WebSocket support
    ]
    
    logger.info("Проверка и установка остальных пакетов...")
    
    for pkg in packages:
        check_cmd = [PYTHON, "-c", f"import {pkg.replace('-', '_')}"]
        try:
            result_ok = subprocess.run(check_cmd, capture_output=True, timeout=10).returncode == 0
        except Exception:
            result_ok = False
        
        if result_ok:
            logger.info(f"  ✅ {pkg} — уже установлен")
        else:
            logger.info(f"  📦 Установка {pkg}...")
            run(pip_cmd + ["install", pkg, "--quiet"] + extra_flags, check=False)
    
    # Проверка CUDA
    gpu_name, vram = gpu_info()
    ram = get_ram_gb()
    
    logger.info("")
    logger.info(f"  🖥️  GPU: {gpu_name or 'Не найден'}")
    if vram > 0:
        logger.info(f"  💾 VRAM: {vram:.1f} GB")
    logger.info(f"  🧠 RAM: {ram:.0f} GB")
    
    return True


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 1: СКАЧИВАНИЕ ДАННЫХ
# ═════════════════════════════════════════════════════════════════════════

def phase_1_download(quick: bool = False):
    """Скачивает все данные для обучения."""
    banner(1, "Скачивание данных")
    
    success = True
    # 1.1 Wikipedia
    wiki_path = DATA / "wiki_ru.txt"
    if quick:
        logger.info("  📚 Wikipedia: пропуск (quick mode, используем встроенный корпус)")
    elif wiki_path.exists() and wiki_path.stat().st_size > 1_000_000:
        wiki_mb = wiki_path.stat().st_size / 1024 / 1024
        logger.info(f"  📚 Wikipedia: уже есть ({wiki_mb:.1f} MB)")
    else:
        logger.info("  📚 Скачивание Wikipedia (100000 статей)...")
        if not run([PYTHON, TRAINING / "download_wiki.py", "--count", "100000"], check=False):
            logger.warning("  ⚠ Wikipedia не скачана — продолжаем")
            success = False
    
    # 1.2 HuggingFace датасеты
    if quick:
        logger.info("  🤗 HuggingFace: пропуск (quick mode)")
    else:
        hf_files = list(DATA.glob("hf_*.txt"))
        if len(hf_files) >= 3:
            total_mb = sum(f.stat().st_size for f in hf_files) / 1024 / 1024
            logger.info(f"  🤗 HuggingFace: уже есть ({len(hf_files)} файлов, {total_mb:.0f} MB)")
        else:
            logger.info("  🤗 Скачивание HuggingFace датасетов (код + чат + агенты)...")
            if not run([PYTHON, TRAINING / "download_hf_dataset.py", "--preset", "all"], check=False):
                logger.warning("  ⚠ HuggingFace не скачан — продолжаем")
                success = False
    
    # 1.3 LEANN embedding model
    emb_path = MODELS / "embeddings"
    if emb_path.exists():
        logger.info(f"  🧠 LEANN embeddings: уже есть ({emb_path})")
    else:
        logger.info("  🧠 Скачивание модели эмбеддингов (all-MiniLM-L6-v2)...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            model.save(str(emb_path))
            logger.info(f"  ✅ Сохранена в {emb_path}")
        except Exception as e:
            logger.warning(f"  ⚠ Embeddings: {e}")
    
    # 1.4 Инициализация LEANN памяти
    logger.info("  🧠 Загрузка данных в LEANN...")
    try:
        sys.path.insert(0, str(TRAINING))
        from ingest_to_leann import ingest_all
        ingest_all()
        logger.info("  ✅ LEANN заполнена")
    except Exception as e:
        logger.info(f"  ℹ LEANN: {e} (не критично)")
    
    # 1.5 Голосовые модели (Whisper CTranslate2 + Piper ONNX + Silero VAD)
    voice_dir = MODELS / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    # 1.5.1 faster-whisper CTranslate2 (для runtime STT)
    whisper_ct2 = voice_dir / "whisper_tiny"
    if (whisper_ct2 / "model.bin").exists():
        logger.info(f"  🎙 Whisper CTranslate2: уже есть ({whisper_ct2})")
    else:
        logger.info("  🎙 Скачивание Whisper Tiny (CTranslate2) для STT...")
        try:
            from faster_whisper import WhisperModel
            _m = WhisperModel("tiny", device="cpu", compute_type="int8",
                              download_root=str(voice_dir))
            # faster-whisper кеширует модель, копируем в нужное место
            import huggingface_hub
            cached = huggingface_hub.snapshot_download("guillaumekln/faster-whisper-tiny")
            if not (whisper_ct2 / "model.bin").exists():
                shutil.copytree(cached, str(whisper_ct2), dirs_exist_ok=True)
            del _m
            logger.info(f"  ✅ Whisper Tiny CTranslate2 сохранён: {whisper_ct2}")
        except Exception as e:
            logger.warning(f"  ⚠ Whisper CTranslate2: {e}")
    
    # 1.5.2 Piper TTS ONNX (русский голос для синтеза речи)
    piper_models = [
        ("ru_RU-irina-medium.onnx",
         "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"),
        ("ru_RU-irina-medium.onnx.json",
         "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json"),
    ]
    for fname, url in piper_models:
        dst = voice_dir / fname
        if dst.exists():
            size_mb = dst.stat().st_size / 1024 / 1024
            logger.info(f"  🗣 Piper {fname}: уже есть ({size_mb:.1f} MB)")
        else:
            logger.info(f"  🗣 Скачивание Piper {fname}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, str(dst))
                size_mb = dst.stat().st_size / 1024 / 1024
                logger.info(f"  ✅ {fname}: {size_mb:.1f} MB")
            except Exception as e:
                logger.warning(f"  ⚠ Piper {fname}: {e}")
    
    # 1.5.3 Silero VAD ONNX (детекция голоса)
    vad_path = voice_dir / "silero_vad.onnx"
    if vad_path.exists():
        logger.info(f"  👂 Silero VAD: уже есть")
    else:
        logger.info("  👂 Скачивание Silero VAD...")
        try:
            vad_url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            import urllib.request
            urllib.request.urlretrieve(vad_url, str(vad_path))
            size_mb = vad_path.stat().st_size / 1024 / 1024
            logger.info(f"  ✅ Silero VAD: {size_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"  ⚠ Silero VAD: {e}")
    
    # Сводка данных
    total_data = 0
    for f in DATA.glob("*.txt"):
        total_data += f.stat().st_size
    for f in DATA.glob("*.json"):
        total_data += f.stat().st_size
    
    logger.info(f"\n  📊 Итого данных: {total_data / 1024 / 1024:.0f} MB")
    return success


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 2: РЕФЛЕКСЫ (TIER 1)
# ═════════════════════════════════════════════════════════════════════════

def phase_2_reflex(quick: bool = False):
    """Обучение рефлексного классификатора."""
    banner(2, "Рефлексы (MinGRU Classifier)")
    
    epochs = "10" if quick else "100"
    return run([PYTHON, TRAINING / "train_reflex.py",
        "--epochs", epochs,
        "--lr", "0.002",
    ])


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 3: MinGRU LANGUAGE MODEL (TIER 1 / System 1)
# ═════════════════════════════════════════════════════════════════════════

def phase_3_mingru(device: str, quick: bool = False):
    """Обучение MinGRU LM — быстрый языковой генератор."""
    banner(3, "MinGRU Language Model (System 1)")
    
    if quick:
        logger.info("  ⚡ Quick mode: dim=256, layers=4, 3 эпохи")
        return run([PYTHON, TRAINING / "train_mingru.py",
            "--epochs", "3",
            "--lr", "3e-3",
            "--dim", "256",
            "--layers", "4",
            "--batch", "16",
            "--seq_len", "128",
        ])
    
    return run([PYTHON, TRAINING / "train_mingru.py",
        "--epochs", "25",           # 25 эпох (быстро на GPU)
        "--lr", "3e-3",
        "--dim", "512",             # Полноценная размерность
        "--layers", "6",            # 6 слоёв MinGRU
        "--batch", "32",            # Большой батч (64GB RAM)
        "--seq_len", "256",
        "--augment",                # + HuggingFace данные
    ])


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 4: MAMBA-2 BRAIN (TIER 2 / System 2) — ОСНОВНОЕ ОБУЧЕНИЕ
# ═════════════════════════════════════════════════════════════════════════

def phase_4_mamba2(device: str, resume: bool = False, quick: bool = False):
    """
    Обучение Mamba-2 brain — полная архитектура.
    
    4 под-фазы:
      Phase 1: Full pretrain (все параметры, 3 эпохи)
      Phase 2: Fine-tune WKV + Fusion (SSD frozen, 2 эпохи)
      Phase 3: Fine-tune MoLE + MatrixPool (1 эпоха)
      Phase 4: Fine-tune WKV Fusion + RAG (1 эпоха)
    """
    banner(4, "Mamba-2 Brain (12×768d, Full Architecture)")
    
    # ═══ Transfer embedding MinGRU → Mamba-2 ═══
    emb_path = None
    mingru_weights = MODELS / "mingru_weights.pt"
    if mingru_weights.exists():
        logger.info("🔗 Перенос эмбеддинга MinGRU → Mamba-2...")
        try:
            import torch
            cp = torch.load(str(mingru_weights), map_location='cpu', weights_only=False)
            state = cp.get('model_state_dict', cp)
            for k in state:
                if 'shared_embedding' in k or 'emb.weight' in k:
                    emb_tensor = state[k]
                    TARS_V3.mkdir(parents=True, exist_ok=True)
                    emb_path = str(TARS_V3 / "_transfer_embedding.pt")
                    torch.save(emb_tensor, emb_path)
                    logger.info(f"  ✅ Embedding ({emb_tensor.shape}) → {emb_path}")
                    break
        except Exception as e:
            logger.warning(f"  ⚠ Transfer failed: {e}")
    
    # ═══ Базовые аргументы для RTX 4090 ═══
    if quick:
        logger.info("  ⚡ Quick mode: d_model=256, n_layers=4, 1 эпоха на фазу")
        base = [
            "--d_model", "256",
            "--n_layers", "4",
            "--vocab_size", "256",
            "--batch", "8",
            "--accum_steps", "2",
            "--device", device,
            "--curriculum",
            "--label_smoothing", "0.1",
            "--max_samples", "500",
            "--no_compile",  # T4 не поддерживает linalg.solve в FP16 через compile
        ]
    else:
        base = [
            "--d_model", "2048",        # TARS 1B
            "--n_layers", "24",         # 24 rich blocks
            "--vocab_size", "256",      # cp1251 байты
            "--batch", "4",             # 4 × seq → fits 16GB VRAM (bf16+ckpt)
            "--accum_steps", "16",      # Effective batch = 64
            "--device", device,
            "--curriculum",             # Curriculum learning (64→128→256)
            "--label_smoothing", "0.1",
        ]
    # ═══ BitMamba-2 optimizations: bf16 + grad checkpoint ═══
    if device != "cpu" and not quick:
        base += ["--bf16", "--grad_ckpt"]
    
    if emb_path:
        base += ["--pretrained_emb", emb_path]
    
    results = {}
    
    # ── Phase 1: Full pretrain (все компоненты) ──
    logger.info("── Phase 1/4: Full pretrain (SSD + WKV + Ω-SSM + MoLE + WaveMerge) ──")
    quick_epochs = "1"
    results["p1"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", quick_epochs if quick else "5",  # Tars.txt §7.3
        "--lr", "3e-4",
        "--phase", "1",
        "--seq_len", "128" if quick else "256",
    ])
    
    # ── Phase 2: Fine-tune WKV + Fusion (SSD frozen) ──
    logger.info("── Phase 2/4: Fine-tune WKV + Fusion (SSD frozen) ──")
    results["p2"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", quick_epochs if quick else "3",  # Tars.txt §7.3
        "--lr", "1e-4",
        "--phase", "2",
        "--seq_len", "128" if quick else "512",
        "--resume",
    ])
    
    # ── Phase 3: Fine-tune MoLE + MatrixPool + WaveMerge ──
    logger.info("── Phase 3/4: Fine-tune MoLE + MatrixPool + WaveMerge ──")
    results["p3"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", quick_epochs if quick else "2",  # Tars.txt §7.3
        "--lr", "3e-5",
        "--phase", "3",
        "--seq_len", "128" if quick else "512",
        "--resume",
    ])
    
    # ── Phase 4: Fine-tune WKV RAG State + Memory Integration ──
    logger.info("── Phase 4/4: Fine-tune WKV + RAG + Memory Injection ──")
    results["p4"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", quick_epochs if quick else "2",  # Tars.txt §7.3
        "--lr", "1.5e-5",
        "--phase", "4",
        "--seq_len", "128" if quick else "512",
        "--resume",
    ])
    
    all_ok = all(results.values())
    if all_ok:
        logger.info("✅ Все 4 фазы Mamba-2 завершены")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.warning(f"⚠ Фазы с ошибками: {failed}")
    
    return all_ok


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 5: КВАНТИЗАЦИЯ 1.58-bit
# ═════════════════════════════════════════════════════════════════════════

def phase_5_quantize(device: str, quick: bool = False):
    """Квантизация FP16 → 1.58-bit + дообучение."""
    banner(5, "Квантизация BitNet 1.58-bit")
    
    # Ищем обученную FP16 модель
    fp16_path = MODELS / "mamba2" / "mamba2_omega.pt"
    
    if not fp16_path.exists():
        logger.info("  🔍 FP16 модель не найдена локально, скачиваем из HuggingFace...")
        fp16_path.parent.mkdir(parents=True, exist_ok=True)
        
        downloaded = False
        # Пробуем скачать pre-trained Mamba-2 из state-spaces
        hf_models = [
            "state-spaces/mamba2-130m",
            "state-spaces/mamba2-370m",
            "state-spaces/mamba-130m",
        ]
        
        for hf_model in hf_models:
            try:
                logger.info(f"  📥 Скачивание {hf_model}...")
                from transformers import AutoModelForCausalLM
                import torch
                
                hf_m = AutoModelForCausalLM.from_pretrained(
                    hf_model, torch_dtype=torch.float16, trust_remote_code=True
                )
                
                # Сохраняем в формате, совместимом с TarsMamba2LM
                # Извлекаем конфиг из HF модели
                hf_cfg = hf_m.config
                d_model = getattr(hf_cfg, 'd_model', getattr(hf_cfg, 'hidden_size', 768))
                n_layers = getattr(hf_cfg, 'n_layer', getattr(hf_cfg, 'num_hidden_layers', 24))
                vocab_size = getattr(hf_cfg, 'vocab_size', 50280)
                
                # Создаём TarsMamba2LM с подходящими размерами
                from brain.mamba2.model import TarsMamba2LM
                tars_model = TarsMamba2LM(
                    d_model=d_model, n_layers=n_layers,
                    vocab_size=vocab_size, quant_mode="fp16",
                )
                
                # Переносим совместимые веса
                hf_state = hf_m.state_dict()
                tars_state = tars_model.state_dict()
                loaded = 0
                for key in tars_state:
                    # Пробуем найти соответствие по имени или форме
                    for hf_key, hf_val in hf_state.items():
                        if hf_val.shape == tars_state[key].shape:
                            if key.split('.')[-1] == hf_key.split('.')[-1]:
                                tars_state[key] = hf_val
                                loaded += 1
                                break
                
                tars_model.load_state_dict(tars_state, strict=False)
                
                # Сохраняем
                checkpoint = {
                    "model_state_dict": tars_model.state_dict(),
                    "config": {
                        "d_model": d_model,
                        "n_layers": n_layers,
                        "vocab_size": vocab_size,
                        "quant_mode": "fp16",
                    },
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "vocab_size": vocab_size,
                    "source": hf_model,
                }
                torch.save(checkpoint, str(fp16_path))
                
                size_mb = fp16_path.stat().st_size / 1024 / 1024
                logger.info(f"  ✅ {hf_model} скачан и конвертирован: {size_mb:.1f} MB")
                logger.info(f"  📊 d_model={d_model}, n_layers={n_layers}, vocab={vocab_size}")
                logger.info(f"  🔗 Перенесено {loaded} тензоров")
                
                del hf_m, tars_model
                downloaded = True
                break
                
            except Exception as e:
                logger.warning(f"  ⚠ {hf_model}: {e}")
                continue
        
        if not downloaded:
            logger.warning(f"  ⚠ Не удалось скачать Mamba-2 из HuggingFace")
            logger.warning("  Пропускаем квантизацию")
            return False
    
    logger.info(f"  Исходная модель: {fp16_path}")
    fp16_size = fp16_path.stat().st_size / 1024 / 1024
    logger.info(f"  Размер FP16: {fp16_size:.1f} MB")
    
    # Квантизация + дообучение 5 эпох
    return run([PYTHON, TRAINING / "train_mamba2.py",
        "--d_model", "768",
        "--n_layers", "12",
        "--batch", "16",
        "--accum_steps", "4",
        "--epochs", "1" if quick else "3",  # Tars.txt §7.5
        "--lr", "5e-5",
        "--phase", "1",
        "--quant",                  # 1.58-bit режим (STE)
        "--resume",
        "--device", device,
        "--seq_len", "256",
        "--label_smoothing", "0.1",
    ])


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 6: ФИНАЛЬНАЯ СБОРКА
# ═════════════════════════════════════════════════════════════════════════

def phase_6_consolidate(results: dict, total_time: float):
    """Собирает все модели в models/tars_v3/ и генерирует config.json."""
    banner(6, "Финальная сборка")
    
    TARS_V3.mkdir(parents=True, exist_ok=True)
    
    copies = {
        "reflex": (MODELS / "reflex" / "reflex_classifier.pt", TARS_V3 / "reflex.pt"),
        "mingru": (MODELS / "mingru_weights.pt", TARS_V3 / "mingru.pt"),
        "mamba2_fp16": (MODELS / "mamba2" / "mamba2_omega.pt", TARS_V3 / "mamba2.pt"),
        "mamba2_158bit": (MODELS / "mamba2" / "mamba2_omega_158bit.pt", TARS_V3 / "mamba2_158bit.pt"),
    }
    
    consolidated = []
    for name, (src, dst) in copies.items():
        if src.exists():
            shutil.copy2(str(src), str(dst))
            size_mb = dst.stat().st_size / 1024 / 1024
            logger.info(f"  📦 {name}: {src.name} → tars_v3/{dst.name} ({size_mb:.1f} MB)")
            consolidated.append(name)
        else:
            logger.info(f"  ⏭ {name}: не найден ({src.name})")
    
    # ═══ Генерация config.json (нужен для load_pretrained) ═══
    # Пытаемся прочитать конфиг из checkpoint, иначе — дефолт
    model_config = {
        "d_model": 768, "n_layers": 12, "vocab_size": 256,
        "d_state": 64, "headdim": 64, "omega_dim": 32,
        "pool_size": 48, "n_experts": 8,
    }
    # Ищем конфиг в .pt файле
    for pt_name in ["mamba2.pt", "mamba2_158bit.pt"]:
        pt_path = TARS_V3 / pt_name
        if pt_path.exists():
            try:
                import torch
                ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                if "config" in ckpt and isinstance(ckpt["config"], dict):
                    cfg = ckpt["config"]
                    model_config.update({
                        "d_model": cfg.get("d_model", model_config["d_model"]),
                        "n_layers": cfg.get("n_layers", model_config["n_layers"]),
                        "vocab_size": cfg.get("vocab_size", model_config["vocab_size"]),
                    })
                    logger.info(f"  📄 config из {pt_name}: d={model_config['d_model']}, L={model_config['n_layers']}")
                elif "d_model" in ckpt:
                    model_config["d_model"] = ckpt["d_model"]
                    model_config["n_layers"] = ckpt.get("n_layers", model_config["n_layers"])
                    model_config["vocab_size"] = ckpt.get("vocab_size", model_config["vocab_size"])
                    logger.info(f"  📄 config из {pt_name}: d={model_config['d_model']}, L={model_config['n_layers']}")
                del ckpt
                break
            except Exception as e:
                logger.warning(f"  ⚠ Не удалось прочитать config из {pt_name}: {e}")
    
    config_json = {
        "name": "tars_v3",
        "version": "3.0",
        "encoding": "cp1251",
        "models": {
            "mamba2": {
                "params": model_config
            }
        }
    }
    config_path = TARS_V3 / "config.json"
    config_path.write_text(json.dumps(config_json, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"  📄 config.json сохранён: {config_path}")
    
    # Лог обучения
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": round(total_time, 1),
        "total_time_human": f"{total_time/3600:.1f} часов",
        "results": {k: ("ok" if v else "failed") for k, v in results.items()},
        "models_consolidated": consolidated,
        "config": model_config,
    }
    
    log_path = TARS_V3 / "training_log.json"
    logs = []
    if log_path.exists():
        try:
            logs = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not isinstance(logs, list):
        logs = []
    logs.append(log_entry)
    log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    
    logger.info(f"\n  📋 Training log: {log_path}")
    return True


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 7: ВАЛИДАЦИЯ
# ═════════════════════════════════════════════════════════════════════════

def phase_7_validate():
    """Тестовая генерация для проверки модели."""
    banner(7, "Валидация")
    
    try:
        sys.path.insert(0, str(ROOT))
        import torch
        from brain.tokenizer import TarsTokenizer
        from brain.mamba2.model import TarsMamba2LM
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = TarsTokenizer()
        
        # Загрузка обученной модели
        model, ckpt = TarsMamba2LM.load_pretrained(device=device)
        model.eval()
        
        if ckpt is None:
            logger.warning("  ⚠ Нет обученных весов — модель случайная")
            return False
        
        # Тестовые промпты
        test_prompts = ["привет", "как дела", "что такое"]
        
        logger.info(f"  Модель: {ckpt}")
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Параметры: {params:,}")
        
        for prompt in test_prompts:
            tokens = tokenizer.encode(prompt)
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            
            with torch.no_grad():
                logits = model(input_ids)
            
            # Берём top-5 предсказаний для следующего токена
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top5 = torch.topk(probs, 5)
            
            predictions = []
            for idx, prob in zip(top5.indices.tolist(), top5.values.tolist()):
                char = tokenizer.decode([idx])
                predictions.append(f"'{char}'({prob:.2%})")
            
            logger.info(f"  \"{prompt}\" → {', '.join(predictions)}")
        
        logger.info("  ✅ Модель работает!")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Валидация провалилась: {e}")
        import traceback
        traceback.print_exc()
        return False


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 8: WHISPER TINY FINE-TUNE (РУССКИЙ)
# ═════════════════════════════════════════════════════════════════════════

def phase_8_whisper(device: str, quick: bool = False):
    """Дообучение Whisper Tiny для русского (LoRA)."""
    banner(8, "Whisper Tiny LoRA (Russian STT)")

    # Установка дополнительных зависимостей
    logger.info("  📦 Установка peft, jiwer...")
    run([PYTHON, "-m", "pip", "install", "peft", "jiwer", "-q"], check=False)

    args_list = [
        PYTHON, TRAINING / "train_whisper.py",
        "--device", device,
    ]
    if quick:
        args_list += ["--samples", "500", "--epochs", "1", "--batch", "8"]
    else:
        args_list += ["--samples", "5000", "--epochs", "3", "--batch", "16"]

    return run(args_list)


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 9: PIPER TTS FINE-TUNE (РУССКИЙ)
# ═════════════════════════════════════════════════════════════════════════

def phase_9_piper(quick: bool = False):
    """Дообучение Piper TTS для русского голоса."""
    banner(9, "Piper TTS (Russian Voice)")

    # Установка дополнительных зависимостей
    logger.info("  📦 Установка piper-tts, piper-phonemize...")
    run([PYTHON, "-m", "pip", "install", "piper-tts", "piper-phonemize", "-q"], check=False)

    args_list = [
        PYTHON, TRAINING / "train_piper.py",
    ]
    if quick:
        args_list += ["--epochs", "100", "--max_samples", "200", "--batch", "8"]
    else:
        args_list += ["--epochs", "1000", "--max_samples", "3000", "--batch", "16"]

    return run(args_list)


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 10: КВАНТИЗАЦИЯ ГОЛОСОВЫХ МОДЕЛЕЙ (INT8)
# ═════════════════════════════════════════════════════════════════════════

def phase_10_quantize_voice():
    """INT8 квантизация Whisper ONNX + Piper ONNX."""
    banner(10, "Квантизация голосовых моделей (INT8)")

    # Whisper Vocabulary Boost
    logger.info("  📝 Генерация Whisper hotwords...")
    run([PYTHON, TRAINING / "whisper_boost.py"], check=False)

    # ONNX квантизация
    return run([PYTHON, TRAINING / "quantize_voice.py"])


# ═════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ТАРС v3 — Полностью автономный пайплайн обучения",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python mega_train.py                        # Полный пайплайн (~15 часов)
  python mega_train.py --skip-download        # Данные уже скачаны
  python mega_train.py --phase 4              # Только Mamba-2
  python mega_train.py --phase 8              # Только Whisper
  python mega_train.py --phase 9              # Только Piper
  python mega_train.py --quick                # Быстрый тест (256d, 4 слоя)
  python mega_train.py --skip-quantize        # Без квантизации
  python mega_train.py --skip-voice           # Без голосовых фаз (8-10)
        """
    )
    
    parser.add_argument("--skip-download", action="store_true",
                        help="Пропустить скачивание данных")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Пропустить квантизацию 1.58-bit")
    parser.add_argument("--skip-voice", action="store_true",
                        help="Пропустить голосовые фазы (Whisper, Piper, квант.)")
    parser.add_argument("--phase", type=int, choices=[0,1,2,3,4,5,6,7,8,9,10],
                        help="Запустить только конкретную фазу")
    parser.add_argument("--quick", action="store_true",
                        help="Быстрый тест (маленькая модель, 1 эпоха)")
    
    args = parser.parse_args()
    
    # ═══ Определение железа ═══
    gpu_name, vram = gpu_info()
    ram = get_ram_gb()
    device = "cuda" if gpu_name else "cpu"
    
    logger.info("")
    logger.info("╔" + "═" * 62 + "╗")
    logger.info("║                                                              ║")
    logger.info("║   ████████╗ █████╗ ██████╗ ███████╗    ██╗   ██╗██████╗     ║")
    logger.info("║      ██╔══╝██╔══██╗██╔══██╗██╔════╝    ██║   ██║╚════██╗    ║")
    logger.info("║      ██║   ███████║██████╔╝███████╗    ██║   ██║ █████╔╝    ║")
    logger.info("║      ██║   ██╔══██║██╔══██╗╚════██║    ╚██╗ ██╔╝ ╚═══██╗    ║")
    logger.info("║      ██║   ██║  ██║██║  ██║███████║     ╚████╔╝ ██████╔╝    ║")
    logger.info("║      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝      ╚═══╝  ╚═════╝     ║")
    logger.info("║                                                              ║")
    logger.info("║           MEGA TRAIN — Autonomous Pipeline                   ║")
    logger.info("║                                                              ║")
    logger.info("╚" + "═" * 62 + "╝")
    logger.info("")
    logger.info(f"  🖥️  GPU:    {gpu_name or 'CPU only'}")
    logger.info(f"  💾 VRAM:   {vram:.1f} GB" if vram > 0 else "  💾 VRAM:   N/A")
    logger.info(f"  🧠 RAM:    {ram:.0f} GB")
    logger.info(f"  🐍 Python: {sys.version.split()[0]}")
    logger.info(f"  📁 Root:   {ROOT}")
    logger.info(f"  ⏰ Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    if args.quick:
        logger.info("  ⚡ QUICK MODE: уменьшенная модель (256d, 4 слоя, 1 эпоха)")
    
    if device == "cuda" and vram < 8:
        logger.warning("⚠ VRAM < 8GB — рекомендуется уменьшить batch size")
    
    t0 = time.time()
    results = {}
    
    # ═══ Выполнение фаз ═══
    phases = {
        0: ("install", lambda: phase_0_install()),
        1: ("download", lambda: phase_1_download(quick=args.quick)),
        2: ("reflex", lambda: phase_2_reflex(quick=args.quick)),
        3: ("mingru", lambda: phase_3_mingru(device, quick=args.quick)),
        4: ("mamba2", lambda: phase_4_mamba2(device, quick=args.quick)),
        5: ("quantize", lambda: phase_5_quantize(device, quick=args.quick)),
        7: ("validate", lambda: phase_7_validate()),
        8: ("whisper", lambda: phase_8_whisper(device, quick=args.quick)),
        9: ("piper", lambda: phase_9_piper(quick=args.quick)),
        10: ("voice_quant", lambda: phase_10_quantize_voice()),
    }
    
    # Если задана конкретная фаза
    if args.phase is not None:
        if args.phase in phases:
            name, func = phases[args.phase]
            results[name] = func()
        elif args.phase == 6:
            # Помечаем все остальные фазы как "skipped"
            for pn in ["install", "download", "reflex", "mingru", "mamba2", "quantize", "validate"]:
                results[pn] = "skipped"
            phase_6_consolidate(results, time.time() - t0)
        else:
            logger.error(f"Неизвестная фаза: {args.phase}")
        total = time.time() - t0
        logger.info(f"\n  Фаза {args.phase} завершена за {total:.0f}s")
        return
    
    # ═══ Полный пайплайн ═══
    
    # Фаза 0: Зависимости
    results["install"] = phase_0_install()
    
    # Переопределяем device после установки torch с CUDA в venv
    gpu_name, vram = gpu_info()
    device = "cuda" if gpu_name else "cpu"
    if gpu_name:
        logger.info(f"  🖥️  GPU обнаружен: {gpu_name} ({vram:.1f} GB VRAM)")
    
    # Фаза 1: Данные
    if not args.skip_download:
        results["download"] = phase_1_download(quick=args.quick)
    else:
        logger.info("⏭ Пропуск скачивания данных (--skip-download)")
        results["download"] = True
    
    # Фаза 2: Рефлексы
    results["reflex"] = phase_2_reflex(quick=args.quick)
    
    # Фаза 3: MinGRU
    results["mingru"] = phase_3_mingru(device, quick=args.quick)
    
    # Фаза 4: Mamba-2 (основное обучение)
    results["mamba2"] = phase_4_mamba2(device, quick=args.quick)
    
    # Фаза 5: Квантизация
    if not args.skip_quantize:
        results["quantize"] = phase_5_quantize(device, quick=args.quick)
    else:
        logger.info("⏭ Пропуск квантизации (--skip-quantize)")
        results["quantize"] = True
    
    # Фаза 6: Сборка (промежуточная)
    phase_6_consolidate(results, time.time() - t0)
    
    # Фаза 7: Валидация мозга
    results["validate"] = phase_7_validate()
    
    # ═══ ГОЛОСОВОЙ ПАЙПЛАЙН (Фазы 8-10) ═══
    if not args.skip_voice:
        # Фаза 8: Whisper STT fine-tune
        results["whisper"] = phase_8_whisper(device, quick=args.quick)
        
        # Фаза 9: Piper TTS fine-tune
        results["piper"] = phase_9_piper(quick=args.quick)
        
        # Фаза 10: Квантизация голосовых ONNX
        results["voice_quant"] = phase_10_quantize_voice()
    else:
        logger.info("⏭ Пропуск голосовых фаз (--skip-voice)")
        results["whisper"] = True
        results["piper"] = True
        results["voice_quant"] = True
    
    # ═══ INSTRUCTION TUNING (Фаза 11) ═══
    banner(11, "Instruction Tuning (→ помощник)", total=11)
    try:
        from training.train_instruct import (
            load_instruction_dataset, train_instruct, download_instruct_datasets
        )
        
        # Скачать datasets если нужно
        download_instruct_datasets()
        
        # Загрузить данные
        instruct_texts = load_instruction_dataset()
        
        if instruct_texts and results.get("mamba2"):
            # Загрузить обученную модель
            from brain.mamba2.model import TarsMamba2LM
            model_path = ROOT / "models" / "tars_v3" / "mamba2.pt"
            if model_path.exists():
                import torch
                state = torch.load(model_path, map_location=device, weights_only=False)
                config = state.get("config", {})
                model = TarsMamba2LM(**config).to(device)
                model.load_state_dict(state["model_state_dict"], strict=False)
                
                def tokenize_cp1251(text):
                    tokens = list(text.encode('cp1251', errors='replace')[:1024])
                    return torch.tensor([tokens], dtype=torch.long)
                
                model = train_instruct(
                    model, tokenize_cp1251, instruct_texts,
                    epochs=2 if args.quick else 3,
                    lr=5e-5, batch_size=4
                )
                
                # Сохранить
                state["model_state_dict"] = model.state_dict()
                torch.save(state, model_path)
                logger.info(f"✅ Instruction-tuned модель сохранена: {model_path}")
                results["instruct"] = True
            else:
                logger.warning("⚠ Модель brain_mamba2.pt не найдена — пропуск instruction tuning")
                results["instruct"] = False
        else:
            logger.info("⏭ Нет данных или модели для instruction tuning")
            results["instruct"] = len(instruct_texts) == 0
    except Exception as e:
        logger.error(f"❌ Instruction tuning failed: {e}")
        import traceback; traceback.print_exc()
        results["instruct"] = False
    
    # ═══ ИТОГИ ═══
    total_time = time.time() - t0
    hours = total_time / 3600
    
    logger.info("")
    logger.info("╔" + "═" * 62 + "╗")
    logger.info("║              ИТОГИ ОБУЧЕНИЯ                                  ║")
    logger.info("╠" + "═" * 62 + "╣")
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        logger.info(f"║  {icon} {name:<20s}                                    ║")
    logger.info("╠" + "═" * 62 + "╣")
    logger.info(f"║  ⏱  Время: {hours:.1f} часов ({total_time:.0f} сек)                         ║")
    logger.info("╚" + "═" * 62 + "╝")
    
    if all(results.values()):
        logger.info("")
        logger.info("  🎯 ВСЕ ФАЗЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        logger.info("  📁 Модели собраны: models/tars_v3/")
        logger.info("  🎤 Голос: Whisper (RU) + Piper (RU) + INT8")
        logger.info("  🚀 Запуск: python launch_tars.py")
        logger.info("")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.info("")
        logger.info(f"  ⚠ Фазы с ошибками: {', '.join(failed)}")
        logger.info("  Проверьте mega_train.log для деталей")
        logger.info("")


if __name__ == "__main__":
    main()
