"""
═══════════════════════════════════════════════════════════════════════════
  mega_train.py — ТАРС v3: Полностью автономный пайплайн обучения
═══════════════════════════════════════════════════════════════════════════

Один скрипт делает ВСЁ:

  Фаза 0: Установка зависимостей (pip install)
  Фаза 1: Скачивание данных (Wikipedia + HuggingFace + LEANN)
  Фаза 2: Обучение рефлексов (MinGRU classifier, ~1 мин)
  Фаза 3: Обучение MinGRU LM (System 1, ~30 мин GPU)
  Фаза 4: Обучение Mamba-2 (12 слоёв × 768d, 4 фазы, ~2-4ч GPU)
  Фаза 5: Квантизация 1.58-bit + дообучение (~30 мин)
  Фаза 6: Финальная сборка в models/tars_v3/
  Фаза 7: Валидация (тестовая генерация)

Оптимизировано для: RTX 4090 (24GB VRAM) + 64GB RAM

Использование:
  python mega_train.py              # Полный пайплайн
  python mega_train.py --skip-download  # Без скачивания (данные есть)
  python mega_train.py --phase 4    # Только Mamba-2
  python mega_train.py --quick      # Быстрый тест (маленькая модель)

═══════════════════════════════════════════════════════════════════════════
"""

import os
import sys
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

def banner(phase: int, title: str, total: int = 7):
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
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
            return name, vram
    except Exception:
        pass
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
                           ("ullTotalPhys", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024**3
        except Exception:
            return 0


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 0: УСТАНОВКА ЗАВИСИМОСТЕЙ
# ═════════════════════════════════════════════════════════════════════════

def phase_0_install():
    """Устанавливает все необходимые пакеты."""
    banner(0, "Установка зависимостей")
    
    global PYTHON
    
    # ═══ Автоматическое создание venv (PEP 668 fix) ═══
    venv_dir = ROOT / "venv"
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
        venv_pip = venv_dir / "Scripts" / "pip.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
        venv_pip = venv_dir / "bin" / "pip"
    
    if not venv_python.exists():
        logger.info("  🔧 Создание виртуального окружения (venv)...")
        try:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            logger.info(f"  ✅ venv создан: {venv_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ❌ Не удалось создать venv: {e}")
            logger.info("  Попытка установки через --break-system-packages...")
            # Fallback: использовать системный pip с --break-system-packages
            PYTHON = sys.executable
            pip_cmd = [PYTHON, "-m", "pip"]
            packages = [
                "torch", "numpy", "einops", "tqdm",
                "sentencepiece", "tokenizers",
                "sentence-transformers", "datasets", "psutil",
            ]
            logger.info("Проверка и установка пакетов...")
            for pkg in packages:
                try:
                    __import__(pkg.replace("-", "_"))
                    logger.info(f"  ✅ {pkg} — уже установлен")
                except ImportError:
                    logger.info(f"  📦 Установка {pkg}...")
                    run(pip_cmd + ["install", pkg, "--quiet", "--break-system-packages"], check=False)
            return True
    
    # Обновляем PYTHON для всего пайплайна
    PYTHON = str(venv_python)
    pip_cmd = [str(venv_pip)]
    logger.info(f"  🐍 Python: {PYTHON}")
    
    # Обязательные пакеты
    packages = [
        # Ядро
        "torch", "numpy", "einops", "tqdm",
        # Токенайзеры
        "sentencepiece", "tokenizers",
        # Память (LEANN)
        "sentence-transformers",
        # Датасеты
        "datasets",
        # Мониторинг
        "psutil",
    ]
    
    logger.info("Проверка и установка пакетов...")
    
    for pkg in packages:
        # Проверяем через venv python
        check_cmd = [PYTHON, "-c", f"import {pkg.replace('-', '_')}"]
        try:
            subprocess.run(check_cmd, capture_output=True, timeout=10)
            result_ok = subprocess.run(check_cmd, capture_output=True, timeout=10).returncode == 0
        except Exception:
            result_ok = False
        
        if result_ok:
            logger.info(f"  ✅ {pkg} — уже установлен")
        else:
            logger.info(f"  📦 Установка {pkg}...")
            run(pip_cmd + ["install", pkg, "--quiet"], check=False)
    
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

def phase_1_download():
    """Скачивает все данные для обучения."""
    banner(1, "Скачивание данных")
    
    success = True
    
    # 1.1 Wikipedia
    wiki_path = DATA / "wiki_ru.txt"
    if wiki_path.exists() and wiki_path.stat().st_size > 1_000_000:
        wiki_mb = wiki_path.stat().st_size / 1024 / 1024
        logger.info(f"  📚 Wikipedia: уже есть ({wiki_mb:.1f} MB)")
    else:
        logger.info("  📚 Скачивание Wikipedia (100 000 статей)...")
        if not run([PYTHON, TRAINING / "download_wiki.py"], check=False):
            logger.warning("  ⚠ Wikipedia не скачана — продолжаем")
            success = False
    
    # 1.2 HuggingFace датасеты
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

def phase_2_reflex():
    """Обучение рефлексного классификатора."""
    banner(2, "Рефлексы (MinGRU Classifier)")
    
    return run([PYTHON, TRAINING / "train_reflex.py",
        "--epochs", "100",          # 100 эпох для максимальной точности
        "--lr", "0.002",
    ])


# ═════════════════════════════════════════════════════════════════════════
#  ФАЗА 3: MinGRU LANGUAGE MODEL (TIER 1 / System 1)
# ═════════════════════════════════════════════════════════════════════════

def phase_3_mingru(device: str):
    """Обучение MinGRU LM — быстрый языковой генератор."""
    banner(3, "MinGRU Language Model (System 1)")
    
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

def phase_4_mamba2(device: str, resume: bool = False):
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
    base = [
        "--d_model", "768",         # Полная модель
        "--n_layers", "12",         # 12 блоков TarsBlock
        "--vocab_size", "256",      # cp1251 байты
        "--batch", "16",            # 16 × seq → 24GB VRAM OK
        "--accum_steps", "4",       # Effective batch = 64
        "--device", device,
        "--curriculum",             # Curriculum learning (64→128→256)
        "--label_smoothing", "0.1",
    ]
    if emb_path:
        base += ["--pretrained_emb", emb_path]
    
    results = {}
    
    # ── Phase 1: Full pretrain (все компоненты) ──
    logger.info("── Phase 1/4: Full pretrain (SSD + WKV + Ω-SSM + MoLE + WaveMerge) ──")
    results["p1"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", "8",            # 8 эпох (24ч сессия → ~6ч на Phase 1)
        "--lr", "3e-4",
        "--phase", "1",
        "--seq_len", "256",
    ])
    
    # ── Phase 2: Fine-tune WKV + Fusion (SSD frozen) ──
    logger.info("── Phase 2/4: Fine-tune WKV + Fusion (SSD frozen) ──")
    results["p2"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", "5",            # 5 эпох (24ч → ~5ч на Phase 2)
        "--lr", "1e-4",
        "--phase", "2",
        "--seq_len", "512",         # Длинный контекст
        "--resume",
    ])
    
    # ── Phase 3: Fine-tune MoLE + MatrixPool + WaveMerge ──
    logger.info("── Phase 3/4: Fine-tune MoLE + MatrixPool + WaveMerge ──")
    results["p3"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", "3",            # 3 эпохи (24ч → ~4ч на Phase 3)
        "--lr", "3e-5",
        "--phase", "3",
        "--seq_len", "512",
        "--resume",
    ])
    
    # ── Phase 4: Fine-tune WKV RAG State + Memory Integration ──
    logger.info("── Phase 4/4: Fine-tune WKV + RAG + Memory Injection ──")
    results["p4"] = run([PYTHON, TRAINING / "train_mamba2.py"] + base + [
        "--epochs", "3",            # 3 эпохи (24ч → ~4ч на Phase 4)
        "--lr", "1.5e-5",
        "--phase", "4",
        "--seq_len", "512",
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

def phase_5_quantize(device: str):
    """Квантизация FP16 → 1.58-bit + дообучение."""
    banner(5, "Квантизация BitNet 1.58-bit")
    
    # Ищем обученную FP16 модель
    fp16_path = MODELS / "mamba2" / "mamba2_omega.pt"
    if not fp16_path.exists():
        logger.warning(f"  ⚠ FP16 модель не найдена: {fp16_path}")
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
        "--epochs", "5",            # 5 эпох STE дообучения (~5ч на 4090)
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
    """Собирает все модели в models/tars_v3/."""
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
    
    # Лог обучения
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": round(total_time, 1),
        "total_time_human": f"{total_time/3600:.1f} часов",
        "results": {k: ("ok" if v else "failed") for k, v in results.items()},
        "models_consolidated": consolidated,
        "config": {
            "encoding": "cp1251",
            "vocab_size": 256,
            "d_model": 768,
            "n_layers": 12,
            "n_experts": 8,
            "omega_dim": 32,
            "pool_size": 48,
        },
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
#  MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ТАРС v3 — Полностью автономный пайплайн обучения",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python mega_train.py                        # Полный пайплайн (~4-6 часов)
  python mega_train.py --skip-download        # Данные уже скачаны
  python mega_train.py --phase 4              # Только Mamba-2
  python mega_train.py --quick                # Быстрый тест (256d, 4 слоя)
  python mega_train.py --skip-quantize        # Без квантизации
        """
    )
    
    parser.add_argument("--skip-download", action="store_true",
                        help="Пропустить скачивание данных")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Пропустить квантизацию 1.58-bit")
    parser.add_argument("--phase", type=int, choices=[0,1,2,3,4,5,6,7],
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
    
    if device == "cuda" and vram < 8:
        logger.warning("⚠ VRAM < 8GB — рекомендуется уменьшить batch size")
    
    t0 = time.time()
    results = {}
    
    # ═══ Выполнение фаз ═══
    phases = {
        0: ("install", lambda: phase_0_install()),
        1: ("download", lambda: phase_1_download()),
        2: ("reflex", lambda: phase_2_reflex()),
        3: ("mingru", lambda: phase_3_mingru(device)),
        4: ("mamba2", lambda: phase_4_mamba2(device)),
        5: ("quantize", lambda: phase_5_quantize(device)),
        7: ("validate", lambda: phase_7_validate()),
    }
    
    # Если задана конкретная фаза
    if args.phase is not None:
        if args.phase in phases:
            name, func = phases[args.phase]
            results[name] = func()
        elif args.phase == 6:
            phase_6_consolidate(results, time.time() - t0)
        else:
            logger.error(f"Неизвестная фаза: {args.phase}")
        total = time.time() - t0
        logger.info(f"\n  Фаза {args.phase} завершена за {total:.0f}s")
        return
    
    # ═══ Полный пайплайн ═══
    
    # Фаза 0: Зависимости
    results["install"] = phase_0_install()
    
    # Фаза 1: Данные
    if not args.skip_download:
        results["download"] = phase_1_download()
    else:
        logger.info("⏭ Пропуск скачивания данных (--skip-download)")
        results["download"] = True
    
    # Фаза 2: Рефлексы
    results["reflex"] = phase_2_reflex()
    
    # Фаза 3: MinGRU
    results["mingru"] = phase_3_mingru(device)
    
    # Фаза 4: Mamba-2 (основное обучение)
    results["mamba2"] = phase_4_mamba2(device)
    
    # Фаза 5: Квантизация
    if not args.skip_quantize:
        results["quantize"] = phase_5_quantize(device)
    else:
        logger.info("⏭ Пропуск квантизации (--skip-quantize)")
        results["quantize"] = True
    
    # Фаза 6: Сборка
    total_time = time.time() - t0
    phase_6_consolidate(results, total_time)
    
    # Фаза 7: Валидация
    results["validate"] = phase_7_validate()
    
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
