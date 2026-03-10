"""
═══════════════════════════════════════════════════════════════════════
  ТАРС HELIX LITE — Единый скрипт обучения (Colab + Local)
═══════════════════════════════════════════════════════════════════════

Один файл для ВСЕГО:
  • Авто-подключение Google Drive (Colab)
  • Авто-обновление кода из Git
  • Hardware benchmark + автотюнинг batch size
  • HELIX LITE тренировка (296M params)
  • Чекпоинты + журнал
  • Smoke test

ИСПОЛЬЗОВАНИЕ:
  python local_train.py                        # Интерактивный выбор
  python local_train.py --level small          # Отладка (~15 мин)
  python local_train.py --level medium         # Стандарт (~3 часа)
  python local_train.py --level max            # Продакшн (~15 часов)
  python local_train.py --drive colab          # + Google Drive
  python local_train.py --test-only            # Только smoke test
  python local_train.py --count-params         # Только счёт параметров
  python local_train.py --resume               # Продолжить с чекпоинта
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, time, math, json, shutil, argparse, subprocess, random, logging
from pathlib import Path
from datetime import datetime

# Fix encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Tars.Train")

# ═══════════════════════════════════════════
# 1. Пути
# ═══════════════════════════════════════════

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
MODELS = ROOT / "models"
SAVE_DIR = MODELS / "tars_lite"
LOG_FILE = ROOT / "train_lite.log"
STATE_FILE = ROOT / "train_state.json"

# ═══════════════════════════════════════════
# 2. CLI аргументы
# ═══════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="ТАРС HELIX LITE — Единый скрипт обучения",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--level", choices=["small", "medium", "max", "marathon"], default=None,
                    help="Уровень: small (~15 мин), medium (~3ч), max (~15ч), marathon (~96ч)")
parser.add_argument("--drive", choices=["none", "colab", "rclone"], default=None,
                    help="Диск: none / colab / rclone")
parser.add_argument("--resume", action="store_true", help="Продолжить с чекпоинта")
parser.add_argument("--test-only", action="store_true", help="Только smoke test")
parser.add_argument("--count-params", action="store_true", help="Показать число параметров")
parser.add_argument("--no-git-pull", action="store_true", help="Не обновлять из Git")
parser.add_argument("--seq-len", type=int, default=None, help="Override seq_len")
parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
parser.add_argument("--d-model", type=int, default=None, help="Override d_model")
parser.add_argument("--n-layers", type=int, default=None, help="Override n_layers")
parser.add_argument("--data", type=str, default=None, help="Путь к текстовому файлу для обучения")
parser.add_argument("--checkpoint-interval", type=int, default=1800, help="Интервал чекпоинтов (сек)")
args = parser.parse_args()


# ═══════════════════════════════════════════
# 3. Google Drive (Colab)
# ═══════════════════════════════════════════

def setup_drive(mode):
    """Подключить Google Drive."""
    if mode == "none":
        return False

    if mode == "colab":
        logger.info("☁️  Google Drive: подключение через Colab...")
        drive_path = Path("/content/drive/MyDrive")
        if drive_path.exists():
            logger.info("✅ Drive уже смонтирован")
            return True
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("✅ Google Drive смонтирован!")
            return True
        except ImportError:
            logger.warning("google.colab недоступен (не Colab?)")
            return False
        except Exception as e:
            if drive_path.exists():
                logger.warning(f"Ошибка mount ({e}), но Drive доступен")
                return True
            logger.error(f"Ошибка: {e}")
            return False

    if mode == "rclone":
        logger.info("☁️  Google Drive: rclone...")
        setup_script = ROOT / "tools" / "setup_gdrive.py"
        if setup_script.exists():
            try:
                r = subprocess.run([PYTHON, str(setup_script), "status"],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode == 0:
                    logger.info("✅ rclone настроен!")
                    return True
            except Exception:
                pass
        logger.warning("rclone не настроен")
        return False
    return False


def drive_backup(mode):
    """Бэкап чекпоинтов на Drive."""
    if mode == "colab":
        # If models/ is already symlinked to Drive, files are already there
        if (ROOT / "models").is_symlink():
            logger.info("☁️  models/ → Drive (symlink) — файлы уже на Drive")
            return True
        drive_models = Path("/content/drive/MyDrive/TarsData/models/tars_lite")
        drive_models.mkdir(parents=True, exist_ok=True)
        if SAVE_DIR.exists():
            for f in SAVE_DIR.glob("*.pt"):
                dst = drive_models / f.name
                # Skip if same file (resolved paths match)
                try:
                    if f.resolve() == dst.resolve():
                        continue
                except Exception:
                    pass
                shutil.copy2(str(f), str(dst))
                logger.info(f"☁️  → Drive: {f.name} ({f.stat().st_size/1e6:.1f} MB)")
            return True
    return False


# ═══════════════════════════════════════════
# 4. Git Auto-Update
# ═══════════════════════════════════════════

def git_pull():
    """Обновить код из Git (если доступен)."""
    git_dir = ROOT / ".git"
    if not git_dir.exists():
        logger.info("⏭  Нет .git — пропускаем обновление")
        return False
    
    logger.info("🔄 Git pull...")
    try:
        # Стеш незафиксированных изменений
        r = subprocess.run(
            ["git", "stash"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=30
        )
        # Pull
        r = subprocess.run(
            ["git", "pull", "--rebase", "origin", "main"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60
        )
        if r.returncode == 0:
            logger.info(f"✅ Git updated: {r.stdout.strip()[:100]}")
        else:
            # Попробуем без rebase
            r = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60
            )
            if r.returncode == 0:
                logger.info(f"✅ Git updated (merge): {r.stdout.strip()[:100]}")
            else:
                logger.warning(f"Git pull failed: {r.stderr.strip()[:100]}")
        # Pop stash
        subprocess.run(
            ["git", "stash", "pop"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=15
        )
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("Git pull timeout")
        return False
    except FileNotFoundError:
        logger.info("Git не найден — пропускаем")
        return False
    except Exception as e:
        logger.warning(f"Git error: {e}")
        return False


# ═══════════════════════════════════════════
# 5. Hardware Benchmark
# ═══════════════════════════════════════════

def detect_gpu():
    """Определяет GPU → (name, vram_gb, device, bf16)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, 0, "cpu", False
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        bf16 = torch.cuda.get_device_capability(0) >= (8, 0)
        return name, vram, "cuda", bf16
    except Exception:
        return None, 0, "cpu", False


def get_ram_gb():
    """Определить RAM в GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        try:
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            s = MS(); s.dwLength = ctypes.sizeof(s)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.ullTotalPhys / 1024**3
        except Exception:
            return 16


def benchmark_hardware():
    """Полный бенчмарк железа."""
    import torch
    gpu_name, vram_total, device, bf16 = detect_gpu()
    ram_gb = get_ram_gb()
    cpu_count = os.cpu_count() or 4

    hw = {
        "gpu_name": gpu_name,
        "vram_total_gb": round(vram_total, 2),
        "vram_usable_gb": 0,
        "device": device,
        "bf16": bf16,
        "fp16": device == "cuda",
        "ram_gb": round(ram_gb, 1),
        "cpu_count": cpu_count,
        "num_workers": min(cpu_count, 4) if device == "cuda" else 0,
        "tflops": 0,
        "compile_ok": False,
        "is_colab": os.path.exists("/content"),
        "pin_memory": device == "cuda",
    }

    if device != "cuda":
        logger.info("⚙️  CPU mode — пропускаем GPU-тесты")
        return hw

    props = torch.cuda.get_device_properties(0)

    # VRAM stress test
    logger.info("⚙️  Тест VRAM...")
    torch.cuda.empty_cache()
    lo, hi = 0.5, vram_total
    best_gb = 0.5
    for _ in range(12):
        mid = (lo + hi) / 2
        try:
            n_floats = int(mid * 1024**3 / 4)
            t = torch.empty(n_floats, dtype=torch.float32, device="cuda")
            del t
            torch.cuda.empty_cache()
            best_gb = mid
            lo = mid
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            hi = mid
            torch.cuda.empty_cache()

    hw["vram_usable_gb"] = round(best_gb * 0.90, 2)
    logger.info(f"  VRAM: {hw['vram_usable_gb']:.1f} GB usable (90% of {best_gb:.1f} GB)")

    # Matmul throughput
    try:
        sz = 2048
        dtype = torch.bfloat16 if bf16 else torch.float16
        a = torch.randn(sz, sz, dtype=dtype, device="cuda")
        b = torch.randn(sz, sz, dtype=dtype, device="cuda")
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        n_iters = 20
        start_ev.record()
        for _ in range(n_iters):
            torch.mm(a, b)
        end_ev.record()
        torch.cuda.synchronize()
        elapsed_ms = start_ev.elapsed_time(end_ev)
        flops = 2 * sz**3 * n_iters
        hw["tflops"] = round(flops / (elapsed_ms / 1000) / 1e12, 1)
        del a, b
        torch.cuda.empty_cache()
        logger.info(f"  Throughput: {hw['tflops']:.1f} TFLOPS ({dtype})")
    except Exception:
        pass

    # torch.compile
    try:
        if hasattr(torch, 'compile'):
            @torch.compile(mode="reduce-overhead", fullgraph=False)
            def _test_fn(x):
                return x * 2 + 1
            _test_fn(torch.tensor([1.0], device="cuda"))
            hw["compile_ok"] = True
            logger.info("  torch.compile: ✅")
    except Exception:
        pass

    if hw["is_colab"]:
        hw["num_workers"] = min(cpu_count, 2)
    else:
        hw["num_workers"] = min(cpu_count // 2, 4) if cpu_count > 1 else 0

    torch.cuda.empty_cache()
    return hw


def print_hw_report(hw):
    """Красивый вывод."""
    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │         ⚙️  HARDWARE BENCHMARK RESULTS               │")
    print("  ├──────────────────────────────────────────────────────┤")
    if hw["gpu_name"]:
        g = hw['gpu_name'][:40]
        print(f"  │  🎮 GPU:  {g:<42}│")
        print(f"  │  💾 VRAM: {hw['vram_total_gb']:.1f} GB → {hw['vram_usable_gb']:.1f} GB usable           │")
        print(f"  │  ⚡ Speed: {hw['tflops']:.1f} TFLOPS                             │")
        dtypes = []
        if hw['bf16']: dtypes.append('bf16')
        if hw['fp16']: dtypes.append('fp16')
        print(f"  │  📐 Types: {'+'.join(dtypes) if dtypes else 'fp32':<40}│")
    else:
        print(f"  │  🖥️  CPU mode (no GPU)                              │")
    print(f"  │  🧠 RAM:  {hw['ram_gb']:.0f} GB                                    │")
    colab_s = 'Да' if hw['is_colab'] else 'Нет'
    print(f"  │  📍 Colab: {colab_s:<40}│")
    print("  └──────────────────────────────────────────────────────┘")
    print()


# ═══════════════════════════════════════════
# 6. Автотюнинг конфигурации
# ═══════════════════════════════════════════

def auto_config(level, hw):
    """Конфиг обучения по уровню + hardware.
    
    GPU tiers (выжимает максимум из каждого):
      A100-80GB : d=1024, L=20, batch=16, seq=1024
      A100-40GB : d=1024, L=20, batch=8,  seq=1024
      L4-24GB   : d=1024, L=20, batch=6,  seq=768
      T4-15GB   : d=768,  L=16, batch=4,  seq=512
      GTX-6GB   : d=512,  L=10, batch=2,  seq=256
      CPU       : d=256,  L=6,  batch=1,  seq=128
    """
    vram = hw["vram_usable_gb"]
    bf16 = hw["bf16"]
    gpu_name = hw.get("gpu_name", "").lower()

    # ═══ GPU-specific tiers — max utilization ═══
    if vram >= 65:
        # A100-80GB / H100
        d, nl, batch, max_seq = 1024, 20, 16, 1024
        grad_ckpt = False
    elif vram >= 30:
        # A100-40GB
        d, nl, batch, max_seq = 1024, 20, 8, 1024
        grad_ckpt = False
    elif vram >= 18:
        # L4-24GB (Ada Lovelace, 24GB, bf16)
        d, nl, batch, max_seq = 1024, 20, 6, 768
        grad_ckpt = True  # save VRAM for bigger effective batch
    elif vram >= 11:
        # T4-15GB (Turing)
        d, nl, batch, max_seq = 768, 16, 4, 512
        grad_ckpt = True
    elif vram >= 5:
        # GTX 1060 6GB / RTX 3060 etc.
        d, nl, batch, max_seq = 512, 10, 2, 256
        grad_ckpt = True
    else:
        # CPU fallback
        d, nl, batch, max_seq = 256, 6, 1, 128
        grad_ckpt = False

    # AMP dtype
    amp = "bf16" if bf16 else ("fp16" if hw["fp16"] else "fp32")

    # Effective batch = batch * accum → target ~16-32 tokens per step
    target_eff_batch = 16 if vram < 30 else 32
    accum = max(1, target_eff_batch // batch)

    # Level-specific (seq_len capped by GPU max)
    configs = {
        "small": {
            "epochs": 2, "steps_per_epoch": 100,
            "seq_len": min(256, max_seq), "accum": accum,
            "lr": 5e-4, "warmup": 50, "max_chunks": 2000,
        },
        "medium": {
            "epochs": 3, "steps_per_epoch": 500,
            "seq_len": min(512, max_seq), "accum": accum,
            "lr": 3e-4, "warmup": 200, "max_chunks": 20000,
        },
        "max": {
            "epochs": 5, "steps_per_epoch": 2000,
            "seq_len": max_seq, "accum": accum,
            "lr": 3e-4, "warmup": 500, "max_chunks": 100000,
        },
        "marathon": {
            "epochs": 15, "steps_per_epoch": 5000,
            "seq_len": max_seq, "accum": accum,
            "lr": 3e-4, "warmup": 1000, "max_chunks": 500000,
        },
    }

    cfg = configs[level]
    cfg.update({
        "d_model": d, "n_layers": nl,
        "batch": batch, "amp": amp,
        "device": hw["device"],
        "num_workers": hw["num_workers"],
        "pin_memory": hw["pin_memory"],
        "compile": hw.get("compile_ok", False),
        "grad_checkpoint": grad_ckpt,
    })

    # CLI overrides
    if args.d_model:    cfg["d_model"] = args.d_model
    if args.n_layers:   cfg["n_layers"] = args.n_layers
    if args.batch_size: cfg["batch"] = args.batch_size
    if args.seq_len:    cfg["seq_len"] = args.seq_len
    if args.lr:         cfg["lr"] = args.lr

    return cfg


# ═══════════════════════════════════════════
# 7. Интерактивное меню
# ═══════════════════════════════════════════

def interactive_menu():
    """Интерактивный выбор."""
    drive = args.drive
    level = args.level

    if drive is None:
        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  💾 Подключить Google Drive?                    │")
        print("  │                                                 │")
        print("  │  [1] Нет (локально)                             │")
        print("  │  [2] Google Drive (Colab)                       │")
        print("  │  [3] Google Drive (rclone)                      │")
        print("  └─────────────────────────────────────────────────┘")
        try:
            ch = input("  Выберите [1/2/3] (default=1): ").strip()
        except (EOFError, KeyboardInterrupt):
            ch = "1"
        drive = {"2": "colab", "3": "rclone"}.get(ch, "none")

    if level is None:
        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  📊 Уровень обучения:                           │")
        print("  │                                                 │")
        print("  │  [1] Малый    — smoke test, ~15 мин             │")
        print("  │  [2] Средний  — стандарт, ~3 часа               │")
        print("  │  [3] Максимум — продакшн, ~15 часов             │")
        print("  │  [4] Марафон  — 4 дня, ~96 часов                │")
        print("  └─────────────────────────────────────────────────┘")
        try:
            ch = input("  Выберите [1/2/3/4] (default=2): ").strip()
        except (EOFError, KeyboardInterrupt):
            ch = "2"
        level = {"1": "small", "3": "max", "4": "marathon"}.get(ch, "medium")

    return drive, level


# ═══════════════════════════════════════════
# 8. State tracking
# ═══════════════════════════════════════════

def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_phases": [], "current_epoch": 0, "best_loss": float('inf'), "total_steps": 0}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# ═══════════════════════════════════════════
# 9. Datasets — Реальные данные
# ═══════════════════════════════════════════

# Vocab = 256 (UTF-8 byte-level, без BPE обучения)
BYTE_VOCAB = 256


def find_text_files():
    """Найти текстовые файлы для обучения."""
    search_dirs = [
        ROOT / "data",
        ROOT / "data" / "datasets",
    ]
    
    found = []
    for d in search_dirs:
        if d.exists():
            found.extend(d.glob("*.txt"))
            found.extend(d.glob("hf_*.txt"))
    
    # Дедупликация
    seen = set()
    unique = []
    for f in found:
        r = f.resolve()
        if r not in seen and f.stat().st_size > 100:
            seen.add(r)
            unique.append(f)
    
    return sorted(unique, key=lambda f: -f.stat().st_size)


def auto_download_data():
    """Авто-скачивание датасетов из HuggingFace (chat preset)."""
    logger.info("📥 Авто-скачивание данных из HuggingFace...")
    
    # Установить datasets если нет
    try:
        import datasets as _
    except ImportError:
        logger.info("  Установка: datasets")
        subprocess.run(
            [PYTHON, "-m", "pip", "install", "datasets", "--quiet"],
            capture_output=True
        )
    
    # Скачать через наш download_hf_dataset.py
    dl_script = ROOT / "training" / "data" / "download_hf_dataset.py"
    if not dl_script.exists():
        logger.warning("⚠️  download_hf_dataset.py не найден")
        return False
    
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    
    try:
        result = subprocess.run(
            [PYTHON, str(dl_script), "--preset", "chat",
             "--count", "5000", "--output", str(data_dir)],
            cwd=str(ROOT), capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            logger.info("✅ Данные скачаны")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-5:]:
                    logger.info(f"  {line.strip()}")
            return True
        else:
            logger.warning(f"⚠️  Скачивание завершилось с ошибкой: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Timeout скачивания (>10 мин)")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Ошибка скачивания: {e}")
        return False


def load_corpus_text(data_path=None):
    """Загрузить текстовый корпус для обучения.
    
    Приоритет:
      1. --data (конкретный файл)
      2. data/*.txt (все текстовые файлы)
      3. Авто-скачивание из HF
      4. Random tokens (fallback)
    
    Returns: (bytes_list, total_size) или None
    """
    # 1. Конкретный файл
    if data_path and os.path.exists(data_path):
        with open(data_path, 'rb') as f:
            raw = f.read()
        mb = len(raw) / 1024**2
        logger.info(f"📂 Файл: {data_path} ({mb:.1f} MB)")
        return raw
    
    # 2. Все .txt из data/
    txt_files = find_text_files()
    if txt_files:
        parts = []
        total = 0
        for f in txt_files:
            try:
                with open(f, 'rb') as fh:
                    data = fh.read()
                parts.append(data)
                total += len(data)
                logger.info(f"  📄 {f.name}: {len(data)/1024:.0f} KB")
            except Exception as e:
                logger.debug(f"  Skipped {f.name}: {e}")
        if parts:
            corpus = b'\n\n'.join(parts)
            logger.info(f"📂 Корпус: {len(txt_files)} файлов, {total/1024**2:.1f} MB")
            return corpus
    
    # 3. Авто-скачивание
    logger.info("📂 Текстовые данные не найдены — пробуем скачать...")
    if auto_download_data():
        txt_files = find_text_files()
        if txt_files:
            parts = []
            for f in txt_files:
                try:
                    with open(f, 'rb') as fh:
                        parts.append(fh.read())
                except Exception:
                    pass
            if parts:
                corpus = b'\n\n'.join(parts)
                logger.info(f"📂 Скачанный корпус: {len(parts)} файлов, {len(corpus)/1024**2:.1f} MB")
                return corpus
    
    # 4. Fallback — нет данных
    return None


def load_dataset(cfg, vocab_size):
    """Загрузить dataset (реальные тексты или random tokens)."""
    import torch
    from torch.utils.data import Dataset, DataLoader

    class TextCorpusDataset(Dataset):
        """UTF-8 byte-level chunked text dataset.
        
        Каждый символ → UTF-8 байт → токен [0..255].
        Chunks с overlap (stride = seq_len // 2).
        """
        def __init__(self, raw_bytes, seq_len, max_chunks=0):
            tokens = list(raw_bytes)
            stride = max(1, seq_len // 2)  # 50% overlap
            
            self.chunks = []
            for i in range(0, len(tokens) - seq_len - 1, stride):
                chunk = tokens[i:i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    self.chunks.append(chunk)
            
            # Перемешать чанки
            random.shuffle(self.chunks)
            
            # Ограничить размер по уровню
            if max_chunks > 0 and len(self.chunks) > max_chunks:
                self.chunks = self.chunks[:max_chunks]
            
            if not self.chunks:
                # Fallback: pad short corpus
                tokens = tokens + [0] * (seq_len + 1 - len(tokens))
                self.chunks = [tokens[:seq_len + 1]]
            
            logger.info(f"📂 TextCorpus: {len(raw_bytes)/1024**2:.1f} MB → "
                        f"{len(self.chunks)} chunks, seq_len={seq_len}")
        
        def __len__(self):
            return len(self.chunks)
        
        def __getitem__(self, idx):
            t = torch.tensor(self.chunks[idx], dtype=torch.long)
            return {'input_ids': t[:-1], 'labels': t[1:]}

    class RandomTokenDataset(Dataset):
        """Random tokens (fallback если нет данных)."""
        def __init__(self, vocab_size, seq_len, n_samples):
            self.data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            t = self.data[idx]
            return {'input_ids': t[:-1], 'labels': t[1:]}

    # Загрузить корпус
    corpus_bytes = load_corpus_text(data_path=args.data)
    max_chunks = cfg.get("max_chunks", 0)
    
    if corpus_bytes is not None:
        dataset = TextCorpusDataset(corpus_bytes, cfg["seq_len"], max_chunks=max_chunks)
    else:
        n_samples = cfg["steps_per_epoch"] * cfg["batch"]
        n_samples = min(n_samples, 5000)
        dataset = RandomTokenDataset(vocab_size, cfg["seq_len"], n_samples)
        logger.warning(f"⚠️  Нет текстовых данных — random tokens ({n_samples} samples)")
        logger.warning(f"   Положите .txt файлы в {ROOT / 'data'}/")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=cfg["pin_memory"],
        drop_last=True,
    )
    return dataloader


# ═══════════════════════════════════════════
# 10. Training Loop
# ═══════════════════════════════════════════

def get_wsd_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """WSD (Warmup-Stable-Decay) LR schedule per HELIX GOLDEN spec.
    
    Phase 1: 5% linear warmup (0 → lr)
    Phase 2: 75% stable (lr constant)
    Phase 3: 20% cosine decay (lr → min_lr)
    """
    import torch
    stable_end = int(total_steps * 0.80)  # warmup(5%) + stable(75%)
    def lr_lambda(step):
        if step < warmup_steps:
            # Phase 1: Linear warmup
            return step / max(1, warmup_steps)
        elif step < stable_end:
            # Phase 2: Stable (full LR)
            return 1.0
        else:
            # Phase 3: Cosine decay
            decay_steps = total_steps - stable_end
            progress = (step - stable_end) / max(1, decay_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model, dataloader, optimizer, scheduler, scaler, cfg, state, drive_mode,
          adamw_group=None, scheduler_adamw=None):
    """Full training loop with checkpointing. Supports Muon+AdamW dual optimizer."""
    import torch

    device = cfg["device"]
    start_epoch = state.get("current_epoch", 0)
    total_steps = state.get("total_steps", 0)
    best_loss = state.get("best_loss", float('inf'))
    last_ckpt_time = time.time()

    amp_enabled = cfg["amp"] in ("fp16", "bf16") and device == "cuda"
    amp_dtype = torch.bfloat16 if cfg["amp"] == "bf16" else torch.float16

    logger.info(f"\n{'='*60}")
    logger.info(f"🧬 HELIX LITE TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"  Model:   {cfg['d_model']}d × {cfg['n_layers']}L")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Params:  {n_params/1e6:.1f}M")
    logger.info(f"  Batch:   {cfg['batch']} × {cfg['accum']} = {cfg['batch'] * cfg['accum']} effective")
    logger.info(f"  SeqLen:  {cfg['seq_len']}")
    logger.info(f"  Epochs:  {cfg['epochs']} (starting from {start_epoch})")
    logger.info(f"  LR:      {cfg['lr']}")
    logger.info(f"  AMP:     {cfg['amp']}")
    logger.info(f"  Device:  {device}")
    logger.info(f"{'='*60}\n")

    model.train()

    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_loss = 0
        epoch_tokens = 0
        epoch_start = time.time()

        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype if amp_enabled else torch.float32):
                result = model(input_ids, labels=labels)
                loss = result['loss'] / cfg['accum']

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (every accum steps)
            if (step + 1) % cfg['accum'] == 0 or (step + 1) == len(dataloader):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                # Step AdamW group (dual optimizer)
                if adamw_group is not None:
                    adamw_group.step()
                # Scheduler AFTER optimizer (PyTorch requirement)
                if scheduler is not None:
                    scheduler.step()
                if scheduler_adamw is not None:
                    scheduler_adamw.step()
                optimizer.zero_grad(set_to_none=True)
                if adamw_group is not None:
                    adamw_group.zero_grad(set_to_none=True)
                total_steps += 1

            # Stats
            real_loss = loss.item() * cfg['accum']
            epoch_loss += real_loss * input_ids.numel()
            epoch_tokens += input_ids.numel()

            # Log every 10 steps
            if (step + 1) % 10 == 0 or step == 0:
                elapsed = time.time() - epoch_start
                avg_loss = epoch_loss / max(epoch_tokens, 1)
                tps = epoch_tokens / max(elapsed, 0.001)
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"  Epoch {epoch+1}/{cfg['epochs']} | "
                    f"Step {step+1}/{len(dataloader)} | "
                    f"Loss: {real_loss:.4f} (avg: {avg_loss:.4f}) | "
                    f"LR: {lr:.2e} | {tps:.0f} tok/s"
                )

            # Periodic checkpoint
            if time.time() - last_ckpt_time > args.checkpoint_interval:
                _save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_steps,
                                epoch_loss / max(epoch_tokens, 1), cfg, "periodic")
                last_ckpt_time = time.time()
                if drive_mode != "none":
                    drive_backup(drive_mode)

            # Max steps per epoch
            if (step + 1) >= cfg["steps_per_epoch"]:
                break

        # End of epoch
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        epoch_time = time.time() - epoch_start
        logger.info(f"\n  ─── Epoch {epoch+1} complete | Loss: {avg_loss:.4f} | Time: {epoch_time/60:.1f} min ───\n")

        # Save checkpoint
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        _save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, total_steps,
                        avg_loss, cfg, "best" if is_best else "epoch")

        # Update state
        state["current_epoch"] = epoch + 1
        state["total_steps"] = total_steps
        state["best_loss"] = best_loss
        save_state(state)

        # Drive backup after each epoch
        if drive_mode != "none":
            drive_backup(drive_mode)

    return best_loss


def _save_checkpoint(model, optimizer, scheduler, scaler, epoch, total_steps, loss, cfg, tag):
    """Save checkpoint."""
    import torch
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = {
        'epoch': epoch,
        'total_steps': total_steps,
        'loss': loss,
        'vocab_size': BYTE_VOCAB,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'd_model': cfg['d_model'],
            'n_layers': cfg['n_layers'],
            'seq_len': cfg['seq_len'],
            'lr': cfg['lr'],
        },
    }
    if scheduler is not None:
        ckpt['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        ckpt['scaler_state_dict'] = scaler.state_dict()

    if tag == "best":
        path = SAVE_DIR / "checkpoint_best.pt"
    elif tag == "periodic":
        path = SAVE_DIR / "checkpoint_latest.pt"
    else:
        path = SAVE_DIR / f"checkpoint_epoch{epoch}.pt"

    torch.save(ckpt, str(path))
    mb = path.stat().st_size / 1e6
    logger.info(f"  💾 Saved {path.name} ({mb:.0f} MB, loss={loss:.4f})")

    # Always save latest
    if tag != "periodic":
        latest = SAVE_DIR / "checkpoint_latest.pt"
        shutil.copy2(str(path), str(latest))


# ═══════════════════════════════════════════
# 11. Smoke Test
# ═══════════════════════════════════════════

def smoke_test(device="cpu"):
    """Quick sanity check."""
    import torch
    from config import TarsConfig
    from brain.mamba2.core.model_lite import TarsHelixLite

    logger.info("=" * 60)
    logger.info("🔬 SMOKE TEST")
    logger.info("=" * 60)

    cfg = TarsConfig(d_model=128, n_layers=2, vocab_size=256, d_state=16, headdim=32)
    model = TarsHelixLite(cfg).to(device)

    params = model.count_parameters()
    logger.info(f"  Params: {params['_total_M']:.2f}M")

    B, L = 2, 64
    x = torch.randint(0, 256, (B, L), device=device)
    labels = torch.randint(0, 256, (B, L), device=device)

    t0 = time.time()
    result = model(x, labels=labels)
    t_fwd = time.time() - t0
    logger.info(f"  Forward: {result['logits'].shape}, loss={result['loss'].item():.4f}, {t_fwd*1000:.0f}ms")

    t0 = time.time()
    result['loss'].backward()
    t_bwd = time.time() - t0
    logger.info(f"  Backward: {t_bwd*1000:.0f}ms")

    # Gradient check
    all_ok = True
    for name, p in model.named_parameters():
        if p.grad is None:
            logger.warning(f"  ❌ No gradient: {name}")
            all_ok = False
            break

    # Generate
    prompt = torch.randint(0, 256, (1, 8), device=device)
    with torch.no_grad():
        gen = model.generate(prompt, max_new_tokens=16)
    logger.info(f"  Generate: {prompt.shape} → {gen.shape}")

    logger.info("=" * 60)
    if all_ok:
        logger.info("✅ SMOKE TEST PASSED")
    else:
        logger.info("❌ SMOKE TEST FAILED")
    logger.info("=" * 60)
    return all_ok


def count_params():
    """Show parameter count for full config."""
    from config import TarsConfig
    from brain.mamba2.core.model_lite import TarsHelixLite

    cfg = TarsConfig()
    # Override from CLI
    if args.d_model: cfg.d_model = args.d_model
    if args.n_layers: cfg.n_layers = args.n_layers

    model = TarsHelixLite(cfg)
    params = model.count_parameters()

    print()
    print("=" * 60)
    print(f"  🧬 TARS HELIX LITE — {params['_total_M']:.1f}M параметров")
    print("=" * 60)
    print(f"  Config: d={cfg.d_model}, L={cfg.n_layers}, V={cfg.vocab_size}")
    print()
    for comp, count in sorted(params.items()):
        if not comp.startswith('_'):
            pct = count / params['_total'] * 100
            print(f"    {comp:20s}: {count/1e6:8.2f}M  ({pct:5.1f}%)")
    print()
    print(f"  FP16:    {params['_total'] * 2 / 1e6:.0f} MB")
    print(f"  Ternary: {params['_total'] * 0.2 / 1e6:.0f} MB")
    print("=" * 60)


# ═══════════════════════════════════════════
# 12. MAIN
# ═══════════════════════════════════════════

def main():
    import torch

    t0_global = time.time()

    # ── Quick modes ──
    if args.test_only:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        smoke_test(device)
        return

    if args.count_params:
        count_params()
        return

    # ═══ Banner ═══
    print()
    print("═" * 65)
    print("  🧬 ТАРС HELIX LITE — Единый скрипт обучения")
    print("═" * 65)
    print()

    # ═══ Hardware Benchmark ═══
    logger.info("⚙️  Тестирование железа...")
    hw = benchmark_hardware()
    print_hw_report(hw)
    device = hw["device"]

    # ═══ Interactive Menu ═══
    drive_mode, level = interactive_menu()

    # ═══ Google Drive ═══
    drive_ok = False
    if drive_mode != "none":
        drive_ok = setup_drive(drive_mode)

    # ═══ Git Update ═══
    if not args.no_git_pull:
        git_pull()

    # ═══ Install dependencies ═══
    logger.info("📦 Проверка зависимостей...")
    missing = []
    for pkg in ["torch", "einops"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.info(f"  Установка: {', '.join(missing)}")
        subprocess.run([PYTHON, "-m", "pip", "install"] + missing + ["--quiet"],
                      capture_output=True)
    logger.info("  ✅ Зависимости OK")

    # ═══ Auto Config ═══
    cfg = auto_config(level, hw)
    state = load_state()

    # ═══ Banner with config ═══
    print()
    print("─" * 65)
    print(f"  📐 Model:   d={cfg['d_model']} × {cfg['n_layers']}L")
    print(f"  📦 Batch:   {cfg['batch']} × {cfg['accum']} = {cfg['batch'] * cfg['accum']} effective")
    print(f"  📏 SeqLen:  {cfg['seq_len']}")
    print(f"  📊 Level:   {level.upper()}")
    print(f"  ⚡ AMP:     {cfg['amp']}")
    print(f"  🔄 Epochs:  {cfg['epochs']}")
    print(f"  💾 Drive:   {drive_mode}")
    if cfg.get('grad_checkpoint'):
        print(f"  🧩 GradCkpt: ON (saves ~30% VRAM)")
    if cfg.get('compile'):
        print(f"  🔥 Compile: ON (torch.compile)")
    if args.resume and state.get("current_epoch", 0) > 0:
        print(f"  🔄 Resume:  epoch {state['current_epoch']}, loss {state.get('best_loss', '?'):.4f}")
    print("─" * 65)
    print()

    # ═══ Smoke Test First ═══
    logger.info("🔬 Quick smoke test...")
    if not smoke_test(device):
        logger.error("❌ Smoke test failed — aborting!")
        return

    # ═══ Build Model ═══
    from config import TarsConfig
    from brain.mamba2.core.model_lite import TarsHelixLite

    model_cfg = TarsConfig(
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        vocab_size=BYTE_VOCAB,  # UTF-8 byte-level (256 tokens)
        quant_mode="fp16",      # training in FP16/BF16, not ternary
    )
    model = TarsHelixLite(model_cfg).to(device)

    # Gradient checkpointing для L4/T4 (saves ~30% VRAM)
    if cfg.get('grad_checkpoint', False):
        from torch.utils.checkpoint import checkpoint as torch_checkpoint
        # Enable gradient checkpointing on model blocks
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("🧩 Gradient checkpointing: ON (native)")
        else:
            # Manual: set flag for forward to use checkpoint
            model._use_gradient_checkpointing = True
            logger.info("🧩 Gradient checkpointing: ON (manual flag)")

    # torch.compile для совместимых GPU
    if cfg.get('compile', False) and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("🔥 torch.compile: ON")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")

    # Resume from checkpoint
    ckpt = None
    ckpt_loaded = False
    if args.resume:
        ckpt_path = SAVE_DIR / "checkpoint_latest.pt"
        if not ckpt_path.exists():
            ckpt_path = SAVE_DIR / "checkpoint_best.pt"
        if ckpt_path.exists():
            logger.info(f"🔄 Loading checkpoint: {ckpt_path}")
            try:
                ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

                # ── Check 1: embedding shape compatibility ──
                sd = ckpt.get('model_state_dict', {})
                emb_w = sd.get('embedding.weight', None)
                if emb_w is not None and emb_w.shape[0] != BYTE_VOCAB:
                    logger.warning(f"⚠️  Checkpoint embedding [{emb_w.shape[0]}] ≠ vocab [{BYTE_VOCAB}]")
                    logger.warning(f"  Удаляю несовместимый чекпоинт...")
                    ckpt_path.unlink(missing_ok=True)
                    ckpt = None
                
                # ── Check 2: NaN in weights ──
                if ckpt is not None:
                    has_nan = False
                    for key, tensor in sd.items():
                        if torch.is_floating_point(tensor) and torch.isnan(tensor).any():
                            has_nan = True
                            logger.warning(f"⚠️  NaN в весах: {key}")
                            break
                    if has_nan:
                        logger.warning(f"  Удаляю NaN-чекпоинт — обучение с нуля")
                        ckpt_path.unlink(missing_ok=True)
                        ckpt = None

                # ── Load if OK ──
                if ckpt is not None:
                    # Skip useless checkpoints (no training progress)
                    ckpt_epoch = ckpt.get('epoch', 0)
                    ckpt_steps = ckpt.get('total_steps', 0)
                    if ckpt_epoch == 0 and ckpt_steps == 0:
                        logger.warning(f"⚠️  Checkpoint epoch=0, step=0 — бесполезный, пропускаем")
                        ckpt_path.unlink(missing_ok=True)
                        ckpt = None
                    else:
                        model.load_state_dict(sd, strict=False)
                        state["current_epoch"] = ckpt_epoch
                        state["total_steps"] = ckpt_steps
                        ckpt_loaded = True
                        logger.info(f"  ✅ Resumed from epoch {ckpt_epoch}, step {ckpt_steps}")
            except Exception as e:
                logger.warning(f"⚠️  Checkpoint corrupted: {e}")
                logger.warning(f"  Удаляю — обучение с нуля...")
                try:
                    ckpt_path.unlink()
                except Exception:
                    pass
                ckpt = None

    # ═══ Dataset ═══
    dataloader = load_dataset(cfg, BYTE_VOCAB)

    # ═══ Sanity check: forward pass before training ═══
    logger.info("🔬 Sanity check: forward pass...")
    model.eval()
    with torch.no_grad():
        test_x = torch.randint(0, BYTE_VOCAB, (1, 64), device=device)
        test_out = model(test_x)
        test_logits = test_out['logits'] if isinstance(test_out, dict) else test_out
        if torch.isnan(test_logits).any() or torch.isinf(test_logits).any():
            logger.warning("⚠️  Модель выдаёт NaN/Inf — переинициализация с нуля!")
            # Re-create model from scratch
            model = TarsHelixLite(model_cfg).to(device)
            ckpt_loaded = False
            ckpt = None
            # Verify fix
            test_out2 = model(test_x)
            test_logits2 = test_out2['logits'] if isinstance(test_out2, dict) else test_out2
            if torch.isnan(test_logits2).any():
                logger.error("❌ Модель NaN даже после переинициализации!")
                return
            logger.info("  ✅ Переинициализация успешна")
        else:
            logger.info(f"  ✅ Forward OK: logits range [{test_logits.min():.2f}, {test_logits.max():.2f}]")
    model.train()

    # ═══ Optimizer: Muon (2D matrices) + AdamW (1D params) ═══
    # HELIX GOLDEN: Muon for weight matrices, AdamW for biases/norms/embeddings
    try:
        from training.muon import Muon
        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and 'embedding' not in name:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        # Muon uses ~10× higher LR than AdamW (orthogonalized gradients)
        muon_lr = cfg["lr"] * 10  # e.g. 3e-4 → 3e-3
        optimizer = Muon(
            [{'params': muon_params, 'lr': muon_lr}],
            lr=muon_lr,
            weight_decay=0.01,
        )
        # Add AdamW group for 1D params
        adamw_group = torch.optim.AdamW(
            adamw_params,
            lr=cfg["lr"],
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        logger.info(f"  ⚡ Muon: {len(muon_params)} matrix params (lr={muon_lr:.1e})")
        logger.info(f"  ⚡ AdamW: {len(adamw_params)} other params (lr={cfg['lr']:.1e})")
        use_dual_optimizer = True
    except ImportError:
        logger.warning("  ⚠️ Muon not available — falling back to AdamW only")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["lr"],
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
        adamw_group = None
        use_dual_optimizer = False

    # ═══ Scheduler: WSD (Warmup-Stable-Decay) per GOLDEN spec ═══
    actual_steps_per_epoch = min(len(dataloader), cfg["steps_per_epoch"])
    # Scheduler steps only when optimizer steps (every accum batches)
    optimizer_steps_per_epoch = max(actual_steps_per_epoch // cfg["accum"], 1)
    total_steps = optimizer_steps_per_epoch * cfg["epochs"]
    warmup_steps = min(cfg["warmup"], int(total_steps * 0.05))  # cap at 5%
    logger.info(f"📊 WSD Schedule: {warmup_steps} warmup → {int(total_steps*0.75)} stable → {int(total_steps*0.20)} decay")
    import warnings
    warnings.filterwarnings("ignore", "Detected call of `lr_scheduler.step\\(\\)` before")
    scheduler = get_wsd_schedule(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1,
    )
    # Mirror scheduler for AdamW group
    scheduler_adamw = None
    if use_dual_optimizer:
        scheduler_adamw = get_wsd_schedule(
            adamw_group,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=0.1,
        )

    # ═══ AMP Scaler ═══
    scaler = None
    if cfg["amp"] == "fp16" and device == "cuda":
        scaler = torch.amp.GradScaler('cuda')

    # ═══ TRAIN! ═══
    try:
        best_loss = train(model, dataloader, optimizer, scheduler, scaler, cfg, state, drive_mode,
                          adamw_group=adamw_group if use_dual_optimizer else None,
                          scheduler_adamw=scheduler_adamw)
    except KeyboardInterrupt:
        logger.info("\n⏸ Прервано. Сохраняю чекпоинт...")
        _save_checkpoint(model, optimizer, scheduler, scaler,
                        state.get("current_epoch", 0), state.get("total_steps", 0),
                        state.get("best_loss", float('inf')), cfg, "periodic")
        if drive_mode != "none":
            drive_backup(drive_mode)
        best_loss = state.get("best_loss", float('inf'))

    # ═══ Results ═══
    elapsed = time.time() - t0_global
    print()
    print("═" * 65)
    print(f"  🧬 HELIX LITE — Training Complete!")
    print("═" * 65)
    print(f"  ⏱  Time:     {elapsed/3600:.1f} hours ({elapsed/60:.0f} min)")
    print(f"  📉 Best loss: {best_loss:.4f}")
    print(f"  📐 Model:    {cfg['d_model']}d × {cfg['n_layers']}L")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  📊 Params:   {n_params/1e6:.1f}M")
    if SAVE_DIR.exists():
        for f in sorted(SAVE_DIR.glob("*.pt")):
            mb = f.stat().st_size / 1e6
            print(f"  💾 {f.name}: {mb:.0f} MB")
    print()

    # Final Drive backup
    if drive_mode != "none":
        logger.info("☁️  Финальный бэкап на Drive...")
        drive_backup(drive_mode)

    print("═" * 65)
    print("  ✅ Готово! Чекпоинты в models/tars_lite/")
    print("═" * 65)


if __name__ == "__main__":
    main()
