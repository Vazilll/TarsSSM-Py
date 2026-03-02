"""
═══════════════════════════════════════════════════════════════════════
  ТАРС v3 — СТАЦИОНАРНОЕ ОБУЧЕНИЕ (Лаборатория, MAX)
═══════════════════════════════════════════════════════════════════════

Максимальное обучение на стационарном GPU (RTX 4090 / 3090 / A6000).
Авто-определение GPU → оптимальный конфиг по VRAM.

  ≥22 GB (4090):  768M params (1024d × 20L), batch=8×4=32, bf16, seq→4096
  ≥14 GB (3090):  400M params (768d × 16L),  batch=4×8=32
  <14 GB (3060):  250M params (768d × 12L),  batch=2×16=32

ПОЛНЫЙ ПАЙПЛАЙН (фазы 0-15):
  0. Dependencies
  1. Download (Wiki 100K + HF all presets + personality corpus)
  2. Reflex classifier (100 epochs)
  3. MinGRU LM (25 epochs)
  4. Mamba-2 Brain (4 sub-phases + personality + second pass)
  5. Quantize 1.58-bit
  6. Consolidate → models/tars_v3/
  7. Validate
  8-10. Voice (Whisper + Piper + Quantize)
  11. Instruction tuning
  12. CoT (Chain-of-Thought)
  13. DPO (preference alignment)
  14. RLVR (verifiable rewards)
  15. Knowledge Distillation (optional, --distill)

ДАННЫЕ С ДИСКА:
  По умолчанию данные берутся из data/ (hf_*.txt, wiki_ru.txt, etc).
  Для больших корпусов → создайте data/shards_*/shard_*.txt
  
  Структура диска:
    data/
    ├── wiki_ru.txt              # Wikipedia 100K статей (~300 MB)
    ├── hf_*.txt                 # HuggingFace датасеты
    ├── tars_personality*.txt    # Личность ТАРС
    ├── tars_identity.txt        # Архитектура ТАРС
    ├── instruct_data.jsonl      # Instruction tuning
    ├── cot_data.jsonl           # CoT задачи
    ├── dpo_pairs.jsonl          # DPO пары
    └── shards_custom/           # Большие корпуса (sharded)
        ├── shard_000.txt
        ├── shard_001.txt
        └── ...

ИСПОЛЬЗОВАНИЕ:
  python local_train.py                    # Авто-конфиг по GPU (полный пайплайн)
  python local_train.py --1b              # Форсировать 768M модель
  python local_train.py --resume          # Продолжить с чекпоинта
  python local_train.py --phase 4         # Только Mamba-2
  python local_train.py --phase 12        # Только CoT
  python local_train.py --download-only   # Только скачать данные
  python local_train.py --distill         # Включить Knowledge Distillation
  python local_train.py --skip-posttrain  # Без post-training (CoT/DPO/RLVR)
  python local_train.py --data-dir D:\datasets  # Данные с другого диска

═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import argparse
import subprocess
import json
import shutil
from pathlib import Path
from datetime import datetime

# Fix encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ═══════════════════════════════════════════
# 1. Paths & Constants
# ═══════════════════════════════════════════

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
TRAINING = ROOT / "training"
DATA = ROOT / "data"
MODELS = ROOT / "models"
TARS_V3 = MODELS / "tars_v3"
LOG_FILE = ROOT / "local_train.log"
STATE_FILE = ROOT / "train_state.json"

# ═══════════════════════════════════════════
# 2. Arguments
# ═══════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="ТАРС v3 — Local Training (RTX 4090 MAX)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Примеры:
  python local_train.py                    # Полный пайплайн (все 15 фаз)
  python local_train.py --1b              # 768M модель
  python local_train.py --resume          # Продолжить с чекпоинта
  python local_train.py --phase 4         # Только Mamba-2
  python local_train.py --phase 12        # Только CoT
  python local_train.py --download-only   # Только скачать все данные
  python local_train.py --distill         # + Knowledge Distillation
  python local_train.py --data-dir D:\\datasets  # Данные с другого диска
  python local_train.py --gdrive          # Синхронизировать данные с Google Drive
  python local_train.py --gdrive --gdrive-backup  # + бэкап моделей после обучения
    """
)
parser.add_argument("--1b", dest="one_billion", action="store_true",
                    help="Форсировать 768M модель (1024d × 20L)")
parser.add_argument("--resume", action="store_true",
                    help="Продолжить с чекпоинта")
parser.add_argument("--phase", type=int, default=None,
                    help="Запустить только конкретную фазу (0-15)")
parser.add_argument("--download-only", action="store_true",
                    help="Только скачать данные")
parser.add_argument("--skip-download", action="store_true",
                    help="Пропустить скачивание (данные есть)")
parser.add_argument("--skip-voice", action="store_true",
                    help="Пропустить голосовые модули")
parser.add_argument("--skip-posttrain", action="store_true",
                    help="Пропустить post-training (CoT, DPO, RLVR)")
parser.add_argument("--distill", action="store_true",
                    help="Включить Knowledge Distillation (фаза 15)")
parser.add_argument("--data-dir", type=str, default=None,
                    help="Директория с данными (default: data/)")
parser.add_argument("--data-preset", default="all",
                    choices=["all", "max", "quality", "massive", "reasoning"],
                    help="Какие данные скачивать (default: all)")
parser.add_argument("--gdrive", action="store_true",
                    help="Синхронизировать данные с Google Drive перед обучением")
parser.add_argument("--gdrive-backup", action="store_true",
                    help="Выгрузить модели на Google Drive после обучения")
parser.add_argument("--checkpoint-interval", type=int, default=1800,
                    help="Интервал сохранения чекпоинтов (сек, default: 1800)")
args = parser.parse_args()

# ═══════════════════════════════════════════
# 3. GPU Detection + Auto-Config
# ═══════════════════════════════════════════

def detect_gpu():
    """Определяет GPU и возвращает (name, vram_gb, device, bf16)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, 0, "cpu", False
        
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Check bf16 support (Ampere+)
        bf16_ok = torch.cuda.get_device_capability(0) >= (8, 0)
        
        return gpu_name, vram_gb, "cuda", bf16_ok
    except Exception:
        return None, 0, "cpu", False


def get_ram_gb():
    """Get system RAM in GB."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        # Fallback for Windows
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / 1024**3
        except Exception:
            return 16  # fallback


def get_config(vram_gb, ram_gb, force_1b=False):
    """Возвращает оптимальный конфиг по VRAM + RAM.
    
    RTX 4090 + 64GB RAM: максимальная модель, большие батчи, длинные контексты.
    """
    if force_1b or vram_gb >= 22:
        # ═══ RTX 4090 / A6000 / A100 — МАКСИМУМ ═══
        # 24GB VRAM: 1024d × 20L = ~768M params
        # batch=8 × accum=4 = 32 effective (больше batch → лучше GPU utilization)
        # seq_len до 4096 (grad_ckpt позволяет)
        batch = 8 if vram_gb >= 22 else 4
        return {
            "name": "768M",
            "d_model": 1024,
            "n_layers": 20,
            "batch": batch,
            "accum": 4,          # effective batch = 32
            "seq_len_start": 512,
            "seq_len_mid": 1024,
            "seq_len_max": 4096,
            # Learning rates по фазам
            "lr_p1": 3e-4,       # Full pretrain
            "lr_p2": 1e-4,       # WKV fine-tune
            "lr_p3": 3e-5,       # MoLE fine-tune
            "lr_p4": 1.5e-5,     # RAG fine-tune
            "lr_p5": 5e-5,       # Personality
            # Эпохи по фазам
            "epochs_p1": 10,
            "epochs_p2": 5,
            "epochs_p3": 3,
            "epochs_p4": 3,
            "epochs_p5": 5,      # personality
            # Post-training epochs
            "epochs_instruct": 3,
            "epochs_cot": 5,
            "epochs_dpo": 3,
            "epochs_rlvr": 3,
            "epochs_distill": 3,
            # Extra flags for 4090
            "use_muon": True,         # 2x faster optimizer
            "use_wsd": True,          # Better scheduler
            "use_mod": False,         # MoD пока экспериментальный
            "wiki_count": 100_000,    # Больше данных с диска
            "second_pass_epochs": 5,  # Больше эпох на длинном контексте
        }
    elif vram_gb >= 14:
        return {
            "name": "400M",
            "d_model": 768,
            "n_layers": 16,
            "batch": 4,
            "accum": 8,
            "seq_len_start": 384,
            "seq_len_mid": 512,
            "seq_len_max": 1024,
            "lr_p1": 3e-4, "lr_p2": 1e-4, "lr_p3": 3e-5,
            "lr_p4": 1.5e-5, "lr_p5": 5e-5,
            "epochs_p1": 10, "epochs_p2": 5, "epochs_p3": 3,
            "epochs_p4": 3, "epochs_p5": 3,
            "epochs_instruct": 3, "epochs_cot": 3,
            "epochs_dpo": 2, "epochs_rlvr": 2, "epochs_distill": 2,
            "use_muon": False, "use_wsd": False, "use_mod": False,
            "wiki_count": 50_000, "second_pass_epochs": 3,
        }
    else:
        return {
            "name": "250M",
            "d_model": 768,
            "n_layers": 12,
            "batch": 2,
            "accum": 16,
            "seq_len_start": 256,
            "seq_len_mid": 384,
            "seq_len_max": 512,
            "lr_p1": 3e-4, "lr_p2": 1e-4, "lr_p3": 3e-5,
            "lr_p4": 1.5e-5, "lr_p5": 5e-5,
            "epochs_p1": 10, "epochs_p2": 5, "epochs_p3": 3,
            "epochs_p4": 3, "epochs_p5": 3,
            "epochs_instruct": 2, "epochs_cot": 2,
            "epochs_dpo": 1, "epochs_rlvr": 1, "epochs_distill": 1,
            "use_muon": False, "use_wsd": False, "use_mod": False,
            "wiki_count": 10_000, "second_pass_epochs": 2,
        }


# ═══════════════════════════════════════════
# 4. State Management (resume support)
# ═══════════════════════════════════════════

def load_state():
    """Загрузить состояние обучения."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"completed_phases": [], "current_phase": None, "started": None}


def save_state(state):
    """Сохранить состояние обучения."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def mark_done(state, phase_key):
    """Отметить фазу как завершённую."""
    state.setdefault("completed_phases", [])
    if phase_key not in state["completed_phases"]:
        state["completed_phases"].append(phase_key)
    save_state(state)


def is_done(state, phase_key):
    """Проверить, завершена ли фаза."""
    return phase_key in state.get("completed_phases", [])


# ═══════════════════════════════════════════
# 5. Training Runner
# ═══════════════════════════════════════════

def run(cmd, timeout=None, label=""):
    """Запустить команду с логированием."""
    cmd = [str(c) for c in cmd]
    
    # Вставляем -u после python для unbuffered output
    if len(cmd) >= 2 and ('python' in cmd[0].lower() or cmd[0] == PYTHON):
        if '-u' not in cmd:
            cmd.insert(1, '-u')
    
    cmd_str = " ".join(cmd)
    if label:
        print(f"  → [{label}] {cmd_str[:120]}...")
    else:
        print(f"  → {cmd_str[:120]}...")
    sys.stdout.flush()
    
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"[{datetime.now()}] {cmd_str}\n")
        log.write(f"{'='*60}\n")
    
    # PYTHONUNBUFFERED=1 → вывод подпроцесса виден в реальном времени
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            timeout=timeout,
            env=env,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ Таймаут ({timeout}s)")
        return False
    except KeyboardInterrupt:
        print(f"\n  ⏸ Прервано пользователем. Чекпоинт сохранён.")
        return False
    except Exception as e:
        print(f"  ❌ {e}")
        return False


def train_mamba_phase(phase_num, config, device, bf16, extra_args=None):
    """Запустить одну фазу обучения Mamba-2."""
    
    phase_keys = {
        1: ("epochs_p1", "lr_p1", "seq_len_start"),
        2: ("epochs_p2", "lr_p2", "seq_len_mid"),
        3: ("epochs_p3", "lr_p3", "seq_len_max"),
        4: ("epochs_p4", "lr_p4", "seq_len_max"),
        5: ("epochs_p5", "lr_p5", "seq_len_mid"),
    }
    
    epoch_key, lr_key, seq_key = phase_keys[phase_num]
    
    cmd = [
        PYTHON, str(TRAINING / "train_mamba2.py"),
        "--d_model", str(config["d_model"]),
        "--n_layers", str(config["n_layers"]),
        "--vocab_size", "256",
        "--batch", str(config["batch"]),
        "--accum_steps", str(config["accum"]),
        "--epochs", str(config[epoch_key]),
        "--lr", str(config[lr_key]),
        "--seq_len", str(config[seq_key]),
        "--phase", str(phase_num),
        "--device", device,
        "--curriculum",
        "--label_smoothing", "0.1",
        "--grad_ckpt",
    ]
    
    if bf16:
        cmd += ["--bf16"]
    
    # RTX 4090 optimizations
    if config.get("use_muon"):
        cmd += ["--muon"]
    if config.get("use_wsd"):
        cmd += ["--wsd"]
    if config.get("use_mod"):
        cmd += ["--mod"]
    
    if phase_num > 1 or args.resume:
        cmd += ["--resume"]
    
    # Data directory support
    if args.data_dir:
        cmd += ["--data_dir", args.data_dir]
    
    if extra_args:
        cmd += extra_args
    
    return run(cmd, label=f"Mamba P{phase_num}")


def train_post_phase(script, config, device, bf16, epochs_key, extra_args=None):
    """Запустить post-training фазу (CoT/DPO/RLVR/Distill)."""
    cmd = [
        PYTHON, str(TRAINING / script),
        "--d_model", str(config["d_model"]),
        "--n_layers", str(config["n_layers"]),
        "--epochs", str(config.get(epochs_key, 3)),
        "--batch", str(config["batch"]),
        "--device", device,
        "--save_dir", str(TARS_V3),
        "--resume",
        "--grad_ckpt",
    ]
    
    if bf16:
        cmd += ["--bf16"]
    
    if extra_args:
        cmd += extra_args
    
    return run(cmd, label=script.replace("train_", "").replace(".py", "").upper())


# ═══════════════════════════════════════════
# 6. Data Download (from disk/network)
# ═══════════════════════════════════════════

def download_all_data(config):
    """Скачать ВСЕ данные: Wiki + HF + Personality + Synthetic."""
    
    data_dir = Path(args.data_dir) if args.data_dir else DATA
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  📂 Data directory: {data_dir}")
    
    # ── 1. Wikipedia (100K статей) ──
    wiki_path = data_dir / "wiki_ru.txt"
    if not wiki_path.exists() or wiki_path.stat().st_size < 100_000:
        print(f"\n  📚 Wikipedia: скачивание {config['wiki_count']:,} статей...")
        run([
            PYTHON, str(TRAINING / "download_wiki.py"),
            "--count", str(config["wiki_count"]),
            "--output", str(wiki_path),
        ], label="Wiki")
    else:
        wiki_mb = wiki_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Wikipedia: {wiki_mb:.1f} MB (кеш)")
    
    # ── 2. HuggingFace datasets (ALL presets) ──
    print(f"\n  📦 HuggingFace datasets (preset: {args.data_preset})...")
    run([
        PYTHON, str(TRAINING / "download_hf_dataset.py"),
        "--preset", args.data_preset,
        "--output", str(data_dir),
    ], label="HF")
    
    # ── 3. Personality corpus ──
    personality = data_dir / "tars_personality_mega.txt"
    if not personality.exists() or personality.stat().st_size < 100_000:
        print("\n  🧠 Generating personality corpus...")
        run([PYTHON, str(TRAINING / "generate_tars_corpus.py")], label="Personality")
    else:
        print(f"  ✓ Personality: {personality.stat().st_size // 1024} KB (кеш)")
    
    # ── 4. Synthetic STEM data ──
    stem_path = data_dir / "synthetic_stem.jsonl"
    if not stem_path.exists():
        print("\n  🔬 Generating synthetic STEM data...")
        run([PYTHON, str(TRAINING / "generate_synthetic.py")], label="STEM")
    else:
        print(f"  ✓ STEM data: {stem_path.stat().st_size // 1024} KB (кеш)")
    
    # ── 5. Show data summary ──
    total_mb = 0
    file_count = 0
    print(f"\n  {'─' * 50}")
    print(f"  📁 Файлы данных в {data_dir}:")
    for f in sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.jsonl")):
        mb = f.stat().st_size / (1024 * 1024)
        if mb > 0.01:
            total_mb += mb
            file_count += 1
            print(f"    {f.name:40s} {mb:8.1f} MB")
    
    # Sharded data
    for shard_dir in sorted(data_dir.glob("shards_*")):
        if shard_dir.is_dir():
            shard_files = list(shard_dir.glob("shard_*.txt"))
            shard_mb = sum(f.stat().st_size for f in shard_files) / (1024 * 1024)
            total_mb += shard_mb
            file_count += len(shard_files)
            print(f"    {shard_dir.name + '/':40s} {shard_mb:8.1f} MB ({len(shard_files)} shards)")
    
    print(f"  {'─' * 50}")
    print(f"  Итого: {file_count} файлов, {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)")
    
    # If data_dir is not default, symlink to data/ for train scripts
    if args.data_dir and Path(args.data_dir).resolve() != DATA.resolve():
        print(f"\n  🔗 Данные в {args.data_dir} — скрипты будут использовать --data_dir")
    
    return total_mb


# ═══════════════════════════════════════════
# 7. Google Drive Integration (AUTO)
# ═══════════════════════════════════════════

# Глобальный флаг — подключён ли GDrive в этом сеансе
gdrive_connected = False


def gdrive_check_and_prompt():
    """Проверить подключение к GDrive.
    
    Если не подключён — спросить пользователя, хочет ли подключить.
    Вызывается АВТОМАТИЧЕСКИ при старте обучения.
    """
    global gdrive_connected
    
    setup_script = ROOT / "setup_gdrive.py"
    if not setup_script.exists():
        print("  ⚠ setup_gdrive.py не найден — GDrive пропущен")
        return False
    
    # Проверить, есть ли rclone и настроен ли remote
    import subprocess as sp
    try:
        r = sp.run(
            [PYTHON, str(setup_script), "status"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print("  ☁️  Google Drive: подключён ✅")
            gdrive_connected = True
            return True
    except Exception:
        pass
    
    # GDrive не подключён — спросить пользователя
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │  ☁️  Google Drive НЕ подключён                      │")
    print("  │                                                     │")
    print("  │  Подключение позволит:                              │")
    print("  │  • Скачать данные для обучения с Drive              │")
    print("  │  • Автоматически выгружать модели после обучения    │")
    print("  │  • Видеть прогресс обучения на Drive в реальном     │")
    print("  │    времени                                          │")
    print("  └─────────────────────────────────────────────────────┘")
    print()
    
    try:
        answer = input("  Подключить Google Drive? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    
    if answer in ("", "y", "yes", "д", "да"):
        print()
        print("  🔧 Запускаю настройку Google Drive...")
        print("     (Следуйте инструкциям в терминале)")
        print()
        ok = run([PYTHON, str(setup_script), "setup"], label="GDrive Setup")
        if ok:
            gdrive_connected = True
            print("  ✅ Google Drive подключён!")
            return True
        else:
            print("  ⚠ Не удалось подключить Drive — продолжаю без него")
            return False
    else:
        print("  ⏭ Пропускаю Google Drive (можно подключить позже)")
        print("     python setup_gdrive.py setup")
        return False


def gdrive_sync():
    """Синхронизировать данные с Google Drive → data/."""
    if not gdrive_connected:
        return False
    
    print("\n  ☁️  Google Drive: синхронизация данных...")
    
    setup_script = ROOT / "setup_gdrive.py"
    ok = run([PYTHON, str(setup_script), "sync"], label="GDrive Sync")
    if ok:
        print("  ✅ Данные с Drive синхронизированы → data/")
    return ok


def gdrive_backup():
    """Выгрузить модели на Google Drive (rclone copy, неразрушающий)."""
    if not gdrive_connected:
        return False
    
    print("\n  ☁️  Google Drive: выгрузка моделей...")
    
    setup_script = ROOT / "setup_gdrive.py"
    if not setup_script.exists():
        return False
    
    ok = run([PYTHON, str(setup_script), "sync-models"], label="GDrive Backup")
    if ok:
        print("  ✅ Модели выгружены на Google Drive")
        print("     Проверьте: Google Drive → TarsLocalModels/")
    return ok


def gdrive_sync_memory():
    """Синхронизировать LEANN память + embeddings с Google Drive."""
    if not gdrive_connected:
        return False
    
    setup_script = ROOT / "setup_gdrive.py"
    if not setup_script.exists():
        return False
    
    ok = run([PYTHON, str(setup_script), "sync-memory"], label="GDrive Memory")
    if ok:
        print("  ✅ Память синхронизирована с Google Drive")
        print("     Проверьте: Google Drive → TarsMemory/")
    return ok


def gdrive_backup_if_enabled():
    """Промежуточный бэкап — вызывается АВТОМАТИЧЕСКИ после ключевых фаз."""
    if gdrive_connected:
        gdrive_backup()
        gdrive_sync_memory()


# ═══════════════════════════════════════════
# 8. Main Pipeline
# ═══════════════════════════════════════════

def main():
    # Detect hardware
    gpu_name, vram_gb, device, bf16 = detect_gpu()
    ram_gb = get_ram_gb()
    
    config = get_config(vram_gb, ram_gb, force_1b=args.one_billion)
    state = load_state()
    
    # ── Google Drive: авто-проверка и подключение ──
    # Запрашивает подключение ПЕРЕД началом обучения
    if not args.phase:  # Не спрашиваем если запуск конкретной фазы
        gdrive_check_and_prompt()
    elif args.gdrive:
        gdrive_check_and_prompt()
    
    # Banner
    print()
    print("═" * 65)
    print("  🤖 ТАРС v3 — LOCAL TRAINING (MAX POWER)")
    print("═" * 65)
    print()
    print(f"  🎮 GPU:    {gpu_name or 'CPU'}")
    print(f"  💾 VRAM:   {vram_gb:.1f} GB")
    print(f"  🧠 RAM:    {ram_gb:.0f} GB")
    print(f"  📐 Model:  {config['name']} ({config['d_model']}d × {config['n_layers']}L)")
    print(f"  📦 Batch:  {config['batch']} × {config['accum']} = {config['batch'] * config['accum']} effective")
    print(f"  📏 SeqLen: {config['seq_len_start']} → {config['seq_len_mid']} → {config['seq_len_max']}")
    print(f"  ⚡ bf16:   {'Yes' if bf16 else 'No'}")
    if config.get("use_muon"):
        print(f"  🚀 Muon:   Yes (2x faster optimizer)")
    if config.get("use_wsd"):
        print(f"  📈 WSD:    Yes (Warmup-Stable-Decay scheduler)")
    print(f"  📁 Data:   {args.data_dir or str(DATA)}")
    if gdrive_connected:
        print(f"  ☁️  GDrive: ON (авто-синхронизация + авто-бэкап)")
    else:
        print(f"  ☁️  GDrive: OFF")
    if args.resume:
        print(f"  🔄 Resume: from checkpoint")
        if state.get("completed_phases"):
            print(f"     Done:   {state['completed_phases']}")
    print()
    
    # ── Estimate total params ──
    est_params = config["d_model"] ** 2 * config["n_layers"] * 12
    print(f"  📊 ~{est_params / 1e6:.0f}M parameters")
    est_size_mb = est_params * 2 / (1024 * 1024)  # fp16
    print(f"  💿 ~{est_size_mb:.0f} MB model size (fp16)")
    est_vram = est_size_mb * 4 / 1024  # ~4x for training (model + grads + optimizer + activations)
    print(f"  ⚙️  ~{est_vram:.1f} GB VRAM needed (training)")
    print()
    print("─" * 65)
    
    t0 = time.time()
    results = {}
    
    # ── Google Drive: синхронизация данных (ДО обучения) ──
    if gdrive_connected:
        ok = gdrive_sync()
        results["gdrive_sync"] = ok
        if not ok:
            print("  ⚠ Не удалось синхронизировать — продолжаю с локальными данными")
    
    def should_run(phase_num):
        """Проверить, нужно ли запускать фазу."""
        return args.phase is None or args.phase == phase_num
    
    # ══════════════════════════════════════════════════
    # Phase 0: Install dependencies
    # ══════════════════════════════════════════════════
    if should_run(0):
        print("\n  📦 Phase 0: Dependencies...")
        results["deps"] = run([PYTHON, "mega_train.py", "--phase", "0"], label="deps")
    
    # ══════════════════════════════════════════════════
    # Phase 1: Download ALL data
    # ══════════════════════════════════════════════════
    if not args.skip_download and (should_run(1) or args.download_only):
        print(f"\n  📚 Phase 1: Download ALL data (preset: {args.data_preset})...")
        data_mb = download_all_data(config)
        results["download"] = data_mb > 1
        
        if args.download_only:
            elapsed = time.time() - t0
            print(f"\n  ✅ Данные скачаны за {elapsed/60:.0f} минут ({data_mb:.0f} MB)")
            return
    
    # ══════════════════════════════════════════════════
    # Phase 2: Reflex classifier
    # ══════════════════════════════════════════════════
    if should_run(2) and not is_done(state, "reflex"):
        print("\n  🔁 Phase 2: Reflex classifier (100 epochs)...")
        ok = run([PYTHON, "mega_train.py", "--phase", "2"], label="Reflex")
        results["reflex"] = ok
        if ok:
            mark_done(state, "reflex")
    
    # ══════════════════════════════════════════════════
    # Phase 3: MinGRU LM
    # ══════════════════════════════════════════════════
    if should_run(3) and not is_done(state, "mingru"):
        print("\n  🧪 Phase 3: MinGRU LM (25 epochs)...")
        cmd = [
            PYTHON, str(TRAINING / "train_mingru.py"),
            "--dim", "512", "--layers", "6",
            "--epochs", "25", "--augment",
        ]
        ok = run(cmd, label="MinGRU")
        results["mingru"] = ok
        if ok:
            mark_done(state, "mingru")
    
    # ══════════════════════════════════════════════════
    # Phase 4: Mamba-2 Brain — THE MAIN EVENT
    # ══════════════════════════════════════════════════
    for mamba_phase in [1, 2, 3, 4]:
        phase_key = f"mamba_p{mamba_phase}"
        if not should_run(4):
            continue
        if is_done(state, phase_key):
            print(f"\n  ⏭ Phase 4.{mamba_phase}: already done")
            continue
        
        phase_names = {
            1: "Full Pretrain (SSD + WKV + Ω-SSM + MoLE)",
            2: "WKV + Fusion Fine-tune (SSD frozen)",
            3: "MoLE + MatrixPool + WaveMerge",
            4: "RAG + Memory + NoveltyGate",
        }
        
        print(f"\n  🧠 Phase 4.{mamba_phase}: {phase_names[mamba_phase]}...")
        print(f"     {config['d_model']}d × {config['n_layers']}L, "
              f"batch={config['batch']}×{config['accum']}")
        
        # Transfer embedding from MinGRU for Phase 1
        extra = []
        if mamba_phase == 1:
            emb_path = ROOT / "models" / "tars_v3" / "_transfer_embedding.pt"
            mingru_path = ROOT / "models" / "mingru_weights.pt"
            if mingru_path.exists() and not emb_path.exists():
                print("     🔗 Transferring MinGRU embedding...")
                try:
                    import torch
                    ckpt = torch.load(str(mingru_path), map_location='cpu', weights_only=False)
                    if 'model_state_dict' in ckpt:
                        for k, v in ckpt['model_state_dict'].items():
                            if 'token_emb' in k or 'embedding' in k:
                                emb_path.parent.mkdir(parents=True, exist_ok=True)
                                torch.save(v, str(emb_path))
                                print(f"     ✅ Embedding saved: {v.shape}")
                                break
                except Exception as e:
                    print(f"     ⚠ Embedding transfer: {e}")
            
            if emb_path.exists():
                extra += ["--pretrained_emb", str(emb_path)]
        
        ok = train_mamba_phase(mamba_phase, config, device, bf16, extra)
        results[phase_key] = ok
        
        if ok:
            mark_done(state, phase_key)
            print(f"  ✅ Phase 4.{mamba_phase} done")
        else:
            print(f"  ⚠️ Phase 4.{mamba_phase} failed → run with --resume to retry")
            break
    
    # ── Промежуточный бэкап после Mamba-2 ──
    gdrive_backup_if_enabled()
    
    # ── Phase 4.5: PersonalityAdapter ──
    if should_run(4) and not is_done(state, "personality"):
        print(f"\n  🎭 Phase 4.5: PersonalityAdapter ({config['epochs_p5']} epochs)...")
        ok = train_mamba_phase(5, config, device, bf16)
        results["personality"] = ok
        if ok:
            mark_done(state, "personality")
    
    # ── Phase 4.6: Second Pass (longer context) ──
    if should_run(4) and not is_done(state, "second_pass"):
        epochs_2nd = config.get("second_pass_epochs", 5)
        print(f"\n  🔄 Phase 4.6: Second Pass (seq_len={config['seq_len_max']}, "
              f"{epochs_2nd} epochs)...")
        cmd = [
            PYTHON, str(TRAINING / "train_mamba2.py"),
            "--d_model", str(config["d_model"]),
            "--n_layers", str(config["n_layers"]),
            "--vocab_size", "256",
            "--batch", str(config["batch"]),
            "--accum_steps", str(config["accum"]),
            "--epochs", str(epochs_2nd),
            "--lr", "5e-5",
            "--seq_len", str(config["seq_len_max"]),
            "--phase", "1",
            "--device", device,
            "--curriculum",
            "--label_smoothing", "0.05",
            "--grad_ckpt",
            "--resume",
        ]
        if bf16:
            cmd += ["--bf16"]
        if config.get("use_muon"):
            cmd += ["--muon"]
        if config.get("use_wsd"):
            cmd += ["--wsd"]
        if args.data_dir:
            cmd += ["--data_dir", args.data_dir]
        
        ok = run(cmd, label="2nd Pass")
        results["second_pass"] = ok
        if ok:
            mark_done(state, "second_pass")
    
    # ══════════════════════════════════════════════════
    # Phase 5: Quantize 1.58-bit
    # ══════════════════════════════════════════════════
    if should_run(5) and not is_done(state, "quantize"):
        print("\n  ⚗️ Phase 5: Quantize 1.58-bit...")
        ok = run([PYTHON, "mega_train.py", "--phase", "5"], label="Quantize")
        results["quantize"] = ok
        if ok:
            mark_done(state, "quantize")
            gdrive_backup_if_enabled()
    
    # ══════════════════════════════════════════════════
    # Phase 6: Consolidate → models/tars_v3/
    # ══════════════════════════════════════════════════
    if should_run(6) and not is_done(state, "consolidate"):
        print("\n  📦 Phase 6: Consolidate...")
        ok = run([PYTHON, "mega_train.py", "--phase", "6"], label="Consolidate")
        results["consolidate"] = ok
        if ok:
            mark_done(state, "consolidate")
    
    # ══════════════════════════════════════════════════
    # Phase 7: Validate
    # ══════════════════════════════════════════════════
    if should_run(7):
        print("\n  ✅ Phase 7: Validate...")
        ok = run([PYTHON, "mega_train.py", "--phase", "7"], label="Validate")
        results["validate"] = ok
    
    # ══════════════════════════════════════════════════
    # Phase 8-10: Voice (Whisper + Piper + Quantize)
    # ══════════════════════════════════════════════════
    if not args.skip_voice and (should_run(8) or should_run(9) or should_run(10)):
        print("\n  🎙 Phase 8-10: Voice (Whisper + Piper + INT8)...")
        
        if should_run(8) and not is_done(state, "whisper"):
            ok = run([PYTHON, "mega_train.py", "--phase", "8"], label="Whisper")
            if ok:
                mark_done(state, "whisper")
            results["whisper"] = ok
        
        if should_run(9) and not is_done(state, "piper"):
            ok = run([PYTHON, "mega_train.py", "--phase", "9"], label="Piper")
            if ok:
                mark_done(state, "piper")
            results["piper"] = ok
        
        if should_run(10) and not is_done(state, "voice_quant"):
            ok = run([PYTHON, "mega_train.py", "--phase", "10"], label="VoiceQ")
            if ok:
                mark_done(state, "voice_quant")
            results["voice_quant"] = ok
        
        # ── Промежуточный бэкап после Voice ──
        gdrive_backup_if_enabled()
    
    # ══════════════════════════════════════════════════
    # Phase 11: Instruction Tuning
    # ══════════════════════════════════════════════════
    if (should_run(11) or args.phase is None) and not is_done(state, "instruct"):
        print(f"\n  📖 Phase 11: Instruction Tuning ({config.get('epochs_instruct', 3)} epochs)...")
        ok = train_post_phase(
            "train_instruct.py", config, device, bf16,
            "epochs_instruct",
            extra_args=["--lr", "5e-5"],
        )
        results["instruct"] = ok
        if ok:
            mark_done(state, "instruct")
    
    # ══════════════════════════════════════════════════
    # Phase 12-14: Post-Training (CoT → DPO → RLVR)
    # ══════════════════════════════════════════════════
    if not args.skip_posttrain:
        
        # Phase 12: Chain-of-Thought
        if (should_run(12) or args.phase is None) and not is_done(state, "cot"):
            print(f"\n  🧩 Phase 12: CoT ({config.get('epochs_cot', 5)} epochs)...")
            ok = train_post_phase(
                "train_cot.py", config, device, bf16,
                "epochs_cot",
                extra_args=["--lr", "3e-5"],
            )
            results["cot"] = ok
            if ok:
                mark_done(state, "cot")
        
        # Phase 13: DPO (preference alignment)
        if (should_run(13) or args.phase is None) and not is_done(state, "dpo"):
            print(f"\n  ⚖️ Phase 13: DPO ({config.get('epochs_dpo', 3)} epochs)...")
            ok = train_post_phase(
                "train_dpo.py", config, device, bf16,
                "epochs_dpo",
                extra_args=["--lr", "1e-5"],
            )
            results["dpo"] = ok
            if ok:
                mark_done(state, "dpo")
        
        # Phase 14: RLVR (verifiable rewards)
        if (should_run(14) or args.phase is None) and not is_done(state, "rlvr"):
            print(f"\n  🎯 Phase 14: RLVR ({config.get('epochs_rlvr', 3)} epochs)...")
            ok = train_post_phase(
                "train_rlvr.py", config, device, bf16,
                "epochs_rlvr",
                extra_args=["--lr", "3e-5"],
            )
            results["rlvr"] = ok
            if ok:
                mark_done(state, "rlvr")
    else:
        print("\n  ⏭ Post-training пропущен (--skip-posttrain)")
    
    # ══════════════════════════════════════════════════
    # Phase 15: Knowledge Distillation (optional)
    # ══════════════════════════════════════════════════
    if args.distill and (should_run(15) or args.phase is None) and not is_done(state, "distill"):
        print(f"\n  🎓 Phase 15: Knowledge Distillation ({config.get('epochs_distill', 3)} epochs)...")
        ok = train_post_phase(
            "train_distill.py", config, device, bf16,
            "epochs_distill",
            extra_args=["--lr", "1e-4", "--temperature", "3.0", "--alpha", "0.7"],
        )
        results["distill"] = ok
        if ok:
            mark_done(state, "distill")
    
    # ══════════════════════════════════════════════════
    # Results
    # ══════════════════════════════════════════════════
    
    elapsed = time.time() - t0
    hours = elapsed / 3600
    
    print()
    print("═" * 65)
    print(f"  🤖 ТАРС v3 — РЕЗУЛЬТАТЫ ({hours:.1f} часов)")
    print("═" * 65)
    print()
    
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"    {icon} {name}")
    
    print()
    
    all_ok = all(results.values()) if results else False
    if all_ok:
        print(f"  🎯 ВСЕ ФАЗЫ ЗАВЕРШЕНЫ!")
        print(f"  📐 Модель: {config['name']} ({config['d_model']}d × {config['n_layers']}L)")
        print()
        
        if TARS_V3.exists():
            total_mb = 0
            for f in sorted(TARS_V3.glob("*.pt")):
                mb = f.stat().st_size / 1024 / 1024
                total_mb += mb
                print(f"    {f.name}: {mb:.1f} MB")
            print(f"    {'─' * 40}")
            print(f"    Итого: {total_mb:.0f} MB")
        
        print()
        print("  🚀 Запуск: python launch_tars.py")
    else:
        failed = [k for k, v in results.items() if not v]
        if failed:
            print(f"  ⚠️ Ошибки: {', '.join(failed)}")
        print(f"  🔄 Продолжить: python local_train.py --resume")
    
    # ── Google Drive: финальный бэкап (АВТО) ──
    if gdrive_connected:
        print()
        print("  ☁️  Финальная выгрузка моделей на Google Drive...")
        gdrive_ok = gdrive_backup()
        results["gdrive_backup"] = gdrive_ok
        if gdrive_ok:
            print("  ✅ Все модели на Drive — проверьте Google Drive → tars_training/models/")
    
    print()
    print("═" * 65)


if __name__ == "__main__":
    main()
