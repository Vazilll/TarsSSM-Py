"""
═══════════════════════════════════════════════════════════════════════
  ТАРС v3 — УНИВЕРСАЛЬНОЕ ОБУЧЕНИЕ (Единый Скрипт)
═══════════════════════════════════════════════════════════════════════

Один скрипт для ВСЕГО обучения. Интерактивный выбор:
  • Подключать диск? (нет / Colab / rclone)
  • Уровень: малый / средний / максимум

ФАЗЫ (0-14):
  0. Зависимости           8.  Whisper STT
  1. Скачивание данных      9.  Piper TTS
  2. Рефлекс-классификатор  11. Instruction tuning
  2.5 RRN маршрутизация     12. Chain-of-Thought
  3. MinGRU LM              13. DPO alignment
  3.5 SNN спайковые         14. RLVR
  4. Mamba-2 Brain          6.  Консолидация
  4.5. Personality          7.  Валидация

  Вынесены в local_train_extras.py:
    4.6 Second Pass, 5 Квант 1.58-bit, 10 Voice Quant, 15 Distillation

ИСПОЛЬЗОВАНИЕ:
  python local_train.py                        # Интерактивный выбор
  python local_train.py --level small          # Быстрая отладка (~15 мин)
  python local_train.py --level medium         # Стандарт (~3 часа)
  python local_train.py --level max            # Продакшн (~15 часов)
  python local_train.py --level marathon        # 4 дня непрерывно (~96 часов)
  python local_train.py --drive colab          # + Google Drive через Colab
  python local_train.py --drive rclone         # + Google Drive через rclone
  python local_train.py --phase 4              # Только Mamba-2
  python local_train.py --resume               # Продолжить с чекпоинта
  python local_train.py --download-only        # Только скачать данные
  python local_train.py --distill              # + Knowledge Distillation
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, time, json, shutil, argparse, subprocess, random
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
# 1. Пути
# ═══════════════════════════════════════════

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

PYTHON = sys.executable
TRAINING = ROOT / "training"
TRAIN_DIR = TRAINING / "train"
DATA_DIR = TRAINING / "data"
EVAL_DIR = TRAINING / "eval"
DATA = ROOT / "data"
MODELS = ROOT / "models"
TARS_V3 = MODELS / "tars_v3"
LOG_FILE = ROOT / "local_train.log"
STATE_FILE = ROOT / "train_state.json"

# ═══════════════════════════════════════════
# 2. Аргументы CLI
# ═══════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="ТАРС v3 — Универсальное обучение",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--level", choices=["small", "medium", "max", "marathon"], default=None,
                    help="Уровень обучения: small (~15 мин), medium (~3ч), max (~15ч), marathon (~96ч)")
parser.add_argument("--drive", choices=["none", "colab", "rclone"], default=None,
                    help="Подключение диска: none / colab / rclone")
parser.add_argument("--resume", action="store_true", help="Продолжить с чекпоинта")
parser.add_argument("--phase", type=int, default=None, help="Запустить только фазу (0-15)")
parser.add_argument("--download-only", action="store_true", help="Только скачать данные")
parser.add_argument("--skip-download", action="store_true", help="Пропустить скачивание")
parser.add_argument("--skip-voice", action="store_true", help="Пропустить голосовые модули")
parser.add_argument("--skip-posttrain", action="store_true", help="Без post-training (CoT/DPO/RLVR)")
parser.add_argument("--extras", action="store_true", help="Запустить доп. фазы из local_train_extras.py")
parser.add_argument("--data-dir", type=str, default=None, help="Директория с данными")
parser.add_argument("--data-preset", default="all",
                    choices=["all", "max", "quality", "massive", "reasoning"],
                    help="HF preset (default: all)")
parser.add_argument("--checkpoint-interval", type=int, default=1800, help="Интервал чекпоинтов (сек)")
args = parser.parse_args()

# ═══════════════════════════════════════════
# 3. Hardware Benchmark & Auto-Tuning
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
    """
    Полный бенчмарк железа (~5 сек).
    Тестирует VRAM (бинарный поиск 90%), matmul TFLOPS, torch.compile,
    оптимальный batch size для каждого d_model.
    """
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
        "cuda_capability": (0, 0),
        "is_colab": os.path.exists("/content"),
        "pin_memory": device == "cuda",
        "max_batch": {},
    }

    if device != "cuda":
        print("  ⚙️  CPU mode — пропускаем GPU-тесты")
        return hw

    props = torch.cuda.get_device_properties(0)
    hw["cuda_capability"] = (props.major, props.minor)

    # ────── Тест 1: VRAM стресс-тест (бинарный поиск 90%) ──────
    print("  ⚙️  Тест VRAM...  ", end="", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    lo, hi = 0.5, vram_total
    best_gb = 0.5
    for _ in range(12):  # 12 итераций бинарного поиска
        mid = (lo + hi) / 2
        try:
            n_floats = int(mid * 1024**3 / 4)  # float32 = 4 bytes
            t = torch.empty(n_floats, dtype=torch.float32, device="cuda")
            del t
            torch.cuda.empty_cache()
            best_gb = mid
            lo = mid
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            hi = mid
            torch.cuda.empty_cache()

    hw["vram_usable_gb"] = round(best_gb * 0.90, 2)  # 90% от доступного
    print(f"{hw['vram_usable_gb']:.1f} GB доступно (90% от {best_gb:.1f} GB)")

    # ────── Тест 2: Throughput (matmul TFLOPS) ──────
    print("  ⚙️  Тест matmul... ", end="", flush=True)
    try:
        sz = 2048
        dtype = torch.bfloat16 if bf16 else torch.float16
        a = torch.randn(sz, sz, dtype=dtype, device="cuda")
        b = torch.randn(sz, sz, dtype=dtype, device="cuda")
        # Прогрев
        for _ in range(3):
            torch.mm(a, b)
        torch.cuda.synchronize()
        # Замер
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        n_iters = 20
        start_ev.record()
        for _ in range(n_iters):
            torch.mm(a, b)
        end_ev.record()
        torch.cuda.synchronize()
        elapsed_ms = start_ev.elapsed_time(end_ev)
        flops = 2 * sz**3 * n_iters  # matmul: 2*N^3 FLOP
        tflops = flops / (elapsed_ms / 1000) / 1e12
        hw["tflops"] = round(tflops, 1)
        del a, b
        torch.cuda.empty_cache()
        print(f"{tflops:.1f} TFLOPS ({dtype})")
    except Exception as e:
        print(f"skip ({e})")

    # ────── Тест 3: torch.compile ──────
    print("  ⚙️  Тест torch.compile... ", end="", flush=True)
    try:
        if hasattr(torch, 'compile'):
            @torch.compile(mode="reduce-overhead", fullgraph=False)
            def _test_fn(x):
                return x * 2 + 1
            _test_fn(torch.tensor([1.0], device="cuda"))
            hw["compile_ok"] = True
            print("✅ поддерживается")
        else:
            print("❌ PyTorch < 2.0")
    except Exception as e:
        print(f"❌ ({e})")

    # ────── Тест 4: Оптимальный batch (расчёт по VRAM) ──────
    print("  ⚙️  Расчёт batch size... ", end="", flush=True)
    torch.cuda.empty_cache()
    max_batch_for_d = {}
    for d_model in [256, 512, 768, 1024]:
        seq = 512
        lo_b, hi_b = 1, 32  # макс 32 для стабильности
        best_b = 1
        # Модель: weights + optimizer(Adam) + gradients
        # nl_est: число слоёв по шкале d_model
        nl_est = {256: 4, 512: 8, 768: 16, 1024: 20}.get(d_model, 12)
        n_params = d_model * d_model * nl_est * 12
        dtype_size = 2 if (bf16 or hw["fp16"]) else 4
        # Model mem: params × dtype + optimizer(2×params×fp32) + grads(×dtype)
        model_mem_gb = (n_params * dtype_size + n_params * 4 * 2 + n_params * dtype_size) / 1024**3
        while lo_b <= hi_b:
            mid_b = (lo_b + hi_b) // 2
            # Activations: batch × seq × d × layers × dtype × 2 (fwd+bwd)
            act_mem_gb = (mid_b * seq * d_model * nl_est * dtype_size * 2) / 1024**3
            total_need = model_mem_gb + act_mem_gb
            if total_need <= hw["vram_usable_gb"]:
                best_b = mid_b
                lo_b = mid_b + 1
            else:
                hi_b = mid_b - 1
        max_batch_for_d[d_model] = max(1, best_b)
    hw["max_batch"] = max_batch_for_d
    print(f"d=768→batch={max_batch_for_d.get(768, '?')}, d=1024→batch={max_batch_for_d.get(1024, '?')}")

    # ────── Тест 5: Optimal num_workers ──────
    if hw["is_colab"]:
        hw["num_workers"] = min(cpu_count, 2)
    else:
        hw["num_workers"] = min(cpu_count // 2, 4) if cpu_count > 1 else 0

    torch.cuda.empty_cache()
    return hw

def print_hw_report(hw):
    """Красивый вывод результатов бенчмарка."""
    print()
    print("  ┌──────────────────────────────────────────────────────┐")
    print("  │            ⚙️  HARDWARE BENCHMARK RESULTS            │")
    print("  ├──────────────────────────────────────────────────────┤")
    if hw["gpu_name"]:
        g = hw['gpu_name'][:40]
        print(f"  │  🎮 GPU:     {g:<40}│")
        print(f"  │  💾 VRAM:    {hw['vram_total_gb']:.1f} GB total → {hw['vram_usable_gb']:.1f} GB usable (90%)    │")
        print(f"  │  ⚡ TFLOPS:  {hw['tflops']:.1f}{'':>40}│")
        cap = hw['cuda_capability']
        dtypes = []
        if hw['bf16']: dtypes.append('bf16')
        if hw['fp16']: dtypes.append('fp16')
        print(f"  │  📐 Compute: sm_{cap[0]}{cap[1]}, {'+'.join(dtypes):<33}│")
        cs = '✅ yes' if hw['compile_ok'] else '❌ no'
        print(f"  │  🔧 Compile: {cs:<40}│")
        for dm in [512, 768, 1024]:
            if dm in hw.get('max_batch', {}):
                b = hw['max_batch'][dm]
                print(f"  │  📦 d={dm}:   max batch={b:<31}│")
    else:
        print(f"  │  🖥️  CPU mode (no GPU){'':>32}│")
    print(f"  │  🧠 RAM:     {hw['ram_gb']:.0f} GB{'':>39}│")
    print(f"  │  🔢 CPU:     {hw['cpu_count']} cores → {hw['num_workers']} workers{'':>23}│")
    colab_s = 'Да' if hw['is_colab'] else 'Нет'
    print(f"  │  📍 Colab:   {colab_s:<40}│")
    print("  └──────────────────────────────────────────────────────┘")
    print()

# ═══════════════════════════════════════════
# 4. Интерактивное меню
# ═══════════════════════════════════════════

def interactive_menu():
    """Интерактивный выбор диска и уровня (если не задано через CLI)."""
    drive = args.drive
    level = args.level

    if drive is None:
        print()
        print("  ┌─────────────────────────────────────────────────┐")
        print("  │  💾 Подключить диск для данных/бэкапа?          │")
        print("  │                                                 │")
        print("  │  [1] Нет (локально)                             │")
        print("  │  [2] Google Drive через Colab                   │")
        print("  │  [3] Google Drive через rclone (локально)       │")
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
        print("  │  [1] Малый    — отладка, ~15 мин                │")
        print("  │  [2] Средний  — стандарт, ~3 часа               │")
        print("  │  [3] Максимум — продакшн, ~15 часов             │")
        print("  │  [4] Марафон  — 4 дня непрерывно, ~96 часов     │")
        print("  └─────────────────────────────────────────────────┘")
        try:
            ch = input("  Выберите [1/2/3/4] (default=2): ").strip()
        except (EOFError, KeyboardInterrupt):
            ch = "2"
        level = {"1": "small", "3": "max", "4": "marathon"}.get(ch, "medium")

    return drive, level

# ═══════════════════════════════════════════
# 5. Конфигурация по уровню + бенчмарку
# ═══════════════════════════════════════════

def get_config(level, hw):
    """Возвращает конфиг обучения по уровню и результатам бенчмарка."""
    vram = hw["vram_usable_gb"]  # уже 90% от реального
    tflops = hw["tflops"]
    ram_gb = hw["ram_gb"]
    max_batch = hw.get("max_batch", {})

    # ── Размер модели по доступному VRAM ──
    if vram >= 18:    # A100(40GB)→36, L4(24GB)→~20, RTX4090(24GB)→~20
        d, nl = 1024, 20
    elif vram >= 11:  # T4(15GB)→~13, RTX3060(12GB)→~10
        d, nl = 768, 16
    elif vram >= 5:
        d, nl = 512, 8
    elif vram >= 2:
        d, nl = 256, 4
    else:
        d, nl = 256, 4

    # ── Batch из бенчмарка (по шкале d_model) ──
    batch = max_batch.get(d, 4)

    # ── Accum для effective batch ≥ 32 ──
    eff_target = 32
    accum = max(1, eff_target // batch)

    # ── Seq_len ограничение по RAM (Colab: ~13 GB) ──
    max_seq_by_ram = min(4096, int(ram_gb * 256))  # ~256 tokens / GB RAM

    # ── Muon: быстрый, но жрёт больше VRAM ──
    use_muon = vram >= 18 and tflops >= 20
    # ── WSD: работает на любом GPU ──
    use_wsd = vram >= 10
    # ── torch.compile ──
    use_compile = hw.get("compile_ok", False)
    # ── AMP dtype ──
    amp_dtype = "bf16" if hw["bf16"] else ("fp16" if hw["fp16"] else "fp32")
    # ── num_workers ──
    num_workers = hw["num_workers"]
    pin_memory = hw["pin_memory"]

    # ── Общие поля для всех уровней ──
    common = {
        "d_model": d, "n_layers": nl,
        "vocab_size": 4096,  # BPE default (TarsTokenizer)
        "batch": batch, "accum": accum,
        "use_muon": use_muon, "use_wsd": use_wsd,
        "use_compile": use_compile,
        "amp_dtype": amp_dtype,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "vram_usable": vram,
    }

    if level == "small":
        # ═══ МАЛЫЙ: отладка, smoke-test ═══
        return {**common,
            "name": f"SMALL ({d}d×{nl}L, batch={batch}×{accum})",
            "seq_start": 256, "seq_mid": 384, "seq_max": min(512, max_seq_by_ram),
            # Эпохи (мало)
            "reflex_ep": 10, "rrn_ep": 10, "mingru_ep": 5,
            "snn_ep": 5, "snn_dim": 256, "snn_heads": 4, "snn_batch": min(batch * 4, 32), "snn_seq": 128,
            "mingru_dim": min(d, 256), "mingru_layers": 4,
            "mamba_ep": [2, 1, 1, 1], "mamba_lr": [3e-4, 1e-4, 5e-5, 1.5e-5],
            "personality_ep": 1, "second_pass_ep": 1,
            "instruct_ep": 1, "cot_ep": 1, "dpo_ep": 1, "rlvr_ep": 1, "distill_ep": 1,
            "wiki_count": 5000,
            "snn_max_mb": 10,
        }
    elif level == "medium":
        # ═══ СРЕДНИЙ: стандарт ═══
        return {**common,
            "name": f"MEDIUM ({d}d×{nl}L, batch={batch}×{accum})",
            "seq_start": 384, "seq_mid": 512, "seq_max": min(1024, max_seq_by_ram),
            "reflex_ep": 30, "rrn_ep": 25, "mingru_ep": 15,
            "snn_ep": 15, "snn_dim": min(d, 384), "snn_heads": 4,
            "snn_batch": min(batch * 6, 48), "snn_seq": 192,
            "mingru_dim": min(d, 512), "mingru_layers": 6,
            "mamba_ep": [5, 3, 2, 2], "mamba_lr": [3e-4, 1e-4, 3e-5, 1.5e-5],
            "personality_ep": 3, "second_pass_ep": 3,
            "instruct_ep": 3, "cot_ep": 3, "dpo_ep": 2, "rlvr_ep": 2, "distill_ep": 2,
            "wiki_count": 50000,
            "snn_max_mb": 20,
        }
    elif level == "max":
        # ═══ МАКСИМУМ: продакшн ═══
        return {**common,
            "name": f"MAX ({d}d×{nl}L, batch={batch}×{accum})",
            "seq_start": 512, "seq_mid": 1024, "seq_max": min(4096, max_seq_by_ram),
            "reflex_ep": 100, "rrn_ep": 50, "mingru_ep": 25,
            "snn_ep": 30, "snn_dim": min(d, 512), "snn_heads": 8,
            "snn_batch": min(batch * 8, 64), "snn_seq": 256,
            "mingru_dim": min(d, 512), "mingru_layers": 6,
            "mamba_ep": [10, 5, 3, 3], "mamba_lr": [3e-4, 1e-4, 3e-5, 1.5e-5],
            "personality_ep": 5, "second_pass_ep": 5,
            "instruct_ep": 3, "cot_ep": 5, "dpo_ep": 3, "rlvr_ep": 3, "distill_ep": 3,
            "wiki_count": 100000,
            "snn_max_mb": 50,
        }
    else:  # marathon
        # ═══ МАРАФОН: 4 дня непрерывного обучения ═══
        return {**common,
            "name": f"MARATHON ({d}d×{nl}L, batch={batch}×{accum}, ~96h)",
            "seq_start": 512, "seq_mid": 1024, "seq_max": min(2048, max_seq_by_ram),
            "reflex_ep": 200, "rrn_ep": 100, "mingru_ep": 50,
            "snn_ep": 60, "snn_dim": min(d, 512), "snn_heads": 8,
            "snn_batch": min(batch * 8, 64), "snn_seq": 256,
            "mingru_dim": min(d, 512), "mingru_layers": 6,
            "mamba_ep": [50, 25, 15, 10], "mamba_lr": [3e-4, 8e-5, 2e-5, 8e-6],
            "personality_ep": 15, "second_pass_ep": 10,
            "instruct_ep": 10, "cot_ep": 15, "dpo_ep": 8, "rlvr_ep": 8, "distill_ep": 5,
            "wiki_count": 500000,
            "snn_max_mb": 100,
            "marathon": True,  # flag для доп. циклов
        }

# ═══════════════════════════════════════════
# 6. CUDA Environment
# ═══════════════════════════════════════════

def setup_cuda_env(device, cfg):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if device == "cuda":
        env["NVIDIA_TF32_OVERRIDE"] = "1"
        env["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    return env

# ═══════════════════════════════════════════
# 7. Runner + State
# ═══════════════════════════════════════════

CUDA_ENV = None  # заполняется в main()

def run(cmd, label="", timeout=None):
    """Запустить команду (tee: консоль + лог)."""
    cmd = [str(c) for c in cmd]
    if len(cmd) >= 2 and ('python' in cmd[0].lower() or cmd[0] == PYTHON):
        if '-u' not in cmd:
            cmd.insert(1, '-u')

    cmd_str = " ".join(cmd)
    print(f"  → [{label}] {cmd_str[:120]}..." if label else f"  → {cmd_str[:120]}...")
    sys.stdout.flush()

    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(f"\n{'='*60}\n[{datetime.now()}] {cmd_str}\n{'='*60}\n")

    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=CUDA_ENV, bufsize=1)
        with open(LOG_FILE, 'a', encoding='utf-8', errors='replace') as log:
            for raw in proc.stdout:
                try:
                    line = raw.decode('utf-8', errors='replace')
                except Exception:
                    line = str(raw)
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
        proc.wait(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"  ⚠️ Таймаут ({timeout}s)")
        return False
    except KeyboardInterrupt:
        proc.kill()
        print(f"\n  ⏸ Прервано. Чекпоинт сохранён.")
        return False
    except Exception as e:
        print(f"  ❌ {e}")
        return False

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"completed_phases": [], "current_phase": None, "started": None}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def mark_done(state, key):
    state.setdefault("completed_phases", [])
    if key not in state["completed_phases"]:
        state["completed_phases"].append(key)
    save_state(state)

def is_done(state, key):
    return key in state.get("completed_phases", [])

# ═══════════════════════════════════════════
# 8. Google Drive (Colab / rclone)
# ═══════════════════════════════════════════

def setup_drive(mode):
    """Настроить Google Drive. mode = 'colab' | 'rclone' | 'none'."""
    if mode == "none":
        return False

    if mode == "colab":
        print("\n  ☁️  Google Drive: подключение через Colab...")
        # Проверяем: уже смонтировано через ноутбук? (симлинки data/ и models/)
        data_sym = (ROOT / "data").is_symlink()
        models_sym = (ROOT / "models").is_symlink()
        drive_path = Path("/content/drive/MyDrive")
        if (data_sym or models_sym) and drive_path.exists():
            print("  ✅ Drive уже смонтирован через ноутбук (симлинки активны)")
            return True
        if drive_path.exists():
            print("  ✅ Drive уже смонтирован")
            return True
        # Попытка монтирования (только если ещё не смонтирован)
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("  ✅ Google Drive смонтирован!")
            return True
        except ImportError:
            print("  ❌ google.colab недоступен (не запущен в Colab)")
            return False
        except Exception as e:
            # Если ошибка монтирования, но Drive уже доступен — продолжаем
            if drive_path.exists():
                print(f"  ⚠️  Ошибка mount ({e}), но Drive доступен")
                return True
            print(f"  ❌ Ошибка монтирования: {e}")
            return False

    if mode == "rclone":
        print("\n  ☁️  Google Drive: подключение через rclone...")
        setup_script = ROOT / "tools" / "setup_gdrive.py"
        if not setup_script.exists():
            print("  ❌ setup_gdrive.py не найден")
            return False
        try:
            r = subprocess.run([PYTHON, str(setup_script), "status"],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                print("  ✅ rclone уже настроен!")
                return True
        except Exception:
            pass
        print("  🔧 Запускаю настройку rclone...")
        ok = run([PYTHON, str(setup_script), "setup"], label="rclone Setup")
        if ok:
            print("  ✅ rclone настроен!")
        return ok

    return False

def drive_sync_data(mode):
    """Синхронизировать данные с Drive."""
    if mode == "colab":
        drive_data = Path("/content/drive/MyDrive/TarsData/data")
        if drive_data.exists():
            print("  ☁️  Копирование данных с Drive → data/...")
            DATA.mkdir(parents=True, exist_ok=True)
            for f in drive_data.glob("*"):
                dst = DATA / f.name
                if not dst.exists():
                    shutil.copy2(str(f), str(dst))
                    print(f"    + {f.name}")
            return True
        return False
    elif mode == "rclone":
        setup_script = ROOT / "tools" / "setup_gdrive.py"
        if setup_script.exists():
            return run([PYTHON, str(setup_script), "sync"], label="GDrive Sync")
    return False

def drive_backup(mode):
    """Бэкап моделей на Drive."""
    if mode == "colab":
        drive_models = Path("/content/drive/MyDrive/TarsData/models")
        drive_models.mkdir(parents=True, exist_ok=True)
        if TARS_V3.exists():
            print("  ☁️  Копирование моделей на Drive...")
            for f in TARS_V3.glob("*.pt"):
                shutil.copy2(str(f), str(drive_models / f.name))
                print(f"    → {f.name}")
            return True
        return False
    elif mode == "rclone":
        setup_script = ROOT / "tools" / "setup_gdrive.py"
        if setup_script.exists():
            return run([PYTHON, str(setup_script), "sync-models"], label="GDrive Backup")
    return False

# ═══════════════════════════════════════════
# 9. Утилиты обучения
# ═══════════════════════════════════════════

def download_all_data(cfg):
    """Скачать ВСЕ данные: Wiki + HF + Personality + Synthetic + Pretrained."""
    data_dir = Path(args.data_dir) if args.data_dir else DATA
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  📂 Data directory: {data_dir}")

    # Wikipedia — используем локальный файл если есть, НЕ скачиваем автоматически
    wiki = data_dir / "datasets" / "wiki_ru.txt"
    if not wiki.exists():
        wiki = data_dir / "wiki_ru.txt"
    if wiki.exists() and wiki.stat().st_size > 1000:
        print(f"  ✓ Wikipedia: {wiki.stat().st_size / 1024 / 1024:.1f} MB (кеш)")
    else:
        print(f"  ⏭ Wikipedia: не найдена (скачайте вручную: python training/data/download_wiki.py)")

    # HuggingFace
    hf_files = list(data_dir.glob("hf_*.txt"))
    if not hf_files or sum(f.stat().st_size for f in hf_files) < 1024:
        print(f"\n  📦 HuggingFace datasets (preset: {args.data_preset})...")
        run([PYTHON, str(DATA_DIR / "download_hf_dataset.py"),
             "--preset", args.data_preset, "--output", str(data_dir)], label="HF")
    else:
        print(f"  ✓ HuggingFace: {len(hf_files)} файлов (кеш)")

    # Personality
    pers = data_dir / "identity" / "tars_personality_mega.txt"
    if not pers.exists() or pers.stat().st_size < 100_000:
        print("\n  🧠 Generating personality corpus...")
        run([PYTHON, str(DATA_DIR / "generate_tars_corpus.py")], label="Personality")
    else:
        print(f"  ✓ Personality: {pers.stat().st_size // 1024} KB (кеш)")

    # Synthetic STEM
    stem = data_dir / "synthetic_stem.jsonl"
    if not stem.exists():
        print("\n  🔬 Generating STEM data...")
        run([PYTHON, str(DATA_DIR / "generate_synthetic.py")], label="STEM")

    # HF pretrained Mamba-2 weights
    pret = MODELS / "pretrained" / "mamba2-130m"
    if not pret.exists() or not list(pret.glob("*.safetensors")):
        print("\n  🧠 Downloading pretrained Mamba-2 base weights...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download("state-spaces/mamba2-130m",
                              local_dir=str(pret), ignore_patterns=["*.bin", "*.msgpack"])
            print(f"  ✅ Pretrained Mamba-2 saved: {pret}")
        except Exception as e:
            print(f"  ⚠ Mamba-2 download failed: {e}")

    # Summary
    total_mb = 0
    cnt = 0
    print(f"\n  {'─' * 50}")
    print(f"  📁 Файлы данных:")
    for f in sorted(data_dir.glob("*.txt")) + sorted(data_dir.glob("*.jsonl")):
        mb = f.stat().st_size / 1024 / 1024
        if mb > 0.01:
            total_mb += mb; cnt += 1
            print(f"    {f.name:40s} {mb:8.1f} MB")
    for sd in sorted(data_dir.glob("shards_*")):
        if sd.is_dir():
            sf = list(sd.glob("shard_*.txt"))
            smb = sum(f.stat().st_size for f in sf) / 1024 / 1024
            total_mb += smb; cnt += len(sf)
            print(f"    {sd.name + '/':40s} {smb:8.1f} MB ({len(sf)} shards)")
    print(f"  {'─' * 50}")
    print(f"  Итого: {cnt} файлов, {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)")
    return total_mb

def generate_dpo_pairs(output_path, n_pairs=500):
    """Генерация DPO пар (chosen vs rejected)."""
    pairs = [
        ("Привет!", "Привет! Я ТАРС, ваш ИИ-ассистент. Чем могу помочь?", "ну привет"),
        ("Кто ты?", "Я ТАРС — ИИ, созданный для помощи людям.", "я бот"),
        ("Что такое Python?", "Python — высокоуровневый язык программирования для ML, веба и автоматизации.", "язык"),
        ("Помоги с кодом", "Конечно! Опишите задачу, и я помогу.", "не могу"),
        ("Сколько будет 2+2?", "2 + 2 = 4.", "4"),
        ("Объясни рекурсию", "Рекурсия — когда функция вызывает саму себя до базового случая. Пример: n! = n × (n-1)!", "это когда функция вызывает себя"),
        ("Расскажи про нейросети", "Нейронные сети — модели ML, состоящие из слоёв нейронов. Типы: CNN, RNN, Mamba, Transformer.", "это типа мозг"),
        ("Спасибо!", "Рад помочь! Обращайтесь.", "ок"),
    ]
    topics = [
        ("machine learning", "машинное обучение — подраздел ИИ, модели учатся на данных", "это ии"),
        ("Docker", "Docker — платформа контейнеризации приложений", "программа"),
        ("API", "API — набор правил для взаимодействия программ. REST: GET, POST, PUT, DELETE", "интерфейс"),
    ]
    for tn, good, bad in topics:
        for t in [f"Что такое {tn}?", f"Объясни {tn}", f"Расскажи про {tn}"]:
            pairs.append((t, good, bad))

    out = []
    for i in range(n_pairs):
        p, c, r = random.choice(pairs)
        out.append({"prompt": p, "chosen": c, "rejected": r})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  ✅ Generated {len(out)} DPO pairs → {output_path}")

def quick_validate(cfg, device):
    """Быстрая валидация: 3 промпта."""
    print("\n  🔍 Quick validation...")
    try:
        import torch
        from brain.mamba2.model import TarsMamba2LM
        from brain.tokenizer import TarsTokenizer
        mp = TARS_V3 / "mamba2.pt"
        if not mp.exists():
            mp = TARS_V3 / "mamba2_omega.pt"
        if not mp.exists():
            print("     ⚠ Нет модели для валидации")
            return
        ckpt = torch.load(str(mp), map_location='cpu', weights_only=True)
        # Определяем vocab_size из чекпоинта
        ckpt_vocab = ckpt.get('vocab_size', ckpt.get('config', {}).get('vocab_size', 256))
        model = TarsMamba2LM(d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                             vocab_size=ckpt_vocab, quant_mode="fp16")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()
        dev = torch.device(device)
        model.to(dev)
        # Инициализируем токенизатор (BPE если обучен, иначе byte)
        tokenizer = TarsTokenizer(mode="auto")
        for prompt in ["Привет!", "Что такое Python?", "2+2="]:
            tokens = tokenizer.encode(prompt)
            x = torch.tensor([tokens], dtype=torch.long, device=dev)
            with torch.no_grad():
                for _ in range(60):
                    logits = model(x)
                    probs = torch.softmax(logits[:, -1, :ckpt_vocab] / 0.7, dim=-1)
                    nt = torch.multinomial(probs, 1)
                    x = torch.cat([x, nt], dim=1)
                    if nt.item() == tokenizer.eos_token_id or x.shape[1] > 200:
                        break
            out = tokenizer.decode(x[0].cpu().tolist())
            print(f"     Q: {prompt}")
            print(f"     A: {out[len(prompt):].strip()[:80]}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  ✅ Validation complete")
    except Exception as e:
        print(f"     ⚠ Validation error: {e}")

def consolidate_models(cfg, results, total_time):
    """Фаза 6: собрать модели в models/tars_v3/."""
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
            mb = dst.stat().st_size / 1024 / 1024
            print(f"  📦 {name}: → tars_v3/{dst.name} ({mb:.1f} MB)")
            consolidated.append(name)
        else:
            print(f"  ⏭ {name}: не найден")

    # config.json
    mc = {"d_model": cfg["d_model"], "n_layers": cfg["n_layers"],
          "d_state": 64, "headdim": 64, "omega_size": 32, "pool_size": 48, "n_experts": 8}
    for pt_name in ["mamba2.pt", "mamba2_158bit.pt"]:
        pt = TARS_V3 / pt_name
        if pt.exists():
            try:
                import torch
                ck = torch.load(str(pt), map_location="cpu", weights_only=True)
                c = ck.get("config", {})
                mc.update({k: c[k] for k in ["d_model", "n_layers", "vocab_size"] if k in c})
                del ck
                break
            except Exception:
                pass
    cj = {"name": "tars_v3", "version": "3.0",
          "models": {"mamba2": {"params": mc}}}
    (TARS_V3 / "config.json").write_text(json.dumps(cj, ensure_ascii=False, indent=2), encoding="utf-8")

    # Training log
    entry = {"timestamp": datetime.now().isoformat(), "total_time_sec": round(total_time, 1),
             "results": {k: ("ok" if v else "failed") for k, v in results.items()},
             "config": mc}
    lp = TARS_V3 / "training_log.json"
    logs = []
    if lp.exists():
        try: logs = json.loads(lp.read_text(encoding="utf-8"))
        except Exception: pass
    if not isinstance(logs, list): logs = []
    logs.append(entry)
    lp.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

# ═══════════════════════════════════════════
# 10. MAIN PIPELINE
# ═══════════════════════════════════════════

def main():
    global CUDA_ENV

    # ── Бенчмарк железа (~5 сек) ──
    print()
    print("═" * 65)
    print("  ⚙️  ТАРС v3 — ТЕСТИРОВАНИЕ ЖЕЛЕЗА...")
    print("═" * 65)
    print()
    hw = benchmark_hardware()
    print_hw_report(hw)

    device = hw["device"]
    bf16 = hw["bf16"]

    # Интерактивный выбор
    drive_mode, level = interactive_menu()

    # Конфигурация на основе бенчмарка
    cfg = get_config(level, hw)
    state = load_state()
    CUDA_ENV = setup_cuda_env(device, cfg)

    # Google Drive
    drive_ok = False
    if drive_mode != "none":
        drive_ok = setup_drive(drive_mode)

    # Banner
    print()
    print("═" * 65)
    print("  🤖 ТАРС v3 — УНИВЕРСАЛЬНОЕ ОБУЧЕНИЕ")
    print("═" * 65)
    print()
    print(f"  📐 Model:   {cfg['name']}")
    print(f"  📦 Batch:   {cfg['batch']} × {cfg['accum']} = {cfg['batch'] * cfg['accum']} effective")
    print(f"  📏 SeqLen:  {cfg['seq_start']} → {cfg['seq_mid']} → {cfg['seq_max']}")
    print(f"  ⚡ AMP:     {cfg['amp_dtype']}")
    print(f"  📊 Уровень: {level.upper()}")
    print(f"  💾 Диск:    {drive_mode}")
    if cfg.get("use_compile"):
        print(f"  🔧 Compile: Да (+30% speed)")
    if cfg.get("use_muon"):
        print(f"  🚀 Muon:    Да (2x faster)")
    if cfg.get("use_wsd"):
        print(f"  📈 WSD:     Да")
    if cfg.get("num_workers", 0) > 0:
        print(f"  🔄 Workers: {cfg['num_workers']} (pin_memory={cfg['pin_memory']})")
    if args.resume and state.get("completed_phases"):
        print(f"  🔄 Resume:  {state['completed_phases']}")
    print()

    est_params = cfg["d_model"] ** 2 * cfg["n_layers"] * 12
    print(f"  📊 ~{est_params / 1e6:.0f}M параметров")
    print()
    print("─" * 65)

    t0 = time.time()
    results = {}

    def should_run(phase):
        return args.phase is None or args.phase == phase

    # Drive sync
    if drive_ok:
        drive_sync_data(drive_mode)

    # ═══ Phase 0: Dependencies ═══
    if should_run(0):
        print("\n  📦 Phase 0: Зависимости...")
        # Проверка ключевых пакетов
        missing = []
        for pkg in ["torch", "numpy", "einops", "tqdm", "datasets", "transformers", "psutil"]:
            try:
                __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)
        if missing:
            print(f"  📦 Установка: {', '.join(missing)}")
            run([PYTHON, "-m", "pip", "install"] + missing + ["--quiet"], label="pip")
        else:
            print("  ✅ Все зависимости установлены")
        results["deps"] = True

    # ═══ Phase 1: Download ═══
    if not args.skip_download and (should_run(1) or args.download_only):
        print(f"\n  📚 Phase 1: Скачивание данных (preset: {args.data_preset})...")
        data_mb = download_all_data(cfg)
        results["download"] = data_mb > 1
        if args.download_only:
            print(f"\n  ✅ Данные скачаны за {(time.time()-t0)/60:.0f} мин ({data_mb:.0f} MB)")
            return

    # ═══ Phase 2: Reflex ═══
    if should_run(2) and not is_done(state, "reflex"):
        reflex_script = TRAIN_DIR / "train_reflex.py"
        if reflex_script.exists():
            print(f"\n  🔁 Phase 2: Рефлекс-классификатор ({cfg['reflex_ep']} epochs)...")
            ok = run([PYTHON, str(reflex_script),
                      "--epochs", str(cfg["reflex_ep"]), "--lr", "0.002"], label="Reflex")
            results["reflex"] = ok
            if ok: mark_done(state, "reflex")
        else:
            print(f"\n  ⏭ Phase 2: train_reflex.py не найден — пропуск")
            results["reflex"] = False

    # ═══ Phase 2.5: RRN ═══
    if should_run(2) and not is_done(state, "rrn"):
        rrn_script = TRAIN_DIR / "train_rrn.py"
        if rrn_script.exists():
            print(f"\n  🦴 Phase 2.5: RRN маршрутизация ({cfg.get('rrn_ep', 10)} epochs)...")
            ok = run([PYTHON, str(rrn_script),
                      "--epochs", str(cfg.get("rrn_ep", 10)),
                      "--brain_dim", str(cfg["d_model"]),
                      "--lr", "0.002", "--batch", "32"], label="RRN")
            results["rrn"] = ok
            if ok: mark_done(state, "rrn")
        else:
            print(f"\n  ⏭ Phase 2.5: train_rrn.py не найден — пропуск")
            mark_done(state, "rrn")  # не блокировать pipeline

    # ═══ Phase 3: MinGRU ═══
    if should_run(3) and not is_done(state, "mingru"):
        mingru_script = TRAIN_DIR / "train_mingru.py"
        if mingru_script.exists():
            print(f"\n  🧪 Phase 3: MinGRU ({cfg.get('mingru_dim', 256)}d×{cfg.get('mingru_layers', 4)}L, {cfg.get('mingru_ep', 5)}ep)...")
            ok = run([PYTHON, str(mingru_script),
                      "--dim", str(cfg.get("mingru_dim", 256)),
                      "--layers", str(cfg.get("mingru_layers", 4)),
                      "--epochs", str(cfg.get("mingru_ep", 5)), "--augment"], label="MinGRU")
            results["mingru"] = ok
            if ok: mark_done(state, "mingru")
        else:
            print(f"\n  ⏭ Phase 3: train_mingru.py не найден — пропуск")
            mark_done(state, "mingru")

    # ═══ Phase 3.5: SNN ═══
    if (should_run(3) or args.phase is None) and not is_done(state, "spiking"):
        snn_script = TRAIN_DIR / "train_spiking.py"
        if snn_script.exists():
            print(f"\n  ⚡ Phase 3.5: SNN ({cfg.get('snn_dim', 256)}d×{cfg.get('snn_heads', 4)}H, {cfg.get('snn_ep', 5)}ep)...")
            snn_cmd = [PYTHON, str(snn_script),
                       "--dim", str(cfg.get("snn_dim", 256)), "--heads", str(cfg.get("snn_heads", 4)),
                       "--beta", "0.9", "--epochs", str(cfg.get("snn_ep", 5)),
                       "--batch", str(cfg.get("snn_batch", 32)), "--seq_len", str(cfg.get("snn_seq", 128)),
                       "--lr", "3e-4", "--device", device,
                       "--max_bytes", str(cfg.get("snn_max_mb", 20))]
            if args.data_dir: snn_cmd += ["--data_dir", args.data_dir]
            if args.resume: snn_cmd += ["--resume"]
            ok = run(snn_cmd, label="SNN")
            results["spiking"] = ok
            if ok: mark_done(state, "spiking")
        else:
            print(f"\n  ⏭ Phase 3.5: train_spiking.py не найден — пропуск")
            mark_done(state, "spiking")

    # ═══ Phase 4: Mamba-2 Brain ═══
    mamba_seq = [cfg["seq_start"], cfg["seq_mid"], cfg["seq_max"], cfg["seq_max"]]
    mamba_names = {1: "Full Pretrain (SSD+WKV+Ω-SSM+MoLE)",
                   2: "WKV + Fusion Fine-tune",
                   3: "MoLE + MatrixPool + WaveMerge",
                   4: "RAG + Memory + NoveltyGate"}
    for mp in [1, 2, 3, 4]:
        pk = f"mamba_p{mp}"
        if not should_run(4): continue
        if is_done(state, pk):
            print(f"\n  ⏭ Phase 4.{mp}: уже выполнена")
            continue
        print(f"\n  🧠 Phase 4.{mp}: {mamba_names[mp]}...")
        print(f"     {cfg['d_model']}d × {cfg['n_layers']}L, "
              f"batch={cfg['batch']}×{cfg['accum']}, "
              f"{cfg['mamba_ep'][mp-1]} epochs")

        cmd = [PYTHON, str(TRAIN_DIR / "train_mamba2.py"),
               "--d_model", str(cfg["d_model"]),
               "--n_layers", str(cfg["n_layers"]),
               "--vocab_size", str(cfg["vocab_size"]),
               "--batch", str(cfg["batch"]),
               "--accum_steps", str(cfg["accum"]),
               "--epochs", str(cfg["mamba_ep"][mp-1]),
               "--lr", str(cfg["mamba_lr"][mp-1]),
               "--seq_len", str(mamba_seq[mp-1]),
               "--phase", str(mp),
               "--device", device,
               "--curriculum", "--label_smoothing", "0.1",
               "--grad_ckpt"]
        # Marathon mode loads Wikipedia for real training data
        if not cfg.get("marathon"):
            cmd += ["--no_wiki"]
        # ── AMP: bf16 для Ampere+, fp16 для T4/Turing ──
        amp = cfg.get("amp_dtype", "fp32")
        if amp == "bf16": cmd += ["--bf16"]
        elif amp == "fp16": cmd += ["--fp16"]
        # ── Оптимизации из бенчмарка ──
        if cfg.get("use_compile"): cmd += ["--compile"]
        if cfg.get("use_muon"): cmd += ["--muon"]
        if cfg.get("use_wsd"): cmd += ["--wsd"]
        if cfg.get("num_workers", 0) > 0:
            cmd += ["--num_workers", str(cfg["num_workers"])]
        if cfg.get("pin_memory"): cmd += ["--pin_memory"]
        if mp > 1 or args.resume: cmd += ["--resume"]
        if args.data_dir: cmd += ["--data_dir", args.data_dir]

        # Pretrained weights для Phase 1
        if mp == 1:
            pret_dir = MODELS / "pretrained" / "mamba2-130m"
            if pret_dir.exists() and list(pret_dir.glob("*.safetensors")):
                cmd += ["--pretrained", str(pret_dir)]
            # MinGRU embedding transfer
            emb_path = TARS_V3 / "_transfer_embedding.pt"
            mingru_path = MODELS / "mingru_weights.pt"
            if mingru_path.exists() and not emb_path.exists():
                try:
                    import torch
                    ck = torch.load(str(mingru_path), map_location='cpu', weights_only=True)
                    if 'model_state_dict' in ck:
                        for k, v in ck['model_state_dict'].items():
                            if 'token_emb' in k or 'embedding' in k:
                                emb_path.parent.mkdir(parents=True, exist_ok=True)
                                torch.save(v, str(emb_path))
                                print(f"     🔗 Embedding transferred: {v.shape}")
                                break
                except Exception:
                    pass
            if emb_path.exists():
                cmd += ["--pretrained_emb", str(emb_path)]

        ok = run(cmd, label=f"Mamba P{mp}")
        results[pk] = ok
        if ok:
            mark_done(state, pk)
        else:
            print(f"  ⚠️ Phase 4.{mp} failed — продолжаем следующие фазы")
            # НЕ останавливаем: personality/second_pass могут работать с тем что есть

    # Drive backup after Mamba-2
    if drive_ok: drive_backup(drive_mode)

    # ═══ Phase 4.5: Personality ═══
    if should_run(4) and not is_done(state, "personality"):
        print(f"\n  🎭 Phase 4.5: PersonalityAdapter ({cfg['personality_ep']} epochs)...")
        cmd = [PYTHON, str(TRAIN_DIR / "train_mamba2.py"),
               "--d_model", str(cfg["d_model"]), "--n_layers", str(cfg["n_layers"]),
               "--vocab_size", str(cfg["vocab_size"]),
               "--batch", str(cfg["batch"]),
               "--accum_steps", str(cfg["accum"]),
               "--epochs", str(cfg["personality_ep"]),
               "--lr", "5e-5", "--seq_len", str(cfg["seq_mid"]),
               "--phase", "5", "--device", device,
               "--curriculum", "--label_smoothing", "0.1",
               "--grad_ckpt", "--resume", "--no_wiki"]
        if bf16: cmd += ["--bf16"]
        if args.data_dir: cmd += ["--data_dir", args.data_dir]
        ok = run(cmd, label="Personality")
        results["personality"] = ok
        if ok: mark_done(state, "personality")

    # ═══ MARATHON: Second Cycle Refinement (phases 2-4 again, lower LR) ═══
    if cfg.get("marathon") and should_run(4) and not is_done(state, "marathon_cycle2"):
        print(f"\n  🔄 MARATHON Cycle 2: Refinement passes (phases 2→3→4, halved LR)...")
        cycle2_eps = [10, 5, 5]   # phases 2,3,4
        cycle2_lrs = [4e-5, 1e-5, 4e-6]
        cycle2_seqs = [cfg["seq_mid"], cfg["seq_max"], cfg["seq_max"]]
        for ci, (mp, ep, lr, sl) in enumerate(zip([2,3,4], cycle2_eps, cycle2_lrs, cycle2_seqs)):
            pk2 = f"marathon_c2_p{mp}"
            if is_done(state, pk2):
                print(f"  ⏭ Cycle2 Phase {mp}: уже выполнена")
                continue
            print(f"\n  🔄 Cycle2 Phase {mp}: {ep} epochs, LR={lr}, seq={sl}")
            cmd = [PYTHON, str(TRAIN_DIR / "train_mamba2.py"),
                   "--d_model", str(cfg["d_model"]),
                   "--n_layers", str(cfg["n_layers"]),
                   "--vocab_size", str(cfg["vocab_size"]),
                   "--batch", str(cfg["batch"]),
                   "--accum_steps", str(cfg["accum"]),
                   "--epochs", str(ep),
                   "--lr", str(lr),
                   "--seq_len", str(sl),
                   "--phase", str(mp),
                   "--device", device,
                   "--curriculum", "--label_smoothing", "0.05",
                   "--grad_ckpt", "--resume"]
            if bf16: cmd += ["--bf16"]
            if cfg.get("use_compile"): cmd += ["--compile"]
            if cfg.get("use_wsd"): cmd += ["--wsd"]
            if cfg.get("num_workers", 0) > 0:
                cmd += ["--num_workers", str(cfg["num_workers"])]
            if cfg.get("pin_memory"): cmd += ["--pin_memory"]
            ok = run(cmd, label=f"C2-P{mp}")
            results[pk2] = ok
            if ok: mark_done(state, pk2)
        mark_done(state, "marathon_cycle2")
        if drive_ok: drive_backup(drive_mode)

    # ═══ Phase 4.6, 5: → local_train_extras.py ═══
    # Second Pass и Квантизация вынесены в local_train_extras.py
    # Запуск: python local_train_extras.py --phase 4.6 / --phase 5

    # ═══ Phase 6: Consolidate ═══
    if should_run(6) and not is_done(state, "consolidate"):
        print("\n  📦 Phase 6: Консолидация моделей...")
        ok = consolidate_models(cfg, results, time.time() - t0)
        results["consolidate"] = ok
        if ok: mark_done(state, "consolidate")

    # ═══ Phase 7: Validate ═══
    if should_run(7) and not is_done(state, "validate"):
        print("\n  ✅ Phase 7: Валидация...")
        qt = EVAL_DIR / "quick_test.py"
        if qt.exists():
            ok = run([PYTHON, str(qt)], label="Validate", timeout=120)
        else:
            quick_validate(cfg, device)
            ok = True
        results["validate"] = ok
        if ok: mark_done(state, "validate")

    # ═══ Phase 8-9: Voice (STT/TTS) ═══
    if not args.skip_voice and (should_run(8) or should_run(9)):
        print("\n  🎙 Phase 8-9: Голосовые модули...")
        if should_run(8) and not is_done(state, "whisper"):
            whisper_script = TRAIN_DIR / "train_whisper.py"
            if whisper_script.exists():
                ok = run([PYTHON, str(whisper_script),
                          "--device", device, "--epochs", "3", "--batch", "16"], label="Whisper")
                if ok: mark_done(state, "whisper")
                results["whisper"] = ok
            else:
                print("  ⚠ train_whisper.py не найден — пропуск")
                results["whisper"] = False
        if should_run(9) and not is_done(state, "piper"):
            piper_script = TRAIN_DIR / "train_piper.py"
            if piper_script.exists():
                ok = run([PYTHON, str(piper_script),
                          "--epochs", "1000", "--batch", "16"], label="Piper")
                if ok: mark_done(state, "piper")
                results["piper"] = ok
            else:
                print("  ⚠ train_piper.py не найден — пропуск")
                results["piper"] = False
        # Phase 10 (Voice Quant) → local_train_extras.py
        if drive_ok: drive_backup(drive_mode)

    # ═══ Phase 11: Instruction Tuning ═══
    if (should_run(11) or args.phase is None) and not is_done(state, "instruct"):
        print(f"\n  📖 Phase 11: Instruction Tuning ({cfg['instruct_ep']} epochs)...")
        ok = run([PYTHON, str(TRAIN_DIR / "train_instruct.py"),
                  "--d_model", str(cfg["d_model"]), "--n_layers", str(cfg["n_layers"]),
                  "--epochs", str(cfg["instruct_ep"]), "--batch", str(cfg["batch"]),
                  "--device", device, "--save_dir", str(TARS_V3),
                  "--resume", "--grad_ckpt", "--lr", "5e-5"]
                 + (["--bf16"] if bf16 else []), label="Instruct")
        results["instruct"] = ok
        if ok:
            mark_done(state, "instruct")
            if drive_ok: drive_backup(drive_mode)
            quick_validate(cfg, device)

    # ═══ Phase 12-14: Post-Training ═══
    if not args.skip_posttrain:
        data_dir = Path(args.data_dir) if args.data_dir else DATA

        # Phase 12: CoT
        if (should_run(12) or args.phase is None) and not is_done(state, "cot"):
            cot_data = data_dir / "cot_reasoning.txt"
            if not cot_data.exists() or cot_data.stat().st_size < 1000:
                print(f"\n  📝 Phase 12a: Генерация CoT данных...")
                run([PYTHON, str(TRAIN_DIR / "train_cot.py"),
                     "--generate", "--n_samples", "20000",
                     "--cot_data", str(cot_data)], label="CoT-Gen")
            print(f"\n  🧩 Phase 12b: CoT Training ({cfg['cot_ep']} epochs)...")
            ok = run([PYTHON, str(TRAIN_DIR / "train_cot.py"),
                      "--d_model", str(cfg["d_model"]), "--n_layers", str(cfg["n_layers"]),
                      "--epochs", str(cfg["cot_ep"]), "--batch", str(cfg["batch"]),
                      "--device", device, "--save_dir", str(TARS_V3),
                      "--resume", "--grad_ckpt", "--lr", "3e-5",
                      "--train", "--cot_data", str(cot_data)]
                     + (["--bf16"] if bf16 else []), label="CoT")
            results["cot"] = ok
            if ok: mark_done(state, "cot")

        # Phase 13: DPO
        if (should_run(13) or args.phase is None) and not is_done(state, "dpo"):
            dpo_data = data_dir / "dpo_pairs.jsonl"
            if not dpo_data.exists() or dpo_data.stat().st_size < 100:
                print(f"\n  📝 Phase 13a: Генерация DPO пар...")
                generate_dpo_pairs(dpo_data)
            if dpo_data.exists() and dpo_data.stat().st_size > 100:
                print(f"\n  ⚖️ Phase 13b: DPO ({cfg['dpo_ep']} epochs)...")
                ok = run([PYTHON, str(TRAIN_DIR / "train_dpo.py"),
                          "--d_model", str(cfg["d_model"]), "--n_layers", str(cfg["n_layers"]),
                          "--epochs", str(cfg["dpo_ep"]), "--batch", str(cfg["batch"]),
                          "--device", device, "--save_dir", str(TARS_V3),
                          "--resume", "--grad_ckpt", "--lr", "1e-5",
                          "--data", str(dpo_data)]
                         + (["--bf16"] if bf16 else []), label="DPO")
                results["dpo"] = ok
                if ok: mark_done(state, "dpo")

        # Phase 14: RLVR
        if (should_run(14) or args.phase is None) and not is_done(state, "rlvr"):
            print(f"\n  🎯 Phase 14: RLVR ({cfg['rlvr_ep']} epochs)...")
            ok = run([PYTHON, str(TRAIN_DIR / "train_rlvr.py"),
                      "--d_model", str(cfg["d_model"]), "--n_layers", str(cfg["n_layers"]),
                      "--epochs", str(cfg["rlvr_ep"]), "--batch", str(cfg["batch"]),
                      "--device", device, "--save_dir", str(TARS_V3),
                      "--resume", "--grad_ckpt", "--lr", "3e-5"]
                     + (["--bf16"] if bf16 else []), label="RLVR")
            results["rlvr"] = ok
            if ok: mark_done(state, "rlvr")

    # ═══ Phase 15: → local_train_extras.py ═══
    # Knowledge Distillation вынесена в local_train_extras.py
    # Запуск: python local_train_extras.py --phase 15

    # ═══ Запуск extras (если --extras) ═══
    if args.extras:
        print("\n  🧪 Запуск дополнительных фаз (extras)...")
        run([PYTHON, str(ROOT / "local_train_extras.py"), "--all",
             "--level", level], label="Extras")

    # ═══════════════════════════════════════════
    # РЕЗУЛЬТАТЫ
    # ═══════════════════════════════════════════
    elapsed = time.time() - t0
    hrs = elapsed / 3600

    print()
    print("═" * 65)
    print(f"  🤖 ТАРС v3 — РЕЗУЛЬТАТЫ ({hrs:.1f} часов)")
    print("═" * 65)
    print()
    for name, ok in results.items():
        print(f"    {'✅' if ok else '❌'} {name}")
    print()

    all_ok = all(results.values()) if results else False
    if all_ok:
        print(f"  🎯 ВСЕ ФАЗЫ ЗАВЕРШЕНЫ!")
        print(f"  📐 Модель: {cfg['name']}")
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

    # Final Drive backup
    if drive_ok:
        print()
        print("  ☁️  Финальная выгрузка на Drive...")
        drive_backup(drive_mode)

    print()
    print("═" * 65)


if __name__ == "__main__":
    main()
