"""
═══════════════════════════════════════════════════════════════════════
  ТАРС v3 — СТАЦИОНАРНОЕ ОБУЧЕНИЕ (Лаборатория, MAX)
═══════════════════════════════════════════════════════════════════════

Максимальное обучение на стационарном GPU (RTX 4090 / 3090 / A6000).
Авто-определение GPU → оптимальный конфиг по VRAM.

  ≥22 GB:  768M params (1024d × 20L), batch=4×8=32
  ≥14 GB:  400M params (768d × 16L),  batch=4×8=32
  <14 GB:  250M params (768d × 12L),  batch=2×16=32

ИСПОЛЬЗОВАНИЕ:
  python local_train.py                    # Авто-конфиг по GPU
  python local_train.py --1b              # Форсировать 768M модель
  python local_train.py --resume          # Продолжить с чекпоинта
  python local_train.py --phase 1         # Только Phase 1
  python local_train.py --download-only   # Только скачать данные

═══════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import json
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
CHECKPOINTS = MODELS / "checkpoints"
TARS_V3 = MODELS / "tars_v3"
LOG_FILE = ROOT / "local_train.log"
STATE_FILE = ROOT / "train_state.json"

# ═══════════════════════════════════════════
# 2. Arguments
# ═══════════════════════════════════════════

parser = argparse.ArgumentParser(description="ТАРС v3 — Local Training (RTX 4090)")
parser.add_argument("--1b", dest="one_billion", action="store_true",
                    help="Форсировать 1B модель (1024d × 20L)")
parser.add_argument("--resume", action="store_true",
                    help="Продолжить с чекпоинта")
parser.add_argument("--phase", type=int, default=None,
                    help="Запустить только конкретную фазу (1-7)")
parser.add_argument("--download-only", action="store_true",
                    help="Только скачать данные")
parser.add_argument("--skip-download", action="store_true",
                    help="Пропустить скачивание (данные есть)")
parser.add_argument("--skip-voice", action="store_true",
                    help="Пропустить голосовые модули")
parser.add_argument("--data-preset", default="max",
                    choices=["all", "max", "quality", "massive", "reasoning"],
                    help="Какие данные скачивать (default: max)")
parser.add_argument("--checkpoint-interval", type=int, default=1800,
                    help="Интервал сохранения чекпоинтов в секундах (default: 1800 = 30 мин)")
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


def get_config(vram_gb, force_1b=False):
    """Возвращает оптимальный конфиг по VRAM."""
    if force_1b or vram_gb >= 22:
        return {
            "name": "768M",
            "d_model": 1024,
            "n_layers": 20,
            "batch": 4,
            "accum": 8,         # effective batch = 32
            "seq_len_start": 512,
            "seq_len_mid": 1024,
            "seq_len_max": 4096,
            "lr_p1": 3e-4,
            "lr_p2": 1e-4,
            "lr_p3": 3e-5,
            "lr_p4": 1.5e-5,
            "lr_p5": 5e-5,
            "epochs_p1": 10,
            "epochs_p2": 5,
            "epochs_p3": 3,
            "epochs_p4": 3,
            "epochs_p5": 5,     # personality
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
            "lr_p1": 3e-4,
            "lr_p2": 1e-4,
            "lr_p3": 3e-5,
            "lr_p4": 1.5e-5,
            "lr_p5": 5e-5,
            "epochs_p1": 10,
            "epochs_p2": 5,
            "epochs_p3": 3,
            "epochs_p4": 3,
            "epochs_p5": 3,
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
            "lr_p1": 3e-4,
            "lr_p2": 1e-4,
            "lr_p3": 3e-5,
            "lr_p4": 1.5e-5,
            "lr_p5": 5e-5,
            "epochs_p1": 10,
            "epochs_p2": 5,
            "epochs_p3": 3,
            "epochs_p4": 3,
            "epochs_p5": 3,
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


# ═══════════════════════════════════════════
# 5. Training Runner
# ═══════════════════════════════════════════

def run(cmd, timeout=None):
    """Запустить команду с логированием."""
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"  → {cmd_str[:100]}...")
    
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"[{datetime.now()}] {cmd_str}\n")
        log.write(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            [str(c) for c in cmd],
            cwd=str(ROOT),
            timeout=timeout,
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
    
    if phase_num > 1 or args.resume:
        cmd += ["--resume"]
    
    if extra_args:
        cmd += extra_args
    
    return run(cmd)


# ═══════════════════════════════════════════
# 6. Main Pipeline
# ═══════════════════════════════════════════

def main():
    # Detect hardware
    gpu_name, vram_gb, device, bf16 = detect_gpu()
    
    config = get_config(vram_gb, force_1b=args.one_billion)
    state = load_state()
    
    # Banner
    print()
    print("═" * 65)
    print("  🤖 ТАРС v3 — LOCAL TRAINING")
    print("═" * 65)
    print()
    print(f"  🎮 GPU:    {gpu_name or 'CPU'}")
    print(f"  💾 VRAM:   {vram_gb:.1f} GB")
    print(f"  🧠 Model:  {config['name']} ({config['d_model']}d × {config['n_layers']}L)")
    print(f"  📦 Batch:  {config['batch']} × {config['accum']} = {config['batch'] * config['accum']} effective")
    print(f"  ⚡ bf16:   {'Yes' if bf16 else 'No'}")
    print(f"  📁 Data:   {args.data_preset}")
    if args.resume:
        print(f"  🔄 Resume: from checkpoint")
        if state.get("completed_phases"):
            print(f"     Done:   {state['completed_phases']}")
    print()
    print("─" * 65)
    
    t0 = time.time()
    results = {}
    
    # ── Phase 0: Install dependencies ──
    if args.phase is None or args.phase == 0:
        print("\n  📦 Phase 0: Dependencies...")
        results["deps"] = run([PYTHON, "mega_train.py", "--phase", "0"])
    
    # ── Phase 1: Download data ──
    if not args.skip_download and (args.phase is None or args.phase == 1 or args.download_only):
        print(f"\n  📚 Phase 1: Download data (preset: {args.data_preset})...")
        results["download"] = run([
            PYTHON, str(TRAINING / "download_hf_dataset.py"),
            "--preset", args.data_preset,
        ])
        
        # Generate personality corpus
        personality = DATA / "tars_personality_mega.txt"
        if not personality.exists() or personality.stat().st_size < 1_000_000:
            print("\n  🧠 Generating personality corpus...")
            run([PYTHON, str(TRAINING / "generate_tars_corpus.py")])
        
        if args.download_only:
            elapsed = time.time() - t0
            print(f"\n  ✅ Данные скачаны за {elapsed/60:.0f} минут")
            return
    
    # ── Phase 2: Reflex classifier ──
    if args.phase is None or args.phase == 2:
        if "reflex" not in state.get("completed_phases", []):
            print("\n  🔁 Phase 2: Reflex classifier (100 epochs)...")
            ok = run([PYTHON, "mega_train.py", "--phase", "2"])
            results["reflex"] = ok
            if ok:
                state.setdefault("completed_phases", []).append("reflex")
                save_state(state)
    
    # ── Phase 3: MinGRU LM ──
    if args.phase is None or args.phase == 3:
        if "mingru" not in state.get("completed_phases", []):
            print("\n  🧪 Phase 3: MinGRU LM (25 epochs)...")
            ok = run([PYTHON, "mega_train.py", "--phase", "3"])
            results["mingru"] = ok
            if ok:
                state.setdefault("completed_phases", []).append("mingru")
                save_state(state)
    
    # ═══ Phase 4: Mamba-2 Brain — THE MAIN EVENT ═══
    for mamba_phase in [1, 2, 3, 4]:
        phase_key = f"mamba_p{mamba_phase}"
        if args.phase is not None and args.phase != 4:
            continue
        if phase_key in state.get("completed_phases", []):
            print(f"\n  ⏭ Phase 4.{mamba_phase}: already done")
            continue
        
        phase_names = {
            1: "Full Pretrain",
            2: "WKV + Fusion Fine-tune",
            3: "MoLE + MatrixPool",
            4: "RAG + Memory Integration",
        }
        
        print(f"\n  🧠 Phase 4.{mamba_phase}: {phase_names[mamba_phase]}...")
        print(f"     {config['d_model']}d × {config['n_layers']}L, "
              f"batch={config['batch']}×{config['accum']}")
        
        ok = train_mamba_phase(mamba_phase, config, device, bf16)
        results[phase_key] = ok
        
        if ok:
            state.setdefault("completed_phases", []).append(phase_key)
            save_state(state)
            print(f"  ✅ Phase 4.{mamba_phase} done, checkpoint saved")
        else:
            print(f"  ⚠️ Phase 4.{mamba_phase} failed, run with --resume to retry")
            break
    
    # ── Phase 5: PersonalityAdapter ──
    if args.phase is None or args.phase == 5:
        if "personality" not in state.get("completed_phases", []):
            print(f"\n  🎭 Phase 5: PersonalityAdapter ({config['epochs_p5']} epochs)...")
            ok = train_mamba_phase(5, config, device, bf16)
            results["personality"] = ok
            if ok:
                state.setdefault("completed_phases", []).append("personality")
                save_state(state)
    
    # ── Phase 6: Second Pass (longer context) ──
    if args.phase is None or args.phase == 6:
        if "second_pass" not in state.get("completed_phases", []):
            print(f"\n  🔄 Phase 6: Second Pass (seq_len={config['seq_len_max']})...")
            # Retrain Phase 1 with longer sequences
            cmd = [
                PYTHON, str(TRAINING / "train_mamba2.py"),
                "--d_model", str(config["d_model"]),
                "--n_layers", str(config["n_layers"]),
                "--vocab_size", "256",
                "--batch", str(config["batch"]),
                "--accum_steps", str(config["accum"]),
                "--epochs", "5",
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
            
            ok = run(cmd)
            results["second_pass"] = ok
            if ok:
                state.setdefault("completed_phases", []).append("second_pass")
                save_state(state)
    
    # ── Phase 7: Quantize + Validate ──
    if args.phase is None or args.phase == 7:
        print("\n  ⚗️ Phase 7: Quantize 1.58-bit...")
        results["quantize"] = run([PYTHON, "mega_train.py", "--phase", "5"])
        
        print("\n  📦 Phase 7b: Consolidate...")
        results["consolidate"] = run([PYTHON, "mega_train.py", "--phase", "6"])
        
        print("\n  ✅ Phase 7c: Validate...")
        results["validate"] = run([PYTHON, "mega_train.py", "--phase", "7"])
    
    # ── Voice (optional) ──
    if not args.skip_voice and (args.phase is None or args.phase == 8):
        print("\n  🎙 Phase 8: Voice (Whisper + Piper)...")
        run([PYTHON, "mega_train.py", "--phase", "8"])
        run([PYTHON, "mega_train.py", "--phase", "9"])
        run([PYTHON, "mega_train.py", "--phase", "10"])
    
    # ═══════════════════════════════════════════
    # Results
    # ═══════════════════════════════════════════
    
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
    
    all_ok = all(results.values())
    if all_ok:
        print(f"  🎯 ВСЕ ФАЗЫ ЗАВЕРШЕНЫ!")
        print(f"  📁 Модель: {config['name']} ({config['d_model']}d × {config['n_layers']}L)")
        print()
        
        if TARS_V3.exists():
            total_mb = 0
            for f in TARS_V3.glob("*.pt"):
                mb = f.stat().st_size / 1024 / 1024
                total_mb += mb
                print(f"    {f.name}: {mb:.1f} MB")
            print(f"    {'─' * 30}")
            print(f"    Итого: {total_mb:.0f} MB")
        
        print()
        print("  🚀 Запуск: python launch_tars.py")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"  ⚠️ Ошибки: {', '.join(failed)}")
        print(f"  Продолжить: python local_train.py --resume")
    
    print()
    print("═" * 65)


if __name__ == "__main__":
    main()
