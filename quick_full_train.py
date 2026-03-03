"""
═══════════════════════════════════════════════════════════════
  ТАРС — Quick Full Architecture Test (1 час, BEAST MODE)
═══════════════════════════════════════════════════════════════

RTX 4090 (24 GB) + 64 GB RAM → МАКСИМАЛЬНАЯ УТИЛИЗАЦИЯ.

Оптимизации:
  • bf16 mixed precision (2× throughput)
  • torch.compile(mode='max-autotune')  
  • gradient_checkpointing (2× batch size)
  • pin_memory + num_workers=4 (64 GB RAM позволяет)
  • CUDA prefetch: перекрытие data loading и compute
  • Large batch через gradient accumulation
  • Sequence length curriculum (256 → 512 → 1024)

Pipeline:

  Phase 1: Reflex Classifier        (~2 мин,  15 epochs, batch=128)
  Phase 2: RRN Spine Router         (~2 мин,  15 epochs, batch=64)
  Phase 3: MinGRU Synapses          (~3 мин,  8 epochs, 512d×6L)
  Phase 4: SNN Spiking Synapses     (~3 мин,  8 epochs, dim=512)
  Phase 5: Mamba-2 Brain P1         (~30 мин, 3 epochs, 1024d×20L)
  Phase 6: Mamba-2 Brain P2 WKV     (~10 мин, 2 epochs)
  Phase 7: Mamba-2 Brain P3 MoLE    (~8 мин,  1 epoch)
  Phase 8: Validate                 (~2 мин)

Итого: ~60 минут.

Использование:
  python quick_full_train.py                # 4090 BEAST MODE
  python quick_full_train.py --skip-brain   # Только Spine (~10 мин)
  python quick_full_train.py --time 120     # 2 часа (больше epochs)

═══════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import math
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

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

TRAINING = ROOT / "training"
PYTHON = sys.executable
LOG_FILE = ROOT / "quick_train.log"

# ═══════════════════════════════════════════
# Args
# ═══════════════════════════════════════════

parser = argparse.ArgumentParser(description="ТАРС — Quick Full Architecture Test (BEAST MODE)")
parser.add_argument("--cpu", action="store_true", help="Force CPU")
parser.add_argument("--skip-brain", action="store_true", help="Skip Mamba-2 (only Spine)")
parser.add_argument("--skip-download", action="store_true", help="Skip data download")
parser.add_argument("--data_dir", type=str, default=None, help="Custom data directory")
parser.add_argument("--time", type=int, default=60, help="Target training time in minutes")
args = parser.parse_args()

TIME_BUDGET = args.time  # minutes


# ═══════════════════════════════════════════
# GPU Detection
# ═══════════════════════════════════════════

def _get_ram_gb():
    """Get RAM without psutil (works on Linux + Windows)."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1024**3
    except ImportError:
        pass
    # Linux fallback: /proc/meminfo
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    # Windows fallback: wmic
    try:
        import subprocess
        out = subprocess.check_output(['wmic', 'OS', 'get', 'TotalVisibleMemorySize'], text=True)
        for line in out.strip().split('\n'):
            line = line.strip()
            if line.isdigit():
                return int(line) / 1024 / 1024
    except Exception:
        pass
    return 64  # Default

def detect_gpu():
    if args.cpu:
        return "CPU", 0, _get_ram_gb(), "cpu", False
    try:
        import torch
    except ImportError:
        return "CPU", 0, _get_ram_gb(), "cpu", False
    
    if not torch.cuda.is_available():
        return "CPU", 0, _get_ram_gb(), "cpu", False
    
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    bf16 = False
    try:
        bf16 = torch.cuda.is_bf16_supported()
    except AttributeError:
        bf16 = torch.cuda.get_device_capability(0) >= (8, 0)
    ram_gb = _get_ram_gb()
    return name, vram, ram_gb, "cuda", bf16

gpu_name, vram_gb, ram_gb, device, bf16 = detect_gpu()


# ═══════════════════════════════════════════
# Config — BEAST MODE (RTX 4090 + 64 GB RAM)
# ═══════════════════════════════════════════

def get_config(vram_gb, ram_gb, time_budget_min):
    """Build config that MAXIMIZES hardware utilization."""
    
    time_scale = time_budget_min / 60.0  # 1.0 = 60 мин
    
    if vram_gb >= 20:
        # ═══ RTX 4090 / A6000 / A100 BEAST MODE ═══
        cfg = {
            "name": "4090 BEAST",
            # Brain (Mamba-2)
            "d_model": 1024,
            "n_layers": 20,
            "batch": 10,           # 4090 с grad_ckpt держит 10
            "accum": 4,            # effective = 40
            "seq_len_start": 512,
            "seq_len_mid": 768,
            "seq_len_max": 1024,
            "brain_epochs_p1": max(2, int(3 * time_scale)),
            "brain_epochs_p2": max(1, int(2 * time_scale)),
            "brain_epochs_p3": max(1, int(1 * time_scale)),
            "brain_lr_p1": 3e-4,
            "brain_lr_p2": 1e-4,
            "brain_lr_p3": 5e-5,
            # Spine
            "reflex_epochs": 15,
            "rrn_epochs": 15,
            "mingru_dim": 512,
            "mingru_layers": 6,
            "mingru_epochs": max(5, int(8 * time_scale)),
            # SNN
            "snn_dim": 512,
            "snn_heads": 8,
            "snn_epochs": max(5, int(8 * time_scale)),
            "snn_batch": 64,
            "snn_seq": 256,
            # Workers (64 GB RAM)
            "num_workers": 4 if ram_gb >= 32 else 2,
            "pin_memory": True,
            # Optimizations
            "compile": True,
            "grad_ckpt": True,
            "bf16": True,
            "use_muon": True,
            "use_wsd": True,
        }
    elif vram_gb >= 14:
        # ═══ RTX 3090 / A5000 ═══
        cfg = {
            "name": "3090 FULL",
            "d_model": 768,
            "n_layers": 16,
            "batch": 6,
            "accum": 4,
            "seq_len_start": 384,
            "seq_len_mid": 512,
            "seq_len_max": 768,
            "brain_epochs_p1": max(2, int(2 * time_scale)),
            "brain_epochs_p2": max(1, int(1 * time_scale)),
            "brain_epochs_p3": 1,
            "brain_lr_p1": 3e-4,
            "brain_lr_p2": 1e-4,
            "brain_lr_p3": 5e-5,
            "reflex_epochs": 15,
            "rrn_epochs": 15,
            "mingru_dim": 384,
            "mingru_layers": 4,
            "mingru_epochs": max(3, int(5 * time_scale)),
            "snn_dim": 384,
            "snn_heads": 4,
            "snn_epochs": max(3, int(5 * time_scale)),
            "snn_batch": 48,
            "snn_seq": 192,
            "num_workers": 4 if ram_gb >= 32 else 2,
            "pin_memory": True,
            "compile": True,
            "grad_ckpt": True,
            "bf16": True,
            "use_muon": False,
            "use_wsd": True,
        }
    elif vram_gb >= 6:
        # ═══ RTX 3060 / 2060 ═══
        cfg = {
            "name": "3060 MED",
            "d_model": 512,
            "n_layers": 8,
            "batch": 4,
            "accum": 4,
            "seq_len_start": 256,
            "seq_len_mid": 384,
            "seq_len_max": 512,
            "brain_epochs_p1": 2,
            "brain_epochs_p2": 1,
            "brain_epochs_p3": 1,
            "brain_lr_p1": 3e-4,
            "brain_lr_p2": 1e-4,
            "brain_lr_p3": 5e-5,
            "reflex_epochs": 10,
            "rrn_epochs": 10,
            "mingru_dim": 256,
            "mingru_layers": 4,
            "mingru_epochs": 5,
            "snn_dim": 256,
            "snn_heads": 4,
            "snn_epochs": 5,
            "snn_batch": 32,
            "snn_seq": 128,
            "num_workers": 2,
            "pin_memory": True,
            "compile": False,
            "grad_ckpt": True,
            "bf16": False,
            "use_muon": False,
            "use_wsd": False,
        }
    else:
        # ═══ CPU ═══
        cfg = {
            "name": "CPU MIN",
            "d_model": 256,
            "n_layers": 4,
            "batch": 2,
            "accum": 8,
            "seq_len_start": 128,
            "seq_len_mid": 192,
            "seq_len_max": 256,
            "brain_epochs_p1": 1,
            "brain_epochs_p2": 1,
            "brain_epochs_p3": 0,
            "brain_lr_p1": 3e-4,
            "brain_lr_p2": 1e-4,
            "brain_lr_p3": 5e-5,
            "reflex_epochs": 5,
            "rrn_epochs": 5,
            "mingru_dim": 128,
            "mingru_layers": 2,
            "mingru_epochs": 3,
            "snn_dim": 128,
            "snn_heads": 2,
            "snn_epochs": 3,
            "snn_batch": 16,
            "snn_seq": 64,
            "num_workers": 0,
            "pin_memory": False,
            "compile": False,
            "grad_ckpt": False,
            "bf16": False,
            "use_muon": False,
            "use_wsd": False,
        }
    
    return cfg

CFG = get_config(vram_gb, ram_gb, TIME_BUDGET)


# ═══════════════════════════════════════════
# CUDA Environment Tuning
# ═══════════════════════════════════════════

def setup_cuda_env():
    """Максимальная утилизация CUDA."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    if device == "cuda":
        # TF32 — 8× faster MatMul on Ampere+
        env["NVIDIA_TF32_OVERRIDE"] = "1"
        # cuDNN autotuner
        env["TORCH_CUDNN_V8_API_ENABLED"] = "1"
        # Memory allocator: expandable segments (less OOM, better reuse)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Number of CPU threads for DataLoader
        cpu_count = os.cpu_count() or 8
        env["TARS_THREADS"] = str(min(cpu_count, 8))
        # Compile mode
        if CFG["compile"]:
            env["TARS_FORCE_COMPILE"] = "1"
    
    return env

CUDA_ENV = setup_cuda_env()


# ═══════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════

def run(cmd, label="", timeout=None):
    """Запустить команду с логированием и CUDA env."""
    cmd = [str(c) for c in cmd]
    if len(cmd) >= 2 and 'python' in cmd[0].lower():
        if '-u' not in cmd:
            cmd.insert(1, '-u')
    
    cmd_short = " ".join(cmd)[:140]
    print(f"  → [{label}] {cmd_short}...")
    sys.stdout.flush()
    
    with open(LOG_FILE, 'a', encoding='utf-8') as log:
        log.write(f"\n{'='*60}\n[{datetime.now()}] {' '.join(cmd)}\n{'='*60}\n")
    
    t = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout, env=CUDA_ENV)
        elapsed = time.time() - t
        ok = result.returncode == 0
        icon = "✅" if ok else "❌"
        print(f"  {icon} [{label}] {'Done' if ok else 'FAILED'} ({elapsed:.0f}s)")
        return ok
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t
        print(f"  ⏰ [{label}] Таймаут ({elapsed:.0f}s / {timeout}s)")
        return False
    except KeyboardInterrupt:
        print(f"\n  ⏸ Прервано. Ctrl+C ещё раз для полной остановки.")
        return False
    except Exception as e:
        print(f"  ❌ [{label}] {e}")
        return False


# ═══════════════════════════════════════════
# VRAM Monitor
# ═══════════════════════════════════════════

def vram_status():
    """Показать текущее использование VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"{used:.1f}/{total:.1f} GB"
    except Exception:
        pass
    return "N/A"


# ═══════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════

def main():
    t0 = time.time()
    results = {}
    data_dir = args.data_dir or str(ROOT / "data")
    
    est_params = CFG["d_model"] ** 2 * CFG["n_layers"] * 12
    eff_batch = CFG["batch"] * CFG["accum"]
    
    print()
    print("═" * 65)
    print("  🤖 ТАРС — BEAST MODE (Maximum Hardware Utilization)")
    print("═" * 65)
    print()
    print(f"  🎮 GPU:        {gpu_name or 'CPU'}")
    print(f"  💾 VRAM:       {vram_gb:.1f} GB")
    print(f"  🧠 RAM:        {ram_gb:.0f} GB")
    print(f"  📐 Brain:      {CFG['d_model']}d × {CFG['n_layers']}L (~{est_params/1e6:.0f}M params)")
    print(f"  📦 Batch:      {CFG['batch']} × {CFG['accum']} = {eff_batch} effective")
    print(f"  📏 SeqLen:     {CFG['seq_len_start']} → {CFG['seq_len_mid']} → {CFG['seq_len_max']}")
    print(f"  ⚡ bf16:       {'ON' if CFG['bf16'] else 'OFF'}")
    print(f"  🔧 compile:    {'max-autotune' if CFG['compile'] else 'OFF'}")
    print(f"  📉 grad_ckpt:  {'ON' if CFG['grad_ckpt'] else 'OFF'}")
    print(f"  🚀 Muon:       {'ON' if CFG['use_muon'] else 'OFF'}")
    print(f"  📈 WSD:        {'ON' if CFG['use_wsd'] else 'OFF'}")
    print(f"  👷 Workers:    {CFG['num_workers']}")
    print(f"  ⏱  Budget:     {TIME_BUDGET} мин")
    print(f"  ⚡ SNN:        {CFG['snn_dim']}d × {CFG['snn_heads']}H, batch={CFG['snn_batch']}")
    print(f"  🧪 MinGRU:     {CFG['mingru_dim']}d × {CFG['mingru_layers']}L")
    print()
    
    # CUDA env info
    if device == "cuda":
        print(f"  🔧 CUDA Tuning:")
        print(f"     TF32:           ON (8× faster MatMul)")
        print(f"     cuDNN autotune: ON")
        print(f"     Alloc:          expandable_segments")
        print(f"     Threads:        {CUDA_ENV.get('TARS_THREADS', 'auto')}")
        print()
    
    print("─" * 65)
    
    # ════════════════════════════════════════
    # Phase 0: Download ALL Data (like local_train.py)
    # ════════════════════════════════════════
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    if not args.skip_download:
        print(f"\n  📚 Phase 0: Data Download...")
        print(f"     📂 Data dir: {data_path}")
        
        # 1. HuggingFace datasets (быстрее чем Wikipedia API)
        hf_script = TRAINING / "download_hf_dataset.py"
        if hf_script.exists():
            max_count = 10000 if vram_gb >= 20 else 5000
            
            # Скачиваем каждый пресет отдельно для лучшего прогресса
            hf_presets = [
                ("instruct", "📝 Instruction tuning (OpenOrca, Alpaca, Saiga)"),
                ("code",     "💻 Code (Magicoder, StarCoder, CommitPack)"),
                ("math",     "🔢 Math & Logic (MathInstruct, GSM8K)"),
                ("thinking", "🧠 Chain-of-Thought (OpenThoughts)"),
                ("reasoning","💡 Reasoning (GigaChat, Vikhr)"),
                ("science",  "🔬 Science & Knowledge (OASST, UltraFeedback)"),
                ("dpo",      "⚖️ DPO/RLHF (Orca DPO, HH-RLHF-ru)"),
            ]
            
            for preset, desc in hf_presets:
                print(f"\n     {desc} (max {max_count:,})...")
                run([
                    PYTHON, str(hf_script),
                    "--preset", preset,
                    "--output", str(data_path),
                    "--count", str(max_count),
                ], label=f"HF-{preset}", timeout=300)
        else:
            print(f"     ⏭ download_hf_dataset.py не найден")
        
        # 2. Personality corpus
        personality = data_path / "tars_personality_mega.txt"
        if not personality.exists() or personality.stat().st_size < 10_000:
            gen_script = TRAINING / "generate_tars_corpus.py"
            if gen_script.exists():
                print(f"\n     🧠 Generating personality corpus...")
                run([PYTHON, str(gen_script)], label="Personality", timeout=300)
            else:
                print(f"     ⏭ generate_tars_corpus.py не найден")
        else:
            print(f"     ✓ Personality: {personality.stat().st_size // 1024} KB (кеш)")
        
        # 3. Synthetic STEM data
        stem_path = data_path / "synthetic_stem.jsonl"
        if not stem_path.exists():
            gen_synth = TRAINING / "generate_synthetic.py"
            if gen_synth.exists():
                print(f"\n     🔬 Generating synthetic STEM data...")
                run([PYTHON, str(gen_synth)], label="STEM", timeout=300)
        else:
            print(f"     ✓ STEM: {stem_path.stat().st_size // 1024} KB (кеш)")
    
    # Data summary
    txt_files = list(data_path.glob("*.txt")) if data_path.exists() else []
    jsonl_files = list(data_path.glob("*.jsonl")) if data_path.exists() else []
    all_data_files = txt_files + jsonl_files
    total_mb = sum(f.stat().st_size for f in all_data_files) / 1024 / 1024
    
    print(f"\n  {'─' * 50}")
    print(f"  📁 Данные ({len(all_data_files)} файлов, {total_mb:.1f} MB):")
    for f in sorted(all_data_files, key=lambda x: x.stat().st_size, reverse=True)[:10]:
        mb = f.stat().st_size / (1024 * 1024)
        if mb > 0.01:
            print(f"     {f.name:35s} {mb:8.1f} MB")
    
    # Check shards
    for shard_dir in sorted(data_path.glob("shards_*")):
        if shard_dir.is_dir():
            shard_files = list(shard_dir.glob("shard_*.txt"))
            shard_mb = sum(f.stat().st_size for f in shard_files) / 1024 / 1024
            total_mb += shard_mb
            print(f"     {shard_dir.name + '/':35s} {shard_mb:8.1f} MB ({len(shard_files)} shards)")
    
    print(f"  {'─' * 50}")
    print(f"  Итого: {total_mb:.1f} MB ({total_mb / 1024:.2f} GB)")
    
    if total_mb < 1.0:
        print(f"\n  ⚠️  МАЛО ДАННЫХ! Обучение будет слабым.")
        print(f"     Скачайте данные: python local_train.py --download-only")
        print(f"     Или укажите папку: python quick_full_train.py --data_dir /path/to/data")
    
    # ════════════════════════════════════════
    # Phase 1: Reflex Classifier
    # ════════════════════════════════════════
    print(f"\n  🔁 Phase 1: Reflex Classifier ({CFG['reflex_epochs']} epochs)...")
    
    reflex_script = TRAINING / "train_reflex.py"
    if reflex_script.exists():
        reflex_cmd = [
            PYTHON, str(reflex_script),
            "--epochs", str(CFG["reflex_epochs"]),
        ]
        if device == "cpu":
            reflex_cmd += ["--cpu"]
        ok = run(reflex_cmd, label="Reflex")
        results["reflex"] = ok
    else:
        print(f"     ⏭ Не найден — пропускаю")
        results["reflex"] = None
    
    # ════════════════════════════════════════
    # Phase 2: RRN Spine Router
    # ════════════════════════════════════════
    print(f"\n  🧬 Phase 2: RRN Spine Router ({CFG['rrn_epochs']} epochs)...")
    
    rrn_script = TRAINING / "train_rrn.py"
    if rrn_script.exists():
        rrn_cmd = [
            PYTHON, str(rrn_script),
            "--epochs", str(CFG["rrn_epochs"]),
        ]
        if device == "cpu":
            rrn_cmd += ["--cpu"]
        ok = run(rrn_cmd, label="RRN")
        results["rrn"] = ok
    else:
        print(f"     ⏭ Не найден — пропускаю")
        results["rrn"] = None
    
    # ════════════════════════════════════════
    # Phase 3: MinGRU — ENLARGED
    # ════════════════════════════════════════
    print(f"\n  🧪 Phase 3: MinGRU ({CFG['mingru_dim']}d×{CFG['mingru_layers']}L, "
          f"{CFG['mingru_epochs']} epochs)...")
    
    mingru_script = TRAINING / "train_mingru.py"
    if mingru_script.exists():
        cmd = [
            PYTHON, str(mingru_script),
            "--dim", str(CFG["mingru_dim"]),
            "--layers", str(CFG["mingru_layers"]),
            "--epochs", str(CFG["mingru_epochs"]),
            "--augment",
        ]
        ok = run(cmd, label="MinGRU")
        results["mingru"] = ok
    else:
        print(f"     ⏭ Не найден — пропускаю")
        results["mingru"] = None
    
    # ════════════════════════════════════════
    # Phase 4: SNN Spiking Synapses — BEAST
    # ════════════════════════════════════════
    print(f"\n  ⚡ Phase 4: SNN ({CFG['snn_dim']}d×{CFG['snn_heads']}H, "
          f"{CFG['snn_epochs']} epochs, batch={CFG['snn_batch']})...")
    
    snn_script = TRAINING / "train_spiking.py"
    if snn_script.exists():
        ok = run([
            PYTHON, str(snn_script),
            "--dim", str(CFG["snn_dim"]),
            "--heads", str(CFG["snn_heads"]),
            "--beta", "0.9",
            "--epochs", str(CFG["snn_epochs"]),
            "--batch", str(CFG["snn_batch"]),
            "--seq_len", str(CFG["snn_seq"]),
            "--lr", "3e-4",
            "--device", device,
        ], label="SNN")
        results["snn"] = ok
    else:
        print(f"     ⏭ Не найден — пропускаю")
        results["snn"] = None
    
    # ════════════════════════════════════════
    # Phase 5-7: Mamba-2 Brain — BEAST MODE
    # ════════════════════════════════════════
    if not args.skip_brain:
        mamba_script = TRAINING / "train_mamba2.py"
        
        if mamba_script.exists():
            phases = [
                (1, "Full Pretrain (SSD+WKV+Ω-SSM+MoLE)", 
                 CFG["brain_epochs_p1"], CFG["brain_lr_p1"], CFG["seq_len_start"]),
                (2, "WKV + Fusion Fine-tune", 
                 CFG["brain_epochs_p2"], CFG["brain_lr_p2"], CFG["seq_len_mid"]),
            ]
            
            if CFG["brain_epochs_p3"] > 0:
                phases.append(
                    (3, "MoLE + MatrixPool + WaveMerge",
                     CFG["brain_epochs_p3"], CFG["brain_lr_p3"], CFG["seq_len_max"])
                )
            
            for phase_num, desc, epochs, lr, seq_len in phases:
                print(f"\n  🧠 Phase {4+phase_num}: Mamba-2 P{phase_num} — {desc}")
                print(f"     ({CFG['d_model']}d×{CFG['n_layers']}L, {epochs}ep, "
                      f"seq={seq_len}, batch={CFG['batch']}×{CFG['accum']}={eff_batch})")
                
                cmd = [
                    PYTHON, str(mamba_script),
                    "--d_model", str(CFG["d_model"]),
                    "--n_layers", str(CFG["n_layers"]),
                    "--vocab_size", "256",
                    "--batch", str(CFG["batch"]),
                    "--accum_steps", str(CFG["accum"]),
                    "--epochs", str(epochs),
                    "--lr", str(lr),
                    "--seq_len", str(seq_len),
                    "--phase", str(phase_num),
                    "--device", device,
                    "--curriculum",
                    "--label_smoothing", "0.1",
                    "--no_wiki",  # Wikipedia скачивается в Phase 0, не при обучении
                ]
                
                if CFG["grad_ckpt"]:
                    cmd += ["--grad_ckpt"]
                if CFG["bf16"]:
                    cmd += ["--bf16"]
                if CFG["use_muon"]:
                    cmd += ["--muon"]
                if CFG["use_wsd"]:
                    cmd += ["--wsd"]
                if phase_num > 1:
                    cmd += ["--resume"]
                if args.data_dir:
                    cmd += ["--data_dir", args.data_dir]
                
                ok = run(cmd, label=f"Mamba P{phase_num}")
                results[f"mamba_p{phase_num}"] = ok
                
                if not ok:
                    print(f"     ⚠️ P{phase_num} failed → пропускаю остальные фазы Brain")
                    break
        else:
            print(f"\n  ⏭ {mamba_script.name} не найден")
    else:
        print(f"\n  ⏭ Brain пропущен (--skip-brain)")
    
    # ════════════════════════════════════════
    # Phase 8: Validate
    # ════════════════════════════════════════
    print(f"\n  ✅ Phase 8: Validate...")
    
    quick_test = TRAINING / "quick_test.py"
    if quick_test.exists():
        ok = run([PYTHON, str(quick_test)], label="Validate", timeout=120)
        results["validate"] = ok
    else:
        try:
            import torch
            from brain.mamba2.model import TarsMamba2LM
            
            model = TarsMamba2LM(
                d_model=CFG["d_model"],
                n_layers=CFG["n_layers"],
                vocab_size=256,
            )
            x = torch.randint(0, 256, (1, 32))
            if device == "cuda":
                model = model.cuda()
                x = x.cuda()
            with torch.no_grad():
                logits = model(x)
            params = sum(p.numel() for p in model.parameters())
            print(f"     Brain: {CFG['d_model']}d×{CFG['n_layers']}L → "
                  f"logits {logits.shape}, {params:,} params")
            
            # SNN validate
            from brain.spiking import SpikingSynapsePool
            pool = SpikingSynapsePool(dim=CFG["snn_dim"], n_synapses=5, num_heads=CFG["snn_heads"])
            sx = torch.randn(1, 4, CFG["snn_dim"])
            so, ss = pool(sx, task_type="action")
            snn_params = sum(p.numel() for p in pool.parameters())
            print(f"     SNN:   {CFG['snn_dim']}d×{CFG['snn_heads']}H → "
                  f"{so.shape}, {snn_params:,} params")
            
            results["validate"] = True
            print("  ✅ [Validate] OK")
        except Exception as e:
            print(f"  ❌ [Validate] {e}")
            results["validate"] = False
    
    # ════════════════════════════════════════
    # Results
    # ════════════════════════════════════════
    elapsed = time.time() - t0
    mins = elapsed / 60
    
    print()
    print("═" * 65)
    print(f"  🤖 ТАРС BEAST MODE — Results ({mins:.0f} мин)")
    print("═" * 65)
    print()
    
    phase_names = {
        "download": "📚 Data Download",
        "reflex":   "🔁 Reflex Classifier",
        "rrn":      "🧬 RRN Spine Router",
        "mingru":   "🧪 MinGRU Synapses",
        "snn":      "⚡ SNN Spiking Synapses",
        "mamba_p1": "🧠 Mamba-2 P1 (Full Pretrain)",
        "mamba_p2": "🧠 Mamba-2 P2 (WKV Fine-tune)",
        "mamba_p3": "🧠 Mamba-2 P3 (MoLE + WaveMerge)",
        "validate": "✅ Validation",
    }
    
    for key, name in phase_names.items():
        if key in results:
            val = results[key]
            if val is None:
                icon = "⏭"
                status = "skipped"
            elif val:
                icon = "✅"
                status = "OK"
            else:
                icon = "❌"
                status = "FAILED"
            print(f"    {icon} {name}: {status}")
    
    print()
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"  📊 Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    print(f"  ⏱  Time: {mins:.1f} мин ({elapsed/3600:.1f} часов)")
    print(f"  🎮 VRAM used: {vram_status()}")
    
    if failed == 0 and passed > 0:
        print()
        print("  ═══════════════════════════════════════")
        print("  🎯 ТАРС ARCHITECTURE FULLY OPERATIONAL")
        print("  ═══════════════════════════════════════")
        print()
        print("  🧬 Nervous System:")
        print(f"    Spine:   Reflex + RRN (Mode 1/2/3)")
        print(f"    MinGRU:  {CFG['mingru_dim']}d×{CFG['mingru_layers']}L (рефлексы)")
        print(f"    SNN:     {CFG['snn_dim']}d×{CFG['snn_heads']}H, SI-LIF {{-1,0,+1}}")
        print(f"    Brain:   {CFG['d_model']}d×{CFG['n_layers']}L (Mamba-2+RWKV-7)")
        print()
        print("  Для продакшн обучения:")
        print("    python local_train.py             # Полное обучение (~15 часов)")
        print("    python local_train.py --phase 4   # Только Mamba-2")
    else:
        print()
        print("  ⚠️ Есть ошибки. Лог: quick_train.log")
    
    print()
    print("═" * 65)


if __name__ == "__main__":
    main()
