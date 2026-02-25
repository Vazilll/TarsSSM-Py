"""
train_all.py — TARS v3 Unified Training Pipeline.

Обучает все компоненты в правильной последовательности:
  Phase 0: OmegaCore C++ (опционально)
  Phase 1: Reflex Classifier (Tier 1) — 30 сек, CPU
  Phase 2: MinGRU Language Model (Tier 1) — 5 мин, CPU/GPU
  Phase 3: Mamba-2 + RWKV-7 Brain 1.58-bit (Tier 2) — 30 мин+, GPU
            Модель сразу создаётся в 1.58-bit режиме (BitNet STE)
            и обучается на ВСЕХ скачанных данных (Wiki + HF)
  Phase 4: Whisper Vocabulary Boost — контекстная настройка STT

Usage:
    python training/train_all.py                     # Всё
    python training/train_all.py --only reflex       # Только рефлексы
    python training/train_all.py --only mamba2       # Только основной мозг
    python training/train_all.py --only mingru       # Только MinGRU
    python training/train_all.py --device cuda       # GPU mode
    python training/train_all.py --data data/wiki.txt  # Свой корпус
    python training/train_all.py --phase 2           # Только Phase 2 Mamba-2
"""
import argparse
import logging
import time
import subprocess
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrainAll")

# Путь к корню проекта
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def run_script(script: str, extra_args: list = None, cwd: str = None):
    """Run a training script as subprocess."""
    cmd = [PYTHON, script] + (extra_args or [])
    logger.info(f"▶ {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info(f"✅ {os.path.basename(script)} → {elapsed:.1f}s")
    else:
        logger.error(f"❌ {os.path.basename(script)} failed (code {result.returncode})")
    return result.returncode == 0


def build_omega_core():
    """Phase 0: Compile OmegaCore C++ kernel (optional)."""
    logger.info("═" * 60)
    logger.info("PHASE 0: Building OmegaCore C++ Kernel")
    logger.info("═" * 60)
    ps1 = os.path.join(ROOT, "brain", "omega_core", "build_omega.ps1")
    if not os.path.exists(ps1):
        logger.warning("build_omega.ps1 not found — skipping C++ build")
        return True
    result = subprocess.run(
        ["powershell", "-ExecutionPolicy", "Bypass", "-File", ps1],
        cwd=ROOT
    )
    return result.returncode == 0


def train_reflex(args):
    """Phase 1: Train Tier-1 Reflex Classifier (~30s, CPU)."""
    logger.info("═" * 60)
    logger.info("PHASE 1: Reflex Classifier (MinGRU Intent)")
    logger.info("═" * 60)
    return run_script(
        os.path.join(TRAINING, "train_reflex.py"),
        ["--epochs", str(args.reflex_epochs), "--lr", str(args.reflex_lr)]
    )


def train_mingru(args):
    """Phase 2: Train MinGRU LM for fast responses (~5 min)."""
    logger.info("═" * 60)
    logger.info("PHASE 2: MinGRU Language Model (System 1)")
    logger.info("═" * 60)
    extra = [
        "--epochs", str(args.mingru_epochs),
        "--lr", str(args.mingru_lr),
    ]
    # train_mingru.py auto-detects CUDA, no --device/--data flags
    return run_script(os.path.join(TRAINING, "train_mingru.py"), extra)


def train_mamba2(args):
    """Phase 3: Train Mamba-2 + RWKV-7 Brain in 1.58-bit mode."""
    logger.info("═" * 60)
    phase = args.phase or 1
    logger.info(f"PHASE 3: Mamba-2 + RWKV-7 Brain 1.58-bit (phase {phase})")
    logger.info("  Модель сразу в 1.58-bit (BitNet STE) — без FP16 стадии")
    logger.info("═" * 60)
    extra = [
        "--epochs", str(args.mamba_epochs),
        "--lr", str(args.mamba_lr),
        "--d_model", str(args.d_model),
        "--n_layers", str(args.n_layers),
        "--phase", str(phase),
        "--quant",  # Сразу 1.58-bit!
    ]
    if args.device != "cpu":
        extra += ["--device", args.device]
    if args.data:
        extra += ["--data", args.data]
    if args.pretrained:
        extra += ["--pretrained", args.pretrained]
    return run_script(os.path.join(TRAINING, "train_mamba2.py"), extra)


def whisper_boost(args):
    """Phase 4: Build Whisper vocabulary boost from corpus."""
    logger.info("═" * 60)
    logger.info("PHASE 4: Whisper Vocabulary Boost")
    logger.info("═" * 60)
    return run_script(os.path.join(TRAINING, "whisper_boost.py"))


def quantize(args):
    """Phase 5: BitNet 1.58-bit quantization + fine-tune."""
    logger.info("═" * 60)
    logger.info("PHASE 5: BitNet 1.58-bit Quantization + Fine-Tune")
    logger.info("═" * 60)
    extra = ["--epochs", "2"]
    if args.device != "cpu":
        extra += ["--device", args.device]
    return run_script(os.path.join(TRAINING, "quantize_models.py"), extra)


def main():
    parser = argparse.ArgumentParser(
        description="TARS v3 Unified Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/train_all.py                          # Full pipeline
  python training/train_all.py --only mamba2 --phase 1  # Mamba-2 pre-train
  python training/train_all.py --only mamba2 --phase 4  # WKV RAG fine-tune
  python training/train_all.py --data data/wiki_ru.txt  # Train on Wikipedia
  python training/train_all.py --device cuda --mamba-epochs 10
        """
    )
    
    # What to train
    parser.add_argument("--only", choices=["reflex", "mingru", "mamba2", "quantize"],
                        help="Train only one component")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4],
                        help="Mamba-2 hybrid training phase (1=all, 2=WKV+Fusion, 3=MoLE, 4=RAG)")
    
    # Hardware
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    
    # Data
    parser.add_argument("--data", type=str, default=None,
                        help="Path to text corpus (.txt)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained weights for fine-tuning")
    
    # Reflex params
    parser.add_argument("--reflex-epochs", type=int, default=10)
    parser.add_argument("--reflex-lr", type=float, default=0.002)
    
    # MinGRU params
    parser.add_argument("--mingru-epochs", type=int, default=5)
    parser.add_argument("--mingru-lr", type=float, default=3e-4)
    
    # Mamba-2 params
    parser.add_argument("--mamba-epochs", type=int, default=1)
    parser.add_argument("--mamba-lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension (256=demo, 768=full)")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of TarsBlocks (4=demo, 12=full)")
    
    # Extras
    parser.add_argument("--skip-omega", action="store_true")
    parser.add_argument("--skip-quantize", action="store_true")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   TARS v3 Training Pipeline (Deep WuNeng Core)          ║")
    logger.info("╚" + "═" * 58 + "╝")
    logger.info(f"  Device:     {args.device}")
    logger.info(f"  Components: {'ALL' if not args.only else args.only.upper()}")
    logger.info(f"  Data:       {args.data or 'built-in corpus'}")
    if args.phase:
        logger.info(f"  Phase:      {args.phase}")
    logger.info("")
    
    t0 = time.time()
    results = {}
    
    # Phase 0: OmegaCore (optional)
    if not args.skip_omega and args.only is None:
        results["omega"] = build_omega_core()
    
    # Phase 1: Reflex
    if args.only in (None, "reflex"):
        results["reflex"] = train_reflex(args)
    
    # Phase 2: MinGRU
    if args.only in (None, "mingru"):
        results["mingru"] = train_mingru(args)
    
    # Phase 3: Mamba-2 + RWKV-7 (1.58-bit — сразу квантованное обучение)
    if args.only in (None, "mamba2"):
        results["mamba2"] = train_mamba2(args)
    
    # Phase 4: Whisper Vocabulary Boost
    if args.only is None:
        results["whisper"] = whisper_boost(args)
    
    # Summary
    total = time.time() - t0
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║   Training Summary                                       ║")
    logger.info("╠" + "═" * 58 + "╣")
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        logger.info(f"║   {icon} {name:<20s}                               ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info(f"║   Total time: {total:.0f}s                                      ║")
    logger.info("╚" + "═" * 58 + "╝")
    
    if all(results.values()):
        logger.info("\n🎯 All training phases completed successfully!")
        logger.info("   Run: python launch_tars.py")
    else:
        logger.error("\n⚠️ Some phases failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
