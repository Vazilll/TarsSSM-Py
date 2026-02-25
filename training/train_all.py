"""
train_all.py — TARS v3 Unified Training Pipeline.

Обучает все компоненты в правильной последовательности:
  Phase 0: OmegaCore C++ (опционально)
  Phase 1: Reflex Classifier (30 эпох) — ~3 мин, CPU
  Phase 2: MinGRU Language Model (20 эпох + HF данные) — ~1.5-2ч, CPU
  Phase 3: Mamba-2 Brain 1.58-bit (прогрессивное обучение):
            Phase 1→ Full pretrain (2 эпохи) → ~3-4ч
            Phase 2→ Fine-tune WKV+Fusion (1 эпоха) → ~1.5-2ч
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
import json
import shutil
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("TrainAll")

# Путь к корню проекта
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
TARS_V3_DIR = os.path.join(ROOT, "models", "tars_v3")


def consolidate_models(results: dict, total_time: float):
    """Копирует все обученные веса в models/tars_v3/ и пишет training_log.json."""
    os.makedirs(TARS_V3_DIR, exist_ok=True)
    
    # Маппинг: откуда → куда
    copies = {
        "reflex": (os.path.join(ROOT, "models", "reflex", "reflex_classifier.pt"),
                   os.path.join(TARS_V3_DIR, "reflex.pt")),
        "mingru": (os.path.join(ROOT, "models", "mingru_weights.pt"),
                   os.path.join(TARS_V3_DIR, "mingru.pt")),
        "mamba2": (os.path.join(ROOT, "models", "mamba2", "mamba2_omega_158bit.pt"),
                   os.path.join(TARS_V3_DIR, "mamba2.pt")),
    }
    
    copied = []
    for name, (src, dst) in copies.items():
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024 * 1024)
            logger.info(f"📦 {name}: {os.path.basename(src)} → tars_v3/{os.path.basename(dst)} ({size_mb:.1f} MB)")
            copied.append(name)
    
    # Записываем лог обучения
    log_path = os.path.join(TARS_V3_DIR, "training_log.json")
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": round(total_time, 1),
        "results": {k: ("ok" if v else "failed") for k, v in results.items()},
        "models_consolidated": copied,
        "encoding": "cp1251",
        "vocab_size": 256,
    }
    
    # Append to existing log
    logs = []
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception:
            pass
    if not isinstance(logs, list):
        logs = []
    logs.append(log_entry)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📋 Training log: {log_path}")


def run_script(script: str, extra_args: list = None, cwd: str = None):
    """Run a training script as subprocess with retry for Windows Defender/Launcher bugs."""
    cmd = [PYTHON, script] + (extra_args or [])
    logger.info(f"▶ {' '.join(cmd)}")
    t0 = time.time()
    
    # Retry logic for Windows process creation bugs
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, cwd=cwd or ROOT)
            
            # Python launcher bug on Windows ("Unable to create process") returns 101
            if result.returncode == 101:
                if attempt < 2:
                    logger.warning(f"⚠ Ошибка лаунчера (код 101), повтор через 3 сек... ({attempt+1}/3)")
                    time.sleep(3)
                    continue
                else:
                    logger.error(f"❌ {os.path.basename(script)} — Ошибка лаунчера после 3 попыток")
                    return False
            break
        except PermissionError:
            if attempt < 2:
                logger.warning(f"⚠ PermissionError (антивирус?), повтор через 3 сек... ({attempt+1}/3)")
                time.sleep(3)
            else:
                logger.error(f"❌ {os.path.basename(script)} — PermissionError после 3 попыток")
                return False
    
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
    """Phase 2: Train MinGRU LM for fast responses."""
    logger.info("═" * 60)
    logger.info("PHASE 2: MinGRU Language Model (System 1)")
    logger.info("  + HuggingFace augmented data for better quality")
    logger.info("═" * 60)
    extra = [
        "--epochs", str(args.mingru_epochs),
        "--lr", str(args.mingru_lr),
        "--augment",  # Возвращаем подкачку с HuggingFace!
    ]
    # Ускоренное обучение на CPU для ночного прогона (макс. качество за ~1.5 часа)
    if args.device == "cpu":
        extra += [
            "--dim", "512",       # Возвращаем 512 для System 1
            "--layers", "6",      # Полноценная глубина
            "--batch", "8",       # Экономия RAM, но высокая частота обновления градиентов
            "--seq_len", "256",   # Контекст достаточный для коротких ответов
            "--max_samples", "15000", # Максимально 15 000 примеров
        ]
    # train_mingru.py auto-detects CUDA, no --device flag
    return run_script(os.path.join(TRAINING, "train_mingru.py"), extra)


def train_mamba2(args):
    """Phase 3: Mamba-2 Brain — Progressive 1.58-bit Training.
    
    Step A: FP16 init
    Step B: Quantize -> 1.58-bit
    Step C: Phase 1 (full pretrain) + Phase 2 (fine-tune WKV/Fusion)
    """
    logger.info("═" * 60)
    logger.info("PHASE 3: Mamba-2 Brain — Progressive 1.58-bit")
    logger.info("  Phase 1: Full pretrain (all params)")
    logger.info("  Phase 2: Fine-tune WKV + Fusion (SSD frozen)")
    logger.info("═" * 60)
    
    base_extra = [
        "--d_model", str(args.d_model),
        "--n_layers", str(args.n_layers),
        "--batch", "8",       # Нормальный батч для стабильных градиентов
        "--seq_len", "256",   # Оптимальный контекст для CPU
        "--max_samples", "50000", # Ограничиваем корпус 50 000 примерами (вместо 1М)
        "--quant",            # 1.58-bit режим
    ]
    if args.device != "cpu":
        base_extra += ["--device", args.device]
    if args.data:
        base_extra += ["--data", args.data]
    if args.pretrained:
        base_extra += ["--pretrained", args.pretrained]
    
    # ═══ Transfer embedding from MinGRU → Mamba-2 ═══
    mingru_weights = os.path.join(ROOT, "models", "mingru_weights.pt")
    if os.path.exists(mingru_weights):
        logger.info("🔗 Transferring MinGRU embedding → Mamba-2 (shared cp1251 matrix)")
        try:
            import torch
            cp = torch.load(mingru_weights, map_location='cpu', weights_only=False)
            state = cp.get('model_state_dict', cp)
            # MinGRU stores shared embedding as shared_embedding.weight
            emb_key = None
            for k in state:
                if 'shared_embedding' in k or 'emb.weight' in k:
                    emb_key = k
                    break
            if emb_key:
                emb_tensor = state[emb_key]
                emb_path = os.path.join(TARS_V3_DIR, "_transfer_embedding.pt")
                os.makedirs(TARS_V3_DIR, exist_ok=True)
                torch.save(emb_tensor, emb_path)
                logger.info(f"  Saved embedding ({emb_tensor.shape}) → {emb_path}")
                # Mamba-2 will pick this up via --pretrained-emb flag
                base_extra += ["--pretrained_emb", emb_path]
        except Exception as e:
            logger.warning(f"  Embedding transfer failed: {e}")
    
    # Если задана конкретная фаза — запускаем только её
    if args.phase:
        extra = base_extra + [
            "--epochs", str(args.mamba_epochs),
            "--lr", str(args.mamba_lr),
            "--phase", str(args.phase),
        ]
        return run_script(os.path.join(TRAINING, "train_mamba2.py"), extra)
    
    # Прогрессивное обучение: Phase 1 -> Phase 2
    # Phase 1: Full pretrain (2 эпохи, полный LR)
    logger.info("── Phase 1/2: Full pretrain ──")
    extra1 = base_extra + [
        "--epochs", str(max(args.mamba_epochs - 1, 1)),
        "--lr", str(args.mamba_lr),
        "--phase", "1",
    ]
    ok1 = run_script(os.path.join(TRAINING, "train_mamba2.py"), extra1)
    
    # Phase 2: Fine-tune WKV + Fusion (1 эпоха, пониженный LR, --resume)
    logger.info("── Phase 2/2: Fine-tune WKV + Fusion ──")
    extra2 = base_extra + [
        "--epochs", "1",
        "--lr", str(args.mamba_lr * 0.3),  # Ниже LR для fine-tune
        "--phase", "2",
        "--resume",          # Продолжить с чекпоинта Phase 1
    ]
    ok2 = run_script(os.path.join(TRAINING, "train_mamba2.py"), extra2)
    
    return ok1 and ok2


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
    parser.add_argument("--reflex-epochs", type=int, default=50) # Максимум точности
    parser.add_argument("--reflex-lr", type=float, default=0.002)
    
    # MinGRU params
    parser.add_argument("--mingru-epochs", type=int, default=15) # И так учится быстро с 15к примерами
    parser.add_argument("--mingru-lr", type=float, default=3e-3)
    
    # Mamba-2 params
    parser.add_argument("--mamba-epochs", type=int, default=2) # 2 эпохи на 50к примеров (~2 часа)
    parser.add_argument("--mamba-lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension (128=test, 256=demo, 768=full)")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of TarsBlocks (2=test, 4=demo, 12=full)")
    
    # Extras
    parser.add_argument("--skip-omega", action="store_true", default=True,
                        help="Пропустить компиляцию OmegaCore (по умолчанию пропускается)")
    parser.add_argument("--build-omega", action="store_true",
                        help="Принудительно скомпилировать OmegaCore C++ ядро")
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
    
    # Phase 0: OmegaCore (optional, requires Zig)
    if args.build_omega and args.only is None:
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
    
    # ═══ Consolidate models into models/tars_v3/ ═══
    consolidate_models(results, total)
    
    if all(results.values()):
        logger.info("\n🎯 All training phases completed successfully!")
        logger.info("   Models consolidated: models/tars_v3/")
        logger.info("   Run: python launch_tars.py")
    else:
        logger.error("\n⚠️ Some phases failed. Check logs above.")


if __name__ == "__main__":
    main()
