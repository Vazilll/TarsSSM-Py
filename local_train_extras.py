"""
═══════════════════════════════════════════════════════════════════════
  ТАРС v3 — ДОПОЛНИТЕЛЬНЫЕ ФАЗЫ ОБУЧЕНИЯ
═══════════════════════════════════════════════════════════════════════

Фазы, вынесенные из основного pipeline для отдельной доработки:
  • Phase 4.6 — Second Pass (повторный проход с макс seq_len)
  • Phase 5   — Квантизация 1.58-bit
  • Phase 10  — Квантизация голоса
  • Phase 15  — Knowledge Distillation

Эти фазы НЕ являются частью основного обучения.
Они полезны для оптимизации и polish, но требуют предварительно
успешного прохождения core pipeline (Phase 0–14).

Запуск:
  python local_train_extras.py --phase 4.6         # Second Pass
  python local_train_extras.py --phase 5            # Квантизация
  python local_train_extras.py --phase 10           # Voice Quant
  python local_train_extras.py --phase 15           # Distillation
  python local_train_extras.py --all                # Все extras
═══════════════════════════════════════════════════════════════════════
"""

import os, sys, argparse, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Reuse runner and config from main local_train
from local_train import (
    PYTHON, TRAINING, MODELS, TARS_V3, LOG_FILE,
    benchmark_hardware, get_config, setup_cuda_env,
    detect_gpu, run, load_state, mark_done, is_done,
)

# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

parser = argparse.ArgumentParser(
    description="ТАРС v3 — Дополнительные фазы обучения",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--phase", type=str, default=None,
                    choices=["4.6", "5", "10", "15"],
                    help="Запустить конкретную фазу")
parser.add_argument("--all", action="store_true",
                    help="Запустить все дополнительные фазы")
parser.add_argument("--level", choices=["small", "medium", "max"], default="medium")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--data-dir", type=str, default=None)
extra_args = parser.parse_args()


# ═══════════════════════════════════════════
# Phase 4.6 — Second Pass
# ═══════════════════════════════════════════

def run_second_pass(cfg, device, bf16, state):
    """
    Повторный проход по данным с максимальным seq_len.
    Улучшает работу с длинными контекстами.
    
    ТЗ статус: Не упоминается отдельно, но помогает п.2.2 (Бесконечный контекст)
    TODO: Проверить эффективность — даёт ли реальное улучшение vs время.
    """
    if is_done(state, "second_pass"):
        print("  ⏭ Phase 4.6: уже выполнена")
        return True
    
    print(f"\n  🔄 Phase 4.6: Second Pass (seq={cfg['seq_max']}, {cfg['second_pass_ep']}ep)...")
    cmd = [PYTHON, str(TRAINING / "train_mamba2.py"),
           "--d_model", str(cfg["d_model"]), "--n_layers", str(cfg["n_layers"]),
           "--vocab_size", str(cfg["vocab_size"]),
           "--batch", str(cfg["batch"]),
           "--accum_steps", str(cfg["accum"]),
           "--epochs", str(cfg["second_pass_ep"]),
           "--lr", "5e-5", "--seq_len", str(cfg["seq_max"]),
           "--phase", "1", "--device", device,
           "--curriculum", "--label_smoothing", "0.05",
           "--grad_ckpt", "--resume", "--no_wiki"]
    if bf16: cmd += ["--bf16"]
    if cfg.get("use_muon"): cmd += ["--muon"]
    if cfg.get("use_wsd"): cmd += ["--wsd"]
    if extra_args.data_dir: cmd += ["--data_dir", extra_args.data_dir]
    
    ok = run(cmd, label="2nd Pass")
    if ok: mark_done(state, "second_pass")
    return ok


# ═══════════════════════════════════════════
# Phase 5 — Квантизация 1.58-bit
# ═══════════════════════════════════════════

def run_quantize(cfg, device, state):
    """
    Квантизация модели до 1.58-bit (BitNet).
    Уменьшает размер с ~260MB до ~60MB.
    
    ТЗ статус: Желательный (#8 — '1.58-bit deployment')
    TODO: Исправить WeightsUnpickler ошибку при загрузке квантизованных весов.
    """
    if is_done(state, "quantize"):
        print("  ⏭ Phase 5: уже выполнена")
        return True
    
    print("\n  ⚗️ Phase 5: Квантизация 1.58-bit...")
    fp16 = MODELS / "mamba2" / "mamba2_omega.pt"
    if not fp16.exists():
        print("  ⚠ FP16 модель не найдена — пропуск")
        return False
    
    try:
        import torch
        ck = torch.load(str(fp16), map_location="cpu", weights_only=True)
        c = ck.get("config", {})
        dm = str(c.get("d_model", cfg["d_model"]))
        nl = str(c.get("n_layers", cfg["n_layers"]))
        del ck
    except Exception:
        dm, nl = str(cfg["d_model"]), str(cfg["n_layers"])
    
    ok = run([PYTHON, str(TRAINING / "train_mamba2.py"),
              "--d_model", dm, "--n_layers", nl,
              "--batch", str(cfg["batch"]), "--accum_steps", str(cfg["accum"]),
              "--epochs", "3", "--lr", "5e-5", "--phase", "1",
              "--quant", "--resume", "--device", device,
              "--seq_len", "256", "--label_smoothing", "0.1",
              "--no_wiki", "--no_compile"], label="Quantize")
    if ok: mark_done(state, "quantize")
    return ok


# ═══════════════════════════════════════════
# Phase 10 — Квантизация голоса
# ═══════════════════════════════════════════

def run_voice_quant(device, state):
    """
    Квантизация и оптимизация голосовых моделей (Whisper + Piper).
    
    ТЗ статус: Не упоминается
    TODO: Воссоздать скрипты whisper_boost.py и quantize_voice.py
    """
    if is_done(state, "voice_quant"):
        print("  ⏭ Phase 10: уже выполнена")
        return True
    
    print("\n  🎙 Phase 10: Квантизация голоса...")
    
    whisper_boost = TRAINING / "whisper_boost.py"
    quantize_voice = TRAINING / "quantize_voice.py"
    
    if not whisper_boost.exists():
        print("  ⚠ whisper_boost.py не найден — нужно создать")
        return False
    if not quantize_voice.exists():
        print("  ⚠ quantize_voice.py не найден — нужно создать")
        return False
    
    run([PYTHON, str(whisper_boost)], label="HotWords")
    ok = run([PYTHON, str(quantize_voice)], label="VoiceQ")
    if ok: mark_done(state, "voice_quant")
    return ok


# ═══════════════════════════════════════════
# Phase 15 — Knowledge Distillation
# ═══════════════════════════════════════════

def run_distillation(cfg, device, bf16, state):
    """
    Knowledge Distillation от teacher-модели (Qwen2.5-1.5B).
    Передаёт знания от сильного LLM в лёгкий ТАРС.
    
    ТЗ статус: Не упоминается
    TODO: Оценить целесообразность — достаточно ли данных для хорошей дистилляции?
    """
    if is_done(state, "distill"):
        print("  ⏭ Phase 15: уже выполнена")
        return True
    
    print(f"\n  🎓 Phase 15: Knowledge Distillation ({cfg['distill_ep']} epochs)...")
    ok = run([PYTHON, str(TRAINING / "train_distill.py"),
              "--d_model", str(cfg["d_model"]), "--n_layers", str(cfg["n_layers"]),
              "--epochs", str(cfg["distill_ep"]), "--batch", str(cfg["batch"]),
              "--device", device, "--save_dir", str(TARS_V3),
              "--resume", "--grad_ckpt", "--lr", "1e-4",
              "--temperature", "3.0", "--alpha", "0.7",
              "--teacher_model", "Qwen/Qwen2.5-1.5B"]
             + (["--bf16"] if bf16 else []), label="Distill")
    if ok: mark_done(state, "distill")
    return ok


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════

def main():
    import local_train
    
    print()
    print("═" * 65)
    print("  🧪 ТАРС v3 — ДОПОЛНИТЕЛЬНЫЕ ФАЗЫ")
    print("═" * 65)
    
    hw = benchmark_hardware()
    device = hw["device"]
    bf16 = hw["bf16"]
    cfg = get_config(extra_args.level, hw)
    state = load_state()
    
    local_train.CUDA_ENV = setup_cuda_env(device, cfg)
    
    results = {}
    t0 = time.time()
    
    phases_to_run = []
    if extra_args.all:
        phases_to_run = ["4.6", "5", "10", "15"]
    elif extra_args.phase:
        phases_to_run = [extra_args.phase]
    else:
        print("\n  Используйте --phase <номер> или --all")
        print("  Доступные фазы: 4.6, 5, 10, 15")
        return
    
    for phase in phases_to_run:
        if phase == "4.6":
            results["second_pass"] = run_second_pass(cfg, device, bf16, state)
        elif phase == "5":
            results["quantize"] = run_quantize(cfg, device, state)
        elif phase == "10":
            results["voice_quant"] = run_voice_quant(device, state)
        elif phase == "15":
            results["distill"] = run_distillation(cfg, device, bf16, state)
    
    elapsed = time.time() - t0
    print(f"\n{'═' * 65}")
    print(f"  🧪 Extras завершены за {elapsed / 60:.1f} мин")
    for name, ok in results.items():
        print(f"    {'✅' if ok else '❌'} {name}")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
