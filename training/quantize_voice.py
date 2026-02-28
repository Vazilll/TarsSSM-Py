"""
═══════════════════════════════════════════════════════════════
  quantize_voice.py — INT8 квантизация голосовых ONNX-моделей
═══════════════════════════════════════════════════════════════

Квантизирует Whisper ONNX и Piper ONNX в INT8 для снижения
размера и ускорения инференса на CPU.

  Whisper encoder: 37 MB → ~10 MB
  Whisper decoder: 114 MB → ~30 MB
  Piper voice:     63 MB → ~16 MB

Использование:
  python training/quantize_voice.py
"""

import os
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOICE_DIR = ROOT / "models" / "voice"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Quantize.Voice")


def quantize_onnx(input_path: str, output_path: str, model_name: str) -> bool:
    """Квантизирует ONNX модель в INT8 (dynamic quantization)."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("onnxruntime не установлен. pip install onnxruntime")
        return False

    if not os.path.exists(input_path):
        logger.info(f"  ⏭ {model_name}: не найден ({input_path})")
        return False

    if os.path.exists(output_path):
        in_size = os.path.getsize(input_path) / 1024 / 1024
        out_size = os.path.getsize(output_path) / 1024 / 1024
        logger.info(f"  ✅ {model_name}: уже квантован ({in_size:.1f} → {out_size:.1f} MB)")
        return True

    in_size = os.path.getsize(input_path) / 1024 / 1024
    logger.info(f"  🔧 {model_name}: {in_size:.1f} MB → INT8...")

    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QInt8,
        )
        out_size = os.path.getsize(output_path) / 1024 / 1024
        ratio = (1 - out_size / in_size) * 100
        logger.info(f"  ✅ {model_name}: {in_size:.1f} → {out_size:.1f} MB (-{ratio:.0f}%)")
        return True
    except Exception as e:
        logger.error(f"  ❌ {model_name}: {e}")
        return False


def main():
    logger.info("═" * 60)
    logger.info("  Квантизация голосовых ONNX-моделей (INT8)")
    logger.info("═" * 60)

    results = {}

    # ═══ Whisper ONNX (encoder + decoder) ═══
    whisper_dir = VOICE_DIR / "whisper"
    if whisper_dir.exists():
        # Поддержка всех размеров: tiny, base, small
        whisper_found = False
        for model_size in ["tiny", "base", "small"]:
            enc_path = whisper_dir / f"{model_size}-encoder.onnx"
            dec_path = whisper_dir / f"{model_size}-decoder.onnx"
            if enc_path.exists():
                results[f"whisper_{model_size}_encoder"] = quantize_onnx(
                    str(enc_path),
                    str(whisper_dir / f"{model_size}-encoder-int8.onnx"),
                    f"Whisper {model_size} Encoder",
                )
                whisper_found = True
            if dec_path.exists():
                results[f"whisper_{model_size}_decoder"] = quantize_onnx(
                    str(dec_path),
                    str(whisper_dir / f"{model_size}-decoder-int8.onnx"),
                    f"Whisper {model_size} Decoder",
                )
                whisper_found = True
        if not whisper_found:
            logger.info("  ⏭ Whisper ONNX не найден в whisper/ (LoRA режим)")
    else:
        logger.info("  ⏭ Whisper ONNX не найден (используется faster-whisper)")

    # ═══ Piper ONNX ═══
    # Проверяем разные имена файлов
    piper_candidates = [
        "voice.onnx",
        "tars_voice_ru.onnx",
        "ru_RU-irina-medium.onnx",
    ]
    for piper_name in piper_candidates:
        piper_path = VOICE_DIR / piper_name
        if piper_path.exists():
            out_name = piper_name.replace(".onnx", "-int8.onnx")
            results[f"piper_{piper_name}"] = quantize_onnx(
                str(piper_path),
                str(VOICE_DIR / out_name),
                f"Piper ({piper_name})",
            )

    # ═══ Итоги ═══
    total_before = 0
    total_after = 0
    for f in VOICE_DIR.rglob("*.onnx"):
        if "int8" not in f.name:
            total_before += f.stat().st_size
        else:
            total_after += f.stat().st_size

    logger.info("")
    logger.info(f"  📊 Итого ONNX: {total_before / 1024 / 1024:.0f} MB (оригинал)")
    if total_after > 0:
        logger.info(f"  📊 Итого INT8: {total_after / 1024 / 1024:.0f} MB (квантованные)")

    ok = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info(f"\n  ✅ Квантовано: {ok}/{total} моделей")

    return ok > 0


if __name__ == "__main__":
    main()
