"""
═══════════════════════════════════════════════════════════════
  train_piper.py — Дообучение Piper TTS для русского голоса
═══════════════════════════════════════════════════════════════

Fine-tune существующей русской модели Piper (VITS) на корпусе RUSLAN.
Результат: более естественный русский голос с правильной интонацией.

Использование:
  python training/train_piper.py                  # стандарт
  python training/train_piper.py --epochs 2000    # больше эпох
  python training/train_piper.py --skip-download  # данные уже есть
"""

import os
import sys
import csv
import argparse
import logging
import subprocess
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Piper.Train")

RUSLAN_URL = "https://ruslan-corpus.github.io/data/ruslan_dataset.tar.gz"
VOICE_DIR = ROOT / "models" / "voice"


def parse_args():
    p = argparse.ArgumentParser(description="Piper TTS Fine-tune (Russian)")
    p.add_argument("--epochs", type=int, default=1000, help="Макс. эпох обучения")
    p.add_argument("--batch", type=int, default=16, help="Размер батча")
    p.add_argument("--max_samples", type=int, default=3000, help="Макс. аудио-примеров")
    p.add_argument("--skip-download", action="store_true", help="Данные уже скачаны")
    p.add_argument("--data_dir", type=str, default=None, help="Путь к данным LJSpeech формата")
    p.add_argument("--checkpoint", type=str, default=None, help="Продолжить обучение")
    return p.parse_args()


def download_ruslan(data_dir: Path):
    """Скачивает RUSLAN corpus (31ч русской речи)."""
    if (data_dir / "wavs").exists():
        n_wavs = len(list((data_dir / "wavs").glob("*.wav")))
        if n_wavs > 100:
            logger.info(f"RUSLAN уже скачан: {n_wavs} файлов")
            return True

    logger.info("Скачивание RUSLAN corpus (~3 GB)...")
    data_dir.mkdir(parents=True, exist_ok=True)

    tar_path = data_dir / "ruslan.tar.gz"
    try:
        import urllib.request
        urllib.request.urlretrieve(RUSLAN_URL, str(tar_path))
        logger.info("Распаковка...")
        import tarfile
        with tarfile.open(str(tar_path)) as tf:
            tf.extractall(str(data_dir))
        tar_path.unlink(missing_ok=True)
        logger.info("✅ RUSLAN скачан и распакован")
        return True
    except Exception as e:
        logger.error(f"❌ Не удалось скачать RUSLAN: {e}")
        logger.info("Попробуйте скачать вручную и указать --data_dir")
        return False


def prepare_ljspeech_format(data_dir: Path, max_samples: int) -> int:
    """Конвертирует данные в формат LJSpeech (metadata.csv + wavs/)."""
    metadata_path = data_dir / "metadata.csv"

    # Если metadata.csv уже есть
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            count = sum(1 for _ in f)
        if count > 100:
            logger.info(f"metadata.csv уже готов: {count} записей")
            return count

    wavs_dir = data_dir / "wavs"
    texts_dir = data_dir / "texts"

    if not wavs_dir.exists():
        # Ищем wav-файлы рекурсивно
        all_wavs = list(data_dir.rglob("*.wav"))
        if not all_wavs:
            logger.error(f"WAV файлы не найдены в {data_dir}")
            return 0
        wavs_dir.mkdir(exist_ok=True)
        for w in all_wavs[:max_samples]:
            shutil.copy2(str(w), str(wavs_dir / w.name))

    entries = []
    wav_files = sorted(wavs_dir.glob("*.wav"))[:max_samples]

    for wav_path in wav_files:
        stem = wav_path.stem
        # Ищем текст
        txt_path = None
        for ext in [".txt", ".lab"]:
            candidate = texts_dir / f"{stem}{ext}" if texts_dir.exists() else None
            if candidate and candidate.exists():
                txt_path = candidate
                break

        if txt_path:
            text = txt_path.read_text(encoding='utf-8').strip()
            if text:
                entries.append(f"{stem}|{text}|{text}")

    if not entries:
        logger.error("Не удалось создать metadata.csv — нет пар wav+txt")
        return 0

    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(entries))

    logger.info(f"✅ metadata.csv: {len(entries)} записей")
    return len(entries)


def train(args):
    """Основной цикл обучения Piper."""

    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data" / "piper_ru"

    # ═══ Скачивание данных ═══
    if not args.skip_download:
        if not download_ruslan(data_dir):
            logger.warning("Используем встроенные данные (если есть)")

    # ═══ Подготовка формата ═══
    n_samples = prepare_ljspeech_format(data_dir, args.max_samples)
    if n_samples == 0:
        logger.error("❌ Нет данных для обучения")
        logger.info("Укажите --data_dir с папкой содержащей wavs/ и texts/")
        return False

    # ═══ Проверка piper_train ═══
    try:
        import piper_train
        has_piper_train = True
    except ImportError:
        has_piper_train = False
        logger.warning("piper_train не установлен")
        logger.info("Установка: pip install piper-tts piper-phonemize")

    if not has_piper_train:
        # Устанавливаем
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "piper-tts", "piper-phonemize", "-q"],
            check=False,
        )

    # ═══ Preprocessing ═══
    logger.info("Preprocessing (фонемизация)...")
    preprocess_dir = data_dir / "preprocessed"
    preprocess_dir.mkdir(exist_ok=True)

    result = subprocess.run(
        [sys.executable, "-m", "piper_train.preprocess",
         "--language", "ru",
         "--input-dir", str(data_dir),
         "--output-dir", str(preprocess_dir),
         "--dataset-format", "ljspeech",
         "--sample-rate", "22050",
         ],
        check=False,
    )
    if result.returncode != 0:
        logger.error("❌ Preprocessing не удался")
        return False

    # ═══ Обучение VITS ═══
    logger.info(f"\n{'='*60}")
    logger.info(f"  Piper TTS Training (VITS, Russian)")
    logger.info(f"  Epochs: {args.epochs} | Batch: {args.batch}")
    logger.info(f"  Data: {n_samples} samples")
    logger.info(f"{'='*60}\n")

    train_cmd = [
        sys.executable, "-m", "piper_train",
        "--dataset-dir", str(preprocess_dir),
        "--batch-size", str(args.batch),
        "--validation-split", "0.05",
        "--max-epochs", str(args.epochs),
        "--checkpoint-epochs", "100",
        "--precision", "16",
    ]

    # GPU
    import torch
    if torch.cuda.is_available():
        train_cmd += ["--accelerator", "gpu", "--devices", "1"]
    else:
        train_cmd += ["--accelerator", "cpu"]

    if args.checkpoint:
        train_cmd += ["--resume_from_checkpoint", args.checkpoint]

    result = subprocess.run(train_cmd, check=False)
    if result.returncode != 0:
        logger.error("❌ Обучение Piper не завершилось. Проверьте логи.")
        return False

    # ═══ Экспорт в ONNX ═══
    logger.info("Экспорт в ONNX...")
    ckpt_dir = preprocess_dir / "lightning_logs"
    last_ckpt = None
    for version_dir in sorted(ckpt_dir.glob("version_*"), reverse=True):
        candidates = list((version_dir / "checkpoints").glob("*.ckpt"))
        if candidates:
            last_ckpt = str(candidates[-1])
            break

    if last_ckpt:
        output_onnx = str(VOICE_DIR / "tars_voice_ru.onnx")
        result = subprocess.run(
            [sys.executable, "-m", "piper_train.export_onnx",
             last_ckpt, output_onnx],
            check=False,
        )
        if result.returncode == 0:
            size_mb = os.path.getsize(output_onnx) / 1024 / 1024
            logger.info(f"✅ Piper ONNX: {output_onnx} ({size_mb:.1f} MB)")
            return True
        else:
            logger.error("❌ Экспорт ONNX не удался")
    else:
        logger.error("❌ Чекпоинт не найден")

    return False


if __name__ == "__main__":
    args = parse_args()
    train(args)
