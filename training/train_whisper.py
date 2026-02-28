"""
═══════════════════════════════════════════════════════════════
  train_whisper.py — Дообучение Whisper Tiny для русского (LoRA)
═══════════════════════════════════════════════════════════════

Использует LoRA адаптеры → обучается только ~5% параметров.
Данные: Common Voice Russian (HuggingFace).

Использование:
  python training/train_whisper.py                    # 5000 примеров, 3 эпохи
  python training/train_whisper.py --samples 10000    # больше данных
  python training/train_whisper.py --epochs 5         # больше эпох
"""

import os
import sys
import argparse
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Whisper.Train")


def parse_args():
    p = argparse.ArgumentParser(description="Whisper LoRA Fine-tune (Russian)")
    p.add_argument("--model", type=str, default="tiny", choices=["tiny", "base", "small"],
                    help="Размер базовой модели Whisper (tiny=39M, base=74M, small=244M)")
    p.add_argument("--samples", type=int, default=5000, help="Кол-во обучающих примеров")
    p.add_argument("--val_samples", type=int, default=500, help="Кол-во валидационных примеров")
    p.add_argument("--epochs", type=int, default=3, help="Кол-во эпох")
    p.add_argument("--batch", type=int, default=16, help="Размер батча")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    p.add_argument("--output", type=str, default=None, help="Путь для сохранения")
    p.add_argument("--device", type=str, default="auto", help="cpu/cuda/auto")
    return p.parse_args()


def train(args):
    import torch
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset, Audio
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    output_dir = args.output or str(ROOT / "models" / "voice" / "whisper_ru_lora")
    os.makedirs(output_dir, exist_ok=True)

    # ═══ Device ═══
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    fp16 = device == "cuda"
    logger.info(f"Device: {device}, FP16: {fp16}")

    # ═══ Модель ═══
    model_name = f"openai/whisper-{args.model}"
    logger.info(f"Загрузка {model_name}...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name)

    # Принудительно русский язык
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="russian", task="transcribe"
    )
    model.config.suppress_tokens = []

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Whisper {args.model}: {total_params:,} параметров")

    # ═══ LoRA ═══
    logger.info(f"Применение LoRA (r={args.lora_r})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="SEQ_2_SEQ_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ═══ Данные ═══
    logger.info(f"Загрузка Common Voice Russian ({args.samples} train, {args.val_samples} val)...")
    ds_train, ds_val = None, None
    
    # Попытка 1: Common Voice 17
    try:
        ds_train = load_dataset(
            "mozilla-foundation/common_voice_17_0", "ru",
            split=f"train[:{args.samples}]",
        )
        ds_val = load_dataset(
            "mozilla-foundation/common_voice_17_0", "ru",
            split=f"validation[:{args.val_samples}]",
        )
        logger.info("✅ Common Voice 17 загружен")
    except Exception as e:
        logger.warning(f"Common Voice 17 недоступен: {e}")
    
    # Попытка 2: Common Voice 11
    if ds_train is None:
        try:
            logger.info("Пробуем Common Voice 11...")
            ds_train = load_dataset(
                "mozilla-foundation/common_voice_11_0", "ru",
                split=f"train[:{args.samples}]",
            )
            ds_val = load_dataset(
                "mozilla-foundation/common_voice_11_0", "ru",
                split=f"validation[:{args.val_samples}]",
            )
            logger.info("✅ Common Voice 11 загружен")
        except Exception as e:
            logger.warning(f"Common Voice 11 недоступен: {e}")
    
    # Попытка 3: Google FLEURS (всегда доступен)
    if ds_train is None:
        try:
            logger.info("Пробуем Google FLEURS (ru_ru)...")
            ds_train = load_dataset(
                "google/fleurs", "ru_ru",
                split=f"train[:{args.samples}]",
            )
            ds_val = load_dataset(
                "google/fleurs", "ru_ru",
                split=f"validation[:{args.val_samples}]",
            )
            # FLEURS uses "transcription" instead of "sentence"
            ds_train = ds_train.rename_column("transcription", "sentence")
            ds_val = ds_val.rename_column("transcription", "sentence")
            logger.info("✅ Google FLEURS загружен")
        except Exception as e:
            logger.error(f"FLEURS тоже недоступен: {e}")
            logger.error("❌ Нет доступных аудио-датасетов. Пропуск Whisper.")
            return

    # Ресемплинг до 16kHz
    ds_train = ds_train.cast_column("audio", Audio(sampling_rate=16000))
    ds_val = ds_val.cast_column("audio", Audio(sampling_rate=16000))

    logger.info(f"Данные: train={len(ds_train)}, val={len(ds_val)}")

    # ═══ Препроцессинг ═══
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=16000, return_tensors="pt"
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    logger.info("Препроцессинг данных...")
    ds_train = ds_train.map(prepare_dataset, remove_columns=ds_train.column_names)
    ds_val = ds_val.map(prepare_dataset, remove_columns=ds_val.column_names)

    # ═══ Data Collator ═══
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any
        decoder_start_token_id: int

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # ═══ Обучение ═══
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"  Whisper {args.model} LoRA Fine-tune (Russian)")
    logger.info(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    logger.info(f"{'='*60}\n")

    trainer.train()

    # ═══ Сохранение ═══
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    logger.info(f"\n✅ Whisper LoRA сохранён: {output_dir}")
    logger.info(f"   Для использования: загрузите с peft.PeftModel.from_pretrained()")


if __name__ == "__main__":
    args = parse_args()
    train(args)

