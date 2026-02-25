"""
═══════════════════════════════════════════════════════════════
  BitNet 1.58-bit Квантизация — Конвертация FP16 → 1.58-bit
═══════════════════════════════════════════════════════════════

Конвертирует обученную FP16 модель TarsMamba2LM в 1.58-bit формат.
Сжатие ~10x, скорость инференса ~3x.

Пайплайн:
  1. Загружает FP16 веса из models/brain/tars_mamba2.pt
  2. Переключает все UniversalLinear → mode="158bit"
  3. Дообучает 1-2 эпохи для адаптации к квантованным весам
  4. Сохраняет в models/brain/tars_mamba2_158bit.pt

Использование:
  python training/quantize_models.py
  python training/quantize_models.py --fine-tune --epochs 2
  python training/quantize_models.py --device cuda
"""

import os
import sys
import time
import argparse
import logging
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Tars.Quantizer")


def quantize_brain(input_path: str = None, output_path: str = None,
                   fine_tune: bool = True, epochs: int = 2,
                   device: str = "auto", lr: float = 1e-4):
    """
    Конвертирует TarsMamba2LM из FP16 в 1.58-bit.
    
    Args:
        input_path: Путь к FP16 весам
        output_path: Куда сохранить 1.58-bit
        fine_tune: Дообучить после квантизации
        epochs: Количество эпох дообучения
        device: cuda/cpu/auto
    """
    from brain.mamba2.model import TarsMamba2LM
    from brain.mamba2.bitnet import convert_model_to_158bit, model_stats
    
    if input_path is None:
        input_path = str(ROOT / "models" / "brain" / "tars_mamba2.pt")
    if output_path is None:
        output_path = str(ROOT / "models" / "brain" / "tars_mamba2_158bit.pt")
    
    if not os.path.exists(input_path):
        logger.error(f"❌ FP16 модель не найдена: {input_path}")
        logger.error("   Сначала обучите: python training/train_mamba2.py --phase 1")
        return False
    
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info("═" * 60)
    logger.info("  BitNet 1.58-bit Квантизация")
    logger.info("═" * 60)
    logger.info(f"  Вход:     {input_path}")
    logger.info(f"  Выход:    {output_path}")
    logger.info(f"  Device:   {device}")
    logger.info(f"  Fine-tune: {fine_tune} ({epochs} эпох)")
    
    # ═══ 1. Загрузка FP16 модели ═══
    logger.info("\n[1/4] Загрузка FP16 модели...")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    
    # Определяем параметры модели из чекпоинта
    config = checkpoint.get("config", {})
    model = TarsMamba2LM(
        d_model=config.get("d_model", 256),
        n_layers=config.get("n_layers", 4),
        vocab_size=config.get("vocab_size", 32000),
        quant_mode="fp16",  # Начинаем в FP16
    )
    
    # Загрузка весов
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    fp16_params = sum(p.numel() for p in model.parameters())
    fp16_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    logger.info(f"  FP16: {fp16_params:,} параметров, {fp16_mb:.1f} MB")
    
    # ═══ 2. Конвертация в 1.58-bit ═══
    logger.info("\n[2/4] Конвертация UniversalLinear → 1.58-bit...")
    convert_model_to_158bit(model)
    
    stats = model_stats(model)
    logger.info(f"  Конвертировано: {stats['universal_linear_count']} слоёв")
    logger.info(f"  Средняя разреженность: {stats['avg_sparsity']:.1%}")
    logger.info(f"  Сжатие: ~{stats['compression_ratio']:.1f}x")
    
    model = model.to(device)
    
    # ═══ 3. Дообучение (опционально) ═══
    if fine_tune:
        logger.info(f"\n[3/4] Дообучение 1.58-bit модели ({epochs} эпох)...")
        
        # Загрузка корпуса
        from training.train_mamba2 import load_corpus, load_tokenizer
        corpus = load_corpus()
        tokenizer = load_tokenizer()
        
        # Токенизация
        tokens = tokenizer.encode(corpus[:500000])  # Первые 500K символов
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        
        seq_len = 256  # Короче для fine-tune
        batch_size = 8
        
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01
        )
        
        model.train()
        total_loss = 0
        steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0
            
            indices = list(range(0, len(token_tensor) - seq_len - 1, seq_len))
            if len(indices) > 500:
                import random
                random.shuffle(indices)
                indices = indices[:500]
            
            for start in indices:
                x = token_tensor[start:start + seq_len].unsqueeze(0).to(device)
                y = token_tensor[start + 1:start + seq_len + 1].unsqueeze(0).to(device)
                
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                steps += 1
            
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"  Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f}")
        
        logger.info(f"  Дообучение завершено ({steps} шагов)")
    else:
        logger.info("\n[3/4] Дообучение пропущено")
    
    # ═══ 4. Сохранение ═══
    logger.info("\n[4/4] Сохранение 1.58-bit модели...")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    save_data = {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": model.d_model,
            "n_layers": model.n_layers,
            "vocab_size": model.vocab_size,
            "quant_mode": "158bit",
        },
        "stats": stats,
    }
    torch.save(save_data, output_path)
    
    output_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"  ✅ Сохранён: {output_path} ({output_mb:.1f} MB)")
    logger.info(f"  Сжатие: {fp16_mb:.1f} MB → {output_mb:.1f} MB ({fp16_mb / output_mb:.1f}x)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="BitNet 1.58-bit квантизация ТАРС")
    parser.add_argument("--input", type=str, default=None, help="FP16 модель")
    parser.add_argument("--output", type=str, default=None, help="Выходной файл")
    parser.add_argument("--fine-tune", action="store_true", default=True,
                        help="Дообучить после квантизации (default: yes)")
    parser.add_argument("--no-fine-tune", action="store_true",
                        help="Не дообучать")
    parser.add_argument("--epochs", type=int, default=2, help="Эпохи дообучения")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    quantize_brain(
        input_path=args.input,
        output_path=args.output,
        fine_tune=not args.no_fine_tune,
        epochs=args.epochs,
        device=args.device,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
