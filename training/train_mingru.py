"""
═══════════════════════════════════════════════════════════════
  ТАРС MinGRU Training Script — RTX 4090 GPU Edition
═══════════════════════════════════════════════════════════════

Полноценный скрипт обучения MinGRU_LM на русском корпусе.
Оптимизирован для RTX 4090 (24GB VRAM, CUDA).

Что делает:
  1. Загружает встроенный русский корпус (train_corpus.py)
  2. Опционально скачивает датасет с HuggingFace для дообучения  
  3. Обучает MinGRU_LM с увеличенными параметрами (dim=512, 6 слоёв)
  4. Сохраняет веса в models/mingru_weights.pt
  5. Показывает примеры генерации каждые 5 эпох

Использование:
  # Базовое обучение на встроенном корпусе (~5 мин на 4090)
  python training/train_mingru.py
  
  # С augmented данными из HuggingFace (~30 мин на 4090) 
  python training/train_mingru.py --augment
  
  # С кастомными параметрами
  python training/train_mingru.py --dim 512 --layers 6 --epochs 100 --batch 32
  
  # Дообучение существующей модели
  python training/train_mingru.py --resume
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Корень проекта
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.min_gru.mingru_lm import MinGRU_LM
from training.train_corpus import get_training_text
from brain.min_gru.utils import tokenize_text, decode_tokens
from brain.min_gru.generate import generate_text


# ═══════════════════════════════════════════
# Аргументы командной строки
# ═══════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="ТАРС MinGRU Training (GPU)")
    p.add_argument('--dim', type=int, default=512, help="Размерность модели (256/512/768)")
    p.add_argument('--layers', type=int, default=6, help="Количество слоёв MinGRU (4/6/8)")
    p.add_argument('--context_dim', type=int, default=1024, help="Размерность контекста Ω-SSM")
    p.add_argument('--seq_len', type=int, default=512, help="Длина контекстного окна")
    p.add_argument('--batch', type=int, default=32, help="Размер батча")
    p.add_argument('--epochs', type=int, default=100, help="Количество эпох")
    p.add_argument('--lr', type=float, default=3e-3, help="Начальная скорость обучения")
    p.add_argument('--wd', type=float, default=1e-2, help="Weight decay")
    p.add_argument('--augment', action='store_true', help="Скачать доп. данные с HuggingFace")
    p.add_argument('--max_samples', type=int, default=0, help="Макс. примеров (0 = без ограничений)")
    p.add_argument('--resume', action='store_true', help="Продолжить обучение с чекпоинта")
    p.add_argument('--save_every', type=int, default=10, help="Сохранять чекпоинт каждые N эпох")
    return p.parse_args()


# ═══════════════════════════════════════════
# Подготовка данных
# ═══════════════════════════════════════════

def prepare_data(text: str, seq_length: int, max_samples: int = 0):
    """Нарезает текст на пары (input, target) для next-byte prediction."""
    tokens = tokenize_text(text)
    
    # Если max_samples задан, ограничиваем длину текста чтобы не делать лишнюю работу
    if max_samples > 0:
        max_chars = max_samples * seq_length
        if len(tokens) > max_chars:
            tokens = tokens[:max_chars]
            
    inputs = []
    targets = []
    
    # Скользящее окно с перекрытием 50%
    stride = seq_length // 2
    for i in range(0, len(tokens) - seq_length - 1, stride):
        inp = tokens[i : i + seq_length]
        tgt = tokens[i + 1 : i + seq_length + 1]
        if len(inp) == seq_length and len(tgt) == seq_length:
            inputs.append(inp)
            targets.append(tgt)
        if max_samples > 0 and len(inputs) >= max_samples:
            break
            
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def augment_with_huggingface():
    """Загружает дополнительный русский текст — сначала из кэша, потом с HuggingFace."""
    
    # 1. Сначала проверяем кэш data/hf_*.txt (уже скачанные файлы)
    hf_dir = ROOT / "data"
    cached_texts = []
    if hf_dir.exists():
        for hf_file in sorted(hf_dir.glob("hf_*.txt")):
            try:
                with open(hf_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                if len(text) > 1000:
                    cached_texts.append(text)
                    size_kb = len(text.encode('cp1251', errors='replace')) / 1024
                    print(f"[Augment] Кэш: {hf_file.name} ({size_kb:.0f} KB)")
            except Exception:
                pass
    
    if cached_texts:
        result = "\n\n".join(cached_texts)
        print(f"[Augment] Итого из кэша: {len(result):,} символов")
        return result
    
    # 2. Если кэша нет — скачиваем с HuggingFace
    try:
        from datasets import load_dataset
        print("[Augment] Кэш не найден, загрузка с HuggingFace...")
        
        try:
            ds = load_dataset("Den4ikAI/russian_instructions_2", split="train", streaming=True)
            texts = []
            for i, item in enumerate(ds):
                if i >= 5000:
                    break
                q = item.get("question", item.get("instruction", "")).strip()
                a = item.get("answer", item.get("output", "")).strip()
                if q and a:
                    texts.append(f"Вопрос: {q}\nОтвет: {a}")
            
            augmented = "\n\n".join(texts)
            print(f"[Augment] Загружено {len(augmented):,} символов из russian_instructions_2")
            return augmented
        except Exception as e1:
            print(f"[Augment] russian_instructions_2: {e1}")
        
        return ""
    except ImportError:
        print("[Augment] Библиотека 'datasets' не установлена. pip install datasets")
        return ""


def load_tars_memories():
    """Загружает воспоминания ТАРС из data/tars_memories.json для дообучения."""
    memories_path = ROOT / "data" / "tars_memories.json"
    if not memories_path.exists():
        return ""
    
    try:
        with open(memories_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        if isinstance(data, list):
            for entry in data:
                memory = entry.get("memory", "")
                if memory:
                    texts.append(memory)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str):
                    texts.append(value)
        
        result = "\n".join(texts)
        print(f"[Memory] Загружено {len(texts)} воспоминаний ТАРС ({len(result)} символов)")
        return result
    except Exception as e:
        print(f"[Memory] Ошибка загрузки: {e}")
        return ""


# ═══════════════════════════════════════════
# Тренировка
# ═══════════════════════════════════════════

def train(args):
    """Основной цикл обучения."""
    
    # Устройство
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("[CPU] GPU не найден, обучение на CPU")
    
    weights_path = ROOT / "models" / "mingru_weights.pt"
    
    # ═══ Сборка корпуса ═══
    print("\n[Data] Сборка обучающего корпуса...")
    corpus_parts = [get_training_text()]
    
    # Воспоминания ТАРС
    tars_mem = load_tars_memories()
    if tars_mem:
        corpus_parts.append(tars_mem)
    
    # Augment из HuggingFace
    if args.augment:
        hf_text = augment_with_huggingface()
        if hf_text:
            corpus_parts.append(hf_text)
    
    corpus = "\n\n".join(corpus_parts)
    
    # Повторяем маленький корпус для лучшего обучения
    corpus_bytes = len(corpus.encode('cp1251', errors='replace'))
    if corpus_bytes < 100_000:
        repeat = max(1, 100_000 // corpus_bytes)
        corpus = ("\n\n" + corpus) * repeat
        print(f"[Data] Корпус повторён {repeat}x для стабильности")
    
    print(f"[Data] Итоговый корпус: {len(corpus)} символов, {corpus_bytes} байт (cp1251)")
    
    # ═══ Подготовка тензоров ═══
    inputs, targets = prepare_data(corpus, args.seq_len, max_samples=args.max_samples)
    print(f"[Data] Создано {len(inputs)} обучающих примеров (seq_len={args.seq_len})")
    
    if len(inputs) < 4:
        print("[!] Слишком мало данных. Расширьте корпус или уменьшите seq_len.")
        return
    
    # Разделение train/test (90/10)
    n_test = max(2, len(inputs) // 10)
    train_inputs, test_inputs = inputs[:-n_test], inputs[-n_test:]
    train_targets, test_targets = targets[:-n_test], targets[-n_test:]
    
    print(f"[Data] Train: {len(train_inputs)}, Test: {len(test_inputs)}")
    
    # ═══ Модель ═══
    model = MinGRU_LM(
        dim=args.dim, 
        num_tokens=256,  # byte-level vocab
        num_layers=args.layers,
        context_dim=args.context_dim
    )
    
    # Загрузка чекпоинта для дообучения
    start_epoch = 0
    if args.resume and weights_path.exists():
        print(f"[Model] Загрузка чекпоинта: {weights_path}")
        checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=False)
        
        # Проверяем совпадение архитектуры
        cp_dim = checkpoint.get('dim', 256)
        cp_layers = checkpoint.get('num_layers', 4)
        if cp_dim == args.dim and cp_layers == args.layers:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"[Model] Продолжаем с эпохи {start_epoch}")
        else:
            print(f"[Model] Архитектура изменилась ({cp_dim}→{args.dim}, {cp_layers}→{args.layers}). Начинаем заново.")
    
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Model] MinGRU_LM: dim={args.dim}, layers={args.layers}, params={param_count:,}")
    
    # ═══ Оптимизатор ═══
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    total_steps = (len(train_inputs) // args.batch + 1) * args.epochs
    warmup_steps = total_steps // 10  # 10% warmup
    
    # Cosine scheduler с warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision для GPU
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    best_loss = float('inf')
    training_start = time.time()
    
    print(f"\n{'═'*70}")
    print(f"  ТАРС MinGRU Training")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print(f"  Device: {device} | AMP: {use_amp}")
    print(f"{'═'*70}\n")
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        
        # ═══ Train ═══
        model.train()
        total_train_loss = 0
        n_batches = 0
        
        perm = torch.randperm(len(train_inputs))
        train_in = train_inputs[perm]
        train_tgt = train_targets[perm]
        
        for i in range(0, len(train_in), args.batch):
            batch_in = train_in[i:i+args.batch].to(device)
            batch_tgt = train_tgt[i:i+args.batch].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(batch_in, labels=None)
                    loss = F.cross_entropy(logits.transpose(1, 2), batch_tgt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_in, labels=None)
                loss = F.cross_entropy(logits.transpose(1, 2), batch_tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            total_train_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = total_train_loss / max(n_batches, 1)
        
        # ═══ Eval ═══
        model.eval()
        with torch.no_grad():
            eval_losses = []
            for j in range(0, len(test_inputs), args.batch):
                test_in = test_inputs[j:j+args.batch].to(device)
                test_tgt = test_targets[j:j+args.batch].to(device)
                logits = model(test_in, labels=None)
                eval_loss = F.cross_entropy(logits.transpose(1, 2), test_tgt).item()
                eval_losses.append(eval_loss)
            eval_loss = np.mean(eval_losses)
        
        epoch_time = time.time() - epoch_start
        
        # ═══ Логирование ═══
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            perplexity = np.exp(min(eval_loss, 20))
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Эпоха {epoch+1:4d} | Train: {avg_train_loss:.4f} | Eval: {eval_loss:.4f} | "
                  f"PPL: {perplexity:.1f} | LR: {lr_now:.2e} | {epoch_time:.1f}s")
            
            # Генерация примеров
            prompts = ["Вопрос: Привет\nОтвет:", "Вопрос: Кто ты?\nОтвет:"]
            for prompt in prompts:
                sample = generate_text(model, start_text=prompt, max_length=80, 
                                       temperature=0.7, device=device)
                answer = sample.split("Ответ:")[-1].strip() if "Ответ:" in sample else sample
                print(f"  → {prompt.split(chr(10))[0].replace('Вопрос: ', 'Q: ')} → {answer[:80]}")
            print()
        
        # ═══ Сохранение ═══
        if eval_loss < best_loss:
            best_loss = eval_loss
            os.makedirs(weights_path.parent, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'dim': args.dim,
                'num_tokens': 256,
                'num_layers': args.layers,
                'context_dim': args.context_dim,
                'epoch': epoch,
                'eval_loss': eval_loss,
                'train_loss': avg_train_loss,
            }, str(weights_path))
        
        # Периодический чекпоинт
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            cp_path = weights_path.parent / f"mingru_checkpoint_epoch{epoch+1}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'dim': args.dim,
                'num_tokens': 256,
                'num_layers': args.layers,
                'context_dim': args.context_dim,
                'epoch': epoch,
                'eval_loss': eval_loss,
            }, str(cp_path))
            print(f"  [Checkpoint] Сохранён: {cp_path.name}")
    
    total_time = time.time() - training_start
    
    print(f"\n{'═'*70}")
    print(f"  Обучение завершено за {total_time/60:.1f} минут")
    print(f"  Best eval loss: {best_loss:.4f} (PPL: {np.exp(min(best_loss, 20)):.1f})")
    print(f"  Веса: {weights_path}")
    print(f"{'═'*70}\n")
    
    # ═══ Финальная демонстрация ═══
    print("Финальные примеры генерации:\n")
    test_prompts = [
        "Вопрос: Привет\nОтвет:",
        "Вопрос: Что ты умеешь?\nОтвет:",
        "Вопрос: Как ты работаешь?\nОтвет:",
        "Вопрос: Помоги мне\nОтвет:",
        "Вопрос: Что такое Python?\nОтвет:",
        "Вопрос: Кто ты?\nОтвет:",
        "Вопрос: пупупу\nОтвет:",
        "Вопрос: Спасибо\nОтвет:",
    ]
    for prompt in test_prompts:
        result = generate_text(model, start_text=prompt, max_length=100, 
                               temperature=0.7, device=device)
        answer = result.split("Ответ:")[-1].strip() if "Ответ:" in result else result
        q = prompt.split("\n")[0].replace("Вопрос: ", "")
        print(f"  Q: {q}")
        print(f"  A: {answer[:120]}\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
