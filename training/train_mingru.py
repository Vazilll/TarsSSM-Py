"""
═══════════════════════════════════════════════════════════════
  ТАРС MinGRU Training Script — MAX GPU Edition
═══════════════════════════════════════════════════════════════

Полноценный скрипт обучения MinGRU_LM на русском корпусе.
Автоматически выжимает максимум из GPU.

Что делает:
  1. Загружает встроенный русский корпус (train_corpus.py)
  2. Опционально скачивает датасет с HuggingFace для дообучения  
  3. Обучает MinGRU_LM с увеличенными параметрами (dim=512, 6 слоёв)
  4. Сохраняет веса в models/mingru_weights.pt
  5. Показывает примеры генерации каждые 5 эпох

Оптимизации GPU:
  - Автоопределение max batch (бинарный поиск по VRAM)
  - DataLoader с pin_memory + num_workers (async loading)
  - torch.compile (PyTorch 2.0+, ~20% kernel fusion speedup)
  - bf16/fp16 autocast + GradScaler
  - Gradient Accumulation (стабильность на больших batch)

Использование:
  python training/train_mingru.py
  python training/train_mingru.py --augment
  python training/train_mingru.py --dim 512 --layers 6 --epochs 25 --batch 32
  python training/train_mingru.py --resume
"""

import os
import sys
import time
import json
import gc
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
    p = argparse.ArgumentParser(description="ТАРС MinGRU Training (MAX GPU)")
    p.add_argument('--dim', type=int, default=512, help="Размерность модели (256/512/768)")
    p.add_argument('--layers', type=int, default=6, help="Количество слоёв MinGRU (4/6/8)")
    p.add_argument('--context_dim', type=int, default=1024, help="Размерность контекста Ω-SSM")
    p.add_argument('--seq_len', type=int, default=512, help="Длина контекстного окна")
    p.add_argument('--batch', type=int, default=32, help="Стартовый размер батча (авто-увеличится)")
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
    
    if max_samples > 0:
        max_chars = max_samples * seq_length
        if len(tokens) > max_chars:
            tokens = tokens[:max_chars]
            
    inputs = []
    targets = []
    
    # Stride = seq_length (без перекрытия, экономит 2x RAM)
    stride = seq_length
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
# GPU Optimization
# ═══════════════════════════════════════════

def _find_max_batch(model, device, seq_len):
    """Бинарный поиск максимального батча, влезающего в VRAM."""
    model.eval()
    lo, hi = 32, 4096
    best = 32
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            dummy = torch.randint(0, 256, (mid, seq_len), device=device)
            with torch.amp.autocast('cuda'), torch.no_grad():
                _ = model(dummy)
            del dummy, _
            torch.cuda.empty_cache()
            best = mid
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            hi = mid - 1
    # 75% от максимума (запас для градиентов и optimizer states)
    safe = max(32, int(best * 0.75))
    # Округляем до кратного 32
    safe = (safe // 32) * 32
    return safe


# ═══════════════════════════════════════════
# Тренировка
# ═══════════════════════════════════════════

def train(args):
    """Основной цикл обучения — максимальное использование GPU."""
    from torch.utils.data import TensorDataset, DataLoader
    
    # ═══ Устройство ═══
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        gpu_mem = 0
        print("[CPU] GPU не найден, обучение на CPU")
    
    weights_path = ROOT / "models" / "mingru_weights.pt"
    
    # ═══ Сборка корпуса ═══
    print("\n[Data] Сборка обучающего корпуса...")
    corpus_parts = [get_training_text()]
    
    tars_mem = load_tars_memories()
    if tars_mem:
        corpus_parts.append(tars_mem)
    
    if args.augment:
        hf_text = augment_with_huggingface()
        if hf_text:
            corpus_parts.append(hf_text)
    
    corpus = "\n\n".join(corpus_parts)
    
    corpus_bytes = len(corpus.encode('cp1251', errors='replace'))
    if corpus_bytes < 100_000:
        repeat = max(1, 100_000 // corpus_bytes)
        corpus = ("\n\n" + corpus) * repeat
        print(f"[Data] Корпус повторён {repeat}x для стабильности")
    
    print(f"[Data] Итоговый корпус: {len(corpus)} символов, {corpus_bytes} байт")
    
    # ═══ Подготовка тензоров ═══
    inputs, targets = prepare_data(corpus, args.seq_len, max_samples=args.max_samples)
    print(f"[Data] Создано {len(inputs)} обучающих примеров (seq_len={args.seq_len})")
    
    # Освобождаем корпус из RAM
    del corpus, corpus_parts
    gc.collect()
    
    if len(inputs) < 4:
        print("[!] Слишком мало данных.")
        return
    
    # Разделение train/test (90/10)
    n_test = max(2, len(inputs) // 10)
    train_inputs, test_inputs = inputs[:-n_test], inputs[-n_test:]
    train_targets, test_targets = targets[:-n_test], targets[-n_test:]
    del inputs, targets
    gc.collect()
    
    data_size_mb = (train_inputs.nbytes + train_targets.nbytes) / 1024**2
    print(f"[Data] Train: {len(train_inputs)}, Test: {len(test_inputs)} ({data_size_mb:.0f} MB)")
    
    # ═══ Модель ═══
    model = MinGRU_LM(
        dim=args.dim, 
        num_tokens=256,
        num_layers=args.layers,
        context_dim=args.context_dim
    )
    
    # Загрузка чекпоинта
    start_epoch = 0
    if args.resume and weights_path.exists():
        print(f"[Model] Загрузка чекпоинта: {weights_path}")
        checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=False)
        cp_dim = checkpoint.get('dim', 256)
        cp_layers = checkpoint.get('num_layers', 4)
        if cp_dim == args.dim and cp_layers == args.layers:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"[Model] Продолжаем с эпохи {start_epoch}")
        else:
            print(f"[Model] Архитектура изменилась. Начинаем заново.")
    
    model.to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    param_mb = sum(p.nbytes for p in model.parameters()) / 1024**2
    print(f"[Model] MinGRU_LM: dim={args.dim}, layers={args.layers}, "
          f"params={param_count:,} ({param_mb:.0f} MB)")
    
    # ═══ torch.compile (PyTorch 2.0+, ~20% speedup) ═══
    compiled = False
    if device.type == 'cuda':
        try:
            model = torch.compile(model, mode="reduce-overhead")
            compiled = True
            print("[Model] torch.compile: ON (reduce-overhead)")
        except Exception:
            try:
                model = torch.compile(model)
                compiled = True
                print("[Model] torch.compile: ON (default)")
            except Exception:
                print("[Model] torch.compile: недоступен, пропуск")
    
    # ═══ Автоопределение max batch ═══
    if device.type == 'cuda':
        max_batch = _find_max_batch(model, device, args.seq_len)
        if max_batch > args.batch:
            print(f"[GPU] Max batch: {max_batch} (было {args.batch})")
            args.batch = max_batch
    
    # ═══ Gradient Accumulation ═══
    accum_steps = 1
    effective_batch = args.batch
    if args.batch >= 512:
        accum_steps = args.batch // 256
        args.batch = 256
        effective_batch = args.batch * accum_steps
        print(f"[GPU] Grad Accum: {accum_steps} steps → effective batch={effective_batch}")
    
    # ═══ DataLoader (pin_memory + async prefetch) ═══
    train_dataset = TensorDataset(train_inputs, train_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    del train_inputs, train_targets, test_inputs, test_targets
    gc.collect()
    
    use_pin = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch, shuffle=True,
        pin_memory=use_pin, num_workers=2 if use_pin else 0,
        drop_last=True, persistent_workers=use_pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch * 2, shuffle=False,
        pin_memory=use_pin, num_workers=0,
    )
    
    # ═══ Оптимизатор ═══
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    total_steps = len(train_loader) * args.epochs // accum_steps
    warmup_steps = total_steps // 10
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ═══ Mixed Precision ═══
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))
    
    best_loss = float('inf')
    training_start = time.time()
    
    # ═══ Баннер ═══
    if device.type == 'cuda':
        gpu_used = torch.cuda.memory_allocated() / 1024**3
        print(f"[GPU] VRAM до обучения: {gpu_used:.2f} GB / {gpu_mem:.1f} GB")
    
    print(f"\n{'═'*70}")
    print(f"  ТАРС MinGRU Training (MAX GPU)")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch}×{accum_steps} = {effective_batch}")
    print(f"  Device: {device} | AMP: {amp_dtype if use_amp else 'off'} | Compiled: {compiled}")
    print(f"  Batches/epoch: {len(train_loader)} | Total steps: {total_steps}")
    print(f"{'═'*70}\n")
    
    # ═══ Training Loop ═══
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        
        model.train()
        total_train_loss = 0
        n_batches = 0
        optimizer.zero_grad()
        
        for step_idx, (batch_in, batch_tgt) in enumerate(train_loader):
            batch_in = batch_in.to(device, non_blocking=True)
            batch_tgt = batch_tgt.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_in, labels=None)
                loss = F.cross_entropy(logits.transpose(1, 2), batch_tgt)
                loss = loss / accum_steps
            
            scaler.scale(loss).backward()
            
            if (step_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            total_train_loss += loss.item() * accum_steps
            n_batches += 1
        
        # Flush remaining gradients
        if n_batches % accum_steps != 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / max(n_batches, 1)
        
        # ═══ Eval ═══
        model.eval()
        with torch.no_grad():
            eval_losses = []
            for test_in, test_tgt in test_loader:
                test_in = test_in.to(device, non_blocking=True)
                test_tgt = test_tgt.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    logits = model(test_in, labels=None)
                    el = F.cross_entropy(logits.transpose(1, 2), test_tgt).item()
                eval_losses.append(el)
            eval_loss = np.mean(eval_losses)
        
        epoch_time = time.time() - epoch_start
        
        # ═══ Логирование ═══
        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            perplexity = np.exp(min(eval_loss, 20))
            lr_now = optimizer.param_groups[0]['lr']
            
            gpu_info = ""
            if device.type == 'cuda':
                gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
                gpu_info = f" | VRAM: {gpu_mb:.0f}MB"
            
            print(f"Эпоха {epoch+1:4d} | Train: {avg_train_loss:.4f} | Eval: {eval_loss:.4f} | "
                  f"PPL: {perplexity:.1f} | LR: {lr_now:.2e} | {epoch_time:.1f}s{gpu_info}")
            
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
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM: {peak:.1f} GB / {gpu_mem:.1f} GB ({peak/gpu_mem*100:.0f}%)")
    print(f"{'═'*70}\n")
    
    # ═══ Квантизация 1.58-bit ═══
    print("[Quant] Конвертация MinGRU → 1.58-bit...")
    try:
        from brain.mamba2.bitnet import convert_model_to_158bit, model_stats
        
        # Загружаем лучшие веса
        best_ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
        quant_model = MinGRU_LM(
            dim=args.dim, num_tokens=256,
            num_layers=args.layers, context_dim=args.context_dim
        )
        quant_model.load_state_dict(best_ckpt['model_state_dict'])
        
        fp32_mb = sum(p.numel() * p.element_size() for p in quant_model.parameters()) / 1048576
        
        # Конвертация всех UniversalLinear → 1.58-bit
        convert_model_to_158bit(quant_model)
        stats = model_stats(quant_model)
        
        # Сохранение
        quant_path = weights_path.parent / "mingru_158bit.pt"
        torch.save({
            'model_state_dict': quant_model.state_dict(),
            'dim': args.dim,
            'num_tokens': 256,
            'num_layers': args.layers,
            'context_dim': args.context_dim,
            'quant_mode': '158bit',
            'eval_loss': best_loss,
        }, str(quant_path))
        
        quant_mb = os.path.getsize(str(quant_path)) / 1048576
        print(f"  ✅ Квантизировано: {stats['universal_linear_count']} слоёв")
        print(f"  Разреженность: {stats['avg_sparsity']:.1%}")
        print(f"  Размер: {fp32_mb:.1f} MB → {quant_mb:.1f} MB ({fp32_mb/max(quant_mb, 0.1):.1f}x сжатие)")
        print(f"  Файл: {quant_path}")
        
        del quant_model
        gc.collect()
    except Exception as e:
        print(f"  ⚠ Квантизация пропущена: {e}")
        print(f"    (Модель сохранена в fp32: {weights_path})")
    
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
