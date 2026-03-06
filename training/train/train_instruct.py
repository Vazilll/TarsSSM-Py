"""
═══════════════════════════════════════════════════════════════
  train_instruct.py — Instruction Tuning для TARS v3
═══════════════════════════════════════════════════════════════

Обучает модель ОТВЕЧАТЬ на вопросы, а не просто предсказывать
следующий токен. Превращает «генератор текста» в «помощника».

Датасеты:
  1. OpenAssistant (русский + английский)
  2. Dolly-15k (Databricks)
  3. Alpaca-RU (русский перевод)
  4. Собственные примеры автоматизации Windows

Формат:
  ### Инструкция: {instruction}
  ### Вход: {input}
  ### Ответ: {output}
"""

import os
import json
import logging
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger("Tars.InstructTuning")

_ROOT = Path(__file__).parent.parent


# ═══════════════════════════════════════════
# Встроенные instruction-данные
# ═══════════════════════════════════════════

BUILTIN_INSTRUCTIONS: List[Dict[str, str]] = [
    # === Автоматизация Windows ===
    {"instruction": "Открой браузер Chrome", 
     "output": "<tool>open_url</tool><params>{\"url\": \"https://google.com\"}</params>"},
    {"instruction": "Запусти калькулятор",
     "output": "<tool>run_command</tool><params>{\"command\": \"calc\"}</params>"},
    {"instruction": "Сделай скриншот экрана",
     "output": "<tool>analyze_workspace</tool><params>{}</params>"},
    {"instruction": "Какое сейчас время?",
     "output": "Сейчас <time>. Чем могу помочь?"},
    {"instruction": "Открой папку с документами",
     "output": "<tool>run_command</tool><params>{\"command\": \"explorer C:\\\\Users\\\\Public\\\\Documents\"}</params>"},
    {"instruction": "Найди все файлы .py на рабочем столе",
     "output": "<tool>run_command</tool><params>{\"command\": \"dir /s /b C:\\\\Users\\\\*\\\\Desktop\\\\*.py\"}</params>"},
    {"instruction": "Закрой текущее окно",
     "output": "<tool>keyboard_combo</tool><params>{\"keys\": [\"alt\", \"F4\"]}</params>"},
    {"instruction": "Переключись на следующее окно",
     "output": "<tool>keyboard_combo</tool><params>{\"keys\": [\"alt\", \"tab\"]}</params>"},
    {"instruction": "Скопируй выделенный текст",
     "output": "<tool>keyboard_combo</tool><params>{\"keys\": [\"ctrl\", \"c\"]}</params>"},
    {"instruction": "Поищи в интернете про нейросети",
     "output": "<tool>search_web</tool><params>{\"query\": \"нейронные сети обучение\"}</params>"},
    
    # === Обучение / помощь ===
    {"instruction": "Объясни что такое рекурсия",
     "output": "Рекурсия — это когда функция вызывает саму себя для решения подзадачи. Пример:\n```python\ndef factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n - 1)\n```\nЗдесь `factorial(5)` вызывает `factorial(4)`, тот — `factorial(3)`, и так до базового случая `n=1`."},
    {"instruction": "Как работает сортировка пузырьком?",
     "output": "Сортировка пузырьком — простейший алгоритм: он проходит по массиву и меняет местами соседние элементы, если они стоят не по порядку. После каждого прохода самый большой элемент «всплывает» в конец. Сложность: O(n²)."},
    {"instruction": "Напиши функцию для вычисления чисел Фибоначчи",
     "output": "```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n# fibonacci(10) → 55\n```"},
    {"instruction": "В чём разница между list и tuple в Python?",
     "output": "list — изменяемый (можно добавлять/удалять элементы), tuple — неизменяемый (нельзя). tuple быстрее и занимает меньше памяти. Используй tuple когда данные не должны меняться."},
    {"instruction": "Что такое ООП?",
     "output": "ООП (Объектно-ориентированное программирование) — парадигма, где код организован в объекты с данными (атрибуты) и поведением (методы). 4 принципа: инкапсуляция, наследование, полиморфизм, абстракция."},
    
    # === Рабочие процессы ===
    {"instruction": "Напомни что я делал вчера",
     "output": "<tool>recall</tool><params>{\"query\": \"вчера\"}</params>\nНа основе моей памяти: [результаты поиска]"},
    {"instruction": "Что у меня запланировано?",
     "output": "<tool>recall</tool><params>{\"query\": \"план задачи\"}</params>\nВот что я помню из наших разговоров: [результаты поиска]"},
    {"instruction": "Создай Python скрипт для скачивания файлов",
     "output": "```python\nimport requests\n\ndef download(url, path):\n    r = requests.get(url, stream=True)\n    with open(path, 'wb') as f:\n        for chunk in r.iter_content(8192):\n            f.write(chunk)\n    print(f'Скачано: {path}')\n```"},
    {"instruction": "Помоги разобраться с ошибкой ImportError",
     "output": "ImportError означает что Python не может найти модуль. Решения:\n1. `pip install <имя_модуля>` — установи модуль\n2. Проверь виртуальное окружение: `which python`\n3. Проверь PYTHONPATH\n4. Убедись что файл модуля существует"},
    {"instruction": "Автоматизируй резервное копирование папки",
     "output": "<tool>execute_script</tool><params>{\"code\": \"import shutil; from datetime import datetime; shutil.copytree('C:/Work', f'D:/Backup/Work_{datetime.now():%Y%m%d_%H%M}')\"}</params>\nРезервная копия создана."},
]


def format_instruction(item: Dict[str, str]) -> str:
    """Форматирует instruction в обучающий промпт."""
    instruction = item.get("instruction", "")
    inp = item.get("input", "")
    output = item.get("output", "")
    
    if inp:
        return f"### Инструкция: {instruction}\n### Вход: {inp}\n### Ответ: {output}"
    else:
        return f"### Инструкция: {instruction}\n### Ответ: {output}"


def load_instruction_dataset(path: Optional[str] = None) -> List[str]:
    """
    Загружает instruction dataset.
    
    1. Встроенные примеры (BUILTIN_INSTRUCTIONS)
    2. Пользовательские из файла (если есть)
    3. Скачанные датасеты (OpenAssistant, Dolly)
    """
    texts = []
    
    # 1. Встроенные
    for item in BUILTIN_INSTRUCTIONS:
        texts.append(format_instruction(item))
    logger.info(f"InstructTuning: {len(BUILTIN_INSTRUCTIONS)} встроенных примеров")
    
    # 2. Пользовательские
    custom_path = _ROOT / "data" / "custom_instructions.json"
    if custom_path.exists():
        try:
            with open(custom_path, "r", encoding="utf-8") as f:
                custom = json.load(f)
            for item in custom:
                texts.append(format_instruction(item))
            logger.info(f"InstructTuning: +{len(custom)} пользовательских примеров")
        except Exception as e:
            logger.warning(f"InstructTuning: ошибка загрузки custom: {e}")
    
    # 3. Скачанные датасеты
    datasets_dir = _ROOT / "data" / "instruct"
    if datasets_dir.exists():
        for fpath in datasets_dir.glob("*.jsonl"):
            count = 0
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line.strip())
                        texts.append(format_instruction(item))
                        count += 1
                logger.info(f"InstructTuning: +{count} из {fpath.name}")
            except Exception as e:
                logger.warning(f"InstructTuning: ошибка {fpath.name}: {e}")
    
    # 4. Если path указан
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                texts.append(format_instruction(item))
            logger.info(f"InstructTuning: +{len(data)} из {path}")
        except Exception as e:
            logger.warning(f"InstructTuning: ошибка {path}: {e}")
    
    logger.info(f"InstructTuning: Итого {len(texts)} обучающих примеров")
    return texts


def train_instruct(model, tokenize_fn, texts: List[str], 
                   epochs: int = 3, lr: float = 5e-5, batch_size: int = 4):
    """
    Instruction fine-tuning loop with AMP acceleration.
    
    Args:
        model: TarsMamba2LM
        tokenize_fn: текст → torch.LongTensor [1, L]
        texts: список отформатированных instruction примеров
        epochs: число эпох
        lr: learning rate (ниже чем pretrain!)
        batch_size: размер батча
    """
    import random
    import time
    
    device = next(model.parameters()).device
    use_amp = device.type == "cuda"
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    
    # Pre-tokenize (cp1251 → маленькие последовательности, быстро)
    logger.info(f"InstructTuning: Токенизация {len(texts)} примеров...")
    # ═══ Support both BPE and byte-level marker detection ═══
    marker_text = "### Ответ:"
    marker_tokens = tokenize_fn(marker_text).squeeze(0).tolist()
    marker_len = len(marker_tokens)
    
    tokenized = []
    max_len = 512  # ограничиваем длину для скорости
    for text in texts:
        tokens = tokenize_fn(text)
        if tokens.shape[1] < 4:
            continue
        if tokens.shape[1] > max_len:
            tokens = tokens[:, :max_len]
        tokenized.append(tokens.squeeze(0))  # [L]
    
    logger.info(f"InstructTuning: {len(tokenized)} примеров после фильтрации")
    
    total_batches = (len(tokenized) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * total_batches
    )
    
    # Eval split (10%)
    n_eval = max(2, len(tokenized) // 10)
    eval_set = tokenized[-n_eval:]
    train_set = tokenized[:-n_eval]
    logger.info(f"InstructTuning: Train={len(train_set)}, Eval={len(eval_set)}")
    
    best_eval_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    model.train()
    
    for epoch in range(epochs):
        random.shuffle(train_set)
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        
        for i in range(0, len(train_set), batch_size):
            batch_tokens = train_set[i:i + batch_size]
            batch_loss = torch.tensor(0.0, device=device)
            valid = 0
            
            optimizer.zero_grad(set_to_none=True)
            
            for tokens_1d in batch_tokens:
                tokens = tokens_1d.unsqueeze(0).to(device)  # [1, L]
                input_ids = tokens[:, :-1]
                labels = tokens[:, 1:]
                
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(input_ids)
                    
                    # Маска: loss только на ответной части
                    masked_labels = labels.clone()
                    token_list = tokens_1d.tolist()
                    answer_start = 0
                    for j in range(len(token_list) - marker_len):
                        if token_list[j:j + marker_len] == marker_tokens:
                            answer_start = j + marker_len
                            break
                    if answer_start > 0:
                        masked_labels[0, :min(answer_start, masked_labels.shape[1])] = -100
                    
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)),
                        masked_labels.reshape(-1)
                    )
                
                scaler.scale(loss / len(batch_tokens)).backward()
                batch_loss += loss.detach()
                valid += 1
            
            if valid > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                total_loss += batch_loss.item() / valid
                n_batches += 1
            
            # Прогресс каждые 50 батчей
            if n_batches % 50 == 0 and n_batches > 0:
                elapsed = time.time() - t0
                speed = n_batches * batch_size / elapsed
                logger.info(f"  [{n_batches}/{total_batches}] loss={total_loss/n_batches:.4f} | {speed:.0f} samples/s")
        
        # Eval
        model.eval()
        eval_loss_total = 0.0
        eval_count = 0
        with torch.no_grad():
            for tokens_1d in eval_set:
                tokens = tokens_1d.unsqueeze(0).to(device)
                input_ids = tokens[:, :-1]
                labels = tokens[:, 1:]
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(input_ids)
                    el = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                eval_loss_total += el.item()
                eval_count += 1
        eval_loss = eval_loss_total / max(eval_count, 1)
        model.train()
        
        avg_loss = total_loss / max(1, n_batches)
        elapsed = time.time() - t0
        is_best = eval_loss < best_eval_loss
        marker = " ★ BEST" if is_best else ""
        logger.info(f"InstructTuning: Epoch {epoch+1}/{epochs}, train={avg_loss:.4f}, eval={eval_loss:.4f}, time={elapsed:.1f}s{marker}")
        
        if is_best:
            best_eval_loss = eval_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"InstructTuning: ⏹ Early stopping ({patience} epochs without improvement)")
                break
    
    logger.info("InstructTuning: ✅ Instruction tuning завершено")
    return model


def download_instruct_datasets():
    """Скачивает instruction datasets для обучения."""
    datasets_dir = _ROOT / "data" / "instruct"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("InstructTuning: Загрузка instruction datasets...")
    
    try:
        from datasets import load_dataset
        
        # 1. Dolly-15k (английский, хорошего качества)
        dolly_path = datasets_dir / "dolly.jsonl"
        if not dolly_path.exists():
            logger.info("Загрузка Dolly-15k...")
            ds = load_dataset("databricks/databricks-dolly-15k", split="train")
            with open(dolly_path, "w", encoding="utf-8") as f:
                for item in ds:
                    f.write(json.dumps({
                        "instruction": item.get("instruction", ""),
                        "input": item.get("context", ""),
                        "output": item.get("response", ""),
                    }, ensure_ascii=False) + "\n")
            logger.info(f"Dolly-15k: {len(ds)} примеров сохранено")
        
        # 2. Alpaca-RU (русский)
        alpaca_path = datasets_dir / "alpaca_ru.jsonl"
        if not alpaca_path.exists():
            try:
                logger.info("Загрузка Alpaca-RU...")
                ds = load_dataset("IlyaGusev/ru_turbo_alpaca", split="train")
                with open(alpaca_path, "w", encoding="utf-8") as f:
                    for item in ds:
                        f.write(json.dumps({
                            "instruction": item.get("instruction", ""),
                            "input": item.get("input", ""),
                            "output": item.get("output", ""),
                        }, ensure_ascii=False) + "\n")
                logger.info(f"Alpaca-RU: {len(ds)} примеров сохранено")
            except Exception as e:
                logger.warning(f"Alpaca-RU не удалось загрузить: {e}")
        
        logger.info("InstructTuning: ✅ Датасеты загружены")
        
    except ImportError:
        logger.warning("datasets не установлен. pip install datasets")
        logger.info("Используем только встроенные примеры.")


def parse_args():
    """CLI arguments for subprocess invocation from local_train.py."""
    import argparse
    p = argparse.ArgumentParser(description="TARS Instruction Tuning")
    p.add_argument('--d_model', type=int, default=2048)
    p.add_argument('--n_layers', type=int, default=24)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--save_dir', type=str, default='models/mamba2')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--grad_ckpt', action='store_true')
    p.add_argument('--data', type=str, default=None,
                   help="Path to instruction data JSON")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    import time
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    
    # Device
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Load model
    try:
        from brain.mamba2.model import TarsMamba2LM
    except ImportError:
        sys.path.insert(0, str(_ROOT))
        from brain.mamba2.model import TarsMamba2LM
    
    print(f"\n  📖 Instruction Tuning: {args.d_model}d × {args.n_layers}L")
    print(f"     epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    print(f"     device={device}")
    
    # Tokenizer (BPE if trained, byte-level fallback)
    from brain.tokenizer import TarsTokenizer
    tokenizer = TarsTokenizer(mode="auto")
    
    # Checkpoint paths (must be defined before use)
    save_dir = Path(args.save_dir)
    save_path = save_dir / "mamba2_omega.pt"
    alt_path = save_dir / "mamba2.pt"
    load_path = alt_path if alt_path.exists() else save_path

    # Read vocab_size from checkpoint if resuming
    ckpt_vocab = 4096  # BPE default
    if args.resume and load_path.exists():
        try:
            ckpt = torch.load(str(load_path), map_location='cpu', weights_only=True)
            ckpt_vocab = ckpt.get('vocab_size', ckpt.get('config', {}).get('vocab_size', 4096))
            del ckpt
        except Exception:
            pass
    actual_vocab = max(ckpt_vocab, tokenizer.vocab_size)
    
    model = TarsMamba2LM(
        vocab_size=actual_vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
    ).to(device)
    
    # Resume from checkpoint
    if args.resume and load_path.exists():
        print(f"  🔄 Resuming from {load_path}")
        ckpt = torch.load(str(load_path), map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
    elif args.resume:
        print("  ⚠ No checkpoint found, training from scratch")
    
    if args.grad_ckpt and hasattr(model, 'use_checkpointing'):
        model.use_checkpointing = True
    
    def tokenize_fn(text: str) -> torch.LongTensor:
        return torch.tensor([tokenizer.encode(text)], dtype=torch.long)
    
    # Load instruction data
    texts = load_instruction_dataset(path=args.data)
    if not texts:
        print("  ❌ No instruction data found!")
        sys.exit(1)
    
    print(f"  📚 {len(texts)} instruction examples loaded")
    
    # Train
    t0 = time.time()
    model = train_instruct(
        model, tokenize_fn, texts,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
    )
    elapsed = time.time() - t0
    
    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'vocab_size': actual_vocab,
            'phase': 'instruct',
        }
    }, str(save_path))
    print(f"\n  ✅ Instruction tuning done in {elapsed/60:.1f} min")
    print(f"     Saved: {save_path}")
