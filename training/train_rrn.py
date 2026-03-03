"""
═══════════════════════════════════════════════════════════════
  RRN (Спинной мозг) Training Script — Маршрутизация запросов
═══════════════════════════════════════════════════════════════

Обучает RRN определять mode (1/2/3) для входящих запросов:
  Mode 1 — Рефлекс (привет, пока, время, статус)
  Mode 2 — Действие (открой, запусти, найди файл)
  Mode 3 — Глубокое мышление (объясни, спроектируй, сравни)

Также обучает input_proj / output_proj для правильного
отображения между MinGRU (256d) и Mamba-2 (2048d).

Использование:
  python training/train_rrn.py --epochs 50
  python training/train_rrn.py --epochs 100 --lr 0.001
"""

import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ════════════════════════════════════════════
# Training Data — 3 категории сложности
# ════════════════════════════════════════════

# Mode 1: Рефлексы (<50ms, MinGRU alone)
MODE1_REFLEX = [
    "привет", "здравствуйте", "хай", "доброе утро", "добрый день",
    "пока", "до свидания", "бай", "увидимся", "спокойной ночи",
    "как дела", "как ты", "что делаешь", "как настроение",
    "сколько времени", "который час", "какая дата", "день недели",
    "ты тут", "ты здесь", "ты онлайн", "ты спишь",
    "спасибо", "благодарю", "ок", "понял", "хорошо", "ладно",
    "да", "нет", "конечно", "наверное", "может быть",
    "кто ты", "как тебя зовут", "что ты такое",
    "привет тарс", "здарова", "йо", "хелло", "приветик",
]

# Mode 2: Действия (200ms, Synapses parallel)
MODE2_ACTION = [
    "открой браузер", "запусти программу", "выключи компьютер",
    "найди файл config", "удали временные файлы", "создай папку",
    "включи музыку", "выключи звук", "поставь на паузу",
    "скопируй текст", "вставь сюда", "сохрани файл",
    "pip install torch", "python script.py", "git push",
    "открой файл main.py", "запусти тест", "перезагрузи сервер",
    "покажи логи", "проверь ошибки", "обнови пакеты",
    "сделай скриншот", "запиши видео", "открой терминал",
    "найди процесс", "убей процесс python", "очисти кэш",
    "скачай файл", "загрузи на сервер", "синхронизируй",
]

# Mode 3: Глубокое мышление (500ms+, Brain full)
MODE3_DEEP = [
    "объясни как работает трансформер",
    "спроектируй архитектуру для стартапа",
    "проанализируй этот код на баги",
    "напиши мне эссе о будущем ИИ",
    "сравни mamba и transformer подробно",
    "реализуй алгоритм быстрой сортировки с объяснением",
    "почему нейросети так хорошо работают",
    "какие есть подходы к обучению с подкреплением",
    "расскажи про квантовые вычисления подробно",
    "как оптимизировать производительность pytorch модели",
    "напиши unit тесты для этого класса",
    "проведи рефакторинг модуля авторизации",
    "объясни разницу между supervised и unsupervised learning",
    "спроектируй базу данных для интернет-магазина",
    "как работает внимание в нейросетях подробно",
    "придумай план обучения модели с нуля",
    "объясни что такое рекурсивные нейронные сети",
    "проанализируй последние тренды в machine learning",
    "помоги оптимизировать SQL запрос с JOIN",
    "напиши документацию для этого API",
]


def augment(text: str) -> list:
    """Генерирует вариации."""
    variants = [text]
    # Uppercase первая буква
    variants.append(text.capitalize())
    # Добавить знаки
    variants.append(text + "?")
    variants.append(text + "!")
    # Слово "пожалуйста"
    if random.random() > 0.5:
        variants.append(text + " пожалуйста")
    return variants


def build_dataset(max_len: int = 128):
    """Строит датасет для обучения маршрутизации."""
    samples = []  # (text, mode_label)
    
    for text in MODE1_REFLEX:
        for v in augment(text):
            samples.append((v, 0))  # mode 1 → label 0
    
    for text in MODE2_ACTION:
        for v in augment(text):
            samples.append((v, 1))  # mode 2 → label 1
    
    for text in MODE3_DEEP:
        for v in augment(text):
            samples.append((v, 2))  # mode 3 → label 2
    
    # Mine from HF datasets (complex = mode 3)
    hf_dir = ROOT / "data"
    if hf_dir.exists():
        for hf_file in sorted(hf_dir.glob("hf_*.txt"))[:3]:
            try:
                with open(hf_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                lines = [l.strip() for l in text.split('\n') if l.strip().startswith('Запрос:')]
                for line in lines[:100]:
                    query = line.replace('Запрос:', '').strip()[:200]
                    if len(query) > 30:
                        samples.append((query, 2))  # complex queries → mode 3
            except Exception:
                pass
    
    random.shuffle(samples)
    print(f"[Dataset] {len(samples)} samples: "
          f"mode1={sum(1 for _,m in samples if m==0)}, "
          f"mode2={sum(1 for _,m in samples if m==1)}, "
          f"mode3={sum(1 for _,m in samples if m==2)}")
    return samples


def text_to_bytes(text: str, max_len: int = 128) -> torch.Tensor:
    """Конвертирует текст в байтовый тензор."""
    b = list(text.encode('cp1251', errors='replace'))[:max_len]
    # Паддинг до max_len
    b = b + [0] * (max_len - len(b))
    return torch.tensor(b, dtype=torch.long)


def train(args):
    """Обучение RRN маршрутизации."""
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    if device.type == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    
    # ═══ Load MinGRU embedding (для семантической проекции) ═══
    mingru_emb = None
    try:
        from brain.min_gru.mingru_lm import MinGRU_LM
        mingru = MinGRU_LM(dim=256, num_tokens=256, num_layers=4)
        weights_path = ROOT / "models" / "mingru" / "mingru_best.pt"
        if weights_path.exists():
            ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
            mingru.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"[Model] MinGRU embedding loaded (epoch {ckpt.get('epoch', '?')})")
        mingru_emb = mingru.embedding  # nn.Embedding(256, 256)
        mingru_emb.to(device)
        mingru_emb.eval()
        for p in mingru_emb.parameters():
            p.requires_grad = False  # Заморожен!
    except Exception as e:
        print(f"[Warning] MinGRU not available: {e}")
    
    # ═══ RRN Model ═══
    from brain.rrn import TarsRRN
    
    rrn_dim = 256
    brain_dim = args.brain_dim
    
    rrn_core = TarsRRN(dim=rrn_dim, mem_slots=4, num_heads=4, max_steps=args.max_steps)
    input_proj = nn.Linear(brain_dim, rrn_dim)
    output_proj = nn.Linear(rrn_dim, brain_dim)
    
    # Классификатор маршрутизации: 256 → 3 (mode 1/2/3)
    mode_head = nn.Sequential(
        nn.Linear(rrn_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    )
    
    # Если MinGRU есть — используем его embedding для входа
    # Иначе — создаём свой
    if mingru_emb is None:
        fallback_emb = nn.Embedding(256, rrn_dim).to(device)
    else:
        fallback_emb = None
    
    # Собираем параметры для обучения
    params = (
        list(rrn_core.parameters()) +
        list(input_proj.parameters()) +
        list(output_proj.parameters()) +
        list(mode_head.parameters())
    )
    if fallback_emb:
        params += list(fallback_emb.parameters())
    
    rrn_core.to(device)
    input_proj.to(device)
    output_proj.to(device)
    mode_head.to(device)
    
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    
    total_params = sum(p.numel() for p in params)
    print(f"[Model] RRN Trainable: {total_params:,} params")
    
    # ═══ Dataset ═══
    dataset = build_dataset()
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    
    # ═══ Training Loop ═══
    best_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'═'*60}")
    print(f"  RRN Spine Training — {args.epochs} epochs")
    print(f"  LR: {args.lr} | brain_dim: {brain_dim}")
    print(f"{'═'*60}\n")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        rrn_core.train()
        mode_head.train()
        
        random.shuffle(train_data)
        total_loss = 0
        correct = 0
        total = 0
        
        for i in range(0, len(train_data), args.batch):
            batch = train_data[i:i+args.batch]
            
            # Encode text → embedding
            byte_tensors = torch.stack([text_to_bytes(t, 128) for t, _ in batch]).to(device)
            labels = torch.tensor([m for _, m in batch], dtype=torch.long).to(device)
            
            if mingru_emb is not None:
                with torch.no_grad():
                    emb = mingru_emb(byte_tensors)  # [B, 128, 256]
            else:
                emb = fallback_emb(byte_tensors)  # [B, 128, 256]
            
            # Mean pooling → [B, 256]
            x = emb.mean(dim=1)
            
            # RRN forward (2 steps for routing — fast)
            z, conf, steps = rrn_core(x, n_steps=2)
            
            # Mode classification
            logits = mode_head(z)
            
            # Loss: cross-entropy + confidence calibration
            ce_loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            
            # Confidence loss: mode 1 should have high conf, mode 3 should have low
            target_conf = torch.zeros_like(conf)
            target_conf[labels == 0] = 0.95  # Рефлекс → высокая уверенность
            target_conf[labels == 1] = 0.60  # Действие → средняя
            target_conf[labels == 2] = 0.30  # Глубокий → низкая
            conf_loss = F.mse_loss(conf.squeeze(-1), target_conf.squeeze(-1))
            
            loss = ce_loss + 0.5 * conf_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
        
        # Eval
        rrn_core.eval()
        mode_head.eval()
        eval_correct = 0
        eval_total = 0
        
        with torch.no_grad():
            for i in range(0, len(test_data), args.batch):
                batch = test_data[i:i+args.batch]
                byte_tensors = torch.stack([text_to_bytes(t, 128) for t, _ in batch]).to(device)
                labels = torch.tensor([m for _, m in batch], dtype=torch.long).to(device)
                
                if mingru_emb is not None:
                    emb = mingru_emb(byte_tensors)
                else:
                    emb = fallback_emb(byte_tensors)
                
                x = emb.mean(dim=1)
                z, conf, steps = rrn_core(x, n_steps=2)
                logits = mode_head(z)
                
                preds = logits.argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_total += len(labels)
        
        train_acc = correct / max(total, 1) * 100
        eval_acc = eval_correct / max(eval_total, 1) * 100
        elapsed = time.time() - t0
        
        is_best = eval_acc > best_acc
        if is_best:
            best_acc = eval_acc
            # Save
            torch.save({
                'rrn_core': rrn_core.state_dict(),
                'input_proj': input_proj.state_dict(),
                'output_proj': output_proj.state_dict(),
                'mode_head': mode_head.state_dict(),
                'epoch': epoch,
                'eval_acc': eval_acc,
                'brain_dim': brain_dim,
            }, str(save_dir / "rrn_spine.pt"))
        
        marker = "★ BEST" if is_best else ""
        if (epoch + 1) % 5 == 0 or epoch == 0 or is_best:
            n_batches = max(total // args.batch, 1)
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Loss: {total_loss/n_batches:.4f} | "
                  f"Train: {train_acc:.1f}% | Eval: {eval_acc:.1f}% | "
                  f"{elapsed:.1f}s {marker}")
    
    print(f"\n✓ Best accuracy: {best_acc:.1f}%")
    print(f"  Saved: {save_dir / 'rrn_spine.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RRN Spine (Mode Routing)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--brain_dim", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="models/rrn")
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
