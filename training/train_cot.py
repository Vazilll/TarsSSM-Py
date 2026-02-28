"""
═══════════════════════════════════════════════════════════════
  Chain-of-Thought Training — ТАРС учится рассуждать пошагово
═══════════════════════════════════════════════════════════════

Источник: Qwen3 (thinking mode), Phi-4, DeepSeek-R1 (2025)

ПОЧЕМУ ЭТО КРИТИЧЕСКИ ВАЖНО:
  Без CoT: "12×15=?" → модель угадывает → "170" ❌
  С CoT:   "12×15=?" → "12×10=120, 12×5=60, 120+60=180" → "180" ✅

  Qwen3-0.6B с CoT ОБГОНЯЕТ модели в 4x больше без CoT.
  Это разница между "тупым" и "умным" маленьким ИИ.

Два этапа:
  1. Генерация CoT данных (from teacher или шаблоны)
  2. Fine-tune TARS на формате <think>...</think><answer>...</answer>

Использование:
  # Генерация CoT данных:
  python training/train_cot.py --generate --n_samples 20000

  # Fine-tune:
  python training/train_cot.py --train --resume --bf16
"""

import argparse
import json
import os
import sys
import time
import random
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════
#  CoT Data Generation — многошаговые рассуждения
# ═══════════════════════════════════════════════════

def generate_math_cot():
    """Generate math problem with step-by-step thinking."""
    problem_type = random.choice(['add', 'multi', 'equation', 'percent', 'compare'])
    
    if problem_type == 'add':
        a, b, c = random.randint(10, 999), random.randint(10, 999), random.randint(10, 999)
        answer = a + b + c
        return (f"Задача: {a} + {b} + {c} = ?\n"
                f"<think>\n"
                f"Шаг 1: Сначала сложу первые два числа: {a} + {b} = {a+b}\n"
                f"Шаг 2: Теперь прибавлю третье: {a+b} + {c} = {answer}\n"
                f"Проверка: {answer} — это сумма трёх чисел.\n"
                f"</think>\n"
                f"<answer>{answer}</answer>")
    
    elif problem_type == 'multi':
        a = random.randint(2, 30)
        b = random.randint(2, 30)
        answer = a * b
        tens_a, ones_a = a // 10, a % 10
        return (f"Задача: {a} × {b} = ?\n"
                f"<think>\n"
                f"Разложу {a} = {tens_a}×10 + {ones_a}\n"
                f"Шаг 1: {tens_a}×10 × {b} = {tens_a * 10 * b}\n"
                f"Шаг 2: {ones_a} × {b} = {ones_a * b}\n"
                f"Шаг 3: {tens_a * 10 * b} + {ones_a * b} = {answer}\n"
                f"</think>\n"
                f"<answer>{answer}</answer>")
    
    elif problem_type == 'equation':
        # ax + b = c → x = (c-b)/a
        a = random.randint(2, 10)
        x_true = random.randint(1, 20)
        b = random.randint(1, 50)
        c = a * x_true + b
        return (f"Задача: Реши уравнение: {a}x + {b} = {c}\n"
                f"<think>\n"
                f"Шаг 1: Перенесу {b} вправо: {a}x = {c} - {b} = {c - b}\n"
                f"Шаг 2: Разделю обе части на {a}: x = {c - b} / {a} = {x_true}\n"
                f"Проверка: {a} × {x_true} + {b} = {a * x_true + b} = {c} ✓\n"
                f"</think>\n"
                f"<answer>x = {x_true}</answer>")
    
    elif problem_type == 'percent':
        base = random.choice([100, 200, 300, 400, 500, 1000])
        pct = random.choice([10, 15, 20, 25, 30, 50, 75])
        answer = base * pct // 100
        return (f"Задача: Найди {pct}% от {base}\n"
                f"<think>\n"
                f"Шаг 1: {pct}% = {pct}/100 = {pct/100}\n"
                f"Шаг 2: {base} × {pct/100} = {answer}\n"
                f"</think>\n"
                f"<answer>{answer}</answer>")
    
    else:  # compare
        a, b = random.randint(1, 1000), random.randint(1, 1000)
        diff = abs(a - b)
        bigger = "первое" if a > b else "второе" if b > a else "равны"
        return (f"Задача: Какое число больше: {a} или {b}? На сколько?\n"
                f"<think>\n"
                f"Сравню: {a} и {b}\n"
                f"Разница: |{a} - {b}| = {diff}\n"
                f"{'Первое больше' if a > b else 'Второе больше' if b > a else 'Числа равны'}\n"
                f"</think>\n"
                f"<answer>Больше {bigger} число, на {diff}</answer>")


def generate_logic_cot():
    """Generate logic problem with step-by-step thinking."""
    templates = [
        # Syllogism
        lambda: (
            f"Задача: Все кошки — млекопитающие. Мурка — кошка. Является ли Мурка млекопитающим?\n"
            f"<think>\n"
            f"Посылка 1: Все кошки являются млекопитающими.\n"
            f"Посылка 2: Мурка является кошкой.\n"
            f"По правилу Барбара (AAA-1):\n"
            f"Если все A ⊂ B и x ∈ A, то x ∈ B.\n"
            f"Значит Мурка ∈ млекопитающие.\n"
            f"</think>\n"
            f"<answer>Да, Мурка является млекопитающим.</answer>"
        ),
        # Contraposition
        lambda: (
            f"Задача: Если идёт дождь, то дорога мокрая. Дорога сухая. Идёт ли дождь?\n"
            f"<think>\n"
            f"Дано: P → Q (если дождь, то дорога мокрая)\n"
            f"Дано: ¬Q (дорога НЕ мокрая, т.е. сухая)\n"
            f"По контрапозиции: ¬Q → ¬P\n"
            f"Если дорога не мокрая, значит дождя нет.\n"
            f"</think>\n"
            f"<answer>Нет, дождь не идёт (по контрапозиции).</answer>"
        ),
        # Transitivity
        lambda: (
            f"Задача: A > B, B > C, C > D. Кто больше всех? Кто меньше всех?\n"
            f"<think>\n"
            f"Шаг 1: A > B (A больше B)\n"
            f"Шаг 2: B > C (B больше C), значит A > B > C\n"
            f"Шаг 3: C > D (C больше D), значит A > B > C > D\n"
            f"Итого: порядок убывания A > B > C > D\n"
            f"</think>\n"
            f"<answer>Больше всех: A. Меньше всех: D.</answer>"
        ),
    ]
    return random.choice(templates)()


def generate_code_cot():
    """Generate coding problem with step-by-step thinking."""
    problems = [
        (
            f"Задача: Напиши функцию, которая проверяет, является ли число простым.\n"
            f"<think>\n"
            f"Шаг 1: Число n простое, если делится только на 1 и на себя.\n"
            f"Шаг 2: Достаточно проверить делители от 2 до √n.\n"
            f"Шаг 3: Если n < 2 — не простое.\n"
            f"Шаг 4: Проверю все числа от 2 до int(√n) + 1.\n"
            f"</think>\n"
            f"<answer>\n"
            f"```python\n"
            f"def is_prime(n):\n"
            f"    if n < 2:\n"
            f"        return False\n"
            f"    for i in range(2, int(n**0.5) + 1):\n"
            f"        if n % i == 0:\n"
            f"            return False\n"
            f"    return True\n"
            f"```\n"
            f"</answer>"
        ),
        (
            f"Задача: Напиши функцию для разворота строки без встроенных методов.\n"
            f"<think>\n"
            f"Шаг 1: Мне нужно пройти строку с конца в начало.\n"
            f"Шаг 2: Могу использовать цикл от len-1 до 0.\n"
            f"Шаг 3: Или могу собирать символы в новую строку.\n"
            f"Шаг 4: Выберу простой подход — цикл.\n"
            f"</think>\n"
            f"<answer>\n"
            f"```python\n"
            f"def reverse_string(s):\n"
            f"    result = ''\n"
            f"    for i in range(len(s) - 1, -1, -1):\n"
            f"        result += s[i]\n"
            f"    return result\n"
            f"```\n"
            f"</answer>"
        ),
        (
            f"Задача: Напиши функцию для поиска второго максимума в списке.\n"
            f"<think>\n"
            f"Шаг 1: Нужно найти два наибольших различных числа.\n"
            f"Шаг 2: Заведу две переменные: max1 и max2.\n"
            f"Шаг 3: Пройду по списку, обновляя оба максимума.\n"
            f"Шаг 4: Если число > max1, то max2 = max1, max1 = число.\n"
            f"</think>\n"
            f"<answer>\n"
            f"```python\n"
            f"def second_max(lst):\n"
            f"    if len(lst) < 2:\n"
            f"        return None\n"
            f"    m1 = m2 = float('-inf')\n"
            f"    for x in lst:\n"
            f"        if x > m1:\n"
            f"            m2, m1 = m1, x\n"
            f"        elif x > m2 and x != m1:\n"
            f"            m2 = x\n"
            f"    return m2 if m2 != float('-inf') else None\n"
            f"```\n"
            f"</answer>"
        ),
    ]
    return random.choice(problems)


def generate_cot_dataset(n_samples, output_path):
    """Generate Chain-of-Thought training data."""
    generators = [generate_math_cot, generate_logic_cot, generate_code_cot]
    weights = [0.5, 0.3, 0.2]  # Math-heavy like Phi-4
    
    samples = []
    for i in range(n_samples):
        gen = random.choices(generators, weights=weights, k=1)[0]
        samples.append(gen())
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(s.strip() + "\n\n")
    
    print(f"✓ Generated {len(samples)} CoT samples → {output_path}")
    return samples


# ═══════════════════════════════════════════════════
#  CoT Fine-tuning
# ═══════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="TARS Chain-of-Thought Training")
    # Mode
    p.add_argument('--generate', action='store_true', help="Generate CoT dataset")
    p.add_argument('--train', action='store_true', help="Fine-tune on CoT data")
    # Generation
    p.add_argument('--n_samples', type=int, default=10000)
    p.add_argument('--cot_data', type=str, default='data/cot_reasoning.txt')
    # Model
    p.add_argument('--d_model', type=int, default=2048)
    p.add_argument('--n_layers', type=int, default=24)
    # Training  
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-5, help="Low LR for fine-tuning")
    p.add_argument('--seq_len', type=int, default=512, help="Longer seq for CoT")
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--save_dir', type=str, default='models/mamba2')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--grad_ckpt', action='store_true')
    # CoT specific
    p.add_argument('--think_weight', type=float, default=0.3,
                   help="Weight for <think> tokens vs <answer> tokens in loss")
    return p.parse_args()


def create_samples(text, seq_len):
    """Create training samples from CoT text."""
    encoded = list(text.encode('cp1251', errors='replace'))
    if len(encoded) < 2:
        return []
    
    samples = []
    for i in range(0, len(encoded) - seq_len, seq_len // 2):
        chunk = encoded[i:i + seq_len + 1]
        if len(chunk) > seq_len:
            samples.append(chunk[:seq_len + 1])
    
    if not samples and len(encoded) > 1:
        # Pad short texts
        padded = encoded + [0] * (seq_len + 1 - len(encoded))
        samples.append(padded[:seq_len + 1])
    
    return samples


def compute_weighted_loss(logits, targets, text_bytes, think_weight=0.3):
    """
    Weighted loss: <answer> tokens get weight 1.0, <think> tokens get think_weight.
    
    This teaches the model that getting the ANSWER right is more important
    than getting the THINKING right, but thinking still matters.
    """
    loss = F.cross_entropy(logits, targets, reduction='none')
    
    # Create weight mask: higher weight for answer tokens
    # <think> = bytes of "<think>" ≈ [60, 116, 104, 105, 110, 107, 62]
    # <answer> = bytes of "<answer>" ≈ [60, 97, 110, 115, 119, 101, 114, 62]
    # Simple heuristic: weight everything after last '>' higher
    weights = torch.ones_like(loss) * think_weight
    
    # Find </think> boundary in each sample
    # For simplicity, give last 40% of tokens higher weight (answer part)
    seq_len = weights.shape[-1] if weights.ndim > 1 else weights.shape[0]
    answer_start = int(seq_len * 0.6)
    if weights.ndim > 1:
        weights[:, answer_start:] = 1.0
    else:
        weights[answer_start:] = 1.0
    
    return (loss * weights).mean()


def train_cot(args):
    """Fine-tune TARS on Chain-of-Thought data."""
    import numpy as np
    
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    device = torch.device(device_str)
    
    # Load CoT data
    if not os.path.exists(args.cot_data):
        print(f"[!] No CoT data found. Generating {args.n_samples} samples...")
        generate_cot_dataset(args.n_samples, args.cot_data)
    
    with open(args.cot_data, 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    print(f"[Data] {len(corpus)/1e6:.1f}M chars from {args.cot_data}")
    
    # Create samples
    all_samples = create_samples(corpus, args.seq_len)
    random.shuffle(all_samples)
    print(f"[Data] {len(all_samples)} training samples (seq_len={args.seq_len})")
    
    # Load model
    from brain.mamba2.model import TarsMamba2LM
    
    model = TarsMamba2LM(
        d_model=args.d_model, n_layers=args.n_layers,
        vocab_size=256, quant_mode="fp16",
    )
    
    save_path = Path(args.save_dir) / "mamba2_omega.pt"
    if args.resume and save_path.exists():
        print(f"[Model] Loading: {save_path}")
        ckpt = torch.load(str(save_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    if args.grad_ckpt:
        model.use_checkpointing = True
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # AMP
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and args.bf16) else torch.float16 if use_amp else torch.float32
    
    print(f"\n{'═'*60}")
    print(f"  TARS Chain-of-Thought Fine-tuning")
    print(f"  Samples: {len(all_samples)} | Epochs: {args.epochs}")
    print(f"  Think weight: {args.think_weight} (answer weight: 1.0)")
    print(f"  Format: <think>reasoning</think><answer>result</answer>")
    print(f"{'═'*60}\n")
    
    for epoch in range(args.epochs):
        random.shuffle(all_samples)
        t0 = time.time()
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(all_samples), args.batch):
            batch_data = all_samples[i:i + args.batch]
            if not batch_data:
                continue
            
            # Pad to same length
            max_len = max(len(s) for s in batch_data)
            padded = [s + [0] * (max_len - len(s)) for s in batch_data]
            
            x = torch.tensor(padded, dtype=torch.long, device=device)
            inputs = x[:, :-1]
            targets = x[:, 1:]
            
            with torch.amp.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(inputs)
                logits_flat = logits.reshape(-1, logits.size(-1))
                targets_flat = targets.reshape(-1)
                loss = compute_weighted_loss(logits_flat, targets_flat,
                                            None, args.think_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            n_batches += 1
        
        elapsed = time.time() - t0
        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | {elapsed:.1f}s")
        
        # Save
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'d_model': args.d_model, 'n_layers': args.n_layers,
                       'vocab_size': 256, 'cot_trained': True},
        }, str(save_path))
        print(f"  ✓ Saved (CoT fine-tuned, epoch {epoch+1})")
    
    print(f"\n✓ CoT training complete! Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    
    if args.generate:
        generate_cot_dataset(args.n_samples, args.cot_data)
    elif args.train:
        train_cot(args)
    else:
        print("Usage:")
        print("  python training/train_cot.py --generate --n_samples 20000")
        print("  python training/train_cot.py --train --resume --bf16")
