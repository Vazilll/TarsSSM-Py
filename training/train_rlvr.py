"""
═══════════════════════════════════════════════════════════════
  RLVR — Reinforcement Learning from Verifiable Rewards
═══════════════════════════════════════════════════════════════

Техника из DeepSeek-R1 и Qwen3 (2025).

Вместо RLHF (нужны люди) или DPO (нужны пары), RLVR использует
задачи с ПРОВЕРЯЕМЫМИ ответами:
  - Математика: "2+3=?" → проверяем: ответ == 5
  - Код: "напиши функцию" → проверяем: тесты проходят
  - Логика: "A→B, B→C, A?" → проверяем: ответ == C

Reward:
  R = +1 если ответ правильный
  R = -1 если неправильный  
  R = -0.5 если формат неверный

Обучение: REINFORCE-style policy gradient:
  ∇J = E[R × ∇log π(a|s)]

Использование:
  python training/train_rlvr.py --epochs 3 --resume --bf16
"""

import argparse
import json
import os
import sys
import time
import random
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══ Verifiable Tasks ═══

def generate_math_task():
    """Generate a math task with verifiable answer."""
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    sign, func = random.choice(ops)
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    answer = func(a, b)
    
    prompt = f"Вычисли: {a} {sign} {b} = "
    return prompt, str(answer)


def generate_logic_task():
    """Generate a logic task with verifiable answer."""
    templates = [
        ("Если A=True и B=False, то A AND B = ", "False"),
        ("Если A=True и B=True, то A AND B = ", "True"),
        ("Если A=True и B=False, то A OR B = ", "True"),
        ("Если A=False и B=False, то A OR B = ", "False"),
        ("NOT True = ", "False"),
        ("NOT False = ", "True"),
    ]
    prompt, answer = random.choice(templates)
    return prompt, answer


def generate_sequence_task():
    """Generate a sequence continuation task."""
    start = random.randint(1, 10)
    step = random.randint(1, 5)
    seq = [start + i * step for i in range(5)]
    answer = start + 5 * step
    
    prompt = f"Продолжи последовательность: {', '.join(map(str, seq))}, "
    return prompt, str(answer)


def generate_task():
    """Generate a random verifiable task."""
    generators = [generate_math_task, generate_logic_task, generate_sequence_task]
    weights = [0.5, 0.25, 0.25]
    gen = random.choices(generators, weights=weights, k=1)[0]
    return gen()


def verify_answer(model_output: str, correct_answer: str) -> float:
    """
    Verify model's answer and return reward.
    
    Returns:
        +1.0: correct answer
        -0.5: wrong format (no number/answer found)  
        -1.0: wrong answer
    """
    # Extract number or keyword from model output
    output_clean = model_output.strip().lower()
    correct_clean = correct_answer.strip().lower()
    
    # Try to find the answer in the output
    # Look for numbers
    numbers = re.findall(r'-?\d+', output_clean)
    if numbers:
        # Check if any extracted number matches
        for num in numbers:
            if num == correct_clean:
                return 1.0
        return -1.0  # Found numbers but wrong
    
    # Check for True/False
    if correct_clean in output_clean:
        return 1.0
    
    return -0.5  # No parseable answer


def parse_args():
    p = argparse.ArgumentParser(description="TARS RLVR Training")
    p.add_argument('--d_model', type=int, default=2048)
    p.add_argument('--n_layers', type=int, default=24)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-5,
                   help="RL requires very low LR")
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--tasks_per_epoch', type=int, default=1000)
    p.add_argument('--max_gen_len', type=int, default=32,
                   help="Max tokens to generate for answer")
    p.add_argument('--save_dir', type=str, default='models/mamba2')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--grad_ckpt', action='store_true')
    # RLVR specific
    p.add_argument('--baseline', type=float, default=0.0,
                   help="Reward baseline (moving average)")
    p.add_argument('--kl_coeff', type=float, default=0.01,
                   help="KL penalty coefficient to avoid collapse")
    return p.parse_args()


def generate_answer(model, prompt_ids, max_len, device):
    """Generate answer tokens from the model."""
    model.eval()
    model.reset_cache()
    
    generated = prompt_ids.clone()
    
    with torch.no_grad():
        logits = model.step(prompt_ids)
    
    all_log_probs = []
    
    for _ in range(max_len):
        # Sample next token
        last_logits = logits[:, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        token = torch.multinomial(probs, 1)  # [B, 1]
        
        log_prob = F.log_softmax(last_logits, dim=-1)
        selected_log_prob = log_prob.gather(-1, token)  # [B, 1]
        all_log_probs.append(selected_log_prob)
        
        generated = torch.cat([generated, token], dim=1)
        
        with torch.no_grad():
            logits = model.step(token)
        
        # Stop if newline or period
        if token.item() in [10, 46, 0]:  # \n, '.', \0
            break
    
    if all_log_probs:
        total_log_prob = torch.cat(all_log_probs, dim=1).sum(dim=1)  # [B]
    else:
        total_log_prob = torch.zeros(1, device=device)
    
    return generated, total_log_prob


def train(args):
    """RLVR training loop."""
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    device = torch.device(device_str)
    
    from brain.mamba2.model import TarsMamba2LM
    
    actual_vocab = 256
    model = TarsMamba2LM(
        d_model=args.d_model, n_layers=args.n_layers,
        vocab_size=actual_vocab, quant_mode="fp16",
    )
    
    save_path = Path(args.save_dir) / "mamba2_omega.pt"
    if args.resume and save_path.exists():
        print(f"[Model] Loading: {save_path}")
        ckpt = torch.load(str(save_path), map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    if args.grad_ckpt:
        model.use_checkpointing = True
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    print(f"\n{'═'*60}")
    print(f"  TARS RLVR — Reinforcement Learning from Verifiable Rewards")
    print(f"  Tasks/epoch: {args.tasks_per_epoch} | Epochs: {args.epochs}")
    print(f"  KL coeff: {args.kl_coeff} | LR: {args.lr}")
    print(f"{'═'*60}\n")
    
    baseline = args.baseline
    
    for epoch in range(args.epochs):
        t0 = time.time()
        total_reward = 0
        total_loss = 0
        correct = 0
        n = 0
        
        for task_i in range(args.tasks_per_epoch):
            # Generate task
            prompt, correct_answer = generate_task()
            
            # Encode prompt as bytes
            prompt_bytes = list(prompt.encode('cp1251', errors='replace'))
            prompt_ids = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
            
            # Generate answer
            model.train()
            generated, log_prob = generate_answer(model, prompt_ids, args.max_gen_len, device)
            
            # Decode answer
            answer_ids = generated[0, len(prompt_bytes):].cpu().tolist()
            try:
                answer_text = bytes(answer_ids).decode('cp1251', errors='replace')
            except Exception:
                answer_text = ""
            
            # Verify and get reward
            reward = verify_answer(answer_text, correct_answer)
            
            # REINFORCE loss: -R × log π(a|s) (with baseline)
            advantage = reward - baseline
            loss = -advantage * log_prob.mean()
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update every batch
            if (task_i + 1) % args.batch == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_reward += reward
            total_loss += loss.item()
            if reward > 0:
                correct += 1
            n += 1
            
            # Update baseline (exponential moving average)
            baseline = 0.95 * baseline + 0.05 * reward
        
        # Flush remaining gradients
        optimizer.step()
        optimizer.zero_grad()
        
        elapsed = time.time() - t0
        acc = correct / max(n, 1)
        avg_r = total_reward / max(n, 1)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Acc: {acc:.1%} | "
              f"Reward: {avg_r:.3f} | "
              f"Loss: {total_loss/n:.4f} | "
              f"Baseline: {baseline:.3f} | {elapsed:.1f}s")
        
        # Save
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {'d_model': args.d_model, 'n_layers': args.n_layers,
                       'vocab_size': actual_vocab, 'rlvr_baseline': baseline},
            'epoch': epoch,
        }, str(save_path))
        print(f"  ✓ Saved (accuracy: {acc:.1%})")
    
    print(f"\n{'═'*60}")
    print(f"  RLVR Training complete!")
    print(f"  Final accuracy: {acc:.1%}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
