"""
═══════════════════════════════════════════════════════════════
  DPO — Direct Preference Optimization для TARS
═══════════════════════════════════════════════════════════════

Техника: DPO (Rafailov et al., 2023) — обучение без reward model.

Вместо RLHF (PPO + reward model), DPO напрямую оптимизирует:
  L_DPO = -log σ(β × (log π_θ(y_w|x) - log π_ref(y_w|x) 
                      - log π_θ(y_l|x) + log π_ref(y_l|x)))

Где:
  y_w — предпочтительный ответ (winner)
  y_l — нежелательный ответ (loser) 
  π_θ — обучаемая модель
  π_ref — замороженная референс-модель
  β — temperature (0.1-0.5)

Данные: пары (prompt, chosen_response, rejected_response)

Использование:
  python training/train_dpo.py --data data/dpo_pairs.jsonl --resume
"""

import argparse
import json
import os
import sys
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="TARS DPO Alignment Training")
    p.add_argument('--d_model', type=int, default=2048)
    p.add_argument('--n_layers', type=int, default=24)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-6,
                   help="DPO requires very low LR (1e-6 to 1e-5)")
    p.add_argument('--beta', type=float, default=0.1,
                   help="DPO temperature (0.1 = strong preference, 0.5 = weak)")
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--data', type=str, required=True,
                   help="Path to DPO pairs JSONL (prompt, chosen, rejected)")
    p.add_argument('--save_dir', type=str, default='models/mamba2')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--grad_ckpt', action='store_true')
    return p.parse_args()


def get_sequence_log_probs(model, input_ids, target_ids, device):
    """
    Compute per-token log probabilities for a sequence.
    
    Returns: mean log probability of target tokens
    """
    input_ids = input_ids.to(device)
    target_ids = target_ids.to(device)
    
    logits = model(input_ids)  # [B, L, V]
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probs for target tokens
    # target_ids: [B, L]
    target_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    
    # Mean over sequence (ignore padding if present)
    mask = (target_ids != 0).float()
    seq_log_prob = (target_log_probs * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    
    return seq_log_prob  # [B]


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    """
    DPO loss (Rafailov et al., 2023).
    
    L = -log σ(β × ((log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))))
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    
    # Metrics
    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
    accuracy = ((chosen_rewards > rejected_rewards).float().mean().item())
    
    return loss, reward_margin, accuracy


def load_dpo_data(path, seq_len):
    """
    Load DPO pairs from JSONL.
    
    Expected format per line:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Encode as bytes (cp1251)
                prompt = item['prompt'].encode('cp1251', errors='replace')
                chosen = item['chosen'].encode('cp1251', errors='replace')
                rejected = item['rejected'].encode('cp1251', errors='replace')
                
                # Create input/target pairs
                chosen_full = prompt + chosen
                rejected_full = prompt + rejected
                
                # Truncate to seq_len
                if len(chosen_full) > seq_len:
                    chosen_full = chosen_full[:seq_len]
                if len(rejected_full) > seq_len:
                    rejected_full = rejected_full[:seq_len]
                
                # Pad to seq_len
                chosen_padded = list(chosen_full) + [0] * (seq_len - len(chosen_full))
                rejected_padded = list(rejected_full) + [0] * (seq_len - len(rejected_full))
                
                pairs.append({
                    'chosen_in': torch.tensor(chosen_padded[:-1], dtype=torch.long),
                    'chosen_tgt': torch.tensor(chosen_padded[1:], dtype=torch.long),
                    'rejected_in': torch.tensor(rejected_padded[:-1], dtype=torch.long),
                    'rejected_tgt': torch.tensor(rejected_padded[1:], dtype=torch.long),
                })
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    print(f"[DPO Data] Loaded {len(pairs)} preference pairs from {path}")
    return pairs


def train(args):
    """DPO training loop."""
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    device = torch.device(device_str)
    
    # Load model
    from brain.mamba2.model import TarsMamba2LM
    
    actual_vocab = 256
    policy = TarsMamba2LM(
        d_model=args.d_model, n_layers=args.n_layers,
        vocab_size=actual_vocab, quant_mode="fp16",
    )
    
    save_path = Path(args.save_dir) / "mamba2_omega.pt"
    if args.resume and save_path.exists():
        print(f"[Policy] Loading: {save_path}")
        ckpt = torch.load(str(save_path), map_location='cpu', weights_only=False)
        policy.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    if args.grad_ckpt:
        policy.use_checkpointing = True
    policy.to(device)
    
    # Create frozen reference model (copy of policy at start)
    print("[Ref] Creating frozen reference model...")
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[Policy] {params/1e6:.1f}M trainable params")
    
    # Data
    pairs = load_dpo_data(args.data, args.seq_len)
    if not pairs:
        print("[!] No DPO pairs found. Create data/dpo_pairs.jsonl first.")
        return
    
    # Optimizer (very low LR for DPO)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    
    print(f"\n{'═'*60}")
    print(f"  TARS DPO Alignment")
    print(f"  β={args.beta} | LR={args.lr} | Pairs: {len(pairs)}")
    print(f"{'═'*60}\n")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        policy.train()
        total_loss = total_margin = total_acc = 0
        n = 0
        
        indices = torch.randperm(len(pairs))
        
        for i in range(0, len(pairs), args.batch):
            batch_idx = indices[i:i+args.batch]
            
            # Collect batch
            chosen_in = torch.stack([pairs[j]['chosen_in'] for j in batch_idx])
            chosen_tgt = torch.stack([pairs[j]['chosen_tgt'] for j in batch_idx])
            rejected_in = torch.stack([pairs[j]['rejected_in'] for j in batch_idx])
            rejected_tgt = torch.stack([pairs[j]['rejected_tgt'] for j in batch_idx])
            
            # Policy log probs
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    pol_chosen = get_sequence_log_probs(policy, chosen_in, chosen_tgt, device)
                    pol_rejected = get_sequence_log_probs(policy, rejected_in, rejected_tgt, device)
            else:
                pol_chosen = get_sequence_log_probs(policy, chosen_in, chosen_tgt, device)
                pol_rejected = get_sequence_log_probs(policy, rejected_in, rejected_tgt, device)
            
            # Reference log probs (frozen, no grad)
            with torch.no_grad():
                ref_chosen = get_sequence_log_probs(ref_model, chosen_in, chosen_tgt, device)
                ref_rejected = get_sequence_log_probs(ref_model, rejected_in, rejected_tgt, device)
            
            # DPO loss
            loss, margin, acc = dpo_loss(
                pol_chosen, pol_rejected,
                ref_chosen, ref_rejected,
                args.beta
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            total_margin += margin
            total_acc += acc
            n += 1
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {total_loss/n:.4f} | "
              f"Margin: {total_margin/n:.3f} | "
              f"Acc: {total_acc/n:.1%} | {elapsed:.1f}s")
        
        # Save
        os.makedirs(args.save_dir, exist_ok=True)
        dpo_path = Path(args.save_dir) / "mamba2_omega_dpo.pt"
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': {'d_model': args.d_model, 'n_layers': args.n_layers,
                       'vocab_size': actual_vocab, 'dpo_beta': args.beta},
            'epoch': epoch,
        }, str(dpo_path))
        # Also update main checkpoint
        torch.save({
            'model_state_dict': policy.state_dict(),
            'config': {'d_model': args.d_model, 'n_layers': args.n_layers,
                       'vocab_size': actual_vocab},
        }, str(save_path))
        print(f"  ✓ Saved: {dpo_path}")
    
    # Cleanup ref model
    del ref_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\n{'═'*60}")
    print(f"  DPO Alignment complete!")
    print(f"{'═'*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
