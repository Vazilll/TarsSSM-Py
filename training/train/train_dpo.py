"""
═══════════════════════════════════════════════════════════════
  DPO (Direct Preference Optimization) Training for TARS
═══════════════════════════════════════════════════════════════

Trains TARS to prefer better responses WITHOUT a reward model.

DPO loss:
  L = -log σ(β * (log π(y_w|x) - log π(y_l|x) 
                  - log π_ref(y_w|x) + log π_ref(y_l|x)))

Where:
  y_w = preferred (winning) response
  y_l = rejected (losing) response
  π_ref = frozen reference policy (TARS before alignment)
  β = temperature controlling deviation from reference

Usage:
  python training/train_dpo.py --data data/preferences.jsonl
                               --model models/mamba2/brain_best.pt
                               --epochs 3 --beta 0.1

Data format (JSONL):
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import os
import sys
import json
import copy
import time
import logging
import argparse
import torch
import torch.nn.functional as F
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("training.dpo")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="TARS DPO Alignment Training")
    # ═══ Standard args (passed by local_train.py) ═══
    p.add_argument("--d_model", type=int, default=2048)
    p.add_argument("--n_layers", type=int, default=24)
    p.add_argument("--batch", type=int, default=2,
                   help="Alias for --batch_size (compat with local_train.py)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--bf16", action="store_true")
    # ═══ DPO-specific args ═══
    p.add_argument("--data", type=str, default="data/preferences.jsonl",
                   help="JSONL with {prompt, chosen, rejected}")
    p.add_argument("--model", type=str, default="models/mamba2/brain_best.pt",
                   help="Path to pretrained TARS checkpoint")
    p.add_argument("--save_dir", type=str, default="models/mamba2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6, help="DPO requires low LR")
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO temperature (0.1=strict, 0.5=loose)")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--label_smoothing", type=float, default=0.0,
                   help="Label smoothing for DPO loss (0=none, 0.1=recommended)")
    return p.parse_args()


def load_preference_data(path: str) -> list:
    """Load preference pairs from JSONL."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if all(k in item for k in ('prompt', 'chosen', 'rejected')):
                    data.append(item)
                else:
                    logger.warning(f"Line {line_no}: missing keys, skipping")
            except json.JSONDecodeError:
                logger.warning(f"Line {line_no}: invalid JSON, skipping")
    
    logger.info(f"Loaded {len(data)} preference pairs from {path}")
    return data


def encode_text(tokenizer, text: str, max_len: int) -> torch.LongTensor:
    """Encode text to token tensor."""
    ids = tokenizer.encode(text)[:max_len]
    return torch.tensor([ids], dtype=torch.long)


def get_log_probs(model, input_ids: torch.Tensor, device: str) -> torch.Tensor:
    """
    Get log probabilities of the sequence under the model.
    
    Returns: scalar log prob (sum of log p(token_i | token_{<i}))
    """
    input_ids = input_ids.to(device)
    
    # Forward
    result = model.think(input_ids)
    if isinstance(result, tuple):
        logits, _ = result
    else:
        logits = result
    
    # logits: [1, L, V], shift for autoregressive
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    # Per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
    
    # Average over sequence (length-normalized for fair comparison)
    return token_log_probs.mean()


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    DPO Loss (Rafailov et al., 2023).
    
    L = -log σ(β * ((log π(y_w|x) - log π_ref(y_w|x)) 
                    - (log π(y_l|x) - log π_ref(y_l|x))))
    """
    # Log ratios
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    # DPO implicit reward difference
    logits = beta * (pi_logratios - ref_logratios)
    
    # Label smoothing (IPO-style)
    if label_smoothing > 0:
        loss = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        loss = -F.logsigmoid(logits)
    
    # Metrics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        reward_margin = chosen_rewards - rejected_rewards
        accuracy = (logits > 0).float()
    
    return loss, chosen_rewards, rejected_rewards, reward_margin, accuracy


def train(args):
    """Main DPO training loop."""
    from brain.mamba2.model import TarsMamba2LM
    from brain.tokenizer import TarsTokenizer
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}...")
    model, _ = TarsMamba2LM.load_pretrained(
        checkpoint_path=args.model, device=device
    )
    model.train()
    
    # Reference model (frozen copy)
    logger.info("Creating reference model (frozen)...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    # Tokenizer
    tokenizer = TarsTokenizer()
    
    # Data
    data = load_preference_data(args.data)
    if not data:
        logger.error("No preference data found!")
        return
    
    # Optimizer (low LR for alignment)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    
    # Training
    save_path = Path(args.save_dir) / "brain_dpo.pt"
    os.makedirs(args.save_dir, exist_ok=True)
    
    total_steps = 0
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        n_batches = 0
        
        random.shuffle(data)
        
        for i, item in enumerate(data):
            prompt = item['prompt']
            chosen = prompt + item['chosen']
            rejected = prompt + item['rejected']
            
            chosen_ids = encode_text(tokenizer, chosen, args.max_len)
            rejected_ids = encode_text(tokenizer, rejected, args.max_len)
            
            # Policy log probs
            policy_chosen_logps = get_log_probs(model, chosen_ids, device)
            policy_rejected_logps = get_log_probs(model, rejected_ids, device)
            
            # Reference log probs (no grad)
            with torch.no_grad():
                ref_chosen_logps = get_log_probs(ref_model, chosen_ids, device)
                ref_rejected_logps = get_log_probs(ref_model, rejected_ids, device)
            
            # DPO loss
            loss, chosen_r, rejected_r, margin, acc = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=args.beta,
                label_smoothing=args.label_smoothing,
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_margin += margin.item()
            n_batches += 1
            total_steps += 1
            
            if (i + 1) % 50 == 0:
                avg_loss = epoch_loss / n_batches
                avg_acc = epoch_acc / n_batches
                logger.info(
                    f"  [{epoch+1}/{args.epochs}] step {i+1}/{len(data)} | "
                    f"loss={avg_loss:.4f} | acc={avg_acc:.1%}"
                )
        
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_acc = epoch_acc / max(n_batches, 1)
        avg_margin = epoch_margin / max(n_batches, 1)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"loss={avg_loss:.4f} | acc={avg_acc:.1%} | margin={avg_margin:.3f}"
        )
        
        if avg_acc > best_accuracy:
            best_accuracy = avg_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'dpo_epoch': epoch + 1,
                'dpo_accuracy': avg_acc,
                'dpo_beta': args.beta,
            }
            torch.save(checkpoint, str(save_path))
            logger.info(f"  ✓ Best DPO saved: {save_path} (acc={avg_acc:.1%})")
    
    logger.info(f"\nDPO complete! Best accuracy: {best_accuracy:.1%}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
