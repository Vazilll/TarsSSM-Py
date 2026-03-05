"""
═══════════════════════════════════════════════════════════════
  KTO (Kahneman-Tversky Optimization) Training for TARS
═══════════════════════════════════════════════════════════════

SIMPLER than DPO — doesn't need preference PAIRS.
Only needs: (prompt, response, is_good: bool)

This is huge because:
  - 10x easier to label (thumbs up/down vs comparing two responses)
  - Works with existing feedback data (ratings, likes)
  - Similar quality to DPO (Ethayarajh et al., 2024)

KTO Loss:
  L = w(y) * max(0, 1 - v(x,y))
  where v(x,y) = β * KL * (r(x,y) - z_ref)
  r(x,y) = log π(y|x) - log π_ref(y|x)

Human value function is LOSS-AVERSE: losses weigh ~2x more than gains
(Kahneman & Tversky Prospect Theory)

Usage:
  python training/train_kto.py \
      --data data/feedback.jsonl \
      --model models/mamba2/brain_best.pt
      
Data format (JSONL):
  {"prompt": "...", "response": "...", "label": true}   # good
  {"prompt": "...", "response": "...", "label": false}  # bad
"""

import os
import sys
import json
import copy
import random
import logging
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("training.kto")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def get_log_probs(model, input_ids, device):
    """Get mean log probability of sequence under model."""
    input_ids = input_ids.to(device)
    result = model.think(input_ids)
    logits = result[0] if isinstance(result, tuple) else result
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
    
    return token_log_probs.mean()


def kto_loss(
    policy_logps: torch.Tensor,      # log π(y|x) under current policy
    ref_logps: torch.Tensor,         # log π_ref(y|x) under reference
    label: bool,                     # True=desirable, False=undesirable
    beta: float = 0.1,
    desirable_weight: float = 1.0,   # λ_D
    undesirable_weight: float = 1.0, # λ_U
    kl_estimate: float = 0.0,        # running KL estimate
) -> tuple:
    """
    KTO Loss (Ethayarajh et al., 2024).
    
    Key insight from Prospect Theory:
      - People feel losses ~2x as strongly as equivalent gains
      - Undesirable responses need stronger penalty
    
    v(x,y) = β * (r(x,y) - KL_ref)
    r(x,y) = log π(y|x) - log π_ref(y|x)  (implicit reward)
    
    Loss = w(y) * max(0, 1 - v(x,y)/σ)
    """
    # Implicit reward
    reward = policy_logps - ref_logps
    
    # Value function
    v = beta * (reward - kl_estimate)
    
    if label:
        # Desirable: maximize v → loss = 1 - σ(v)
        loss = desirable_weight * (1 - torch.sigmoid(v))
    else:
        # Undesirable: minimize v → loss = 1 - σ(-v) = σ(v) 
        loss = undesirable_weight * torch.sigmoid(v)
    
    with torch.no_grad():
        implicit_reward = beta * reward
        accuracy = torch.tensor(1.0 if (label and v > 0) or (not label and v < 0) else 0.0)
    
    return loss, implicit_reward, accuracy


def train(args):
    """Main KTO training loop."""
    from brain.mamba2.model import TarsMamba2LM
    from brain.tokenizer import TarsTokenizer
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    
    logger.info(f"Device: {device}")
    
    # Load model + reference
    logger.info(f"Loading model from {args.model}...")
    model, _ = TarsMamba2LM.load_pretrained(args.model, device=device)
    model.train()
    
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    
    tokenizer = TarsTokenizer()
    
    # Load feedback data
    data = []
    n_good, n_bad = 0, 0
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if all(k in item for k in ('prompt', 'response', 'label')):
                    data.append(item)
                    if item['label']:
                        n_good += 1
                    else:
                        n_bad += 1
            except json.JSONDecodeError:
                pass
    
    logger.info(f"Loaded {len(data)} samples: {n_good} good, {n_bad} bad")
    
    if not data:
        logger.error("No feedback data!")
        return
    
    # Balance weights based on data ratio (Prospect Theory: loss aversion)
    # Undesirable should weigh more (λ_U ≈ 2 × λ_D)
    ratio = max(n_good, 1) / max(n_bad, 1)
    desirable_w = 1.0
    undesirable_w = max(1.0, ratio) * args.loss_aversion
    logger.info(f"Weights: desirable={desirable_w:.2f}, undesirable={undesirable_w:.2f} "
                f"(loss_aversion={args.loss_aversion})")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    save_path = Path(args.save_dir) / "brain_kto.pt"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Running KL estimate (for reference baseline)
    kl_running = 0.0
    kl_momentum = 0.9
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        random.shuffle(data)
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_steps = 0
        
        for i, item in enumerate(data):
            text = item['prompt'] + '\n' + item['response']
            ids = tokenizer.encode(text)[:args.max_len]
            
            if len(ids) < 10:
                continue
            
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            label = bool(item['label'])
            
            # Policy log probs
            policy_logps = get_log_probs(model, input_ids, device)
            
            # Reference log probs
            with torch.no_grad():
                ref_logps = get_log_probs(ref_model, input_ids, device)
                # Update running KL
                kl = (policy_logps - ref_logps).detach()
                kl_running = kl_momentum * kl_running + (1 - kl_momentum) * kl.item()
            
            # KTO loss  
            loss, reward, acc = kto_loss(
                policy_logps, ref_logps, label,
                beta=args.beta,
                desirable_weight=desirable_w,
                undesirable_weight=undesirable_w,
                kl_estimate=kl_running,
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            n_steps += 1
            
            if (i + 1) % 50 == 0:
                avg_loss = epoch_loss / n_steps
                avg_acc = epoch_acc / n_steps
                logger.info(f"  [{epoch+1}/{args.epochs}] step {i+1}/{len(data)} | "
                           f"loss={avg_loss:.4f} acc={avg_acc:.1%} kl={kl_running:.4f}")
        
        avg_acc = epoch_acc / max(n_steps, 1)
        avg_loss = epoch_loss / max(n_steps, 1)
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f} acc={avg_acc:.1%}")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'kto_epoch': epoch + 1,
                'kto_accuracy': avg_acc,
            }
            torch.save(checkpoint, str(save_path))
            logger.info(f"  ✓ Best KTO saved: {save_path} (acc={avg_acc:.1%})")
    
    logger.info(f"\nKTO complete! Best accuracy: {best_acc:.1%}")


def main():
    p = argparse.ArgumentParser(description="TARS KTO Training")
    p.add_argument("--data", type=str, default="data/feedback.jsonl")
    p.add_argument("--model", type=str, default="models/mamba2/brain_best.pt")
    p.add_argument("--save_dir", type=str, default="models/mamba2")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--loss_aversion", type=float, default=2.0,
                   help="Loss aversion multiplier (Prospect Theory: ~2.0)")
    p.parse_args()
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
