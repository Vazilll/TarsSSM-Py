"""
═══════════════════════════════════════════════════════════════
  TARS HELIX LITE — Colab Training Script
═══════════════════════════════════════════════════════════════

Run on Google Colab (T4/A100 GPU) for test training.

Usage:
  1. Upload the TarsSSM-Py project to Colab (or clone from git)
  2. Run this script: python train_lite.py
  3. Check loss curve and parameter count

Quick start (Colab cell):
  !pip install torch einops
  !python train_lite.py --test-only   # smoke test (no GPU needed)
  !python train_lite.py               # full training
"""

import os
import sys
import math
import time
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import TarsConfig
from brain.mamba2.core.model_lite import TarsHelixLite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("Tars.TrainLite")


# ═══════════════════════════════════════
# Dummy Dataset (for smoke test / initial training)
# ═══════════════════════════════════════

class RandomTokenDataset(Dataset):
    """Random token dataset for smoke testing and sanity checks."""
    
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        # Pre-generate for reproducibility
        torch.manual_seed(42)
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        return {
            'input_ids': tokens[:-1],   # [seq_len]
            'labels': tokens[1:],       # [seq_len] (shifted by 1)
        }


class TextFileDataset(Dataset):
    """Simple text file dataset — tokenize and chunk."""
    
    def __init__(self, file_path: str, tokenizer, seq_len: int = 512):
        self.seq_len = seq_len
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = tokenizer.encode(text)
        # Chunk into sequences
        n_chunks = len(tokens) // (seq_len + 1)
        tokens = tokens[:n_chunks * (seq_len + 1)]
        self.chunks = torch.tensor(tokens).view(n_chunks, seq_len + 1)
        logger.info(f"Loaded {file_path}: {len(tokens)} tokens → {n_chunks} chunks of {seq_len}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        tokens = self.chunks[idx]
        return {
            'input_ids': tokens[:-1],
            'labels': tokens[1:],
        }


# ═══════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════

def train_epoch(model, dataloader, optimizer, scheduler, device, cfg, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward
        result = model(input_ids, labels=labels)
        loss = result['loss']
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if scheduler is not None:
            scheduler.step()
        
        # Stats
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - start_time
            avg_loss = total_loss / total_tokens
            tok_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch} | Step {step+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) | "
                f"LR: {lr:.2e} | {tok_per_sec:.0f} tok/s"
            )
    
    return total_loss / total_tokens


def get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine LR schedule with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════

def smoke_test(device="cpu"):
    """Quick sanity check: forward + backward with tiny config."""
    logger.info("=" * 60)
    logger.info("SMOKE TEST — verifying model builds and trains")
    logger.info("=" * 60)
    
    # Tiny config for fast testing
    cfg = TarsConfig(
        d_model=128,
        n_layers=2,
        vocab_size=256,
        d_state=16,
        headdim=32,
        expand=2,
        batch_size=2,
    )
    
    logger.info(f"Config: d={cfg.d_model}, L={cfg.n_layers}, V={cfg.vocab_size}")
    
    model = TarsHelixLite(cfg).to(device)
    params = model.count_parameters()
    logger.info(f"Parameters: {params['_total_M']:.2f}M total")
    for comp, count in sorted(params.items()):
        if not comp.startswith('_'):
            logger.info(f"  {comp}: {count/1e3:.1f}K")
    
    # Forward
    B, L = 2, 64
    input_ids = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    labels = torch.randint(0, cfg.vocab_size, (B, L), device=device)
    
    logger.info(f"\nForward pass: B={B}, L={L}")
    t0 = time.time()
    result = model(input_ids, labels=labels)
    t_fwd = time.time() - t0
    logger.info(f"  Logits shape: {result['logits'].shape}")
    logger.info(f"  Loss: {result['loss'].item():.4f}")
    logger.info(f"  Forward time: {t_fwd*1000:.1f}ms")
    
    # Backward
    t0 = time.time()
    result['loss'].backward()
    t_bwd = time.time() - t0
    logger.info(f"  Backward time: {t_bwd*1000:.1f}ms")
    
    # Check gradients exist
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            component = name.split('.')[0]
            if component not in grad_norms:
                grad_norms[component] = 0
            grad_norms[component] += p.grad.norm().item()
    
    logger.info(f"\nGradient flow:")
    for comp, norm in sorted(grad_norms.items()):
        status = "✅" if norm > 0 else "❌ NO GRADIENT"
        logger.info(f"  {comp}: {norm:.4f} {status}")
    
    all_ok = all(n > 0 for n in grad_norms.values())
    
    # Generate test
    logger.info(f"\nGeneration test:")
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=16, temperature=1.0)
    logger.info(f"  Prompt: {prompt.shape} → Generated: {generated.shape}")
    
    logger.info("=" * 60)
    if all_ok:
        logger.info("✅ SMOKE TEST PASSED — model builds, forward/backward work, gradients flow")
    else:
        logger.info("❌ SMOKE TEST FAILED — some components have no gradients")
    logger.info("=" * 60)
    
    return all_ok


def full_param_count():
    """Show parameter count for full LITE config."""
    logger.info("=" * 60)
    logger.info("FULL HELIX LITE PARAMETER COUNT")
    logger.info("=" * 60)
    
    cfg = TarsConfig()
    model = TarsHelixLite(cfg)
    params = model.count_parameters()
    
    logger.info(f"\nTotal: {params['_total_M']:.1f}M parameters")
    logger.info(f"Config: d={cfg.d_model}, L={cfg.n_layers}, V={cfg.vocab_size}, "
                f"d_state={cfg.d_state}, expand={cfg.expand}")
    
    for comp, count in sorted(params.items()):
        if not comp.startswith('_'):
            pct = count / params['_total'] * 100
            logger.info(f"  {comp:20s}: {count/1e6:8.2f}M  ({pct:5.1f}%)")
    
    # Memory estimate
    fp16_mb = params['_total'] * 2 / 1e6
    ternary_mb = params['_total'] * 2 / 8 / 1e6  # 1.58 bits ≈ 2 bits packed
    logger.info(f"\nMemory estimate:")
    logger.info(f"  FP16:    {fp16_mb:.0f} MB")
    logger.info(f"  Ternary: {ternary_mb:.0f} MB")
    
    return params


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TARS HELIX LITE Training")
    parser.add_argument("--test-only", action="store_true", help="Run smoke test only")
    parser.add_argument("--count-params", action="store_true", help="Show parameter count for full config")
    parser.add_argument("--data", type=str, default=None, help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=100, help="Max steps (for random data)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detect if not set)")
    parser.add_argument("--save-dir", type=str, default="models/tars_lite", help="Save directory")
    parser.add_argument("--d-model", type=int, default=None, help="Override d_model")
    parser.add_argument("--n-layers", type=int, default=None, help="Override n_layers")
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    logger.info(f"Device: {device}")
    
    # Smoke test
    if args.test_only:
        smoke_test(device)
        return
    
    # Param count
    if args.count_params:
        full_param_count()
        return
    
    # ═══ Config ═══
    cfg = TarsConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        max_seq_len=args.seq_len,
    )
    # Override if specified
    if args.d_model is not None:
        cfg.d_model = args.d_model
    if args.n_layers is not None:
        cfg.n_layers = args.n_layers
    
    # ═══ Model ═══
    model = TarsHelixLite(cfg).to(device)
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # ═══ Dataset ═══
    if args.data and os.path.exists(args.data):
        # TODO: integrate proper tokenizer
        logger.info(f"Text file training not yet implemented — using random data")
        dataset = RandomTokenDataset(cfg.vocab_size, args.seq_len, n_samples=args.steps * args.batch_size)
    else:
        logger.info("No data file — using random tokens (test training)")
        dataset = RandomTokenDataset(cfg.vocab_size, args.seq_len, n_samples=args.steps * args.batch_size)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
    )
    
    # ═══ Optimizer (AdamW for now, Muon is Phase 2) ═══
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    
    total_steps = len(dataloader) * args.epochs
    scheduler = get_cosine_schedule(
        optimizer, 
        warmup_steps=min(100, total_steps // 10),
        total_steps=total_steps,
        min_lr_ratio=cfg.lr_min / cfg.lr,
    )
    
    # ═══ Training Loop ═══
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {args.epochs} epochs, {len(dataloader)} steps/epoch")
    logger.info(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    logger.info(f"  Seq len: {args.seq_len}, Batch: {args.batch_size}")
    logger.info(f"  LR: {cfg.lr}, Weight decay: {cfg.weight_decay}")
    logger.info(f"  Device: {device}")
    logger.info(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, device, cfg, epoch)
        
        logger.info(f"\n--- Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} ---\n")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.save_dir, exist_ok=True)
            ckpt_path = os.path.join(args.save_dir, "checkpoint_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': cfg.to_dict(),
            }, ckpt_path)
            logger.info(f"Saved best checkpoint: {ckpt_path} (loss: {avg_loss:.4f})")
    
    logger.info(f"\nTraining complete! Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoint: {os.path.join(args.save_dir, 'checkpoint_best.pt')}")


if __name__ == "__main__":
    main()
