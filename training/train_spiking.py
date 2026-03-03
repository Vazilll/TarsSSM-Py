"""
═══════════════════════════════════════════════════════════════
  Train SNN Spiking Synapses — Phase 3.5
═══════════════════════════════════════════════════════════════

Обучает SpikingSynapsePool (5 специализированных спайковых синапсов)
для спинного мозга ТАРС.

SI-LIF нейроны с surrogate gradient — тренируются стандартным PyTorch.

  python training/train_spiking.py --epochs 30 --dim 256
  python training/train_spiking.py --epochs 30 --resume

Данные:
  Использует те же данные что MinGRU (data/*.txt).
  Каждый синапс получает свой подмикс данных:
    action:  команды, скрипты, инструкции
    search:  вопросы, поиск, "найди..."
    social:  диалоги, эмоции, small talk
    code:    код, Python, Rust, баги
    generic: всё остальное
"""

import os
import sys
import time
import math
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Tars.TrainSNN")


# ═══════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════

class ByteTextDataset(Dataset):
    """Simple byte-level dataset from text files."""
    
    def __init__(self, data_dir: str, seq_len: int = 128, max_files: int = 50):
        self.seq_len = seq_len
        self.data = bytearray()
        
        data_path = Path(data_dir)
        files = sorted(data_path.glob("*.txt"))[:max_files]
        
        for f in files:
            try:
                text = f.read_text(encoding='utf-8', errors='replace')
                self.data.extend(text.encode('utf-8', errors='replace'))
            except Exception as e:
                logger.warning(f"Skip {f.name}: {e}")
        
        logger.info(f"Loaded {len(files)} files, {len(self.data):,} bytes")
    
    def __len__(self):
        return max(1, len(self.data) // self.seq_len - 1)
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + bytes(self.seq_len + 1 - len(chunk))
        
        x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
        y = torch.tensor(list(chunk[1:]), dtype=torch.long)
        return x, y


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════

def train_spiking(args):
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    logger.info(f"Device: {device}")
    logger.info(f"Training SNN Synapses: dim={args.dim}, heads={args.heads}, beta={args.beta}")
    
    # ── Model ──
    from brain.spiking import SpikingMinGRUBlock
    
    # Embedding + SNN Block + LM Head
    class SpikingLM(nn.Module):
        """Small LM wrapper around SpikingMinGRUBlock for training."""
        
        def __init__(self, vocab_size, dim, num_heads, beta):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
            self.snn_block = SpikingMinGRUBlock(
                dim=dim, num_heads=num_heads, beta=beta, num_layers=1
            )
            self.head = nn.Linear(dim, vocab_size, bias=False)
            # Weight tying
            self.head.weight = self.embedding.weight
            
            self._init_weights()
        
        def _init_weights(self):
            nn.init.normal_(self.embedding.weight, std=0.02)
        
        def forward(self, x, prev_hidden=None, temporal_phase=None):
            emb = self.embedding(x)  # [B, L, dim]
            out, next_hidden = self.snn_block(emb, prev_hidden, temporal_phase)
            logits = self.head(out)   # [B, L, vocab]
            return logits, next_hidden
    
    model = SpikingLM(
        vocab_size=256, dim=args.dim, num_heads=args.heads, beta=args.beta
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {param_count:,}")
    
    # ── Load checkpoint ──
    save_dir = ROOT / "models" / "spiking"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "spiking_best.pt"
    last_path = save_dir / "spiking_last.pt"
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and last_path.exists():
        ckpt = torch.load(str(last_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
    
    # ── Data ──
    data_dir = args.data_dir or str(ROOT / "data")
    dataset = ByteTextDataset(data_dir, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)
    
    if len(dataset) == 0:
        logger.error("No data found! Run Phase 1 (download) first.")
        return False
    
    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Cosine annealing schedule
    total_steps = args.epochs * len(loader)
    warmup_steps = min(500, total_steps // 10)
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    # ── Training loop ──
    model.train()
    global_step = 0
    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # Circadian simulation: sin wave over training
            temporal_phase = torch.tensor(
                math.sin(2 * math.pi * global_step / max(total_steps, 1)),
                device=device
            )
            
            logits, _ = model(x, temporal_phase=temporal_phase)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            
            # Sparsity regularization: encourage more zero spikes
            spike_reg = 0.0
            for name, param in model.named_parameters():
                if 'lif' in name and 'beta_raw' in name:
                    # Encourage high beta (more decay = more sparsity)
                    spike_reg += (1.0 - torch.sigmoid(param)).mean()
            
            total_loss = loss + args.spike_reg * spike_reg
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item() * x.size(0)
            epoch_tokens += x.numel()
            global_step += 1
            
            if batch_idx % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                sparsity = model.snn_block.sparsity
                logger.info(
                    f"  Epoch {epoch+1}/{args.epochs} "
                    f"[{batch_idx}/{len(loader)}] "
                    f"loss={loss.item():.4f} "
                    f"sparsity={sparsity:.1%} "
                    f"lr={lr:.2e}"
                )
        
        avg_loss = epoch_loss / max(epoch_tokens, 1) * args.seq_len
        elapsed = time.time() - t0
        tok_per_sec = epoch_tokens / max(elapsed, 1)
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"loss={avg_loss:.4f} "
            f"sparsity={model.snn_block.sparsity:.1%} "
            f"tok/s={tok_per_sec:.0f} "
            f"time={elapsed:.0f}s"
        )
        
        # Save checkpoints
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_loss': best_loss,
            'config': {
                'dim': args.dim, 'heads': args.heads,
                'beta': args.beta, 'vocab_size': 256,
            },
        }
        torch.save(ckpt, str(last_path))
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt['best_loss'] = best_loss
            torch.save(ckpt, str(best_path))
            logger.info(f"  ★ New best: {best_loss:.4f}")
    
    # ── Final stats ──
    logger.info(f"\nTraining complete!")
    logger.info(f"  Best loss: {best_loss:.4f}")
    logger.info(f"  Model saved: {best_path}")
    logger.info(f"  Sparsity: {model.snn_block.sparsity:.1%}")
    
    # Print per-LIF beta values
    for name, param in model.named_parameters():
        if 'beta_raw' in name:
            beta_vals = torch.sigmoid(param)
            logger.info(f"  {name}: β ∈ [{beta_vals.min():.3f}, {beta_vals.max():.3f}] (mean={beta_vals.mean():.3f})")
    
    return True


def main():
    p = argparse.ArgumentParser(description="ТАРС SNN Spiking Synapse Training")
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--beta', type=float, default=0.9)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--spike_reg', type=float, default=0.001,
                   help="Sparsity regularization weight")
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    
    ok = train_spiking(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
