"""
═══════════════════════════════════════════════════════════════
  DoubtEngine Training Script
═══════════════════════════════════════════════════════════════

Standalone training for DoubtEngine CoherenceHead.
SafetyHead = hardcoded rules, RepeatHead = n-gram formula.

Dataset: пары (query_emb, response_emb) → label (good=1, shuffled=0)
Brain model is frozen — only used to generate embeddings.

Usage:
  python training/train_doubt.py --brain_model models/mamba2/mamba2_omega.pt --epochs 10
  python training/train_doubt.py --brain_model models/mamba2/mamba2_omega.pt --data data/train_corpus.txt
  python training/train_doubt.py --epochs 1 --dry_run  # smoke test
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path

# Fix Windows encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("training.doubt")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description="DoubtEngine CoherenceHead Training")
    p.add_argument('--brain_model', type=str, default=None,
                   help="Path to pretrained Brain checkpoint (for embedding extraction)")
    p.add_argument('--data', type=str, default=None,
                   help="Path to training corpus (.txt)")
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=2048,
                   help="Brain embedding dimension (must match Brain model)")
    p.add_argument('--save_dir', type=str, default='models/doubt')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--neg_ratio', type=float, default=1.0,
                   help="Ratio of negative (shuffled) to positive pairs (1.0 = balanced)")
    p.add_argument('--dry_run', action='store_true',
                   help="Smoke test with synthetic data (no model/data needed)")
    p.add_argument('--max_pairs', type=int, default=10000,
                   help="Maximum number of training pairs to generate")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  Dataset: Contrastive Coherence Pairs
# ═══════════════════════════════════════════════════════════════

class CoherencePairDataset(torch.utils.data.Dataset):
    """
    Dataset of (query_emb ⊕ response_emb) → coherence label.
    
    Positive pairs: Brain hidden state from (query, matching response) → label=1
    Negative pairs: Brain hidden state from (query, random response) → label=0

    DoubtEngine CoherenceHead Training

    Per TZ Section 2.2a:
      - CoherenceHead: contrastive pairs (query, response) → good/bad — NEURAL, trained here
      - SafetyHead: hardcoded regex rules in SafetyGate — NOT neural, NOT trained
      - RepeatHead: n-gram overlap formula — NOT neural, NOT trained
    """
    
    def __init__(self, query_embs: torch.Tensor, response_embs: torch.Tensor,
                 labels: torch.Tensor):
        """
        Args:
            query_embs: [N, d_model]
            response_embs: [N, d_model]  
            labels: [N] — 1.0 for genuine, 0.0 for shuffled
        """
        assert len(query_embs) == len(response_embs) == len(labels)
        self.query_embs = query_embs
        self.response_embs = response_embs
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (self.query_embs[idx], self.response_embs[idx], self.labels[idx])


def generate_synthetic_pairs(n_pairs: int, d_model: int) -> CoherencePairDataset:
    """
    Generate synthetic training pairs for dry-run testing.
    
    Positive: similar embeddings (cosine > 0.8)
    Negative: random embeddings (cosine ≈ 0)
    """
    logger.info(f"Generating {n_pairs} synthetic pairs (d_model={d_model})")
    
    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos
    
    # Positive pairs: q and r are correlated
    base = torch.randn(n_pos, d_model)
    noise = torch.randn(n_pos, d_model) * 0.2
    q_pos = F.normalize(base, dim=-1)
    r_pos = F.normalize(base + noise, dim=-1)
    
    # Negative pairs: q and r are random
    q_neg = F.normalize(torch.randn(n_neg, d_model), dim=-1)
    r_neg = F.normalize(torch.randn(n_neg, d_model), dim=-1)
    
    query_embs = torch.cat([q_pos, q_neg], dim=0)
    response_embs = torch.cat([r_pos, r_neg], dim=0)
    labels = torch.cat([
        torch.ones(n_pos),
        torch.zeros(n_neg),
    ])
    
    # Shuffle
    perm = torch.randperm(n_pairs)
    return CoherencePairDataset(
        query_embs[perm], response_embs[perm], labels[perm]
    )


def generate_pairs_from_brain(
    brain_model,
    tokenizer,
    corpus: str,
    d_model: int,
    max_pairs: int = 10000,
    neg_ratio: float = 1.0,
    device: str = 'cpu',
) -> CoherencePairDataset:
    """
    Generate contrastive pairs using Brain model embeddings.
    
    For each paragraph in corpus:
      1. Split into (first_half=query, second_half=response) 
      2. Get Brain embedding for each
      3. Genuine pair → label=1
      4. Shuffle response from different paragraph → label=0
    """
    logger.info("Generating coherence pairs from Brain embeddings...")
    
    # Split corpus into paragraphs
    paragraphs = [p.strip() for p in corpus.split('\n\n') if len(p.strip()) > 50]
    random.shuffle(paragraphs)
    paragraphs = paragraphs[:max_pairs * 2]  # limit
    
    brain_model.eval()
    query_embs = []
    response_embs = []
    
    with torch.no_grad():
        for i, para in enumerate(paragraphs):
            if len(query_embs) >= max_pairs:
                break
            
            # Split paragraph into query and response
            mid = len(para) // 2
            query_text = para[:mid]
            response_text = para[mid:]
            
            if len(query_text) < 20 or len(response_text) < 20:
                continue
            
            try:
                # Get embeddings from Brain
                q_ids = tokenizer.encode(query_text)[:256]
                r_ids = tokenizer.encode(response_text)[:256]
                
                q_input = torch.tensor([q_ids], dtype=torch.long, device=device)
                r_input = torch.tensor([r_ids], dtype=torch.long, device=device)
                
                # Use embedding layer + mean pooling
                q_emb = brain_model.embedding(q_input).mean(dim=1)  # [1, d_model]
                r_emb = brain_model.embedding(r_input).mean(dim=1)  # [1, d_model]
                
                query_embs.append(q_emb.cpu())
                response_embs.append(r_emb.cpu())
            except Exception as e:
                logger.debug(f"Skipped paragraph {i}: {e}")
                continue
            
            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i+1}/{len(paragraphs)} paragraphs, "
                           f"{len(query_embs)} pairs so far")
    
    if not query_embs:
        logger.warning("No pairs generated, falling back to synthetic")
        return generate_synthetic_pairs(1000, d_model)
    
    query_embs = torch.cat(query_embs, dim=0)      # [N, d_model]
    response_embs = torch.cat(response_embs, dim=0)  # [N, d_model]
    n_pos = len(query_embs)
    
    # Create negative pairs by shuffling responses
    n_neg = int(n_pos * neg_ratio)
    neg_indices = torch.randperm(n_pos)[:n_neg]
    # Shift by at least 1 to guarantee mismatch
    neg_shift = torch.randint(1, max(n_pos, 2), (n_neg,))
    neg_response_indices = (neg_indices + neg_shift) % n_pos
    
    all_query = torch.cat([query_embs, query_embs[neg_indices]], dim=0)
    all_response = torch.cat([response_embs, response_embs[neg_response_indices]], dim=0)
    all_labels = torch.cat([
        torch.ones(n_pos),
        torch.zeros(n_neg),
    ])
    
    # Shuffle
    perm = torch.randperm(len(all_labels))
    
    logger.info(f"Generated {n_pos} positive + {n_neg} negative = {len(all_labels)} pairs")
    
    return CoherencePairDataset(
        all_query[perm], all_response[perm], all_labels[perm]
    )


# ═══════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════

def train_doubt(args):
    """Train DoubtEngine CoherenceHead."""
    
    # Device
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    device = torch.device(device_str)
    logger.info(f"Device: {device}")
    
    # ── Import DoubtEngine ──
    try:
        from brain.doubt_engine import DoubtEngine
    except ImportError:
        logger.error("brain/doubt_engine.py not found! "
                     "Agent 1 must create it first (T01).")
        logger.info("Creating a minimal DoubtEngine for training...")
        
        # Minimal inline DoubtEngine for standalone training
        class DoubtEngine(nn.Module):
            """Minimal DoubtEngine (fallback if brain/doubt_engine.py not yet created)."""
            def __init__(self, d_model, d_doubt=128):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Linear(d_model * 2, d_doubt), nn.SiLU(),
                    nn.Linear(d_doubt, d_doubt),
                )
                self.coherence_head = nn.Linear(d_doubt, 1)
                self.safety_head = nn.Linear(d_doubt, 1)
                self.repeat_head = nn.Linear(d_doubt, 1)
            
            def forward(self, query_emb, response_emb):
                x = torch.cat([query_emb, response_emb], dim=-1)
                h = self.stem(x)
                return {
                    'coherence': torch.sigmoid(self.coherence_head(h)).squeeze(-1),
                    'safety': torch.sigmoid(self.safety_head(h)).squeeze(-1),
                    'repetition': torch.sigmoid(self.repeat_head(h)).squeeze(-1),
                }
        
        logger.info("Using minimal fallback DoubtEngine")
    
    # ── Create DoubtEngine ──
    doubt_engine = DoubtEngine(args.d_model).to(device)
    n_params = sum(p.numel() for p in doubt_engine.parameters())
    logger.info(f"DoubtEngine: {n_params:,} params ({n_params/1e3:.1f}K)")
    
    # ── Load existing weights ──
    save_path = Path(args.save_dir) / "doubt_engine_best.pt"
    if save_path.exists():
        logger.info(f"Loading existing DoubtEngine weights: {save_path}")
        try:
            state = torch.load(str(save_path), map_location=device, weights_only=True)
            if isinstance(state, dict) and 'model_state_dict' in state:
                doubt_engine.load_state_dict(state['model_state_dict'], strict=False)
            else:
                doubt_engine.load_state_dict(state, strict=False)
            logger.info("DoubtEngine weights loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load weights: {e}, training from scratch")
    
    # ── Generate Dataset ──
    if args.dry_run:
        dataset = generate_synthetic_pairs(2000, args.d_model)
        logger.info("Dry run: using synthetic data")
    else:
        # Try to load Brain model for embedding generation
        brain_model = None
        tokenizer = None
        
        if args.brain_model and os.path.exists(args.brain_model):
            try:
                from brain.mamba2.model import TarsMamba2LM
                from brain.tokenizer import TarsTokenizer
                
                brain_model, _ = TarsMamba2LM.load_pretrained(
                    args.brain_model, device=device
                )
                brain_model.eval()
                for p in brain_model.parameters():
                    p.requires_grad = False
                tokenizer = TarsTokenizer()
                logger.info("Brain model loaded for embedding generation")
            except Exception as e:
                logger.warning(f"Failed to load Brain: {e}")
        
        # Load corpus
        corpus = None
        if args.data and os.path.exists(args.data):
            with open(args.data, 'r', encoding='utf-8', errors='replace') as f:
                corpus = f.read()
            logger.info(f"Corpus loaded: {len(corpus):,} chars")
        elif not args.data:
            # Try built-in corpus
            try:
                from training.train_corpus import get_training_text
                corpus = get_training_text()
                logger.info(f"Built-in corpus: {len(corpus):,} chars")
            except ImportError:
                pass
        
        if brain_model is not None and corpus:
            dataset = generate_pairs_from_brain(
                brain_model, tokenizer, corpus, args.d_model,
                max_pairs=args.max_pairs, neg_ratio=args.neg_ratio,
                device=device_str,
            )
            # Free Brain model memory
            del brain_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logger.info("No Brain model or corpus — using synthetic data")
            dataset = generate_synthetic_pairs(
                min(args.max_pairs, 5000), args.d_model
            )
    
    # ── Split train/test ──
    n_test = max(10, len(dataset) // 10)
    n_train = len(dataset) - n_test
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch, shuffle=False
    )
    
    logger.info(f"Dataset: {n_train} train + {n_test} test = {len(dataset)} pairs")
    
    # ── Optimizer (AdamW — separate from Brain) ──
    optimizer = torch.optim.AdamW(
        doubt_engine.parameters(), lr=args.lr, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    
    # ── Training loop ──
    best_acc = 0.0
    
    print(f"\n{'═' * 60}")
    print(f"  DoubtEngine CoherenceHead Training")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print(f"  Pairs: {n_train} train + {n_test} test")
    print(f"{'═' * 60}\n")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        doubt_engine.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for q_emb, r_emb, labels in train_loader:
            q_emb = q_emb.to(device)
            r_emb = r_emb.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = doubt_engine(q_emb, r_emb)
            coherence = outputs['coherence']  # [B]
            
            # Binary cross-entropy
            loss = F.binary_cross_entropy(coherence, labels)
            
            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(doubt_engine.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = (coherence > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
        
        scheduler.step()
        
        # ── Eval ──
        doubt_engine.eval()
        eval_loss = 0.0
        eval_correct = 0
        eval_total = 0
        
        with torch.no_grad():
            for q_emb, r_emb, labels in test_loader:
                q_emb = q_emb.to(device)
                r_emb = r_emb.to(device)
                labels = labels.to(device)
                
                outputs = doubt_engine(q_emb, r_emb)
                coherence = outputs['coherence']
                
                eval_loss += F.binary_cross_entropy(coherence, labels).item()
                preds = (coherence > 0.5).float()
                eval_correct += (preds == labels).sum().item()
                eval_total += labels.numel()
        
        train_acc = 100.0 * correct / max(total, 1)
        eval_acc = 100.0 * eval_correct / max(eval_total, 1)
        avg_train_loss = total_loss / max(len(train_loader), 1)
        avg_eval_loss = eval_loss / max(len(test_loader), 1)
        elapsed = time.time() - t0
        
        is_best = eval_acc > best_acc
        marker = " ★ BEST" if is_best else ""
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {avg_train_loss:.4f} (acc {train_acc:.1f}%) | "
              f"Eval: {avg_eval_loss:.4f} (acc {eval_acc:.1f}%) | "
              f"{elapsed:.1f}s{marker}")
        
        # Save best
        if is_best:
            best_acc = eval_acc
            os.makedirs(args.save_dir, exist_ok=True)
            checkpoint = {
                'model_state_dict': doubt_engine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'd_model': args.d_model,
                'epoch': epoch,
                'eval_acc': eval_acc,
                'eval_loss': avg_eval_loss,
            }
            torch.save(checkpoint, str(save_path))
            print(f"  ✓ Saved: {save_path} (acc={eval_acc:.1f}%)")
    
    print(f"\n{'═' * 60}")
    print(f"  Done! Best eval accuracy: {best_acc:.1f}%")
    print(f"  Weights: {save_path}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    args = parse_args()
    train_doubt(args)
