"""
═══════════════════════════════════════════════════════════════
  Knowledge Distillation — обучение TARS на soft logits учителя
═══════════════════════════════════════════════════════════════

Техника: Strong-to-Weak Distillation (Qwen3/Llama/Phi-4 подход).

Student (TARS 1B) учится предсказывать:
  1. Hard targets: настоящие токены (CE loss)
  2. Soft targets: logit distribution учителя (KL loss)

  L = α × KL(student_logits/T || teacher_logits/T) + (1-α) × CE(student, labels)

Где T — temperature (обычно 2-4), α — вес soft loss (0.5-0.9).

Использование:
  # Сначала сгенерировать soft targets от учителя (Qwen3-4B):
  python training/train_distill.py --teacher_logits data/teacher_logits.pt

  # Или дистиллировать on-the-fly (медленнее, но проще):
  python training/train_distill.py --teacher_model Qwen/Qwen2.5-1.5B
"""

import argparse
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="TARS Knowledge Distillation Training")
    # Model
    p.add_argument('--d_model', type=int, default=2048)
    p.add_argument('--n_layers', type=int, default=24)
    # Training
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--device', type=str, default='auto')
    # Distillation
    p.add_argument('--temperature', type=float, default=3.0,
                   help="Distillation temperature (2-4 optimal)")
    p.add_argument('--alpha', type=float, default=0.7,
                   help="Weight of soft loss (0.7 = 70% teacher, 30% hard labels)")
    p.add_argument('--teacher_logits', type=str, default=None,
                   help="Pre-computed teacher logits file (.pt)")
    p.add_argument('--teacher_model', type=str, default=None,
                   help="HuggingFace teacher model for on-the-fly distillation")
    # Paths
    p.add_argument('--data', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='models/mamba2')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--grad_ckpt', action='store_true')
    return p.parse_args()


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    Combined distillation + hard label loss.
    
    Args:
        student_logits: [B, L, V] — student model output
        teacher_logits: [B, L, V] — teacher model output  
        labels: [B, L] — ground truth token ids
        temperature: float — softmax temperature
        alpha: float — weight of soft loss
    
    Returns:
        total_loss, soft_loss, hard_loss
    """
    V = student_logits.size(-1)
    
    # Soft targets: KL divergence with temperature scaling
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    # KL(P||Q) where P=teacher, Q=student
    soft_loss = F.kl_div(
        soft_student.view(-1, V),
        soft_teacher.view(-1, V),
        reduction='batchmean'
    ) * (temperature ** 2)  # Scale by T² to match gradient magnitudes
    
    # Hard targets: standard cross-entropy
    hard_loss = F.cross_entropy(
        student_logits.view(-1, V),
        labels.view(-1),
        label_smoothing=0.1,
    )
    
    # Combined loss
    total = alpha * soft_loss + (1 - alpha) * hard_loss
    return total, soft_loss.item(), hard_loss.item()


def load_teacher_hf(model_name, device):
    """Load HuggingFace teacher model for on-the-fly distillation."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[Teacher] Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        teacher = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        n_params = sum(p.numel() for p in teacher.parameters())
        print(f"[Teacher] {model_name}: {n_params/1e6:.0f}M params loaded")
        return teacher, tokenizer
    except Exception as e:
        print(f"[!] Failed to load teacher: {e}")
        return None, None


def generate_teacher_logits_online(teacher, tokenizer, text_batch, seq_len, device):
    """Generate teacher logits for a batch of text on-the-fly."""
    # Tokenize with teacher's tokenizer
    encoded = tokenizer(
        text_batch,
        max_length=seq_len + 1,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    ).to(device)
    
    input_ids = encoded['input_ids'][:, :-1]
    
    with torch.no_grad():
        outputs = teacher(input_ids)
        # Only keep top-k logits to save memory (top-32 is enough)
        logits = outputs.logits
        # Compress: keep only top-32 values, zero rest
        topk_vals, topk_idx = logits.topk(32, dim=-1)
        compressed = torch.zeros_like(logits).scatter_(-1, topk_idx, topk_vals)
        
    return compressed


def train(args):
    """Knowledge Distillation training loop."""
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    device = torch.device(device_str)
    
    # Load student model (TARS)
    from brain.mamba2.model import TarsMamba2LM
    
    actual_vocab = 256
    student = TarsMamba2LM(
        d_model=args.d_model,
        n_layers=args.n_layers,
        vocab_size=actual_vocab,
        quant_mode="fp16",
    )
    
    # Resume from pre-trained weights
    save_path = Path(args.save_dir) / "mamba2_omega.pt"
    if args.resume and save_path.exists():
        print(f"[Student] Loading checkpoint: {save_path}")
        ckpt = torch.load(str(save_path), map_location='cpu', weights_only=False)
        student.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    if args.grad_ckpt:
        student.use_checkpointing = True
    student.to(device)
    
    params = sum(p.numel() for p in student.parameters())
    print(f"[Student] TARS: {params/1e6:.1f}M params")
    
    # Load teacher
    teacher = None
    tokenizer = None
    teacher_logits_cache = None
    
    if args.teacher_logits and os.path.exists(args.teacher_logits):
        print(f"[Teacher] Loading pre-computed logits: {args.teacher_logits}")
        teacher_logits_cache = torch.load(args.teacher_logits, map_location='cpu')
        print(f"[Teacher] {len(teacher_logits_cache)} samples loaded")
    elif args.teacher_model:
        teacher, tokenizer = load_teacher_hf(args.teacher_model, device)
    else:
        print("[!] No teacher specified. Use --teacher_logits or --teacher_model")
        print("[!] Falling back to self-distillation (label smoothing only)")
    
    # Data
    from training.train_mamba2 import load_corpus, prepare_byte_data
    corpus = load_corpus(data_path=args.data)
    inputs, targets = prepare_byte_data(corpus, args.seq_len, actual_vocab)
    
    # Train
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    
    print(f"\n{'═'*60}")
    print(f"  TARS Knowledge Distillation")
    print(f"  Temperature: {args.temperature} | Alpha: {args.alpha}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch}")
    print(f"{'═'*60}\n")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        student.train()
        total_loss = total_soft = total_hard = 0
        n = 0
        
        perm = torch.randperm(len(inputs))
        for i in range(0, len(inputs), args.batch):
            batch_in = inputs[perm[i:i+args.batch]].to(device)
            batch_tgt = targets[perm[i:i+args.batch]].to(device)
            
            # Student forward
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    student_logits = student(batch_in)
            else:
                student_logits = student(batch_in)
            
            # Get teacher logits
            if teacher_logits_cache is not None:
                # Pre-computed: map student vocab to teacher logits
                t_logits = teacher_logits_cache[perm[i:i+args.batch]].to(device)
                # Resize if vocab mismatch
                if t_logits.size(-1) != student_logits.size(-1):
                    t_logits = t_logits[:, :, :actual_vocab]
            elif teacher is not None:
                # On-the-fly: use teacher model
                with torch.no_grad():
                    t_out = teacher(batch_in)
                    t_logits = t_out.logits if hasattr(t_out, 'logits') else t_out
                    if t_logits.size(-1) != actual_vocab:
                        t_logits = t_logits[:, :, :actual_vocab]
            else:
                # Self-distillation: use student's own logits (detached)
                t_logits = student_logits.detach()
            
            # Distillation loss
            loss, sl, hl = distillation_loss(
                student_logits, t_logits, batch_tgt,
                args.temperature, args.alpha
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            total_soft += sl
            total_hard += hl
            n += 1
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {total_loss/n:.4f} | "
              f"Soft: {total_soft/n:.4f} | Hard: {total_hard/n:.4f} | "
              f"{elapsed:.1f}s")
        
        # Save
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': student.state_dict(),
            'config': {
                'd_model': args.d_model, 'n_layers': args.n_layers,
                'vocab_size': actual_vocab, 'distill_from': args.teacher_model or 'logits',
            },
            'epoch': epoch,
        }, str(save_path))
        print(f"  ✓ Saved: {save_path}")
    
    print(f"\n{'═'*60}")
    print(f"  Distillation complete!")
    print(f"{'═'*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
