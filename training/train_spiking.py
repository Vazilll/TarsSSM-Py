"""
═══════════════════════════════════════════════════════════════
  Train SNN Spiking Synapses — Phase 3.5 (MAX GPU Edition)
═══════════════════════════════════════════════════════════════

Обучает SpikingMinGRUBlock (спайковые синапсы) для ТАРС.
SI-LIF нейроны с surrogate gradient.

Оптимизации GPU (как MinGRU):
  - Автоопределение max batch (бинарный поиск по VRAM)
  - AMP (bf16/fp16 autocast + GradScaler)
  - torch.compile (~20% speedup)
  - DataLoader: pin_memory + num_workers + persistent_workers
  - Data subsampling (SNN — System 1, не нужен полный 605MB корпус)
  - Gradient accumulation для стабильности
  - Train/eval split + early stopping

Advanced features (from train_utils):
  - TrainLogger:       JSONL logging (0 dependencies)
  - WSD Schedule:      Warmup-Stable-Decay (идеален для Colab)
  - Muon Optimizer:    2x faster via orthogonal updates
  - GradientMonitor:   Per-layer gradient norm tracking
  - SNNMetrics:        Spike rate, membrane potential, beta distribution
  - CurriculumSchedule: seq_len growth (64 → 128 → 256)
  - ThroughputTracker: Tokens/sec + VRAM monitoring

  python training/train_spiking.py --epochs 20 --dim 256
  python training/train_spiking.py --epochs 20 --schedule wsd --muon
  python training/train_spiking.py --epochs 20 --curriculum --use_memory
"""

import os
import sys
import time
import math
import gc
import argparse
import logging
import platform
from pathlib import Path

# sys.path BEFORE relative imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from training.train_utils import (
    TrainLogger, WSDSchedule, GradientMonitor, SNNMetrics,
    ThroughputTracker, CurriculumSchedule, make_optimizer, make_lr_schedule,
)

_IS_WINDOWS = platform.system() == 'Windows'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Tars.TrainSNN")


# ═══════════════════════════════════════════
# Dataset (оптимизированный)
# ═══════════════════════════════════════════

class ByteTextDataset(Dataset):
    """Byte-level dataset with data size limiting and train/eval split."""
    
    def __init__(self, data_dir: str, seq_len: int = 128, max_files: int = 50,
                 max_bytes: int = 0, split: str = 'train', split_ratio: float = 0.9):
        self.seq_len = seq_len
        self.data = bytearray()
        
        data_path = Path(data_dir)
        files = sorted(data_path.glob("*.txt"))[:max_files]
        
        for f in files:
            try:
                raw = f.read_bytes()
                self.data.extend(raw)
                # Лимит по размеру данных (SNN не нужен полный корпус)
                if max_bytes > 0 and len(self.data) >= max_bytes:
                    self.data = self.data[:max_bytes]
                    logger.info(f"Data limit reached: {max_bytes / 1024**2:.0f} MB "
                               f"(SNN — System 1, не нужен полный корпус)")
                    break
            except Exception as e:
                logger.warning(f"Skip {f.name}: {e}")
        
        logger.info(f"Loaded {len(files)} files, {len(self.data):,} bytes")
        
        # Train/eval split
        n_total = len(self.data) // seq_len - 1
        split_point = int(n_total * split_ratio)
        
        if split == 'train':
            self.offset = 0
            self.n_samples = split_point
        else:
            self.offset = split_point * seq_len
            self.n_samples = n_total - split_point
        
        logger.info(f"  {split} split: {self.n_samples:,} samples")
    
    def __len__(self):
        return max(1, self.n_samples)
    
    def __getitem__(self, idx):
        start = self.offset + idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + bytes(self.seq_len + 1 - len(chunk))
        
        x = torch.tensor(list(chunk[:-1]), dtype=torch.long)
        y = torch.tensor(list(chunk[1:]), dtype=torch.long)
        return x, y


# ═══════════════════════════════════════════
# GPU Optimization (порт из MinGRU)
# ═══════════════════════════════════════════

def _find_max_batch(model, device, seq_len):
    """Бинарный поиск максимального батча, влезающего в VRAM."""
    was_training = model.training
    model.train()
    lo, hi = 32, 2048
    best = 32
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            dummy_in = torch.randint(0, 256, (mid, seq_len), device=device)
            dummy_tgt = torch.randint(0, 256, (mid, seq_len), device=device)
            temporal = torch.tensor(0.0, device=device)
            with torch.amp.autocast(device.type):
                logits, _ = model(dummy_in, temporal_phase=temporal)
                loss = F.cross_entropy(logits.view(-1, 256), dummy_tgt.view(-1))
            loss.backward()
            del dummy_in, dummy_tgt, logits, loss
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            torch.cuda.empty_cache()
            best = mid
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError, AssertionError):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = None
            torch.cuda.empty_cache()
            hi = mid - 1
    if not was_training:
        model.eval()
    # 50% запас для optimizer states (AdamW хранит 2 копии)
    safe = max(32, int(best * 0.50))
    safe = (safe // 32) * 32  # Кратно 32
    return safe


def _get_data_limit_mb():
    """Лимит данных для SNN по GPU. SNN = System 1, НЕ нужен полный корпус."""
    try:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if vram_gb >= 35:   return 100  # A100
            elif vram_gb >= 20: return 50   # L4 / RTX 4090
            elif vram_gb >= 14: return 30   # T4
            else:               return 20   # маленький GPU
    except Exception as e:
        logger.debug(f"GPU detection error: {e}")
    return 20  # CPU


# ═══════════════════════════════════════════
# LEANN Memory Bridge (опционально)
# ═══════════════════════════════════════════

def _load_leann_embeddings():
    """Загружает LEANN int8 эмбеддинги для memory conditioning."""
    npz_path = ROOT / "memory" / "leann.npz"
    texts_path = ROOT / "memory" / "leann.texts.json"
    
    if not (npz_path.exists() and texts_path.exists()):
        # Проверяем альтернативные пути
        alt_npz = ROOT / ".." / "drive" / "MyDrive" / "TarsMemory" / "leann.npz"
        if alt_npz.exists():
            npz_path = alt_npz
            texts_path = alt_npz.parent / "leann.texts.json"
        else:
            return None, None
    
    try:
        import json
        data = np.load(str(npz_path))
        embs = data['embeddings']  # int8 [N, 384]
        scales = data['scales']     # float32 [N]
        with open(str(texts_path), 'r', encoding='utf-8') as f:
            texts = json.load(f)
        
        # Деквантизация → float32
        embs_f32 = embs.astype(np.float32) * scales[:, np.newaxis]
        logger.info(f"LEANN: Загружено {len(texts)} memory vectors ({embs_f32.shape})")
        return embs_f32, texts
    except Exception as e:
        logger.warning(f"LEANN load failed: {e}")
        return None, None


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════

def train_spiking(args):
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    
    logger.info(f"Training SNN Synapses: dim={args.dim}, heads={args.heads}, beta={args.beta}")
    
    # ── Model ──
    from brain.spiking import SpikingMinGRUBlock
    
    class SpikingLM(nn.Module):
        """LM wrapper around SpikingMinGRUBlock with optional memory conditioning."""
        
        def __init__(self, vocab_size, dim, num_heads, beta, num_layers=1, use_memory=False):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, dim)
            self.num_layers = num_layers
            
            # Stack multiple SNN layers
            self.snn_blocks = nn.ModuleList([
                SpikingMinGRUBlock(dim=dim, num_heads=num_heads, beta=beta, num_layers=num_layers)
                for _ in range(num_layers)
            ])
            
            self.head = nn.Linear(dim, vocab_size, bias=False)
            # Weight tying
            self.head.weight = self.embedding.weight
            
            # Optional LEANN memory bridge
            self.use_memory = use_memory
            if use_memory:
                self.memory_proj = nn.Linear(384, dim, bias=False)
                self.memory_gate = nn.Parameter(torch.tensor(0.1))  # learnable strength
            
            self._init_weights()
        
        def _init_weights(self):
            nn.init.normal_(self.embedding.weight, std=0.02)
            if self.use_memory:
                nn.init.normal_(self.memory_proj.weight, std=0.01)
        
        def forward(self, x, prev_hidden=None, temporal_phase=None, memory_vec=None):
            emb = self.embedding(x)  # [B, L, dim]
            
            # Memory conditioning: add context from LEANN
            if self.use_memory and memory_vec is not None:
                # memory_vec: [B, 384] → [B, 1, dim] → broadcast to [B, L, dim]
                mem_proj = self.memory_proj(memory_vec).unsqueeze(1)  # [B, 1, dim]
                emb = emb + self.memory_gate * mem_proj
            
            out = emb
            next_hiddens = []
            if prev_hidden is None:
                prev_hidden = [None] * self.num_layers
            elif not isinstance(prev_hidden, (list, tuple)):
                prev_hidden = [prev_hidden] * self.num_layers
            for i, block in enumerate(self.snn_blocks):
                out, nh = block(out, prev_hidden[i], temporal_phase)
                next_hiddens.append(nh)
            
            logits = self.head(out)   # [B, L, vocab]
            return logits, next_hiddens
        
        @property
        def snn_block(self):
            """Backward compatibility: return first SNN block."""
            return self.snn_blocks[0]
    
    model = SpikingLM(
        vocab_size=256, dim=args.dim, num_heads=args.heads, beta=args.beta,
        num_layers=args.num_layers, use_memory=args.use_memory
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {param_count:,} (layers={args.num_layers})")
    
    # ── Auto batch (ДО compile) ──
    if device.type == 'cuda':
        max_batch = _find_max_batch(model, device, args.seq_len)
        if max_batch > args.batch:
            logger.info(f"Max batch: {max_batch} (было {args.batch})")
            args.batch = max_batch
    
    # ── torch.compile ──
    compiled = False
    if device.type == 'cuda':
        try:
            model = torch.compile(model, mode="default")
            compiled = True
            logger.info("torch.compile: ON (default)")
        except Exception as e:
            logger.info(f"torch.compile: недоступен ({e})")
    
    # ── Gradient Accumulation ──
    accum_steps = 1
    effective_batch = args.batch
    if args.batch >= 512:
        accum_steps = args.batch // 256
        args.batch = 256
        effective_batch = args.batch * accum_steps
        logger.info(f"Grad Accum: {accum_steps} steps → effective batch={effective_batch}")
    
    # ── Load checkpoint ──
    save_dir = ROOT / "models" / "spiking"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "spiking_best.pt"
    last_path = save_dir / "spiking_last.pt"
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and last_path.exists():
        ckpt = torch.load(str(last_path), map_location=device, weights_only=True)
        _raw = getattr(model, '_orig_mod', model)
        _raw.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_epoch = ckpt.get('epoch', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.4f}")
    
    # ── Data ──
    data_dir = args.data_dir or str(ROOT / "data")
    max_bytes = args.max_bytes * 1024 * 1024  # MB → bytes
    if max_bytes == 0:
        max_bytes = _get_data_limit_mb() * 1024 * 1024
        logger.info(f"Auto data limit: {max_bytes // 1024**2} MB")
    
    # num_workers=0 на Windows (spawn вместо fork → возможны зависания)
    use_pin = device.type == 'cuda'
    _num_workers = 0 if _IS_WINDOWS else (2 if use_pin else 0)
    _persist = use_pin and _num_workers > 0
    
    current_seq_len = args.seq_len
    
    def _build_loaders(seq_len):
        """Build train/eval dataloaders with given seq_len."""
        td = ByteTextDataset(data_dir, seq_len=seq_len, max_bytes=max_bytes, split='train')
        ed = ByteTextDataset(data_dir, seq_len=seq_len, max_bytes=max_bytes, split='eval')
        tl = DataLoader(
            td, batch_size=args.batch, shuffle=True,
            num_workers=_num_workers, drop_last=True,
            pin_memory=use_pin, persistent_workers=_persist,
        )
        el = DataLoader(
            ed, batch_size=args.batch * 2, shuffle=False,
            num_workers=0, pin_memory=use_pin,
        )
        return td, ed, tl, el
    
    train_dataset, eval_dataset, train_loader, eval_loader = _build_loaders(current_seq_len)
    
    if len(train_dataset) == 0:
        logger.error("No data found! Run Phase 1 (download) first.")
        return False
    
    # ── LEANN Memory (optional) ──
    leann_embs, leann_texts = None, None
    if args.use_memory:
        leann_embs, leann_texts = _load_leann_embeddings()
        if leann_embs is not None:
            leann_embs_t = torch.tensor(leann_embs, dtype=torch.float32, device=device)
            logger.info(f"LEANN memory: {len(leann_texts)} vectors ready for conditioning")
        else:
            logger.warning("LEANN memory not found, training without memory conditioning")
            args.use_memory = False
    
    # ── Optimizer (train_utils) ──
    optimizer = make_optimizer(
        model, lr=args.lr, weight_decay=0.01,
        use_muon=args.muon, muon_lr=args.lr * 10,
    )
    
    # ── LR Schedule (cosine or WSD) ──
    total_steps = args.epochs * len(train_loader) // accum_steps
    warmup_steps = min(1000, total_steps // 10)
    scheduler = make_lr_schedule(
        optimizer, total_steps=total_steps, warmup_steps=warmup_steps,
        schedule=args.schedule, min_lr_ratio=0.1,
    )
    
    # ── Mixed Precision ──
    device_type = device.type
    use_amp = device_type == 'cuda'
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler(device_type, enabled=(use_amp and amp_dtype == torch.float16))
    
    # ── Monitoring tools ──
    train_log = TrainLogger(
        path=str(save_dir / "snn_train.jsonl"),
        model_name=f"SNN-{args.dim}d-{args.heads}H-{args.num_layers}L",
    )
    train_log.log_config(
        dim=args.dim, heads=args.heads, beta=args.beta,
        num_layers=args.num_layers, lr=args.lr, batch=args.batch,
        effective_batch=effective_batch, schedule=args.schedule,
        muon=args.muon, use_memory=args.use_memory,
        seq_len=args.seq_len, total_steps=total_steps,
    )
    grad_monitor = GradientMonitor(model, check_every=max(1, total_steps // 20))
    snn_metrics = SNNMetrics(model)
    throughput = ThroughputTracker()
    
    # ── Curriculum (optional) ──
    curriculum = None
    if args.curriculum:
        curriculum = CurriculumSchedule(
            seq_min=64, seq_max=args.seq_len, total_epochs=args.epochs,
        )
        logger.info(f"Curriculum: {curriculum}")
    
    # ── Banner ──
    opt_name = 'Muon' if args.muon else 'AdamW'
    print(f"\n{'═'*70}")
    print(f"  ТАРС SNN Training (MAX GPU + Advanced)")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch}×{accum_steps} = {effective_batch}")
    print(f"  Device: {device} | AMP: {amp_dtype if use_amp else 'off'} | Compiled: {compiled}")
    print(f"  Optimizer: {opt_name} | Schedule: {args.schedule.upper()}")
    print(f"  Layers: {args.num_layers} | Memory: {args.use_memory}")
    print(f"  Curriculum: {curriculum or 'off'} | Batches/epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps} | Log: {save_dir / 'snn_train.jsonl'}")
    print(f"{'═'*70}\n")
    
    # ── Training loop ──
    global_step = 0
    patience_counter = 0
    training_start = time.time()
    
    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        
        # ── Curriculum: update seq_len if needed ──
        if curriculum is not None:
            new_seq_len = curriculum.get_seq_len(epoch - start_epoch)
            if new_seq_len != current_seq_len:
                current_seq_len = new_seq_len
                logger.info(f"Curriculum: seq_len → {current_seq_len}")
                train_dataset, eval_dataset, train_loader, eval_loader = _build_loaders(current_seq_len)
        
        model.train()
        
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        total_batches = len(train_loader)
        log_every = max(1, total_batches // 10)  # ~10 логов на эпоху
        
        for batch_idx, (x, y) in enumerate(train_loader):
            throughput.start_step()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Circadian simulation
            temporal_phase = torch.tensor(
                math.sin(2 * math.pi * global_step / max(total_steps, 1)),
                device=device
            )
            
            # Memory conditioning (optional)
            memory_vec = None
            if args.use_memory and leann_embs is not None:
                with torch.no_grad():
                    rand_idx = torch.randint(0, len(leann_embs_t), (x.size(0),))
                    memory_vec = leann_embs_t[rand_idx]  # [B, 384]
            
            with torch.amp.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(x, temporal_phase=temporal_phase, memory_vec=memory_vec)
                loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                
                # Sparsity regularization
                spike_reg = 0.0
                _raw = getattr(model, '_orig_mod', model)
                for name, param in _raw.named_parameters():
                    if 'lif' in name and 'beta_raw' in name:
                        spike_reg += (1.0 - torch.sigmoid(param)).mean()
                
                total_loss = (loss + args.spike_reg * spike_reg) / accum_steps
            
            scaler.scale(total_loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Gradient monitoring (every N steps)
                if global_step % grad_monitor.check_every == 0:
                    grad_monitor.log_to(train_log, global_step)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            throughput.end_step(x.numel())
            
            if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
                avg_loss = epoch_loss / max(n_batches, 1)
                lr = optimizer.param_groups[0]['lr']
                pct = (batch_idx + 1) / total_batches * 100
                _raw_log = getattr(model, '_orig_mod', model)
                sparsity = _raw_log.snn_block.sparsity if hasattr(_raw_log, 'snn_block') else 0
                tps = throughput.format_status()
                gpu_str = ""
                if device.type == 'cuda':
                    gpu_gb = torch.cuda.memory_allocated() / 1024**3
                    gpu_str = f" | VRAM: {gpu_gb:.1f}GB"
                eta_s = (total_batches - batch_idx - 1) * (time.time() - epoch_start) / max(batch_idx + 1, 1)
                print(f"  [{epoch+1:2d}] {batch_idx+1:6d}/{total_batches} ({pct:4.1f}%) | "
                      f"Loss: {avg_loss:.4f} | Sparsity: {sparsity:.1%} | "
                      f"LR: {lr:.2e} | {tps} | ETA: {eta_s/60:.1f}min{gpu_str}", flush=True)
                
                # Log to JSONL
                train_log.log(global_step, loss=round(avg_loss, 4), lr=lr,
                             sparsity=round(sparsity, 4),
                             tokens_per_sec=throughput.tokens_per_sec,
                             epoch=epoch+1, batch=batch_idx+1)
        
        # Flush remaining gradients
        if n_batches % accum_steps != 0:
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        
        # ── Eval ──
        model.eval()
        eval_loss = 0.0
        eval_n_batches = 0
        with torch.no_grad():
            for x, y in eval_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                temporal = torch.tensor(0.0, device=device)
                with torch.amp.autocast(device_type, dtype=amp_dtype, enabled=use_amp):
                    logits, _ = model(x, temporal_phase=temporal)
                    loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
                eval_loss += loss.item()
                eval_n_batches += 1
        
        avg_eval_loss = eval_loss / max(eval_n_batches, 1) if eval_n_batches > 0 else float('inf')
        eval_ppl = np.exp(min(avg_eval_loss, 20))
        _raw = getattr(model, '_orig_mod', model)
        sparsity = _raw.snn_block.sparsity if hasattr(_raw, 'snn_block') else 0
        
        total_elapsed = time.time() - training_start
        remaining = (args.epochs - (epoch - start_epoch + 1)) * epoch_time
        
        is_best = avg_eval_loss < best_loss
        marker = " ★ BEST" if is_best else ""
        
        # SNN-specific metrics
        snn_report = snn_metrics.collect()
        snn_metrics.reset_counters()
        
        print(f"\n{'─'*70}")
        print(f"Эпоха {epoch+1:3d}/{start_epoch + args.epochs} | "
              f"Train: {avg_train_loss:.4f} | Eval: {avg_eval_loss:.4f} | "
              f"PPL: {eval_ppl:.1f} | Sparsity: {sparsity:.1%} | "
              f"{epoch_time:.0f}s{marker}")
        print(f"  ⏱ Прошло: {total_elapsed/60:.0f} мин | Осталось: ~{remaining/60:.0f} мин")
        print(f"  📊 {throughput.format_status()} | Total: {throughput.report()['total_tokens']:,} tokens")
        
        # Print SNN metrics
        for key, val in sorted(snn_report.items()):
            if isinstance(val, dict):
                print(f"  🧬 {key}: [{val.get('min', '?')}, {val.get('max', '?')}] mean={val.get('mean', '?')}")
            elif isinstance(val, float):
                print(f"  🧬 {key}: {val:.4f}")
        print(f"{'─'*70}\n", flush=True)
        
        # Log epoch to JSONL
        train_log.log_epoch(
            epoch+1, train_loss=round(avg_train_loss, 4),
            eval_loss=round(avg_eval_loss, 4), ppl=round(eval_ppl, 2),
            sparsity=round(sparsity, 4), epoch_time=round(epoch_time, 1),
            **{k: v for k, v in snn_report.items() if not isinstance(v, dict)},
        )
        
        # ── Save checkpoints ──
        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': _raw.state_dict(),
            'best_loss': best_loss,
            'config': {
                'dim': args.dim, 'heads': args.heads,
                'beta': args.beta, 'vocab_size': 256,
                'num_layers': args.num_layers,
                'use_memory': args.use_memory,
            },
        }
        torch.save(ckpt, str(last_path))
        
        if is_best:
            best_loss = avg_eval_loss
            ckpt['best_loss'] = best_loss
            torch.save(ckpt, str(best_path))
            patience_counter = 0
            logger.info(f"  ★ New best: {best_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  ⏹ Early stopping: {args.patience} эпох без улучшения (best={best_loss:.4f})")
            break
    
    # ── Final stats ──
    total_time = time.time() - training_start
    print(f"\n{'═'*70}")
    print(f"  Обучение завершено за {total_time/60:.1f} минут")
    print(f"  Best loss: {best_loss:.4f} (PPL: {np.exp(min(best_loss, 20)):.1f})")
    print(f"  Model saved: {best_path}")
    if device.type == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Peak VRAM: {peak:.1f} GB / {gpu_mem:.1f} GB ({peak/gpu_mem*100:.0f}%)")
    
    _raw = getattr(model, '_orig_mod', model)
    sparsity = _raw.snn_block.sparsity if hasattr(_raw, 'snn_block') else 0
    print(f"  Sparsity: {sparsity:.1%}")
    
    # Print per-LIF beta values
    for name, param in _raw.named_parameters():
        if 'beta_raw' in name:
            beta_vals = torch.sigmoid(param)
            logger.info(f"  {name}: β ∈ [{beta_vals.min():.3f}, {beta_vals.max():.3f}] "
                       f"(mean={beta_vals.mean():.3f})")
    
    # Final throughput report
    tp = throughput.report()
    print(f"  ⚡ Throughput: {tp['tokens_per_sec']:,} tok/s | {tp['avg_step_ms']:.1f} ms/step")
    print(f"  📊 Total tokens: {tp['total_tokens']:,} | Time: {tp['total_time_min']:.1f} min")
    
    # Close logger
    train_log.summary()
    print(f"{'═'*70}\n")
    
    return True


def main():
    p = argparse.ArgumentParser(description="ТАРС SNN Spiking Synapse Training (MAX GPU)")
    p.add_argument('--dim', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--beta', type=float, default=0.9)
    p.add_argument('--num_layers', type=int, default=1,
                   help="Number of SNN layers (1-3)")
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=32,
                   help="Starting batch size (auto-increased)")
    p.add_argument('--seq_len', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--spike_reg', type=float, default=0.003,
                   help="Sparsity regularization weight")
    p.add_argument('--max_bytes', type=int, default=0,
                   help="Max data size in MB (0=auto by GPU)")
    p.add_argument('--patience', type=int, default=5,
                   help="Early stopping patience (0=off)")
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--data_dir', type=str, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--use_memory', action='store_true',
                   help="Enable LEANN memory conditioning")
    # Advanced training features
    p.add_argument('--schedule', type=str, default='cosine',
                   choices=['cosine', 'wsd', 'constant'],
                   help="LR schedule: cosine (default), wsd (warmup-stable-decay), constant")
    p.add_argument('--muon', action='store_true',
                   help="Use Muon optimizer (2x faster via orthogonal updates)")
    p.add_argument('--curriculum', action='store_true',
                   help="Curriculum learning: seq_len grows 64→128→256")
    args = p.parse_args()
    
    ok = train_spiking(args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
