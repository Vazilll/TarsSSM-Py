"""
═══════════════════════════════════════════════════════════════
  ТАРС Training Utilities — Shared Infrastructure
═══════════════════════════════════════════════════════════════

Общие утилиты для всех скриптов обучения ТАРС:
  - TrainLogger:        JSONL логгер без зависимостей
  - WSDSchedule:        Warmup-Stable-Decay LR schedule
  - Muon:               Muon optimizer (ортогональные обновления)
  - GradientMonitor:    Per-layer gradient norm tracking
  - CurriculumSampler:  Curriculum learning (seq_len growth)
  - ThroughputTracker:  Tokens/sec + VRAM monitoring
"""

import os
import sys
import time
import math
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger("Tars.TrainUtils")


# ═══════════════════════════════════════════
# 1. TrainLogger — JSONL логгер (0 зависимостей)
# ═══════════════════════════════════════════

class TrainLogger:
    """
    Лёгкий JSONL-based логгер обучения.
    
    Каждая строка — JSON объект с метриками одного шага.
    Файл можно читать в реальном времени: tail -f logs/train_log.jsonl | python -m json.tool
    
    Использование:
        log = TrainLogger("logs/snn_train.jsonl")
        log.log(step=0, loss=2.5, lr=1e-3, sparsity=0.7)
        log.log(step=100, loss=1.8, lr=9e-4, epoch=1)
        log.summary()
    """
    
    def __init__(self, path: str = "logs/train_log.jsonl", model_name: str = "unknown"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.start_time = time.time()
        self._entries = []
        self._best = {"loss": float("inf"), "step": 0}
        
        # Write header
        self._write({"event": "start", "model": model_name,
                      "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    
    def _write(self, entry: dict):
        entry["_t"] = round(time.time() - self.start_time, 2)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def log(self, step: int, **metrics):
        """Записать метрики шага."""
        entry = {"step": step, **metrics}
        self._write(entry)
        self._entries.append(entry)
        
        # Track best
        if "eval_loss" in metrics and metrics["eval_loss"] < self._best["loss"]:
            self._best = {"loss": metrics["eval_loss"], "step": step}
    
    def log_epoch(self, epoch: int, **metrics):
        """Записать метрики эпохи."""
        entry = {"epoch": epoch, **metrics}
        self._write(entry)
        self._entries.append(entry)
        
        if "eval_loss" in metrics and metrics["eval_loss"] < self._best["loss"]:
            self._best = {"loss": metrics["eval_loss"], "step": epoch}
    
    def log_config(self, **config):
        """Записать конфигурацию обучения."""
        self._write({"event": "config", **config})
    
    def log_gradient_norms(self, step: int, norms: Dict[str, float]):
        """Записать нормы градиентов per-layer."""
        self._write({"event": "grad_norms", "step": step, "norms": norms})
    
    def summary(self):
        """Печать итогового отчёта."""
        elapsed = time.time() - self.start_time
        print(f"\n  📊 Training Log: {self.path}")
        print(f"  ⏱  Duration: {elapsed/60:.1f} min")
        print(f"  📈 Steps logged: {len(self._entries)}")
        if self._best["loss"] < float("inf"):
            print(f"  ★  Best loss: {self._best['loss']:.4f} (step {self._best['step']})")
        self._write({"event": "end", "total_steps": len(self._entries),
                      "best_loss": self._best["loss"], "duration_sec": round(elapsed, 1)})


# ═══════════════════════════════════════════
# 2. WSD Schedule — Warmup-Stable-Decay
# ═══════════════════════════════════════════

class WSDSchedule:
    """
    Warmup-Stable-Decay LR schedule.
    
    Три фазы:
      1. Warmup:  linear ramp 0 → lr_max  (warmup_frac %)
      2. Stable:  constant lr_max          (middle %)
      3. Decay:   cosine lr_max → lr_min   (decay_frac %)
    
    Преимущество: можно прервать обучение в любой момент Stable фазы,
    затем быстрый decay → хорошая модель. Не нужно знать total_steps заранее.
    
    Использование:
        wsd = WSDSchedule(total_steps=10000, warmup_frac=0.05, decay_frac=0.10)
        for step in range(10000):
            lr_mult = wsd.get_lr_mult(step)
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * lr_mult
    """
    
    def __init__(self, total_steps: int, warmup_frac: float = 0.05,
                 decay_frac: float = 0.10, min_lr_ratio: float = 0.01):
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = int(total_steps * warmup_frac)
        self.decay_steps = int(total_steps * decay_frac)
        self.stable_steps = total_steps - self.warmup_steps - self.decay_steps
        self.min_lr_ratio = min_lr_ratio
    
    def get_lr_mult(self, step: int) -> float:
        """Возвращает множитель LR ∈ [min_lr_ratio, 1.0]."""
        if step < self.warmup_steps:
            # Phase 1: Linear warmup
            return max(self.min_lr_ratio, step / max(self.warmup_steps, 1))
        
        step_after_warmup = step - self.warmup_steps
        
        if step_after_warmup < self.stable_steps:
            # Phase 2: Stable (constant LR)
            return 1.0
        
        # Phase 3: Cosine decay
        decay_progress = (step_after_warmup - self.stable_steps) / max(self.decay_steps, 1)
        decay_progress = min(decay_progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return max(self.min_lr_ratio, cosine)
    
    def as_lambda(self):
        """Для использования с LambdaLR."""
        return lambda step: self.get_lr_mult(step)
    
    def __repr__(self):
        return (f"WSD(warmup={self.warmup_steps}, stable={self.stable_steps}, "
                f"decay={self.decay_steps}, total={self.total_steps})")


# ═══════════════════════════════════════════
# 3. Muon Optimizer
# ═══════════════════════════════════════════

class Muon(torch.optim.Optimizer):
    """
    Muon optimizer — ортогональные обновления через polar decomposition.
    
    Для матричных параметров (dim >= 2): SVD → U @ Vt (direction without scale).
    Для скалярных params (bias, gamma, beta): обычный AdamW.
    
    Рекомендации:
      - Использовать для Linear и Conv weights
      - НЕ использовать для Embedding, LayerNorm  
      - LR: 0.02 (для Muon) vs 3e-4 (для AdamW на остальных)
    
    Paper: "Muon: An optimizer for hidden layers in neural networks"
    
    Args:
        muon_params: params для Muon (матрицы)
        adam_params: params для AdamW (всё остальное)
        lr: learning rate для Muon part
        adam_lr: learning rate для Adam part (default: lr/10)
        momentum: momentum для Muon (default: 0.95)
        weight_decay: weight decay для обеих частей
    """
    
    def __init__(self, muon_params, adam_params=None,
                 lr=0.02, adam_lr=None, momentum=0.95,
                 weight_decay=0.01, nesterov=True):
        adam_lr = adam_lr or lr * 0.1
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, adam_lr=adam_lr)
        
        param_groups = []
        
        # Group 1: Muon params (matrices)
        muon_list = list(muon_params) if not isinstance(muon_params, list) else muon_params
        if muon_list:
            param_groups.append({
                "params": muon_list,
                "use_muon": True,
                "lr": lr,
            })
        
        # Group 2: Adam params (everything else)
        if adam_params is not None:
            adam_list = list(adam_params) if not isinstance(adam_params, list) else adam_params
            if adam_list:
                param_groups.append({
                    "params": adam_list,
                    "use_muon": False,
                    "lr": adam_lr,
                })
        
        super().__init__(param_groups, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            wd = group.get("weight_decay", 0.01)
            momentum = group.get("momentum", 0.95)
            nesterov = group.get("nesterov", True)
            use_muon = group.get("use_muon", False)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Weight decay (decoupled)
                if wd > 0:
                    p.data.mul_(1.0 - lr * wd)
                
                state = self.state[p]
                
                if use_muon and p.dim() >= 2:
                    # ── Muon: orthogonal update ──
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    
                    # Apply Nesterov momentum
                    if nesterov:
                        update = grad + momentum * buf
                    else:
                        update = buf
                    
                    # Newton-Schulz orthogonalization (faster than SVD)
                    # Approximates polar decomposition: W → U
                    update_2d = update.view(update.shape[0], -1) if update.dim() > 2 else update
                    
                    if update_2d.shape[0] <= update_2d.shape[1]:
                        # Широкая матрица: ортогонализируем строки
                        ortho = self._newton_schulz(update_2d, steps=5)
                    else:
                        # Высокая матрица: ортогонализируем столбцы
                        ortho = self._newton_schulz(update_2d.T, steps=5).T
                    
                    p.data.add_(ortho.view_as(p), alpha=-lr)
                else:
                    # ── AdamW для скалярных params ──
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                        state["step_count"] = 0
                    
                    state["step_count"] += 1
                    t = state["step_count"]
                    
                    state["exp_avg"].mul_(0.9).add_(grad, alpha=0.1)
                    state["exp_avg_sq"].mul_(0.999).addcmul_(grad, grad, value=0.001)
                    
                    bc1 = 1 - 0.9 ** t
                    bc2 = 1 - 0.999 ** t
                    
                    step_size = lr / bc1
                    denom = (state["exp_avg_sq"] / bc2).sqrt().add_(1e-8)
                    
                    p.data.addcdiv_(state["exp_avg"], denom, value=-step_size)
        
        return loss
    
    @staticmethod
    def _newton_schulz(M, steps=5):
        """Newton-Schulz iteration для приближения polar decomposition."""
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = M / (M.norm() + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = a * A + b * (A @ A) + c * (A @ A @ A)  # polynomial
            X = B @ X
        return X


def make_optimizer(model, lr=3e-4, weight_decay=0.01, use_muon=False, muon_lr=0.02):
    """
    Создаёт optimizer с правильным разделением параметров.
    
    Args:
        model: nn.Module
        lr: base learning rate (для AdamW или Adam-part Muon)
        weight_decay: для матричных параметров
        use_muon: использовать Muon для матриц
        muon_lr: LR для Muon-части (только если use_muon=True)
    
    Returns:
        optimizer
    """
    _raw = getattr(model, '_orig_mod', model)
    
    decay_params = []
    no_decay_params = []
    muon_params = []
    
    for name, param in _raw.named_parameters():
        if not param.requires_grad:
            continue
        
        is_matrix = param.dim() >= 2
        is_norm = 'norm' in name or 'gamma' in name
        is_bias = 'bias' in name
        is_embed = 'embed' in name or 'token_emb' in name
        
        if is_norm or is_bias:
            no_decay_params.append(param)
        elif use_muon and is_matrix and not is_embed:
            muon_params.append(param)
        else:
            decay_params.append(param)
    
    if use_muon and muon_params:
        adam_params = decay_params + no_decay_params
        optimizer = Muon(
            muon_params=muon_params,
            adam_params=adam_params,
            lr=muon_lr,
            adam_lr=lr,
            weight_decay=weight_decay,
        )
        logger.info(f"Muon optimizer: {len(muon_params)} matrix params (lr={muon_lr}), "
                    f"{len(adam_params)} other params (lr={lr})")
    else:
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr)
        logger.info(f"AdamW optimizer: {len(decay_params)} decay + {len(no_decay_params)} no-decay params")
    
    return optimizer


# ═══════════════════════════════════════════
# 4. GradientMonitor — отслеживание здоровья градиентов
# ═══════════════════════════════════════════

class GradientMonitor:
    """
    Мониторинг градиентов per-layer.
    
    Отслеживает:
      - Норму градиента по каждому слою
      - Ratio grad_norm / weight_norm (должен быть ~0.001-0.01)
      - Детекцию vanishing (< 1e-7) и exploding (> 100) градиентов
    
    Использование:
        gmon = GradientMonitor(model)
        loss.backward()
        report = gmon.check()  # returns dict with alerts
        gmon.log_to(train_logger, step)
    """
    
    def __init__(self, model: nn.Module, check_every: int = 100):
        self._model = model
        self.check_every = check_every
        self._step = 0
        self._history: List[Dict] = []
    
    def check(self) -> Dict[str, Any]:
        """Проверить состояние градиентов. Возвращает report dict."""
        self._step += 1
        _raw = getattr(self._model, '_orig_mod', self._model)
        
        norms = {}
        alerts = []
        total_norm = 0.0
        
        for name, param in _raw.named_parameters():
            if param.grad is None:
                continue
            
            grad_norm = param.grad.data.norm(2).item()
            weight_norm = param.data.norm(2).item()
            ratio = grad_norm / max(weight_norm, 1e-8)
            
            # Simplify name
            short_name = name.split(".")[-2] + "." + name.split(".")[-1] if "." in name else name
            norms[short_name] = round(grad_norm, 6)
            total_norm += grad_norm ** 2
            
            # Alerts
            if grad_norm < 1e-7:
                alerts.append(f"⚠ VANISHING: {short_name} grad_norm={grad_norm:.2e}")
            elif grad_norm > 100:
                alerts.append(f"🔥 EXPLODING: {short_name} grad_norm={grad_norm:.2e}")
            elif ratio > 0.1:
                alerts.append(f"📈 HIGH_RATIO: {short_name} ratio={ratio:.3f}")
        
        total_norm = total_norm ** 0.5
        
        report = {
            "total_grad_norm": round(total_norm, 4),
            "n_params": len(norms),
            "alerts": alerts,
            "norms": norms,
        }
        
        self._history.append({"step": self._step, "total_norm": total_norm})
        return report
    
    def log_to(self, train_logger: TrainLogger, step: int):
        """Логировать нормы в TrainLogger."""
        report = self.check()
        train_logger.log_gradient_norms(step, report["norms"])
        
        # Print alerts
        for alert in report["alerts"]:
            logger.warning(alert)
        
        return report


# ═══════════════════════════════════════════
# 5. CurriculumSampler — рост seq_len
# ═══════════════════════════════════════════

class CurriculumSchedule:
    """
    Curriculum learning: постепенное увеличение seq_len.
    
    Идея: начинать с коротких последовательностей (быстро, стабильно),
    постепенно увеличивать до полной длины.
    
    Фазы:
      Phase 1 (0-30% обучения):  seq_len_min
      Phase 2 (30-70%):          linear interpolation
      Phase 3 (70-100%):         seq_len_max
    
    Использование:
        curriculum = CurriculumSchedule(64, 512, total_epochs=25)
        for epoch in range(25):
            current_seq_len = curriculum.get_seq_len(epoch)
            # перестроить DataLoader с новым seq_len
    """
    
    def __init__(self, seq_min: int, seq_max: int, total_epochs: int,
                 warmup_frac: float = 0.3, full_frac: float = 0.7):
        self.seq_min = seq_min
        self.seq_max = seq_max
        self.total_epochs = max(total_epochs, 1)
        self.warmup_frac = warmup_frac
        self.full_frac = full_frac
    
    def get_seq_len(self, epoch: int) -> int:
        """Получить seq_len для текущей эпохи."""
        progress = epoch / self.total_epochs
        
        if progress < self.warmup_frac:
            return self.seq_min
        elif progress < self.full_frac:
            # Linear interpolation
            phase_progress = (progress - self.warmup_frac) / (self.full_frac - self.warmup_frac)
            seq = self.seq_min + (self.seq_max - self.seq_min) * phase_progress
            # Round to nearest 32
            return max(self.seq_min, (int(seq) // 32) * 32)
        else:
            return self.seq_max
    
    def __repr__(self):
        return f"Curriculum({self.seq_min} → {self.seq_max} over {self.total_epochs} epochs)"


# ═══════════════════════════════════════════
# 6. ThroughputTracker — tokens/sec + VRAM
# ═══════════════════════════════════════════

class ThroughputTracker:
    """
    Отслеживание tokens/sec и VRAM usage.
    
    Использование:
        tracker = ThroughputTracker()
        for batch in loader:
            tracker.start_step()
            # ... training step ...
            tracker.end_step(batch_tokens=batch_size * seq_len)
        
        tracker.report()
    """
    
    def __init__(self):
        self._step_start = None
        self._total_tokens = 0
        self._total_time = 0.0
        self._step_count = 0
        self._start_time = time.time()
        self._peak_vram = 0.0
    
    def start_step(self):
        self._step_start = time.time()
    
    def end_step(self, batch_tokens: int):
        if self._step_start is None:
            return
        step_time = time.time() - self._step_start
        self._total_tokens += batch_tokens
        self._total_time += step_time
        self._step_count += 1
        
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            self._peak_vram = max(self._peak_vram, vram)
    
    @property
    def tokens_per_sec(self) -> float:
        if self._total_time == 0:
            return 0.0
        return self._total_tokens / self._total_time
    
    @property
    def avg_step_time(self) -> float:
        if self._step_count == 0:
            return 0.0
        return self._total_time / self._step_count
    
    def report(self) -> dict:
        elapsed = time.time() - self._start_time
        return {
            "tokens_per_sec": round(self.tokens_per_sec),
            "avg_step_ms": round(self.avg_step_time * 1000, 1),
            "total_tokens": self._total_tokens,
            "total_time_min": round(elapsed / 60, 1),
            "peak_vram_gb": round(self._peak_vram, 2),
            "steps": self._step_count,
        }
    
    def format_status(self) -> str:
        tps = self.tokens_per_sec
        if tps > 1e6:
            return f"{tps/1e6:.1f}M tok/s"
        elif tps > 1e3:
            return f"{tps/1e3:.1f}K tok/s"
        return f"{tps:.0f} tok/s"


# ═══════════════════════════════════════════
# 7. SNN-specific metrics
# ═══════════════════════════════════════════

class SNNMetrics:
    """
    Метрики специфичные для Spiking Neural Networks.
    
    Отслеживает:
      - Per-layer spike rate (% ненулевых спайков)
      - Membrane potential statistics (mean, std)
      - Per-head sparsity
      - Beta (decay) distribution
    """
    
    def __init__(self, model: nn.Module):
        self._model = model
        self._hooks = []
        self._spike_data = {}
        self._membrane_data = {}
    
    def collect(self) -> Dict[str, Any]:
        """Собрать метрики из текущего состояния модели."""
        _raw = getattr(self._model, '_orig_mod', self._model)
        metrics = {}
        
        # Beta distribution
        betas = []
        for name, param in _raw.named_parameters():
            if 'beta_raw' in name:
                beta_vals = torch.sigmoid(param).detach()
                betas.append(beta_vals)
                metrics[f"beta/{name.split('.')[0]}"] = {
                    "min": round(beta_vals.min().item(), 4),
                    "max": round(beta_vals.max().item(), 4),
                    "mean": round(beta_vals.mean().item(), 4),
                }
        
        # Per-head sparsity (if SpikingMinGRUBlock)
        for name, module in _raw.named_modules():
            if hasattr(module, 'sparsity'):
                metrics[f"sparsity/{name}"] = round(module.sparsity, 4)
            if hasattr(module, '_spike_count') and hasattr(module, '_total_count'):
                if module._total_count > 0:
                    rate = (module._spike_count / module._total_count).item()
                    metrics[f"spike_rate/{name}"] = round(rate, 4)
        
        # Theta (threshold) distribution
        thetas = []
        for name, param in _raw.named_parameters():
            if name.endswith('.theta') and 'lif' in name:
                thetas.append(param.detach())
                metrics[f"theta/{name.split('.')[0]}"] = {
                    "min": round(param.min().item(), 4),
                    "max": round(param.max().item(), 4),
                    "mean": round(param.mean().item(), 4),
                }
        
        return metrics
    
    def reset_counters(self):
        """Сбросить счётчики спайков (вызывать после каждой эпохи)."""
        _raw = getattr(self._model, '_orig_mod', self._model)
        for module in _raw.modules():
            if hasattr(module, '_spike_count'):
                module._spike_count.zero_()
            if hasattr(module, '_total_count'):
                module._total_count.zero_()
    
    def format_report(self) -> str:
        """Форматированный отчёт для печати."""
        metrics = self.collect()
        lines = ["  🧬 SNN Metrics:"]
        for key, val in sorted(metrics.items()):
            if isinstance(val, dict):
                lines.append(f"    {key}: [{val.get('min', '?')}, {val.get('max', '?')}] "
                           f"mean={val.get('mean', '?')}")
            else:
                lines.append(f"    {key}: {val}")
        return "\n".join(lines)


# ═══════════════════════════════════════════
# 8. Convenience: make_lr_schedule
# ═══════════════════════════════════════════

def make_lr_schedule(optimizer, total_steps: int, warmup_steps: int = 0,
                     schedule: str = "cosine", min_lr_ratio: float = 0.1,
                     wsd_decay_frac: float = 0.10):
    """
    Создать LR scheduler.
    
    Args:
        scheduler: "cosine" | "wsd" | "constant"
    
    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """
    if schedule == "wsd":
        wsd = WSDSchedule(
            total_steps=total_steps,
            warmup_frac=warmup_steps / max(total_steps, 1),
            decay_frac=wsd_decay_frac,
            min_lr_ratio=min_lr_ratio,
        )
        logger.info(f"LR Schedule: {wsd}")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, wsd.as_lambda())
    
    elif schedule == "cosine":
        def cosine_lambda(step):
            if step < warmup_steps:
                return max(min_lr_ratio, step / max(warmup_steps, 1))
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, cosine)
        
        logger.info(f"LR Schedule: cosine (warmup={warmup_steps}, total={total_steps})")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_lambda)
    
    elif schedule == "constant":
        logger.info("LR Schedule: constant")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
