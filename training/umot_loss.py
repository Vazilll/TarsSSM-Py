"""
═══════════════════════════════════════════════════════════════
  UMOT — Unified Multi-Objective Training Loss
═══════════════════════════════════════════════════════════════

5-component loss with ALTERNATING BATCHES (not mixed):
  1. CE    — Cross-Entropy (next token prediction)
  2. SFT   — Supervised Fine-Tuning (instruction following)
  3. IPO   — Identity Preference Optimization (replaces DPO)
  4. Safety — Contrastive safety head loss
  5. Personality — Personality consistency loss

Key design decisions (from TZ v3 Debate R4.1):
  • Alternating batches: each step = single loss type
  • Step ratio from curriculum (not λ-weighted mixture)
  • No PCGrad/CAGrad needed → 0% overhead
  • Each DPO/IPO step = 100% gradient (not 5% diluted)
  • 5% smooth λ-ramp for phase transitions (R5.5)

Usage:
  from training.umot_loss import UMOTAlternatingScheduler, UMOTLoss

  scheduler = UMOTAlternatingScheduler()
  umot_loss = UMOTLoss(vocab_size=48256, d_model=1024)

  for step in range(total_steps):
      loss_type = scheduler.get_loss_type(step, total_steps)
      batch = get_batch(loss_type)
      loss = umot_loss.compute(loss_type, model, batch)
      loss.backward()
      optimizer.step()
"""

import math
import random
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Tars.UMOT")


# ═══════════════════════════════════════════
# 1. UMOT Alternating Scheduler
# ═══════════════════════════════════════════

class UMOTAlternatingScheduler:
    """
    Selects loss type per step based on curriculum progress.
    
    Each batch = single loss type. Step ratio = curriculum.
    No mixing → no gradient conflicts → no PCGrad needed.
    
    Ratios at progress p ∈ [0, 1]:
      CE:          max(1.0 - p, 0.30)        [100% → 30%]
      SFT:         min(p, 0.30)               [0% → 30%]
      IPO:         max(0, p - 0.3) * 0.20     [0% → 14%]
      Safety:      max(0, p - 0.5) * 0.15     [0% → 7.5%]
      Personality: max(0, p - 0.8) * 1.0      [0% → 20%]
    """
    
    LOSS_TYPES = ['CE', 'SFT', 'IPO', 'Safety', 'Personality']
    
    def __init__(self, ramp_window: float = 0.05):
        """
        Args:
            ramp_window: smooth transition window for λ changes (5% default, TZ R5.5)
        """
        self.ramp_window = ramp_window
        self._step_counts = defaultdict(int)
    
    def get_ratios(self, progress: float) -> Dict[str, float]:
        """
        Get step ratios for each loss at current progress.
        
        Args:
            progress: training progress ∈ [0, 1]
        
        Returns:
            dict {loss_type: ratio}
        """
        p = max(0.0, min(1.0, progress))
        
        ratios = {
            'CE':          max(1.0 - p, 0.30),
            'SFT':         min(p, 0.30),
            'IPO':         self._smooth_ramp(p, start=0.3) * 0.20,
            'Safety':      self._smooth_ramp(p, start=0.5) * 0.15,
            'Personality': self._smooth_ramp(p, start=0.8) * 1.0,
        }
        
        # Normalize
        total = sum(ratios.values())
        if total > 0:
            ratios = {k: v / total for k, v in ratios.items()}
        
        return ratios
    
    def _smooth_ramp(self, p: float, start: float) -> float:
        """
        Smooth ramp from 0 → 1 starting at `start` with 5% transition window.
        Prevents loss explosion at phase transitions (TZ v3 R5.5).
        """
        if p < start:
            return 0.0
        elif p < start + self.ramp_window:
            # Smooth cosine ramp over window
            t = (p - start) / self.ramp_window
            return 0.5 * (1 - math.cos(math.pi * t))
        else:
            return p - start
    
    def get_loss_type(self, step: int, total_steps: int) -> str:
        """
        Select loss type for this step via stochastic sampling.
        
        Args:
            step: current step
            total_steps: total training steps
        
        Returns:
            one of LOSS_TYPES
        """
        progress = step / max(total_steps, 1)
        ratios = self.get_ratios(progress)
        
        types = list(ratios.keys())
        weights = [ratios[t] for t in types]
        
        # Filter out zero-weight types
        active = [(t, w) for t, w in zip(types, weights) if w > 1e-6]
        if not active:
            return 'CE'
        
        types, weights = zip(*active)
        selected = random.choices(types, weights=weights, k=1)[0]
        
        self._step_counts[selected] += 1
        return selected
    
    def get_stats(self) -> Dict[str, int]:
        """Get step counts per loss type."""
        return dict(self._step_counts)
    
    def format_status(self, progress: float) -> str:
        """Format current ratios as string."""
        ratios = self.get_ratios(progress)
        parts = [f"{k}={v:.0%}" for k, v in ratios.items() if v > 0.01]
        return f"UMOT({', '.join(parts)})"


# ═══════════════════════════════════════════
# 2. Individual Loss Components
# ═══════════════════════════════════════════

def ce_loss(logits: torch.Tensor, targets: torch.Tensor,
            z_loss_alpha: float = 1e-4) -> torch.Tensor:
    """
    Cross-Entropy loss with Z-loss regularization.
    
    L = CE(logits, targets) + α × log²(Σ exp(logits))
    """
    # Shift for autoregressive
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100,
    )
    
    # Z-loss for logit stability
    if z_loss_alpha > 0:
        log_z = torch.logsumexp(shift_logits, dim=-1)
        loss = loss + z_loss_alpha * (log_z ** 2).mean()
    
    return loss


def sft_loss(logits: torch.Tensor, targets: torch.Tensor,
             instruction_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Supervised Fine-Tuning loss.
    
    Same as CE but optionally masks out instruction tokens
    (only train on response tokens).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    
    if instruction_mask is not None:
        # Mask: 1 = response token (compute loss), 0 = instruction (skip)
        shift_mask = instruction_mask[:, 1:].contiguous()
        shift_targets = shift_targets.clone()
        shift_targets[shift_mask == 0] = -100
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100,
    )
    
    return loss


def ipo_loss(policy_chosen_logps: torch.Tensor,
             policy_rejected_logps: torch.Tensor,
             beta: float = 0.1) -> Tuple[torch.Tensor, dict]:
    """
    IPO (Identity Preference Optimization) loss.
    
    Replaces DPO (TZ v3 Debate R7): no reference model needed!
    
    L_IPO = (log π(y_w|x) - log π(y_l|x) - 1/(2β))²
    
    Benefits over DPO:
      - No frozen reference model → 50% less memory
      - Quadratic loss → smoother gradients
      - More stable with alternating batches
    
    Args:
        policy_chosen_logps: log P(chosen | prompt) under policy
        policy_rejected_logps: log P(rejected | prompt) under policy
        beta: temperature controlling deviation strength
    
    Returns:
        (loss, metrics_dict)
    """
    # IPO: squared difference from target margin
    log_ratio = policy_chosen_logps - policy_rejected_logps
    target_margin = 1.0 / (2.0 * beta)
    
    loss = (log_ratio - target_margin) ** 2
    
    # Metrics
    with torch.no_grad():
        accuracy = (log_ratio > 0).float()
        margin = log_ratio.detach()
    
    metrics = {
        'accuracy': accuracy.mean().item(),
        'margin': margin.mean().item(),
        'target_margin': target_margin,
    }
    
    return loss.mean(), metrics


def safety_loss(hidden_states: torch.Tensor,
                safety_labels: torch.Tensor,
                safety_head: nn.Module) -> torch.Tensor:
    """
    Contrastive safety head loss.
    
    SafetyHead(h) → p_safe ∈ [0, 1]
    
    Trained on 1500 examples + hard negatives.
    Labels: 1 = safe, 0 = unsafe.
    """
    p_safe = safety_head(hidden_states).squeeze(-1)
    return F.binary_cross_entropy_with_logits(p_safe, safety_labels.float())


def personality_loss(logits: torch.Tensor, targets: torch.Tensor,
                     personality_mask: torch.Tensor) -> torch.Tensor:
    """
    Personality consistency loss.
    
    CE loss only on personality-tagged tokens (brand consistency).
    α=0.3 weight ensures personality is always present.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = targets[:, 1:].contiguous()
    shift_mask = personality_mask[:, 1:].contiguous()
    
    # Mask non-personality targets
    masked_targets = shift_targets.clone()
    masked_targets[shift_mask == 0] = -100
    
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        masked_targets.view(-1),
        ignore_index=-100,
    )
    
    return loss


# ═══════════════════════════════════════════
# 3. UMOT Unified Loss
# ═══════════════════════════════════════════

class UMOTLoss(nn.Module):
    """
    Unified Multi-Objective Training loss.
    
    Call with a loss_type and appropriate batch → returns scalar loss.
    """
    
    def __init__(self, vocab_size: int = 48256, d_model: int = 1024,
                 beta: float = 0.1, z_loss_alpha: float = 1e-4):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.beta = beta
        self.z_loss_alpha = z_loss_alpha
        
        # Safety head (small: 3 linear layers)
        self.safety_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
    
    def compute(self, loss_type: str, model: nn.Module,
                batch: dict, **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        Compute loss for the given type.
        
        Args:
            loss_type: one of 'CE', 'SFT', 'IPO', 'Safety', 'Personality'
            model: the language model (forward: input_ids → logits)
            batch: dict with keys depending on loss_type:
              CE:    {'input_ids': [B, L]}
              SFT:   {'input_ids': [B, L], 'instruction_mask': [B, L]}
              IPO:   {'chosen_ids': [B, L], 'rejected_ids': [B, L]}
              Safety: {'input_ids': [B, L], 'safety_labels': [B]}
              Personality: {'input_ids': [B, L], 'personality_mask': [B, L]}
        
        Returns:
            (loss, metrics_dict)
        """
        if loss_type == 'CE':
            return self._compute_ce(model, batch)
        elif loss_type == 'SFT':
            return self._compute_sft(model, batch)
        elif loss_type == 'IPO':
            return self._compute_ipo(model, batch)
        elif loss_type == 'Safety':
            return self._compute_safety(model, batch)
        elif loss_type == 'Personality':
            return self._compute_personality(model, batch)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _compute_ce(self, model, batch):
        input_ids = batch['input_ids']
        logits = self._forward(model, input_ids)
        loss = ce_loss(logits, input_ids, self.z_loss_alpha)
        return loss, {'ce_loss': loss.item()}
    
    def _compute_sft(self, model, batch):
        input_ids = batch['input_ids']
        logits = self._forward(model, input_ids)
        mask = batch.get('instruction_mask')
        loss = sft_loss(logits, input_ids, mask)
        return loss, {'sft_loss': loss.item()}
    
    def _compute_ipo(self, model, batch):
        chosen_ids = batch['chosen_ids']
        rejected_ids = batch['rejected_ids']
        
        chosen_logps = self._get_log_probs(model, chosen_ids)
        rejected_logps = self._get_log_probs(model, rejected_ids)
        
        loss, metrics = ipo_loss(chosen_logps, rejected_logps, self.beta)
        metrics['ipo_loss'] = loss.item()
        return loss, metrics
    
    def _compute_safety(self, model, batch):
        input_ids = batch['input_ids']
        hidden = self._get_hidden(model, input_ids)
        labels = batch['safety_labels']
        loss = safety_loss(hidden, labels, self.safety_head)
        return loss, {'safety_loss': loss.item()}
    
    def _compute_personality(self, model, batch):
        input_ids = batch['input_ids']
        logits = self._forward(model, input_ids)
        mask = batch['personality_mask']
        loss = personality_loss(logits, input_ids, mask)
        return loss, {'personality_loss': loss.item()}
    
    def _forward(self, model, input_ids):
        """Forward pass, handling different model return types."""
        result = model(input_ids)
        if isinstance(result, tuple):
            return result[0]
        return result
    
    def _get_log_probs(self, model, input_ids):
        """Get average log probability of sequence."""
        logits = self._forward(model, input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)
        return token_log_probs.mean(dim=-1)
    
    def _get_hidden(self, model, input_ids):
        """Get last hidden state (mean-pooled)."""
        _raw = getattr(model, '_orig_mod', model)
        if hasattr(_raw, 'get_hidden_state'):
            return _raw.get_hidden_state(input_ids)
        # Fallback: use embedding layer
        if hasattr(_raw, 'embedding'):
            return _raw.embedding(input_ids).mean(dim=1)
        # Fallback: forward and use logits as proxy
        logits = self._forward(model, input_ids)
        return logits.mean(dim=1)


# ═══════════════════════════════════════════
# 4. UMOT Health Monitor
# ═══════════════════════════════════════════

class UMOTHealthMonitor:
    """
    Tracks per-loss convergence, alerts on divergence.
    
    Monitors:
      - Per-loss EMA (exponential moving average)
      - Loss spike detection (> 3× EMA)
      - Convergence check (is loss still decreasing?)
    """
    
    def __init__(self, window: int = 100, spike_threshold: float = 3.0):
        self.window = window
        self.spike_threshold = spike_threshold
        self._ema: Dict[str, float] = {}
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._alpha = 2.0 / (window + 1)
    
    def update(self, loss_type: str, loss_value: float) -> Optional[str]:
        """
        Update monitor with new loss value.
        
        Returns: alert string if anomaly detected, None otherwise.
        """
        self._history[loss_type].append(loss_value)
        
        if loss_type not in self._ema:
            self._ema[loss_type] = loss_value
            return None
        
        ema = self._ema[loss_type]
        
        # Update EMA
        self._ema[loss_type] = self._alpha * loss_value + (1 - self._alpha) * ema
        
        # Spike detection
        if loss_value > self.spike_threshold * ema and ema > 0.01:
            alert = (f"🔥 SPIKE: {loss_type} loss={loss_value:.4f} "
                     f"(>{self.spike_threshold}× EMA={ema:.4f})")
            logger.warning(alert)
            return alert
        
        # NaN/Inf detection
        if not math.isfinite(loss_value):
            alert = f"💀 NaN/Inf: {loss_type} loss={loss_value}"
            logger.error(alert)
            return alert
        
        return None
    
    def is_converging(self, loss_type: str, lookback: int = 50) -> bool:
        """Check if loss is still decreasing over recent window."""
        history = self._history.get(loss_type, [])
        if len(history) < lookback * 2:
            return True  # Not enough data
        
        recent = sum(history[-lookback:]) / lookback
        earlier = sum(history[-lookback * 2:-lookback]) / lookback
        return recent < earlier * 1.05  # Allow 5% margin
    
    def report(self) -> Dict[str, dict]:
        """Full health report."""
        report = {}
        for lt in self._history:
            h = self._history[lt]
            report[lt] = {
                'ema': round(self._ema.get(lt, 0), 4),
                'last': round(h[-1], 4) if h else 0,
                'min': round(min(h), 4) if h else 0,
                'steps': len(h),
                'converging': self.is_converging(lt),
            }
        return report


# ═══════════════════════════════════════════
# Test / Demo
# ═══════════════════════════════════════════

if __name__ == "__main__":
    print("═══ UMOT Alternating Scheduler Demo ═══\n")
    
    scheduler = UMOTAlternatingScheduler()
    
    # Show ratios at different training progress
    for p in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        ratios = scheduler.get_ratios(p)
        active = {k: f"{v:.0%}" for k, v in ratios.items() if v > 0.01}
        print(f"  p={p:.1f}: {active}")
    
    # Simulate step selection
    print("\n═══ Step Selection (1000 steps) ═══")
    total = 1000
    for step in range(total):
        scheduler.get_loss_type(step, total)
    
    stats = scheduler.get_stats()
    for lt, count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {lt:12s}: {count:4d} steps ({count / total:.1%})")
    
    print("\n✅ UMOT ready")
