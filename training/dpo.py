"""
═══════════════════════════════════════════════════════════════
  DPO / IPO — Preference Optimization for TARS
═══════════════════════════════════════════════════════════════

High-level preference optimization module.

Two variants:
  1. DPO (Direct Preference Optimization):
     L = -log σ(β × (log π(y_w|x) - log π(y_l|x)
                     - log π_ref(y_w|x) + log π_ref(y_l|x)))
     Requires frozen reference model → 2× memory.

  2. IPO (Identity Preference Optimization) [RECOMMENDED]:
     L = (log π(y_w|x) - log π(y_l|x) - 1/(2β))²
     No reference model needed → 50% less memory.
     Smoother gradients → better with alternating batches.

TZ v3 Decision: Use IPO, not DPO (Debate R7).

Usage:
  from training.dpo import dpo_loss, ipo_loss, DPOTrainer

  # Quick loss computation
  loss, metrics = ipo_loss(chosen_logps, rejected_logps, beta=0.1)

  # Full trainer
  trainer = DPOTrainer(model, tokenizer, use_ipo=True)
  trainer.train(preference_data, epochs=3)
"""

import os
import sys
import copy
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Tars.DPO")


# ═══════════════════════════════════════════
# 1. Loss Functions
# ═══════════════════════════════════════════

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Standard DPO Loss (Rafailov et al., 2023).

    L = -log σ(β × ((log π(y_w|x) - log π_ref(y_w|x))
                    - (log π(y_l|x) - log π_ref(y_l|x))))

    Requires reference model (frozen copy of policy before training).

    Args:
        policy_chosen_logps: log P(chosen | x) under current policy
        policy_rejected_logps: log P(rejected | x) under current policy
        ref_chosen_logps: log P(chosen | x) under reference policy
        ref_rejected_logps: log P(rejected | x) under reference policy
        beta: temperature (0.1 = strict, 0.5 = loose)
        label_smoothing: IPO-style label smoothing (0 = none)

    Returns:
        (loss, metrics_dict)
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = beta * (pi_logratios - ref_logratios)

    if label_smoothing > 0:
        loss = (
            -F.logsigmoid(logits) * (1 - label_smoothing)
            - F.logsigmoid(-logits) * label_smoothing
        )
    else:
        loss = -F.logsigmoid(logits)

    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        reward_margin = chosen_rewards - rejected_rewards
        accuracy = (logits > 0).float()

    metrics = {
        'loss': loss.mean().item(),
        'accuracy': accuracy.mean().item(),
        'chosen_reward': chosen_rewards.mean().item(),
        'rejected_reward': rejected_rewards.mean().item(),
        'margin': reward_margin.mean().item(),
    }

    return loss.mean(), metrics


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    IPO (Identity Preference Optimization) loss.

    L = (log π(y_w|x) - log π(y_l|x) - 1/(2β))²

    Benefits over DPO:
      - No reference model → 50% less memory
      - Quadratic loss → smoother gradients, no saturation
      - Compatible with alternating batches (UMOT)

    Args:
        policy_chosen_logps: log P(chosen | x)
        policy_rejected_logps: log P(rejected | x)
        beta: temperature (smaller = more conservative)

    Returns:
        (loss, metrics_dict)
    """
    log_ratio = policy_chosen_logps - policy_rejected_logps
    target_margin = 1.0 / (2.0 * beta)

    loss = (log_ratio - target_margin) ** 2

    with torch.no_grad():
        accuracy = (log_ratio > 0).float()

    metrics = {
        'loss': loss.mean().item(),
        'accuracy': accuracy.mean().item(),
        'margin': log_ratio.mean().item(),
        'target_margin': target_margin,
    }

    return loss.mean(), metrics


# ═══════════════════════════════════════════
# 2. Utility: log-probability computation
# ═══════════════════════════════════════════

def get_log_probs(model: nn.Module, input_ids: torch.Tensor,
                  device: str = 'cpu') -> torch.Tensor:
    """
    Get average log probability of a sequence under the model.

    Returns: scalar mean log prob (length-normalized).
    """
    input_ids = input_ids.to(device)

    result = model(input_ids)
    logits = result[0] if isinstance(result, tuple) else result

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    return token_log_probs.mean(dim=-1)


# ═══════════════════════════════════════════
# 3. DPO/IPO Trainer
# ═══════════════════════════════════════════

class DPOTrainer:
    """
    Full DPO/IPO training pipeline.

    Supports both DPO (with reference model) and IPO (reference-free).
    """

    def __init__(self, model: nn.Module, tokenizer, device: str = 'auto',
                 use_ipo: bool = True, beta: float = 0.1,
                 lr: float = 5e-6, max_len: int = 512):
        """
        Args:
            model: language model
            tokenizer: tokenizer with encode/decode
            use_ipo: True = IPO (no ref model), False = DPO (with ref model)
            beta: DPO/IPO temperature
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_ipo = use_ipo
        self.beta = beta
        self.lr = lr
        self.max_len = max_len

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)

        # Create reference model only for DPO
        self.ref_model = None
        if not use_ipo:
            logger.info("DPO mode: creating frozen reference model...")
            self.ref_model = copy.deepcopy(model)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        mode = "IPO (reference-free)" if use_ipo else "DPO (with reference)"
        logger.info(f"DPOTrainer: {mode}, β={beta}, lr={lr}")

    def train(self, data: List[Dict], epochs: int = 3,
              save_dir: Optional[str] = None) -> dict:
        """
        Train on preference data.

        Args:
            data: list of {'prompt': str, 'chosen': str, 'rejected': str}
            epochs: number of training epochs
            save_dir: directory to save best checkpoint

        Returns:
            training stats dict
        """
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )

        best_accuracy = 0.0
        all_metrics = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            n_batches = 0

            random.shuffle(data)

            for i, item in enumerate(data):
                prompt = item['prompt']
                chosen_text = prompt + item['chosen']
                rejected_text = prompt + item['rejected']

                chosen_ids = self._encode(chosen_text)
                rejected_ids = self._encode(rejected_text)

                # Policy log probs
                chosen_logps = get_log_probs(self.model, chosen_ids, self.device)
                rejected_logps = get_log_probs(self.model, rejected_ids, self.device)

                if self.use_ipo:
                    loss, metrics = ipo_loss(chosen_logps, rejected_logps, self.beta)
                else:
                    with torch.no_grad():
                        ref_chosen = get_log_probs(self.ref_model, chosen_ids, self.device)
                        ref_rejected = get_log_probs(self.ref_model, rejected_ids, self.device)
                    loss, metrics = dpo_loss(
                        chosen_logps, rejected_logps,
                        ref_chosen, ref_rejected, self.beta
                    )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += metrics['loss']
                epoch_acc += metrics['accuracy']
                n_batches += 1

                if (i + 1) % 50 == 0:
                    avg_loss = epoch_loss / n_batches
                    avg_acc = epoch_acc / n_batches
                    logger.info(
                        f"  [{epoch + 1}/{epochs}] step {i + 1}/{len(data)} | "
                        f"loss={avg_loss:.4f} | acc={avg_acc:.1%}"
                    )

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_acc = epoch_acc / max(n_batches, 1)
            all_metrics.append({'epoch': epoch + 1, 'loss': avg_loss, 'accuracy': avg_acc})

            logger.info(f"Epoch {epoch + 1}: loss={avg_loss:.4f} | acc={avg_acc:.1%}")

            if save_dir and avg_acc > best_accuracy:
                best_accuracy = avg_acc
                save_path = Path(save_dir) / "brain_dpo.pt"
                os.makedirs(save_dir, exist_ok=True)
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'dpo_accuracy': avg_acc,
                    'beta': self.beta,
                    'use_ipo': self.use_ipo,
                }
                torch.save(checkpoint, str(save_path))
                logger.info(f"  ✓ Best saved: {save_path} (acc={avg_acc:.1%})")

        return {
            'epochs': all_metrics,
            'best_accuracy': best_accuracy,
            'mode': 'IPO' if self.use_ipo else 'DPO',
        }

    def _encode(self, text: str) -> torch.Tensor:
        """Encode text to token tensor."""
        ids = self.tokenizer.encode(text)[:self.max_len]
        return torch.tensor([ids], dtype=torch.long)
