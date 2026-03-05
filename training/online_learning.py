"""
═══════════════════════════════════════════════════════════════
  Online Continual Learning for TARS
═══════════════════════════════════════════════════════════════

Learn from user interactions in real-time WITHOUT catastrophic
forgetting. TARS improves as it talks to users.

Key techniques:
  1. Experience Replay: mix new data with old memories
  2. Elastic Weight Consolidation (EWC): protect important params
  3. LoRA-only updates: safe, reversible fine-tuning
  4. Quality gate: only learn from high-quality interactions

Architecture:
  User chat → Quality Filter → Experience Buffer → 
  → Micro-Training (LoRA) → Updated Model

Usage:
  from training.online_learning import OnlineLearner
  
  learner = OnlineLearner(model, tokenizer)
  
  # After each good interaction:
  learner.add_experience("user prompt", "good response")
  
  # Periodically:
  learner.train_step()  # micro gradient update
"""

import os
import sys
import json
import time
import random
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from typing import Optional, List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("online_learning")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class ExperienceBuffer:
    """
    Circular buffer for experience replay.
    
    Stores (prompt, response, quality_score) tuples.
    Prioritized sampling: higher quality = more likely to be replayed.
    """
    
    def __init__(self, max_size: int = 10000, save_path: str = None):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.save_path = save_path
        
        if save_path and os.path.exists(save_path):
            self._load()
    
    def add(self, prompt: str, response: str, quality: float = 1.0):
        """Add experience. quality ∈ [0, 1]."""
        self.buffer.append({
            "prompt": prompt,
            "response": response,
            "quality": quality,
            "timestamp": time.time(),
        })
    
    def sample(self, batch_size: int = 4) -> List[Dict]:
        """Sample batch with quality-weighted priority."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Quality-weighted sampling
        items = list(self.buffer)
        weights = [max(item['quality'], 0.01) for item in items]
        total = sum(weights)
        probs = [w / total for w in weights]
        
        indices = random.choices(range(len(items)), weights=probs, k=batch_size)
        return [items[i] for i in indices]
    
    def save(self):
        """Persist buffer to disk."""
        if self.save_path:
            os.makedirs(os.path.dirname(self.save_path) or '.', exist_ok=True)
            with open(self.save_path, 'w', encoding='utf-8') as f:
                for item in self.buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _load(self):
        """Load buffer from disk."""
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    self.buffer.append(item)
            logger.info(f"Loaded {len(self.buffer)} experiences from {self.save_path}")
        except Exception as e:
            logger.warning(f"Failed to load experiences: {e}")
    
    def __len__(self):
        return len(self.buffer)


class EWCRegularizer:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).
    
    Protects important weights from being overwritten by new data.
    Importance = diagonal of Fisher Information Matrix.
    
    L_total = L_new_data + λ * Σ F_i * (θ_i - θ*_i)²
    """
    
    def __init__(self, model, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optimal_params = {}
    
    def compute_fisher(self, model, data_loader, n_samples: int = 200):
        """Estimate Fisher Information from current data."""
        model.eval()
        
        # Store current params as optimal
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
                self.fisher[name] = torch.zeros_like(param.data)
        
        # Compute Fisher = E[∇log p(y|x)²]
        n_computed = 0
        for batch in data_loader:
            if n_computed >= n_samples:
                break
            
            if isinstance(batch, torch.Tensor):
                input_ids = batch
            else:
                continue
                
            model.zero_grad()
            result = model.think(input_ids)
            logits = result[0] if isinstance(result, tuple) else result
            
            # Use model's own predictions as targets (self-supervised)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            loss = F.nll_loss(
                log_probs.reshape(-1, log_probs.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
            
            n_computed += 1
        
        # Average
        for name in self.fisher:
            self.fisher[name] /= max(n_computed, 1)
        
        logger.info(f"Fisher computed from {n_computed} samples")
    
    def penalty(self, model) -> torch.Tensor:
        """EWC penalty: Σ F_i * (θ_i - θ*_i)²."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if name in self.fisher and param.requires_grad:
                loss += (self.fisher[name].to(param.device) * 
                        (param - self.optimal_params[name].to(param.device)) ** 2).sum()
        
        return self.lambda_ewc * loss


class OnlineLearner:
    """
    Online continual learning system for TARS.
    
    Features:
      - Experience replay buffer (circular, quality-weighted)
      - EWC regularization (anti-catastrophic forgetting)
      - LoRA-only updates (safe, reversible)
      - Micro gradient steps (1-2 steps per interaction)
      - Quality gating (learn only from good interactions)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        buffer_size: int = 5000,
        save_dir: str = "data/online_learning",
        lr: float = 1e-5,
        ewc_lambda: float = 500.0,
        quality_threshold: float = 0.5,
        use_lora: bool = True,
        lora_rank: int = 8,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.quality_threshold = quality_threshold
        
        # Experience buffer
        buffer_path = os.path.join(save_dir, "experience_buffer.jsonl")
        self.buffer = ExperienceBuffer(max_size=buffer_size, save_path=buffer_path)
        
        # EWC regularizer
        self.ewc = EWCRegularizer(model, lambda_ewc=ewc_lambda)
        
        # LoRA for safe updates
        if use_lora:
            try:
                from training.lora import apply_lora
                apply_lora(model, rank=lora_rank, alpha=lora_rank * 2,
                          target_modules=['in_proj', 'out_proj'])
                logger.info(f"LoRA enabled for online learning (rank={lora_rank})")
            except ImportError:
                logger.warning("LoRA not available, using full model updates")
        
        # Optimizer (only trainable params)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
        
        self.total_steps = 0
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def add_experience(self, prompt: str, response: str, 
                       quality: float = 0.8):
        """
        Add a user interaction to the experience buffer.
        
        Args:
            prompt: user's input
            response: model's response  
            quality: quality score [0-1] (can be from thumbs up/down, auto-judge, etc.)
        """
        if quality < self.quality_threshold:
            logger.debug(f"Skipping low-quality experience (q={quality:.2f})")
            return
        
        self.buffer.add(prompt, response, quality)
        
        # Auto-save periodically
        if len(self.buffer) % 100 == 0:
            self.buffer.save()
    
    def train_step(self, n_steps: int = 1, batch_size: int = 2) -> dict:
        """
        Perform micro-training on buffered experiences.
        
        Args:
            n_steps: number of gradient steps (1-2 for online, more for batch)
            batch_size: experiences per step
        
        Returns:
            training stats dict
        """
        if len(self.buffer) < batch_size:
            return {"status": "not_enough_data", "buffer_size": len(self.buffer)}
        
        self.model.train()
        total_loss = 0
        
        for step in range(n_steps):
            batch = self.buffer.sample(batch_size)
            step_loss = 0
            
            for item in batch:
                text = item['prompt'] + '\n' + item['response']
                ids = self.tokenizer.encode(text)[:512]
                
                if len(ids) < 10:
                    continue
                
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
                
                result = self.model.think(input_ids)
                logits = result[0] if isinstance(result, tuple) else result
                
                # Cross-entropy loss
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                
                # EWC penalty (anti-catastrophic forgetting)
                ewc_loss = self.ewc.penalty(self.model)
                
                loss = ce_loss + ewc_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 
                    1.0
                )
                self.optimizer.step()
                
                step_loss += loss.item()
            
            total_loss += step_loss / max(len(batch), 1)
        
        self.total_steps += n_steps
        self.model.eval()
        
        stats = {
            "loss": total_loss / n_steps,
            "total_steps": self.total_steps,
            "buffer_size": len(self.buffer),
        }
        
        # Save checkpoint periodically
        if self.total_steps % 50 == 0:
            self.save_checkpoint()
        
        return stats
    
    def snapshot_fisher(self, data_loader=None):
        """Update EWC Fisher matrix (call periodically, e.g. daily)."""
        if data_loader is None:
            # Use buffer as data source
            items = list(self.buffer.buffer)[-200:]
            tensors = []
            for item in items:
                text = item['prompt'] + '\n' + item['response']
                ids = self.tokenizer.encode(text)[:256]
                if len(ids) >= 10:
                    tensors.append(torch.tensor([ids], dtype=torch.long, device=self.device))
            data_loader = tensors
        
        self.ewc.compute_fisher(self.model, data_loader)
    
    def save_checkpoint(self):
        """Save online learning state."""
        path = os.path.join(self.save_dir, "online_checkpoint.pt")
        
        try:
            from training.lora import save_lora
            lora_path = os.path.join(self.save_dir, "online_lora.pt")
            save_lora(self.model, lora_path)
        except ImportError:
            torch.save(
                {'model_state_dict': self.model.state_dict()},
                path
            )
        
        self.buffer.save()
        logger.info(f"Online learning checkpoint saved (step {self.total_steps})")
