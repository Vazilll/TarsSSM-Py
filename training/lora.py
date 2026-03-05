"""
═══════════════════════════════════════════════════════════════
  LoRA (Low-Rank Adaptation) for TARS
═══════════════════════════════════════════════════════════════

Fine-tune TARS with only 0.5-1% of parameters.
Instead of updating W (d×d), we learn A·B (d×r × r×d) where r << d.

Benefits:
  - 10x less GPU memory
  - Train on single RTX 3060
  - Multiple LoRA adapters for different tasks
  - Merge back for zero-overhead inference

Usage:
  # Add LoRA to model
  from training.lora import apply_lora, merge_lora, save_lora
  
  model = TarsMamba2LM(...)
  apply_lora(model, rank=8, alpha=16, target_modules=['proj'])
  
  # Train normally - only LoRA params have grad
  optimizer = torch.optim.AdamW(
      [p for p in model.parameters() if p.requires_grad],
      lr=2e-4
  )
  
  # After training - merge and save
  merge_lora(model)
  torch.save(model.state_dict(), 'brain_lora_merged.pt')
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Optional, Set

logger = logging.getLogger("Tars.LoRA")


class LoRALinear(nn.Module):
    """
    LoRA: Low-Rank Adaptation of Large Language Models.
    
    Replaces nn.Linear forward:
        y = Wx + (α/r) * BAx
    
    Where:
        W: original frozen weights (d_out × d_in)
        A: down-projection (r × d_in), init Kaiming
        B: up-projection (d_out × r), init zeros
        α: scaling factor
        r: rank (typically 4-32)
    """
    
    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        d_in = original.in_features
        d_out = original.out_features
        
        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        
        # Initialize A with Kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original(x)
        
        if not self._merged:
            # LoRA delta: (α/r) * x @ A^T @ B^T
            lora_out = self.lora_dropout(x)
            lora_out = F.linear(lora_out, self.lora_A)  # x @ A^T
            lora_out = F.linear(lora_out, self.lora_B)  # ... @ B^T
            result = result + lora_out * self.scaling
        
        return result
    
    def merge(self):
        """Merge LoRA weights into original — zero overhead at inference."""
        if not self._merged:
            with torch.no_grad():
                # W_new = W + (α/r) * B @ A
                delta = (self.lora_B @ self.lora_A) * self.scaling
                self.original.weight.add_(delta)
            self._merged = True
            logger.debug(f"Merged LoRA (rank={self.rank})")
    
    def unmerge(self):
        """Reverse merge for continued training."""
        if self._merged:
            with torch.no_grad():
                delta = (self.lora_B @ self.lora_A) * self.scaling
                self.original.weight.sub_(delta)
            self._merged = False
    
    @property
    def lora_params(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> dict:
    """
    Apply LoRA to a model.
    
    Args:
        model: any nn.Module
        rank: LoRA rank (4=minimal, 8=default, 16-32=high quality)
        alpha: scaling factor (usually 2x rank)
        dropout: LoRA dropout
        target_modules: list of module name patterns to apply LoRA to.
            Default: ['proj', 'linear', 'dense', 'out_proj', 'in_proj']
    
    Returns:
        dict with statistics
    """
    if target_modules is None:
        target_modules = ['proj', 'linear', 'dense', 'w_gate', 'w_value', 'w_out']
    
    total_params = sum(p.numel() for p in model.parameters())
    lora_count = 0
    lora_params = 0
    
    # Freeze all original params
    for p in model.parameters():
        p.requires_grad = False
    
    # Find and replace target Linear modules
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if name matches any target pattern
            if any(t in name.split('.')[-1] for t in target_modules):
                replacements.append((name, module))
    
    for name, module in replacements:
        # Navigate to parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace with LoRA version
        lora_module = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_module)
        
        lora_count += 1
        lora_params += lora_module.lora_params
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total_params': total_params,
        'lora_modules': lora_count,
        'lora_params': lora_params,
        'trainable_params': trainable,
        'trainable_pct': trainable / max(total_params, 1) * 100,
    }
    
    logger.info(
        f"LoRA applied: {lora_count} modules, "
        f"rank={rank}, α={alpha}, "
        f"{trainable:,} trainable ({stats['trainable_pct']:.2f}% of {total_params:,})"
    )
    
    return stats


def merge_lora(model: nn.Module):
    """Merge all LoRA weights into original model for zero-overhead inference."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            count += 1
    logger.info(f"Merged {count} LoRA modules")


def unmerge_lora(model: nn.Module):
    """Unmerge all LoRA weights for continued training."""
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
            count += 1
    logger.info(f"Unmerged {count} LoRA modules")


def save_lora(model: nn.Module, path: str):
    """Save only LoRA parameters (tiny file)."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
            lora_state[f"{name}.rank"] = module.rank
            lora_state[f"{name}.alpha"] = module.alpha
    
    torch.save(lora_state, path)
    size_kb = os.path.getsize(path) / 1024
    logger.info(f"LoRA saved: {path} ({size_kb:.1f} KB, {len(lora_state)//3} modules)")


def load_lora(model: nn.Module, path: str, device: str = 'cpu'):
    """Load LoRA parameters into model (model must have apply_lora already)."""
    lora_state = torch.load(path, map_location=device, weights_only=True)
    
    loaded = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_state and b_key in lora_state:
                module.lora_A.data.copy_(lora_state[a_key])
                module.lora_B.data.copy_(lora_state[b_key])
                loaded += 1
    
    logger.info(f"LoRA loaded: {loaded} modules from {path}")

