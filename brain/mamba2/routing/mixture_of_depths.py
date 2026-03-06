"""
═══════════════════════════════════════════════════════════════
  Mixture of Depths (MoD) — Per-token Adaptive Computation
═══════════════════════════════════════════════════════════════

Источник: Raposo et al. "Mixture-of-Depths" (2024)

ИДЕЯ:
  Не все токены одинаково важны. Артикли ("а", "в", "и")
  не нужно обрабатывать через все 24 слоя. Сложные токены
  ("доказательство", "интеграл") — нужно.

  MoD добавляет ROUTER в каждый блок, который решает:
    - Этот токен НУЖНО обработать → через полный блок
    - Этот токен можно ПРОПУСТИТЬ → residual bypass

  Результат: -30% compute при том же качестве.

  ТАРС уже имеет IntegralAuditor (адаптивность на уровне ЗАПРОСА).
  MoD добавляет адаптивность на уровне КАЖДОГО ТОКЕНА.

Использование в TarsBlock:
  from brain.mamba2.mixture_of_depths import MoDRouter
  
  self.mod_router = MoDRouter(d_model, capacity_factor=0.5)  # 50% tokens skip
  
  # В forward:
  x, skipped = self.mod_router(x, training=self.training)
  if not skipped:
      x = block(x)  # full processing
  else:
      # some tokens bypass the block entirely
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MoDRouter(nn.Module):
    """
    Mixture of Depths Router.
    
    Для каждого токена решает: обработать через полный блок (processing)
    или пропустить (residual bypass).
    
    Args:
        d_model: model dimension
        capacity_factor: fraction of tokens to process (0.5 = 50%)
                        Lower = faster but less quality
    """
    
    def __init__(self, d_model, capacity_factor=0.5):
        super().__init__()
        self.capacity_factor = capacity_factor
        
        # Router: linear → scalar per token
        self.router = nn.Linear(d_model, 1, bias=False)
        
        # Aux loss weight (load balancing)
        self.aux_loss_weight = 0.01
    
    def forward(self, x, training=True):
        """
        Route tokens: decide which ones need full processing.
        
        Args:
            x: [B, L, D] input hidden states
            training: bool, whether in training mode
        
        Returns:
            routing_mask: [B, L] bool tensor — True = process, False = skip
            router_logits: [B, L] raw router scores (for aux loss)
            aux_loss: scalar load-balancing loss
        """
        B, L, D = x.shape
        
        # Capacity: how many tokens to process
        capacity = max(1, int(L * self.capacity_factor))
        
        # Router scores
        router_logits = self.router(x).squeeze(-1)  # [B, L]
        
        if training:
            # Добавляем gumbel noise для exploration (как в MoE)
            noise = torch.randn_like(router_logits) * 0.1
            router_logits_noisy = router_logits + noise
        else:
            router_logits_noisy = router_logits
        
        # Top-k selection: process the top `capacity` tokens
        _, top_indices = router_logits_noisy.topk(capacity, dim=-1, sorted=False)
        
        # Create routing mask
        routing_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        routing_mask.scatter_(1, top_indices, True)
        
        # Aux loss: encourage uniform routing (prevent collapse)
        # Each position should be routed ~capacity_factor of the time
        router_probs = torch.sigmoid(router_logits)
        aux_loss = self.aux_loss_weight * (
            (router_probs.mean(dim=-1) - self.capacity_factor) ** 2
        ).mean()
        
        return routing_mask, router_logits, aux_loss


class MoDBlock(nn.Module):
    """
    Wrapper that applies Mixture of Depths to any block.
    
    Tokens selected by the router go through the full block.
    Tokens NOT selected bypass via residual connection.
    
    Usage:
        mod_block = MoDBlock(original_block, d_model=2048, capacity=0.5)
        output, aux_loss = mod_block(x)
    """
    
    def __init__(self, block, d_model, capacity_factor=0.5):
        super().__init__()
        self.block = block
        self.router = MoDRouter(d_model, capacity_factor)
        self.d_model = d_model
    
    def forward(self, x, *args, **kwargs):
        """
        Forward with per-token routing.
        
        Tokens with high router scores → full block processing.
        Others → skip (residual only).
        """
        B, L, D = x.shape
        
        # Get routing decision
        routing_mask, router_logits, aux_loss = self.router(
            x, training=self.training
        )
        
        # Output starts as residual (all tokens "skip" by default)
        output = x.clone()
        
        # Process only selected tokens
        # For efficiency: gather selected tokens, process, scatter back
        for b in range(B):
            selected_idx = routing_mask[b].nonzero(as_tuple=True)[0]
            
            if len(selected_idx) == 0:
                continue
            
            # Gather selected tokens
            selected_tokens = x[b, selected_idx].unsqueeze(0)  # [1, n_selected, D]
            
            # Process through block
            # Note: we call the block's forward directly
            # The block should handle variable-length sequences
            with torch.amp.autocast(x.device.type, enabled=False):
                if hasattr(self.block, 'forward'):
                    processed = self.block(selected_tokens, *args, **kwargs)
                    # If block returns tuple (like TarsBlock), take first element
                    if isinstance(processed, tuple):
                        processed = processed[0]
                else:
                    processed = selected_tokens
            
            # Scatter back
            output[b, selected_idx] = processed.squeeze(0)
        
        # Soft mixing using router weights (differentiable)
        router_weights = torch.sigmoid(router_logits).unsqueeze(-1)  # [B, L, 1]
        
        # Weighted: high router score → more of block output, low → more residual
        final = router_weights * output + (1 - router_weights) * x
        
        return final, aux_loss


def add_mod_to_model(model, capacity_factor=0.5, skip_first_last=True):
    """
    Add Mixture of Depths routing to an existing model.
    
    Args:
        model: TarsMamba2LM instance
        capacity_factor: fraction of tokens to process per block
        skip_first_last: don't add MoD to first and last blocks
                        (they need all tokens for embedding/prediction)
    """
    n_blocks = len(model.blocks)
    d_model = model.blocks[0].core.d_model if hasattr(model.blocks[0], 'core') else 2048
    
    mod_count = 0
    for i, block in enumerate(model.blocks):
        # Skip first and last blocks
        if skip_first_last and (i == 0 or i == n_blocks - 1):
            continue
        
        # Wrap block with MoD
        model.blocks[i] = MoDBlock(block, d_model, capacity_factor)
        mod_count += 1
    
    print(f"  ⚡ Mixture of Depths: {mod_count}/{n_blocks} blocks wrapped "
          f"(capacity={capacity_factor:.0%})")
    print(f"  ⚡ Estimated compute savings: ~{(1-capacity_factor)*100:.0f}%")
    
    return model
