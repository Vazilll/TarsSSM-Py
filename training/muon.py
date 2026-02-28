"""
═══════════════════════════════════════════════════════════════
  Muon Optimizer — 2x faster than AdamW for LLM training
═══════════════════════════════════════════════════════════════

From: "Muon: An optimizer for hidden layers in LLMs" (2025)
Used by: Moonlight 3B/16B MoE model.

Key ideas:
  1. Matrix orthogonalization of weight updates (vs element-wise AdamW)
  2. Single momentum buffer (vs 2 in AdamW → 50% less optimizer VRAM)
  3. 2x compute-efficient: same quality in half the training steps

Usage:
  from training.muon import Muon
  optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
"""

import torch
from torch.optim import Optimizer
import math


class Muon(Optimizer):
    """
    Muon: Matrix-orthogonalized update optimizer.
    
    Uses Newton-Schulz iteration to orthogonalize gradient matrices,
    ensuring weight updates explore the loss landscape more efficiently
    than element-wise methods like AdamW.
    
    Args:
        params: model parameters
        lr: learning rate (default: 0.02, higher than AdamW)
        momentum: momentum factor (default: 0.95)
        nesterov: use Nesterov momentum (default: True)
        weight_decay: decoupled weight decay (default: 0.01)
        ns_steps: Newton-Schulz orthogonalization steps (default: 5)
    """
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.01, ns_steps=5):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            weight_decay=weight_decay, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)
    
    @staticmethod
    def _newton_schulz_orthogonalize(G, steps=5):
        """
        Approximate matrix orthogonalization via Newton-Schulz iteration.
        
        Iteratively computes: X_{k+1} = X_k (3I - X_k^T X_k) / 2
        Converges to orthogonal matrix in ~5 steps.
        
        This is the key innovation over AdamW: instead of element-wise
        scaling, Muon orthogonalizes the entire gradient matrix.
        """
        assert G.ndim >= 2, "Newton-Schulz requires 2D+ tensors"
        
        # Reshape to 2D if needed
        original_shape = G.shape
        if G.ndim > 2:
            G = G.reshape(G.shape[0], -1)
        
        rows, cols = G.shape
        
        # Transpose if more rows than cols (work with smaller dimension)
        transposed = False
        if rows > cols:
            G = G.T
            transposed = True
            rows, cols = G.shape
        
        # Normalize 
        scale = max(G.norm(), 1e-6)
        X = G / scale
        
        # Newton-Schulz iterations
        # X_{k+1} = X_k @ (3I - X_k^T @ X_k) / 2
        I = torch.eye(cols, device=G.device, dtype=G.dtype)
        for _ in range(steps):
            A = X @ X.T  # [rows, rows]
            # 3I - A ≈ orthogonal correction
            B = torch.eye(rows, device=G.device, dtype=G.dtype) * 3 - A
            X = B @ X / 2
        
        if transposed:
            X = X.T
        
        return X.reshape(original_shape) * scale
    
    @torch.no_grad()
    def step(self, closure=None):
        """Single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Decoupled weight decay (same as AdamW)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                
                # Get or init momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                
                buf = state['momentum_buffer']
                
                # Apply orthogonalization for 2D+ params (weight matrices)
                # For 1D params (biases, norms), use standard SGD
                if grad.ndim >= 2 and min(grad.shape) > 1:
                    grad = self._newton_schulz_orthogonalize(grad, ns_steps)
                
                # Momentum update
                buf.mul_(momentum).add_(grad)
                
                if nesterov:
                    # Nesterov: update = grad + momentum * buf
                    update = grad + momentum * buf
                else:
                    update = buf
                
                p.add_(update, alpha=-lr)
        
        return loss


def get_optimizer(model, args):
    """
    Create optimizer based on args.
    
    Returns Muon for weight matrices, AdamW for everything else
    (biases, norms, embeddings) — this is the recommended setup.
    """
    # Separate parameters: 2D (matrices) use Muon, 1D use AdamW
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if param.ndim >= 2 and 'embedding' not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    muon_lr = getattr(args, 'muon_lr', 0.02)
    adamw_lr = getattr(args, 'lr', 3e-4)
    wd = getattr(args, 'weight_decay', 0.01)
    
    optimizer = torch.optim.AdamW([
        {'params': adamw_params, 'lr': adamw_lr, 'weight_decay': wd},
    ])
    
    if muon_params:
        # Add Muon group for matrix params
        muon_opt = Muon(muon_params, lr=muon_lr, weight_decay=wd)
        print(f"  ⚡ Muon: {len(muon_params)} matrix params (lr={muon_lr})")
        print(f"  ⚡ AdamW: {len(adamw_params)} other params (lr={adamw_lr})")
        return muon_opt, optimizer  # Return both
    
    return None, optimizer
