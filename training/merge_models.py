"""
═══════════════════════════════════════════════════════════════
  Model Merging for TARS (DARE / TIES / SLERP)
═══════════════════════════════════════════════════════════════

Combine multiple fine-tuned models into one stronger model.
No training required — just weighted parameter mixing.

Methods:
  1. DARE: Drop And REscale (random sparsification + rescale)
  2. TIES: Trim, Elect, Merge (resolve sign conflicts)
  3. SLERP: Spherical Linear Interpolation (smooth blending)
  4. Linear: Simple weighted average

Use cases:
  - Merge DPO-aligned + LoRA-finetuned models
  - Combine instruction + personality models  
  - Create stronger model from diverse fine-tunes

Usage:
  python training/merge_models.py \
      --base models/mamba2/brain_best.pt \
      --models models/mamba2/brain_dpo.pt models/mamba2/brain_lora.pt \
      --method dare --output models/mamba2/brain_merged.pt
"""

import os
import sys
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("merge")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_state_dict(path: str, device: str = "cpu") -> dict:
    """Load state dict from checkpoint."""
    try:
        cp = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        from collections import OrderedDict
        torch.serialization.add_safe_globals([OrderedDict, np.ndarray, np.dtype])
        cp = torch.load(path, map_location=device, weights_only=True)
    return cp.get('model_state_dict', cp)


def compute_task_vectors(base: dict, models: List[dict]) -> List[dict]:
    """Compute task vectors (delta from base) for each model."""
    vectors = []
    for model_sd in models:
        delta = {}
        for key in base:
            if key in model_sd and base[key].shape == model_sd[key].shape:
                delta[key] = model_sd[key].float() - base[key].float()
        vectors.append(delta)
    return vectors


def merge_linear(base: dict, vectors: List[dict], 
                 weights: List[float] = None) -> dict:
    """Simple linear merge: base + Σ(w_i * δ_i)."""
    if weights is None:
        weights = [1.0 / len(vectors)] * len(vectors)
    
    merged = {}
    for key in base:
        merged[key] = base[key].float().clone()
        for vec, w in zip(vectors, weights):
            if key in vec:
                merged[key] += w * vec[key]
        merged[key] = merged[key].to(base[key].dtype)
    
    return merged


def merge_dare(base: dict, vectors: List[dict],
               weights: List[float] = None,
               density: float = 0.5) -> dict:
    """
    DARE: Drop And REscale (Yu et al., 2023).
    
    1. Randomly drop (1-density)% of delta parameters
    2. Rescale remaining by 1/density to preserve magnitude
    3. Merge rescaled deltas
    
    This reduces interference between models' updates.
    """
    if weights is None:
        weights = [1.0 / len(vectors)] * len(vectors)
    
    merged = {}
    for key in base:
        merged[key] = base[key].float().clone()
        
        for vec, w in zip(vectors, weights):
            if key not in vec:
                continue
            
            delta = vec[key].clone()
            
            # Random drop mask
            mask = torch.bernoulli(torch.full_like(delta, density)).bool()
            delta[~mask] = 0.0
            
            # Rescale to preserve expected magnitude
            if density > 0:
                delta = delta / density
            
            merged[key] += w * delta
        
        merged[key] = merged[key].to(base[key].dtype)
    
    return merged


def merge_ties(base: dict, vectors: List[dict],
               weights: List[float] = None,
               density: float = 0.5) -> dict:
    """
    TIES: Trim, Elect Sign, Merge (Yadav et al., 2023).
    
    1. Trim: keep only top-k% largest magnitude deltas
    2. Elect: resolve sign conflicts by majority vote
    3. Merge: average aligned deltas
    
    Better than DARE when models have conflicting updates.
    """
    if weights is None:
        weights = [1.0 / len(vectors)] * len(vectors)
    
    merged = {}
    for key in base:
        merged[key] = base[key].float().clone()
        
        deltas = [vec[key].clone() for vec in vectors if key in vec]
        if not deltas:
            merged[key] = merged[key].to(base[key].dtype)
            continue
        
        # Step 1: Trim — keep only top density% by magnitude
        trimmed = []
        for delta in deltas:
            threshold = torch.quantile(delta.abs().float(), 1.0 - density)
            mask = delta.abs() >= threshold
            trimmed_delta = delta.clone()
            trimmed_delta[~mask] = 0.0
            trimmed.append(trimmed_delta)
        
        # Step 2: Elect sign — majority vote
        signs = torch.stack([torch.sign(d) for d in trimmed])
        elected_sign = torch.sign(signs.sum(dim=0))  # majority
        
        # Step 3: Merge — average only values matching elected sign
        result = torch.zeros_like(base[key].float())
        counts = torch.zeros_like(base[key].float())
        
        for delta, w in zip(trimmed, weights):
            agree = (torch.sign(delta) == elected_sign) | (delta == 0)
            aligned = delta.clone()
            aligned[~agree] = 0.0
            result += w * aligned
            counts += agree.float() * w
        
        # Normalize by count
        counts = counts.clamp(min=1e-8)
        merged[key] += result / counts * counts.sum() / counts.numel()
        merged[key] = merged[key].to(base[key].dtype)
    
    return merged


def merge_slerp(sd_a: dict, sd_b: dict, t: float = 0.5) -> dict:
    """
    SLERP: Spherical Linear Interpolation.
    
    Smoother than linear interpolation — preserves magnitude
    and interpolates along the geodesic on the hypersphere.
    
    Only works with 2 models. For more, chain SLERP pairs.
    """
    merged = {}
    for key in sd_a:
        if key not in sd_b:
            merged[key] = sd_a[key]
            continue
        
        a = sd_a[key].float().flatten()
        b = sd_b[key].float().flatten()
        
        # Normalize
        a_norm = a.norm()
        b_norm = b.norm()
        
        if a_norm < 1e-8 or b_norm < 1e-8:
            merged[key] = ((1 - t) * sd_a[key].float() + t * sd_b[key].float()).to(sd_a[key].dtype)
            continue
        
        a_unit = a / a_norm
        b_unit = b / b_norm
        
        # Angle between vectors
        cos_theta = torch.clamp(torch.dot(a_unit, b_unit), -1.0, 1.0)
        theta = torch.acos(cos_theta)
        
        if theta.abs() < 1e-6:
            # Vectors are parallel, use linear
            merged[key] = ((1 - t) * sd_a[key].float() + t * sd_b[key].float()).to(sd_a[key].dtype)
        else:
            sin_theta = torch.sin(theta)
            # SLERP formula
            interp = (torch.sin((1 - t) * theta) / sin_theta * a + 
                      torch.sin(t * theta) / sin_theta * b)
            # Interpolate magnitude
            mag = (1 - t) * a_norm + t * b_norm
            interp = interp / interp.norm() * mag
            merged[key] = interp.reshape(sd_a[key].shape).to(sd_a[key].dtype)
    
    return merged


def main():
    p = argparse.ArgumentParser(description="TARS Model Merging")
    p.add_argument("--base", type=str, required=True, help="Base model checkpoint")
    p.add_argument("--models", type=str, nargs='+', required=True, help="Model(s) to merge")
    p.add_argument("--method", type=str, default="dare",
                   choices=["linear", "dare", "ties", "slerp"])
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--weights", type=float, nargs='*', default=None)
    p.add_argument("--density", type=float, default=0.5, 
                   help="DARE/TIES density (0.3-0.7)")
    p.add_argument("--slerp_t", type=float, default=0.5,
                   help="SLERP interpolation parameter")
    args = p.parse_args()
    
    logger.info(f"Method: {args.method}")
    logger.info(f"Base: {args.base}")
    logger.info(f"Models: {args.models}")
    
    base_sd = load_state_dict(args.base)
    model_sds = [load_state_dict(m) for m in args.models]
    
    if args.method == "slerp":
        if len(model_sds) != 1:
            logger.error("SLERP requires exactly 1 model + base (2 total)")
            return
        merged = merge_slerp(base_sd, model_sds[0], t=args.slerp_t)
    else:
        vectors = compute_task_vectors(base_sd, model_sds)
        
        if args.method == "linear":
            merged = merge_linear(base_sd, vectors, args.weights)
        elif args.method == "dare":
            merged = merge_dare(base_sd, vectors, args.weights, args.density)
        elif args.method == "ties":
            merged = merge_ties(base_sd, vectors, args.weights, args.density)
    
    # Save
    output = args.output or args.base.replace('.pt', f'_{args.method}.pt')
    torch.save({'model_state_dict': merged}, output)
    
    size_mb = os.path.getsize(output) / 1024 / 1024
    logger.info(f"Saved: {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
