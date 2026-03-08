"""
═══════════════════════════════════════════════════════════════
  TARS HELIX LITE — Clean Model for Training
═══════════════════════════════════════════════════════════════

Simplified model using TarsCoreBlock (SSD + WKV + Fusion) in a
sequential stack. No waves, no metacortex, no matrix pool.

Architecture:
  Embedding → 20 × HelixBlock → RMSNorm → LM Head (weight-tied)

Each HelixBlock:
  RMSNorm → TarsCoreBlock(SSD + WKV + Fusion + SwiGLU) → Residual
  + optional LoRA adapters

Usage:
    from brain.mamba2.core.model_lite import TarsHelixLite
    from config import TarsConfig
    cfg = TarsConfig()
    model = TarsHelixLite(cfg)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from brain.mamba2.core.ssd import TarsCoreBlock
from brain.mamba2.core.bitnet import RMSNorm, UniversalLinear

logger = logging.getLogger("Tars.HelixLite")


class HelixBlock(nn.Module):
    """
    Single HELIX block: RMSNorm → TarsCoreBlock → Residual.
    
    TarsCoreBlock internally handles:
      - Shared in_proj → SSD path + WKV path
      - Fusion gate (σ-gated blend of SSD and WKV outputs)
      - SwiGLU output gating
      - Shared out_proj
    """
    
    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm = RMSNorm(cfg.d_model)
        self.core = TarsCoreBlock(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            headdim=cfg.headdim,
            ngroups=cfg.ngroups,
            chunk_size=cfg.chunk_size,
            quant_mode=cfg.quant_mode,
            n_meta_tokens=0,  # LITE: no Hymba meta-tokens
        )
    
    def forward(
        self,
        x: torch.Tensor,
        wkv_state: Optional[torch.Tensor] = None,
        x_prev: Optional[torch.Tensor] = None,
        ssd_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ):
        """
        x: [B, L, d_model]
        Returns: (x_out, wkv_state, x_last, ssd_state, conv_state)
        """
        residual = x
        h = self.norm(x)
        out, wkv_state, x_last, ssd_state, conv_state = self.core(
            h, wkv_state, x_prev, ssd_state, conv_state
        )
        return residual + out, wkv_state, x_last, ssd_state, conv_state


class TarsHelixLite(nn.Module):
    """
    TARS HELIX LITE — Minimal viable model.
    
    380M params (ternary → ~100MB packed, FP16 → ~760MB)
    
    Architecture:
      Embedding(48256, 1024) → 20 × HelixBlock → RMSNorm → LM Head
    
    LM Head is weight-tied with Embedding for parameter efficiency.
    Embedding uses Vaswani scaling: x * √d_model
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            from config import TarsConfig
            cfg = TarsConfig()
        
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.n_layers = cfg.n_layers
        self.vocab_size = cfg.vocab_size
        
        # ═══ Embedding (weight-tied with LM head) ═══
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embed_scale = math.sqrt(cfg.d_model)  # Vaswani scaling
        
        # ═══ 20 HelixBlocks (uniform, no graduated for LITE) ═══
        self.blocks = nn.ModuleList([
            HelixBlock(cfg, layer_idx=i)
            for i in range(cfg.n_layers)
        ])
        
        # ═══ Final RMSNorm ═══
        self.final_norm = RMSNorm(cfg.d_model)
        
        # ═══ LM Head (weight-tied) ═══
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Tie weights: lm_head.weight = embedding.weight
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self._init_weights()
        
        # Log param count
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"TarsHelixLite: {n_params/1e6:.1f}M params "
            f"({n_trainable/1e6:.1f}M trainable), "
            f"d={cfg.d_model}, L={cfg.n_layers}, V={cfg.vocab_size}"
        )
    
    def _init_weights(self):
        """Initialize weights with scaled initialization."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                # Scaled init: deeper layers get smaller init
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> dict:
        """
        Training forward pass.
        
        Args:
            input_ids: [B, L] token ids
            labels: [B, L] target token ids (shifted inside)
            return_hidden: if True, return last hidden state
        
        Returns:
            dict with:
              'logits': [B, L, V]
              'loss': scalar (if labels provided)
              'hidden': [B, L, d_model] (if return_hidden)
        """
        B, L = input_ids.shape
        
        # 1. Embed + scale
        x = self.embedding(input_ids) * self.embed_scale
        
        # 2. Sequential block processing with gradient checkpointing
        for block in self.blocks:
            if self.training and self.n_layers > 4:
                # Gradient checkpointing: ~40% memory savings at cost of ~30% compute
                x, _, _, _, _ = torch.utils.checkpoint.checkpoint(
                    block, x, use_reentrant=False
                )
            else:
                x, _, _, _, _ = block(x)
        
        # 3. Final norm
        x = self.final_norm(x)
        
        # 4. LM Head
        logits = self.lm_head(x)
        
        # 5. Loss (if labels provided)
        result = {'logits': logits}
        
        if labels is not None:
            # Labels are ALREADY shifted by the dataset (input=t[:-1], labels=t[1:])
            # So logits[i] should predict labels[i] directly — NO additional shift needed
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            result['loss'] = loss
        
        if return_hidden:
            result['hidden'] = x
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> torch.Tensor:
        """
        Autoregressive generation (no speculative decoding for LITE).
        
        Args:
            input_ids: [B, L] prompt tokens
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling threshold
            top_k: top-k sampling
        
        Returns:
            generated: [B, L + max_new_tokens] full sequence
        """
        self.eval()
        generated = input_ids.clone()
        
        # Initialize states for each block
        states = [{
            'wkv_state': None,
            'x_prev': None,
            'ssd_state': None,
            'conv_state': None,
        } for _ in range(self.n_layers)]
        
        # Prefill: process entire prompt
        x = self.embedding(input_ids) * self.embed_scale
        for i, block in enumerate(self.blocks):
            x, wkv, x_last, ssd, conv = block(
                x, **states[i]
            )
            states[i] = {
                'wkv_state': wkv,
                'x_prev': x_last,
                'ssd_state': ssd,
                'conv_state': conv,
            }
        
        h = self.final_norm(x)
        logits = self.lm_head(h[:, -1:, :])  # last token only
        
        for step in range(max_new_tokens):
            # Sample
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float('-inf')
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Forward one token through all blocks (step mode)
            x = self.embedding(next_token) * self.embed_scale
            for i, block in enumerate(self.blocks):
                x, wkv, x_last, ssd, conv = block(
                    x, **states[i]
                )
                states[i] = {
                    'wkv_state': wkv,
                    'x_prev': x_last,
                    'ssd_state': ssd,
                    'conv_state': conv,
                }
            
            h = self.final_norm(x)
            logits = self.lm_head(h)
            
            # Check for EOS
            if next_token.item() == 0:  # EOS token
                break
        
        return generated
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {}
        total = 0
        for name, p in self.named_parameters():
            component = name.split('.')[0]
            if component not in counts:
                counts[component] = 0
            counts[component] += p.numel()
            total += p.numel()
        counts['_total'] = total
        counts['_total_M'] = total / 1e6
        return counts
