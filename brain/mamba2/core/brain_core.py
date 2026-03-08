"""
═══════════════════════════════════════════════════════════════
  BrainCore — Core Forward Pipeline (Training + Step)
═══════════════════════════════════════════════════════════════

Извлечён из TarsMamba2LM (model.py) при декомпозиции T07.
Содержит:
  - RotaryPositionEmbedding
  - WaveConsolidation
  - GlobalWorkspace
  - SharedGlobalAttention
  - WaveScratchpad
  - TTTLoRA
  - BrainCore (nn.Module) — embedding, blocks, wave loop, lm_head
"""

import math
import os
import json
import copy
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from brain.mamba2.tars_block import TarsBlock, MemoryInjector
from brain.mamba2.matrix_pool import MatrixPool
from brain.mamba2.novelty import NoveltyGate
from brain.mamba2.logger import ThinkingLogger
from brain.mamba2.thinking_chain import ThinkingChain
from brain.mamba2.personality_adapter import PersonalityAdapter
from brain.mamba2.query_router import QueryRouter, ProgressiveMemoryManager
from brain.mamba2.bitnet import (
    UniversalLinear, convert_model_to_158bit,
    convert_model_to_fp16, model_stats, replace_linear_with_universal,
    RMSNorm,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# Reusable Components (extracted from model.py)
# ═══════════════════════════════════════════════════════════════

class RotaryPositionEmbedding(nn.Module):
    """
    Unified RoPE: Rotary Position Embedding.
    
    Uses base=500,000 (TARS default) for up to 32K+ context.
    Applied to WKV branch for positional awareness in SSM-Attention hybrid.
    
    Math: RoPE(x, pos) = x * cos(θ_pos) + rotate_half(x) * sin(θ_pos)
    where θ_i = pos / base^(2i/d)
    """
    
    def __init__(self, dim, base=500_000, max_seq_len=32768):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos/sin tables
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    @staticmethod
    def _rotate_half(x):
        """Rotate half of the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x, seq_len=None, offset=0):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        cos = self.cos_cached[offset:offset + seq_len, :self.dim]
        sin = self.sin_cached[offset:offset + seq_len, :self.dim]
        
        if x.ndim == 4:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        
        x_rope = x[..., :self.dim]
        x_rest = x[..., self.dim:]
        x_rope = x_rope * cos + self._rotate_half(x_rope) * sin
        
        return torch.cat([x_rope, x_rest], dim=-1) if x_rest.size(-1) > 0 else x_rope


class WaveConsolidation(nn.Module):
    """
    WuNeng Fusion — информационное горлышко (HELIX v6).
    
    Архитектура:
      1. Dimension-wise Gate: σ(W · [h_L; h_R]) ∈ (0,1)^d
      2. WuNeng Bottleneck: [h_L; h_R] → 2d → 192 → GELU → d
      3. Reflex integration: сигнал от MoLE stats обоих блоков
      4. Output: (1-gate) * x_left + gate * x_right + fusion + reflex
    """
    
    BOTTLENECK_DIM = 192  # HELIX v6: информационное горлышко
    
    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.d_model = d_model
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        # WuNeng bottleneck: 2d → 192 → GELU → d (per HELIX v6 spec)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, self.BOTTLENECK_DIM),
            nn.GELU(),
            nn.Linear(self.BOTTLENECK_DIM, d_model),
        )
        
        self.reflex_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )
        
        self.norm = RMSNorm(d_model)
        self.fusion_scale = nn.Parameter(torch.tensor(0.1))
        self.reflex_scale = nn.Parameter(torch.tensor(0.05))
    
    def forward(
        self,
        x_left: torch.Tensor,
        x_right: torch.Tensor,
        stats_l: dict = None,
        stats_r: dict = None,
    ) -> Tuple[torch.Tensor, float]:
        h_left = x_left.mean(dim=1)
        h_right = x_right.mean(dim=1)
        h_cat = torch.cat([h_left, h_right], dim=-1)
        
        alpha = self.gate(h_cat)
        alpha_3d = alpha.unsqueeze(1)
        x_gated = (1 - alpha_3d) * x_left + alpha_3d * x_right
        
        fusion = self.fusion(h_cat)
        reflex_signal = self.reflex_gate(h_cat)
        
        x = x_gated + self.fusion_scale * fusion.unsqueeze(1) \
                     + self.reflex_scale * reflex_signal.unsqueeze(1)
        
        return self.norm(x), alpha.mean().detach()


class GlobalWorkspace(nn.Module):
    """
    Global Workspace Theory (Baars, 1988).
    Lazy-initialized, used during inference only.
    """
    
    def __init__(self, d_model: int = 768, n_blocks: int = 12):
        super().__init__()
        self.n_blocks = n_blocks
        self.competition = nn.Linear(d_model * n_blocks, n_blocks)
        self.broadcast_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.mix = nn.Parameter(torch.tensor(0.1))
        self.norm = RMSNorm(d_model)
    
    def forward(self, block_outputs: list, x_current: torch.Tensor) -> torch.Tensor:
        if len(block_outputs) < 2:
            return x_current
        
        h_means = [h.mean(dim=1) for h in block_outputs]
        while len(h_means) < self.n_blocks:
            h_means.append(torch.zeros_like(h_means[0]))
        h_means = h_means[:self.n_blocks]
        
        h_cat = torch.cat(h_means, dim=-1)
        gates = F.softmax(self.competition(h_cat), dim=-1)
        h_stack = torch.stack(h_means, dim=1)
        broadcast = torch.einsum('bn,bnd->bd', gates, h_stack)
        broadcast = self.broadcast_proj(broadcast)
        
        return self.norm(x_current + self.mix * broadcast.unsqueeze(1))


class SharedGlobalAttention(nn.Module):
    """
    Zamba-pattern Shared Global Attention.
    GQA with 4 heads. Applied after every N-th wave.
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, n_kv_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads
        
        self.norm = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.1))
        # BUG-4 fix: learnable temperature for QK-norm (Qwen3-style)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.0)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm(x)
        
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # BUG-4 fix: learnable temperature instead of 1/√d (killed attention with QK-norm)
        scale = torch.exp(self.log_temperature)
        attn = (q @ k.transpose(-2, -1)) * scale
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)
        
        return x + self.gate * out


class WaveScratchpad(nn.Module):
    """
    WaveScratchpad — последовательное мышление между волнами.
    """
    
    def __init__(self, d_model: int, compress_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.compress_dim = compress_dim or d_model // 4
        
        self.compress = nn.Linear(d_model, self.compress_dim, bias=False)
        self.expand = nn.Linear(self.compress_dim, d_model, bias=False)
        self.inject_gate = nn.Parameter(torch.tensor(-1.0))
        self.norm = nn.LayerNorm(self.compress_dim)
    
    def summarize(self, x: torch.Tensor) -> torch.Tensor:
        h_mean = x.mean(dim=1)
        summary = self.norm(self.compress(h_mean))
        return summary
    
    def inject(self, x: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        hint = self.expand(summary)
        gate = torch.sigmoid(self.inject_gate)
        return x + gate * hint.unsqueeze(1)


class TTTLoRA(nn.Module):
    """
    Test-Time Training with LoRA (MIT, 2025).
    Lazy-initialized: only needed during think().
    """
    
    def __init__(self, d_model: int, rank: int = 4, alpha: float = 8.0,
                 n_steps: int = 3, lr: float = 1e-3):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.n_steps = n_steps
        self.lr = lr
        
        self.lora_A = nn.Parameter(torch.randn(d_model, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_model))
    
    @torch.no_grad()
    def adapt(self, x: torch.Tensor, model_forward_fn):
        orig_A = self.lora_A.data.clone()
        orig_B = self.lora_B.data.clone()
        
        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)
        
        for step in range(self.n_steps):
            delta = (self.lora_A @ self.lora_B) * self.scale
            x_adapted = x + F.linear(x, delta)
            
            with torch.enable_grad():
                logits = model_forward_fn(x_adapted)
                shift_logits = logits[:, :-1].contiguous()
                shift_targets = logits[:, 1:].argmax(dim=-1)
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1)
                )
            
            loss.backward()
            if self.lora_A.grad is not None:
                self.lora_A.data -= self.lr * self.lora_A.grad
                self.lora_A.grad = None
            if self.lora_B.grad is not None:
                self.lora_B.data -= self.lr * self.lora_B.grad
                self.lora_B.grad = None
        
        adapted_delta = (self.lora_A @ self.lora_B) * self.scale
        
        self.lora_A.data.copy_(orig_A)
        self.lora_B.data.copy_(orig_B)
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)
        
        return adapted_delta
    
    def apply_delta(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        return x + F.linear(x, delta)


# ═══════════════════════════════════════════════════════════════
# BrainCore — Core Forward Pipeline
# ═══════════════════════════════════════════════════════════════

class BrainCore(nn.Module):
    """
    BrainCore — ядро forward pipeline ТАРС.
    
    Содержит:
      - Embedding + TemporalEmbedding
      - TarsBlock × n_layers (Cortical Columns)
      - WaveConsolidation × n_waves
      - SharedGlobalAttention, WaveScratchpad
      - PersonalityAdapter, LM Head
      - MatrixPool (IDME), NoveltyGate
      - ThinkingChain, QueryRouter
      - Memory projections (to_memory_space / from_memory_space)
    
    Извлечён из TarsMamba2LM для тестируемости и модульности.
    """
    
    def __init__(
        self,
        d_model: int = 1024,     # TARS HELIX (matches TarsConfig)
        n_layers: int = 20,      # 20 HelixBlocks
        vocab_size: int = 48256, # 48K text + 256 tool tokens
        d_state: int = 64,       # SSM state dimension
        headdim: int = 64,
        omega_dim: int = 32,
        pool_size: int = 48,
        n_experts: int = 8,
        expert_rank: int = 8,
        quant_mode: str = "ternary",
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.quant_mode = quant_mode
        self.logger = logging.getLogger("Tars.BrainCore")
        
        # ═══ Embedding ═══
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # ═══ Основные блоки (n_layers × TarsBlock) ═══
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            depth_ratio = i / max(n_layers - 1, 1)
            layer_omega = int(omega_dim * (0.5 + 1.5 * depth_ratio))
            layer_experts = max(4, int(n_experts * (0.5 + 0.5 * depth_ratio)))
            layer_dropout = 0.05 + 0.10 * depth_ratio
            
            self.blocks.append(TarsBlock(
                d_model=d_model, d_state=d_state,
                headdim=headdim, omega_dim=layer_omega,
                n_experts=layer_experts, layer_idx=i,
                quant_mode=quant_mode,
                dropout=layer_dropout,
            ))
        
        # ═══ Wave Consolidation ═══
        n_waves = n_layers // 2
        self.wave_consolidations = nn.ModuleList([
            WaveConsolidation(d_model) for _ in range(n_waves)
        ])
        
        # ═══ Shared Global Attention ═══
        self.shared_attn_interval = max(1, n_layers // 4)
        self.shared_global_attn = SharedGlobalAttention(d_model, n_heads=4, n_kv_heads=2)
        
        # ═══ WaveScratchpad ═══
        self.wave_scratchpad = WaveScratchpad(d_model)
        
        # ═══ IDME Matrix Pool ═══
        self.matrix_pool = MatrixPool(d_model, pool_size, quant_mode=quant_mode)
        
        # ═══ NoveltyGate (для IDME) ═══
        self.novelty_gate = NoveltyGate(d_model)
        
        # ═══ ThinkingLogger ═══
        self.thinking_logger = ThinkingLogger()
        
        # ═══ Titans Memory Hook (384d LTM) ═══
        self.mem_dim = 384
        self.to_memory_space = nn.Linear(d_model, self.mem_dim, bias=False)
        self.from_memory_space = nn.Linear(self.mem_dim, d_model, bias=False)
        self.titans_memory = None
        
        # ═══ Shared Memory Injector ═══
        self.shared_mem_injector = MemoryInjector(d_model, mem_dim=384, quant_mode=quant_mode)
        
        # ═══ Output head ═══
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # ═══ ThinkingChain ═══
        self.thinking_chain = ThinkingChain(
            d_model, d_memory=384, n_max_waves=n_layers // 2, vocab_size=vocab_size
        )
        
        # ═══ QueryRouter ═══
        self.query_router = QueryRouter(d_model, mem_dim=384, quant_mode=quant_mode)
        
        # ═══ PersonalityAdapter ═══
        self.personality = PersonalityAdapter(
            d_model=d_model,
            style_dim=128,
            max_passes=3,
            convergence_threshold=0.01,
        )
        
        # ═══ Speculative Decoding draft head ═══
        self.spec_draft_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, vocab_size),
        )
        
        # ═══ RoPE (base=500K per TARS spec, matches config.py) ═══
        self.rope = RotaryPositionEmbedding(d_model // (d_model // 64), base=500_000)
        
        # ═══ Wave-Parallel Input Diversification ═══
        self.proj_left = nn.Linear(d_model, d_model, bias=False)
        self.proj_right = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.proj_left.weight)
        nn.init.eye_(self.proj_right.weight)
        self.proj_left.weight.data += torch.randn_like(self.proj_left.weight) * 0.01
        self.proj_right.weight.data += torch.randn_like(self.proj_right.weight) * 0.01
        
        # ═══ Cached SSM states ═══
        self._gen_cache = None
        self._prefix_cache = None
        
        # Gradient checkpointing flag
        self.use_checkpointing = False
        
        # MoLE auxiliary loss
        self.mole_aux_loss = torch.tensor(0.0)
        
        # ═══ Lazy-init placeholders (inference-only) ═══
        self._ttt_lora = None
        self._global_workspace = None
    
    @property
    def ttt_lora(self):
        if self._ttt_lora is None:
            device = next(self.parameters()).device
            self._ttt_lora = TTTLoRA(self.d_model, rank=4, alpha=8.0, n_steps=3).to(device)
        return self._ttt_lora
    
    @property
    def global_workspace(self):
        if self._global_workspace is None:
            self._global_workspace = GlobalWorkspace(self.d_model, self.n_layers)
        return self._global_workspace
    
    def reset_cache(self):
        """Сброс кеша SSM-состояний."""
        self._gen_cache = {
            "wkv_states": [None] * self.n_layers,
            "x_prevs": [None] * self.n_layers,
            "ssd_states": [None] * self.n_layers,
            "conv_states": [None] * self.n_layers,
            "memory_vec": None,
        }
    
    def prefix_cache_save(self):
        if self._gen_cache is not None:
            self._prefix_cache = copy.deepcopy(self._gen_cache)
    
    def prefix_cache_load(self):
        if self._prefix_cache is not None:
            self._gen_cache = copy.deepcopy(self._prefix_cache)
            return True
        return False
    
    @torch.no_grad()
    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Fast single-step forward для генерации."""
        if self._gen_cache is None:
            self.reset_cache()
        
        c = self._gen_cache
        x = self.embedding(token_ids) * math.sqrt(self.d_model)  # Vaswani scaling
        
        n_waves = self.n_layers // 2
        wave_summary = None
        
        for wave_idx in range(n_waves):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            if wave_summary is not None:
                x = self.wave_scratchpad.inject(x, wave_summary)
            
            mem_signal = None
            if c["memory_vec"] is not None:
                h_mean = x.mean(dim=1)
                mem_signal = self.shared_mem_injector.compute_signal(h_mean, c["memory_vec"])
            
            x_left, c["wkv_states"][b_left], c["x_prevs"][b_left], _, \
                c["ssd_states"][b_left], c["conv_states"][b_left] = self.blocks[b_left](
                    x, c["wkv_states"][b_left], c["x_prevs"][b_left],
                    None, None,
                    c["ssd_states"][b_left], c["conv_states"][b_left],
                    None, mem_signal
                )
            x_right, c["wkv_states"][b_right], c["x_prevs"][b_right], _, \
                c["ssd_states"][b_right], c["conv_states"][b_right] = self.blocks[b_right](
                    x, c["wkv_states"][b_right], c["x_prevs"][b_right],
                    None, None,
                    c["ssd_states"][b_right], c["conv_states"][b_right],
                    None, mem_signal
                )
            
            x, _ = self.wave_consolidations[wave_idx](x_left, x_right)
            
            if (wave_idx + 1) % self.shared_attn_interval == 0:
                x = self.shared_global_attn(x)
            
            wave_summary = self.wave_scratchpad.summarize(x)
            
            if wave_idx < n_waves - 1:
                h_curr = x.mean(dim=1)
                h_for_mem = self.to_memory_space(h_curr)
                if c["memory_vec"] is None:
                    c["memory_vec"] = h_for_mem
                else:
                    novelty = 1.0 - F.cosine_similarity(
                        c["memory_vec"], h_for_mem, dim=-1
                    ).mean()
                    if novelty > 0.05:
                        c["memory_vec"] = 0.7 * c["memory_vec"] + 0.3 * h_for_mem
        
        if c["memory_vec"] is not None:
            mem_signal = self.from_memory_space(c["memory_vec"])
            x = x + 0.1 * mem_signal.unsqueeze(1)
        
        x = self.norm_f(x)
        return self.lm_head(x)
    
    def _sample(self, logits, temperature=0.7, top_k=50):
        """Sample a token from logits."""
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_k > 0:
            v, _ = logits.topk(top_k, dim=-1)
            logits[logits < v[:, -1:]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_vec: Optional[torch.Tensor] = None,
        rag_state: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (training).
        
        Pipeline: Embedding → TemporalEmbedding → Wave Loop →
                  RMSNorm → PersonalityAdapter → LM Head → Loss
        """
        x = self.embedding(input_ids) * math.sqrt(self.d_model)  # Vaswani scaling
        
        # ═══ Temporal Embedding (if available on parent) ═══
        if hasattr(self, 'temporal_embedding'):
            x = self.temporal_embedding(x)
        
        wkv_states = [None] * self.n_layers
        x_prevs = [None] * self.n_layers
        ssd_states = [None] * self.n_layers
        conv_states = [None] * self.n_layers
        
        mole_aux_total = torch.tensor(0.0, device=input_ids.device)
        
        n_waves = self.n_layers // 2
        wave_summary = None
        
        for wave_idx in range(n_waves):
            b_left = wave_idx * 2
            b_right = wave_idx * 2 + 1
            
            if b_right >= self.n_layers:
                break
            
            if wave_summary is not None:
                x = self.wave_scratchpad.inject(x, wave_summary)
            
            # ═══ Wave-Parallel Memory ═══
            mem_signal = None
            if memory_vec is not None:
                h_mean = x.mean(dim=1)
                mem_signal = self.shared_mem_injector.compute_signal(h_mean, memory_vec)
            
            x_in_left = self.proj_left(x)
            x_in_right = self.proj_right(x)
            
            if self.use_checkpointing and self.training:
                x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                    ssd_states[b_left], conv_states[b_left] = grad_checkpoint(
                    self.blocks[b_left], x_in_left, wkv_states[b_left],
                    x_prevs[b_left], None, rag_state,
                    ssd_states[b_left], conv_states[b_left], None, mem_signal,
                    use_reentrant=True
                )
                x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                    ssd_states[b_right], conv_states[b_right] = grad_checkpoint(
                    self.blocks[b_right], x_in_right, wkv_states[b_right],
                    x_prevs[b_right], None, rag_state,
                    ssd_states[b_right], conv_states[b_right], None, mem_signal,
                    use_reentrant=True
                )
            else:
                x_left, wkv_states[b_left], x_prevs[b_left], stats_l, \
                    ssd_states[b_left], conv_states[b_left] = self.blocks[b_left](
                    x_in_left, wkv_states[b_left], x_prevs[b_left], None, rag_state,
                    ssd_states[b_left], conv_states[b_left], None, mem_signal
                )
                x_right, wkv_states[b_right], x_prevs[b_right], stats_r, \
                    ssd_states[b_right], conv_states[b_right] = self.blocks[b_right](
                    x_in_right, wkv_states[b_right], x_prevs[b_right], None, rag_state,
                    ssd_states[b_right], conv_states[b_right], None, mem_signal
                )
            
            for stats in [stats_l, stats_r]:
                if isinstance(stats, dict) and "mole_aux_loss" in stats:
                    mole_aux_total = mole_aux_total + stats["mole_aux_loss"]
            
            # ═══ NaN/Inf Guard (T08) ═══
            if torch.isnan(x_left).any() or torch.isinf(x_left).any():
                x_left = torch.nan_to_num(x_left, nan=0.0, posinf=1.0, neginf=-1.0)
                self.logger.warning(f"NaN in x_left at wave {wave_idx}, replaced")
            if torch.isnan(x_right).any() or torch.isinf(x_right).any():
                x_right = torch.nan_to_num(x_right, nan=0.0, posinf=1.0, neginf=-1.0)
                self.logger.warning(f"NaN in x_right at wave {wave_idx}, replaced")
            
            x, _ = self.wave_consolidations[wave_idx](x_left, x_right, stats_l, stats_r)
            
            # ═══ Per-wave health check (T08) ═══
            x_norm = x.float().norm().item()
            if x_norm < 0.01 or x_norm > 100:
                x = x.clamp(-10, 10)
                self.logger.warning(f"Wave {wave_idx}: norm={x_norm:.4f}, clamped")
            
            if (wave_idx + 1) % self.shared_attn_interval == 0:
                x = self.shared_global_attn(x)
            
            wave_summary = self.wave_scratchpad.summarize(x)
            
            if wave_idx < n_waves - 1:
                h_curr = x.mean(dim=1)
                h_for_mem = self.to_memory_space(h_curr)
                if memory_vec is None:
                    memory_vec = h_for_mem
                else:
                    novelty = 1.0 - F.cosine_similarity(memory_vec, h_for_mem, dim=-1).mean()
                    update_mask = (novelty > 0.05).float()
                    new_mem = 0.7 * memory_vec + 0.3 * h_for_mem
                    memory_vec = update_mask * new_mem + (1.0 - update_mask) * memory_vec
        
        # Handle odd number of blocks
        if self.n_layers % 2 == 1:
            last_block = self.blocks[-1]
            x, _, _, stats_last, _, _ = last_block(x, None, None, memory_vec, rag_state)
            if isinstance(stats_last, dict) and "mole_aux_loss" in stats_last:
                mole_aux_total = mole_aux_total + stats_last["mole_aux_loss"]
        
        self.mole_aux_loss = mole_aux_total
        
        x = self.norm_f(x)
        x = self.personality(x)
        logits = self.lm_head(x)
        
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss = lm_loss + mole_aux_total
            return logits, total_loss
        
        return logits
    
    def encode_rag(self, rag_tokens: torch.Tensor) -> torch.Tensor:
        """Encode RAG document into compressed WKV state."""
        with torch.no_grad():
            x = self.embedding(rag_tokens)
            wkv_state = None
            x_prev = None
            for block in self.blocks:
                x, wkv_state, x_prev, _, _, _ = block(x, wkv_state, x_prev)
        return wkv_state
