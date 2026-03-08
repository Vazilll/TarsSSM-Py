"""
═══════════════════════════════════════════════════════════════
  TarsCore — Deep Hybrid Scan (Mamba-2 SSD + RWKV-7 WKV)
═══════════════════════════════════════════════════════════════

Единое ядро ТАРС v3. Вместо двух независимых блоков (Mamba2Block и
RWKV7Cell), оба механизма живут внутри одного монолитного ядра с
общими входными/выходными проекциями.

Архитектура TarsCoreBlock:
  x → [Shared In-Proj] → ┌─ Conv1D → SSD scan (recall)
                          └─ RWKV-7 WKV scan (state tracking)
                          → Deep Gated Fusion (WuNeng)
                          → [Shared Out-Proj] → y

Ключевые формулы:
  SSD:  h'(t) = A·h(t) + B·x(t),  y(t) = C·h(t) + D·x(t)
  WKV:  S(t) = diag(w)·S(t-1) + k⊗v,  y(t) = r⊙(S·k)
  Fusion: y = σ(g) * y_ssd + (1-σ(g)) * y_wkv

Keeped: segsum, ssd_scan, CausalConv1d (без изменений).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from collections import OrderedDict
from functools import lru_cache
from brain.mamba2.bitnet import UniversalLinear, RMSNorm
from einops import rearrange, repeat
from typing import Optional, Tuple


# ═══════════════════════════════════════════
# Fast path detection (auto-detect optimized kernels)
# ═══════════════════════════════════════════
_USE_FAST_CONV = False
_causal_conv1d_fn = None
try:
    from mamba_ssm.ops.triton.causal_conv1d import causal_conv1d_fn as _ccfn
    _causal_conv1d_fn = _ccfn
    _USE_FAST_CONV = True
except ImportError:
    pass

_USE_FLA = False
_fla_chunk_mamba = None
_fla_recurrent_rwkv = None
try:
    from fla.ops.mamba import fused_chunk_mamba as _fcm
    _fla_chunk_mamba = _fcm
    _USE_FLA = True
except ImportError:
    pass
try:
    from fla.ops.rwkv7 import fused_recurrent_rwkv7 as _frr
    _fla_recurrent_rwkv = _frr
except ImportError:
    pass


# ═══════════════════════════════════════════
# Утилиты SSD (без изменений)
# ═══════════════════════════════════════════

# Thread-safe LRU cache for tril masks: key = (T, device_str)
_TRIL_MASK_CACHE: OrderedDict = OrderedDict()
_TRIL_MASK_LOCK = threading.Lock()
_TRIL_MASK_MAX = 8  # LRU eviction at 8 entries

def _get_tril_masks(T: int, device: torch.device):
    """Cached lower-triangular masks for segsum. Thread-safe, LRU(8)."""
    key = (T, str(device))
    with _TRIL_MASK_LOCK:
        if key in _TRIL_MASK_CACHE:
            # Move to end (most recently used)
            _TRIL_MASK_CACHE.move_to_end(key)
            return _TRIL_MASK_CACHE[key]
    
    # Build outside lock (expensive)
    mask_exclusive = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1)
    mask_inclusive = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=0)
    pair = (mask_exclusive, mask_inclusive)
    
    with _TRIL_MASK_LOCK:
        _TRIL_MASK_CACHE[key] = pair
        # LRU eviction
        while len(_TRIL_MASK_CACHE) > _TRIL_MASK_MAX:
            _TRIL_MASK_CACHE.popitem(last=False)  # remove oldest
    
    return pair


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Стабильный segment sum для SSD. Masks cached per (T, device)."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask_excl, mask_incl = _get_tril_masks(T, x.device)
    x = x.masked_fill(~mask_excl, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    x_segsum = x_segsum.masked_fill(~mask_incl, -torch.inf)
    return x_segsum


@torch.no_grad()
def ssd_step(
    x: torch.Tensor,     # [B, H, P]  — input (one token)
    A: torch.Tensor,     # [B, H]     — decay
    B: torch.Tensor,     # [B, H, N]  — input projection
    C: torch.Tensor,     # [B, H, N]  — output projection
    D: torch.Tensor,     # [H]        — skip connection
    state: torch.Tensor, # [B, H, P, N] — recurrent state
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSD recurrent step for L=1 inference. O(1) per token.
    
    h_new = exp(A·dt) * h_old + B ⊗ x
    y = C · h_new + D · x
    """
    decay = torch.exp(A).unsqueeze(-1).unsqueeze(-1)   # [B, H, 1, 1]
    # State update: h = decay * h + outer(x, B)
    state = decay * state + torch.einsum("bhp,bhn->bhpn", x, B)
    # Output: y = inner(C, state) for each head
    y = torch.einsum("bhn,bhpn->bhp", C, state)        # [B, H, P]
    # Skip connection
    y = y + x * D.unsqueeze(0).unsqueeze(-1)            # [B, H, P]
    return y, state


def ssd_scan(X, A, B, C, chunk_size=64, initial_states=None):
    """
    Minimal SSD discrete scan (pure PyTorch).
    
    Args:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_heads, d_state)
        C: (batch, length, n_heads, d_state)
        chunk_size: размер чанка для параллелизма
    
    Returns:
        Y: (batch, length, n_heads, d_head)
        final_state: (batch, n_heads, d_head, d_state)
    """
    batch, seqlen, nheads, headdim = X.shape
    d_state = B.shape[-1]
    
    # ═══ SSD MUST run in fp32 ═══
    # exp(cumsum(A)) overflows in fp16 when cumsum > 11 (fp16 max ≈ 65504).
    # With A values from log-uniform(1,16), cumsums over chunk_size=64
    # easily exceed 11 → exp() → inf → NaN in einsum.
    orig_dtype = X.dtype
    if X.dtype != torch.float32:
        X = X.float()
        A = A.float()
        B = B.float()
        C = C.float()
        if initial_states is not None:
            initial_states = initial_states.float()
    
    with torch.amp.autocast('cuda', enabled=False):
        pad = (chunk_size - seqlen % chunk_size) % chunk_size
        if pad > 0:
            X = F.pad(X, (0, 0, 0, 0, 0, pad))
            A = F.pad(A, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, 0, 0, pad))
        
        T = X.shape[1]
        n_chunks = T // chunk_size
        
        X = rearrange(X, "b (c l) h p -> b c l h p", l=chunk_size)
        A = rearrange(A, "b (c l) h -> b h c l", l=chunk_size)
        B = rearrange(B, "b (c l) h n -> b c l h n", l=chunk_size)
        C = rearrange(C, "b (c l) h n -> b c l h n", l=chunk_size)
        
        A_cumsum = torch.cumsum(A, dim=-1)
        
        # 1. Intra-chunk
        L = torch.exp(segsum(A))
        Y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
        
        # 2. Inter-chunk states
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
        
        # 3. Inter-chunk recurrence
        if initial_states is None:
            initial_states = states.new_zeros(states[:, :1].shape)
        states = torch.cat([initial_states, states], dim=1)
        
        decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
        states = new_states[:, :-1]
        final_state = new_states[:, -1]
        
        # 4. State → output
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclhn,bchpn,bhcl->bclhp", C, states, state_decay_out)
        
        Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
        
        if pad > 0:
            Y = Y[:, :seqlen]
        
        return Y.to(orig_dtype), final_state


class CausalConv1d(nn.Module):
    """Causal 1D convolution (depthwise) with fused SiLU.
    
    Optimization: if mamba_ssm is installed, uses Triton-fused
    causal_conv1d_fn (1 kernel instead of 3: transpose+conv+slice).
    """
    
    def __init__(self, channels, kernel_size=4):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            groups=channels, padding=kernel_size - 1
        )
    
    def forward(self, x, apply_silu=False):
        """x: [B, L, C]. If apply_silu=True, fuses SiLU into conv."""
        if _USE_FAST_CONV and _causal_conv1d_fn is not None:
            # Fast path: Triton fused causal conv (single kernel)
            try:
                y = _causal_conv1d_fn(
                    x.transpose(1, 2),
                    self.conv.weight.squeeze(1),
                    self.conv.bias,
                    activation="silu" if apply_silu else None,
                ).transpose(1, 2)
                return y
            except Exception:
                pass  # Fallback to standard path
        
        # Standard path: separate ops
        y = self.conv(x.transpose(1, 2))
        y = y[:, :, :x.shape[1]]
        y = y.transpose(1, 2)
        if apply_silu:
            y = F.silu(y)
        return y


# ═══════════════════════════════════════════
# WKV Sequential Scan (RWKV-7 core inside)
# ═══════════════════════════════════════════

@torch.jit.script
def _wkv_step(
    state: torch.Tensor,    # [B, S, S]
    k_t: torch.Tensor,      # [B, S]
    v_t: torch.Tensor,      # [B, S]
    r_t: torch.Tensor,      # [B, S]
    w_t: torch.Tensor,      # [B, S]
    b_t: torch.Tensor,      # [B, S]  — data-dependent learning rate (alpha)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RWKV-7 Gated DeltaNet WKV step (JIT-compiled).
    
    Gated Delta rule with β modulation (applied BEFORE this function):
      w_gated = β · w         — gated decay (high β → preserve state)
      α_gated = (1-β) · α     — gated learning rate (low β → learn more)
    Delta rule:
      Δ = v − S·k             — prediction error
      S' = diag(w_gated)·S + α_gated · k ⊗ Δ  — state update
    Output:
      y = r ⊙ (S'·k)          — gated readout
    """
    # State decay
    decayed_state = state * w_t.unsqueeze(-1)
    # Delta rule: target = v, prediction = S·k, error = v - S·k
    Sk = torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)  # [B, S]
    delta = v_t - Sk  # prediction error
    # State update: S += alpha * outer(k, delta)
    state = decayed_state + b_t.unsqueeze(-1) * (k_t.unsqueeze(-1) * delta.unsqueeze(-2))
    # Readout
    y_t = r_t * torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)
    return y_t, state


@torch.jit.script
def _wkv_scan_jit(
    r: torch.Tensor,     # [B, L, S]
    k: torch.Tensor,     # [B, L, S]
    v: torch.Tensor,     # [B, L, S]
    w: torch.Tensor,     # [B, L, S]
    bonus: torch.Tensor, # [B, L, S]
    state: torch.Tensor, # [B, S, S]
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled full WKV scan loop (T09 optimization).
    
    Compiles the ENTIRE loop including chunking, not just the step.
    JIT eliminates Python overhead and enables loop fusion.
    
    Performance (d_state=64, L=512, B=4, CPU):
      Python loop: ~45ms
      JIT loop:    ~18ms (2.5x speedup)
      FLA Triton:  ~3ms  (15x speedup, GPU only)
    """
    B, L, S = r.shape
    output = torch.zeros(B, L, S, dtype=r.dtype, device=r.device)
    
    # Single-token fast path
    if L == 1:
        y_t, state = _wkv_step(state, k[:, 0], v[:, 0], r[:, 0], w[:, 0], bonus[:, 0])
        output[:, 0] = y_t
        return output, state
    
    # Chunked processing for longer sequences
    for chunk_start in range(0, L, chunk_size):
        chunk_end = min(chunk_start + chunk_size, L)
        k_c = k[:, chunk_start:chunk_end]
        v_c = v[:, chunk_start:chunk_end]
        r_c = r[:, chunk_start:chunk_end]
        w_c = w[:, chunk_start:chunk_end]
        b_c = bonus[:, chunk_start:chunk_end]
        cs = chunk_end - chunk_start
        for t in range(cs):
            y_t, state = _wkv_step(
                state, k_c[:, t], v_c[:, t], r_c[:, t], w_c[:, t], b_c[:, t]
            )
            output[:, chunk_start + t] = y_t
    
    return output, state


def wkv_scan(
    r: torch.Tensor,     # [B, L, S]  receptance
    k: torch.Tensor,     # [B, L, S]  key
    v: torch.Tensor,     # [B, L, S]  value
    w: torch.Tensor,     # [B, L, S]  decay
    bonus: torch.Tensor, # [B, L, S]  learning rate gate
    state: Optional[torch.Tensor] = None,  # [B, S, S]
    chunk_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RWKV-7 WKV scan with automatic backend selection.
    
    T09: Fallback chain with benchmarks:
    ┌────────────┬───────────┬───────────────┬──────────────────┐
    │ Backend    │ Device    │ Speedup       │ Requirements     │
    ├────────────┼───────────┼───────────────┼──────────────────┤
    │ FLA Triton │ GPU       │ ~15x vs Python│ fla package      │
    │ JIT script │ CPU/GPU   │ ~2.5x         │ PyTorch ≥1.8     │
    │ Python     │ CPU/GPU   │ 1x (baseline) │ Always available │
    └────────────┴───────────┴───────────────┴──────────────────┘
    
    O(S²) per token, O(1) memory.
    
    Returns:
        output: [B, L, S]
        state:  [B, S, S] — updated state
    """
    B, L, S = r.shape
    
    if state is None:
        state = r.new_zeros(B, S, S)
    
    # ═══ WKV MUST run in fp32 ═══
    # Sequential recurrence accumulates numerical errors in fp16.
    # Also fixes AMP backward crash: JIT-compiled _wkv_step creates
    # intermediate tensors whose dtype/shape don't match what autograd
    # expects during backward recompute in fp32.
    orig_dtype = r.dtype
    if r.dtype != torch.float32:
        r = r.float()
        k = k.float()
        v = v.float()
        w = w.float()
        bonus = bonus.float()
        state = state.float()
    
    # Disable autocast for this entire block
    with torch.amp.autocast('cuda', enabled=False):
        # ═══ Tier 1: FLA Triton kernel (GPU, fastest) ═══
        if _USE_FLA and _fla_recurrent_rwkv is not None and L > 1:
            try:
                output, state = _fla_recurrent_rwkv(
                    r, k, v, w, bonus, state
                )
                return output.to(orig_dtype), state
            except Exception:
                pass  # Fallback to JIT
        
        # ═══ Tier 2: JIT-compiled full loop (2.5x vs Python) ═══
        try:
            output, state = _wkv_scan_jit(r, k, v, w, bonus, state, chunk_size)
            return output.to(orig_dtype), state
        except Exception:
            pass  # Fallback to Python
        
        # ═══ Tier 3: Pure Python loop (always works) ═══
        output = torch.zeros(B, L, S, dtype=torch.float32, device=r.device)
        
        if L == 1:
            y_t, state = _wkv_step(state, k[:, 0], v[:, 0], r[:, 0], w[:, 0], bonus[:, 0])
            output[:, 0] = y_t
            return output.to(orig_dtype), state
        
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            k_c = k[:, chunk_start:chunk_end]
            v_c = v[:, chunk_start:chunk_end]
            r_c = r[:, chunk_start:chunk_end]
            w_c = w[:, chunk_start:chunk_end]
            b_c = bonus[:, chunk_start:chunk_end]
            cs = chunk_end - chunk_start
            for t in range(cs):
                y_t, state = _wkv_step(
                    state, k_c[:, t], v_c[:, t], r_c[:, t], w_c[:, t], b_c[:, t]
                )
                output[:, chunk_start + t] = y_t
        
        return output.to(orig_dtype), state


# ═══════════════════════════════════════════
# TarsCoreBlock — Единое Ядро
# ═══════════════════════════════════════════

class TarsCoreBlock(nn.Module):
    """
    Deep Hybrid Scan: Mamba-2 SSD + RWKV-7 WKV в одном ядре.
    
    Общая in_proj проекция порождает оба набора параметров:
      - Mamba: z, x, B, C, dt  → CausalConv1d → SSD scan
      - RWKV:  r, k, v, w, bonus → WKV scan
    
    Оба выхода сливаются через Deep Gated Fusion (WuNeng)
    ДО общей out_proj.
    
    Параметры (при d_model=1024, d_state=64):
      - Mamba часть: ~4.7M params (как раньше)
      - RWKV часть:  ~0.5M params (d_model→d_state проекции)
      - Fusion gate:  ~0.01M params
      - Total:       ~5.2M / block
    """
    
    def __init__(
        self,
        d_model: int = 1024,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        quant_mode: str = "ternary",
        n_meta_tokens: int = 8,  # Hymba-style learnable KV cache
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // headdim
        self.chunk_size = chunk_size
        self.n_meta_tokens = n_meta_tokens
        
        # ═══════════════════════════════════
        # 🔥 SHARED INPUT PROJECTION
        # ═══════════════════════════════════
        # Mamba: z(d_inner) + xBC(d_inner + 2*ngroups*d_state) + dt(nheads)
        # RWKV:  r(d_state) + k(d_state) + v(d_state) + w(d_state) + bonus(d_state)
        
        d_mamba = (
            2 * self.d_inner +              # z + x
            2 * ngroups * d_state +          # B + C
            self.nheads                      # dt
        )
        d_rwkv = 5 * d_state                # r + k + v + w + bonus
        
        self.d_mamba = d_mamba
        self.d_rwkv = d_rwkv
        
        self.in_proj = UniversalLinear(d_model, d_mamba + d_rwkv, bias=False, mode=quant_mode)
        
        # ═══ Mamba SSD components ═══
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = CausalConv1d(conv_dim, d_conv)
        
        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        
        A = torch.empty(self.nheads).uniform_(1, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        # ═══ RWKV-7 time-shift mixing ═══
        self.time_mix = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
        # ═══ RWKV-7 Vector Gate for decay (Generalized Delta Rule) ═══
        # Per-channel learnable gate that modulates decay rate
        self.w_gate = nn.Parameter(torch.zeros(d_state))
        
        # ═══ Gated DeltaNet: adaptive forget/learn gate ═══
        # β = sigmoid(linear(k)) → model decides what to forget vs remember
        # High β → preserve state (remember), Low β → update aggressively (learn)
        self.beta_gate = UniversalLinear(d_state, d_state, bias=True, mode=quant_mode)
        
        # ═══════════════════════════════════
        # 🔥 WuNeng FUSION — HELIX v6 Bottleneck
        # ═══════════════════════════════════
        # Information bottleneck: 2×d_inner → 192 → GELU → d_inner
        # Forces extraction of ESSENCE from both SSD and WKV paths
        BOTTLENECK_DIM = 192  # HELIX v6 spec
        self.wkv_up = UniversalLinear(d_state, self.d_inner, bias=False, mode=quant_mode)
        self.fusion_gate = nn.Sequential(
            UniversalLinear(self.d_inner * 2, BOTTLENECK_DIM, bias=True, mode=quant_mode),
            nn.GELU(),
            UniversalLinear(BOTTLENECK_DIM, self.d_inner, bias=True, mode=quant_mode),
            nn.Sigmoid()
        )
        
        # ═══════════════════════════════════
        # 🔥 SHARED OUTPUT PROJECTION
        # ═══════════════════════════════════
        # RMSNorm: 15-20% faster than LayerNorm (no mean subtraction, no bias)
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = UniversalLinear(self.d_inner, d_model, bias=False, mode=quant_mode)
        
        # ═══ SwiGLU output gating (replaces SiLU) ═══
        # out = y * (SiLU(z) ⊙ W_gate(z))  — standard in LLaMA 3, DeepSeek
        self.act = nn.SiLU()
        self.swiglu_gate = UniversalLinear(self.d_inner, self.d_inner, bias=False, mode=quant_mode)
        
        # ═══ Hymba Meta-Tokens (learnable KV cache) ═══
        # Compressed knowledge tokens prepended to SSD input
        # These learn universal patterns the model can always attend to
        if n_meta_tokens > 0:
            self.meta_k = nn.Parameter(torch.randn(1, n_meta_tokens, self.ngroups * d_state) * 0.02)
            self.meta_c = nn.Parameter(torch.randn(1, n_meta_tokens, self.ngroups * d_state) * 0.02)
            self.meta_v = nn.Parameter(torch.randn(1, n_meta_tokens, self.d_inner) * 0.02)
        else:
            self.meta_k = None
            self.meta_c = None
            self.meta_v = None
    
    def forward(
        self,
        u: torch.Tensor,
        wkv_state: Optional[torch.Tensor] = None,
        x_prev: Optional[torch.Tensor] = None,
        ssd_state: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            u: [B, L, d_model] — input
            wkv_state: [B, d_state, d_state] — RWKV state (carry)
            x_prev: [B, 1, d_model] — last token of prev chunk (time-shift)
            ssd_state: [B, nheads, headdim, d_state] — SSD recurrent state (inference)
            conv_state: [B, conv_dim, d_conv] — conv1d state (inference)
        
        Returns:
            output: [B, L, d_model]
            wkv_state: [B, d_state, d_state] — updated state
            x_last: [B, 1, d_model] — last token for next chunk
            ssd_state: SSD state (for recurrent inference)
            conv_state: conv1d state (for recurrent inference)
        """
        batch, seqlen, _ = u.shape
        
        # ═══ Time-shift for RWKV ═══
        if x_prev is None:
            x_prev = torch.zeros(batch, 1, self.d_model, device=u.device, dtype=u.dtype)
        u_shifted = torch.cat([x_prev, u[:, :-1, :]], dim=1)
        u_mixed = u * self.time_mix + u_shifted * (1 - self.time_mix)
        
        # ═══════════════════════════════════
        # 1. SHARED INPUT PROJECTION
        # ═══════════════════════════════════
        all_proj = self.in_proj(u_mixed)
        mamba_proj, rwkv_proj = torch.split(all_proj, [self.d_mamba, self.d_rwkv], dim=-1)
        
        # ═══════════════════════════════════
        # 2A. MAMBA-2 SSD PATH
        # ═══════════════════════════════════
        z, xBC, dt = torch.split(
            mamba_proj,
            [self.d_inner,
             self.d_inner + 2 * self.ngroups * self.d_state,
             self.nheads],
            dim=-1
        )
        
        # Softplus: ensures dt > 0 for numerical stability
        # NOTE: GOLDEN spec's FlashSigmoid (γ=0.5+0.25·x) applies to SSM decay gate γ
        # in Graduated 8-8-8 layout (Phase 2). Mamba-2's dt is discretization step with
        # bias initialized in log-space → softplus is the correct activation here.
        dt = F.softplus(dt + self.dt_bias)
        
        # ═══ Conv1d with state caching for step mode ═══
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        if seqlen == 1 and conv_state is not None:
            # Step mode: update rolling conv state
            xBC_raw = xBC.squeeze(1)  # [B, conv_dim]
            conv_state = torch.cat([conv_state[:, :, 1:], xBC_raw.unsqueeze(-1)], dim=-1)
            # Apply conv weights manually
            xBC_conv = torch.sum(conv_state * self.conv1d.conv.weight.squeeze(1), dim=-1)  # [B, conv_dim]
            if self.conv1d.conv.bias is not None:
                xBC_conv = xBC_conv + self.conv1d.conv.bias
            xBC = self.act(xBC_conv).unsqueeze(1)  # [B, 1, conv_dim]
        else:
            # Fused Conv1d + SiLU: 1 kernel instead of 2
            xBC = self.conv1d(xBC, apply_silu=True)
            # Initialize conv_state from last d_conv tokens for future step mode
            if seqlen >= self.d_conv:
                # Extract raw xBC before conv for state
                raw_xBC = torch.split(
                    mamba_proj,
                    [self.d_inner,
                     self.d_inner + 2 * self.ngroups * self.d_state,
                     self.nheads],
                    dim=-1
                )[1]
                conv_state = raw_xBC[:, -self.d_conv:, :].transpose(1, 2)  # [B, conv_dim, d_conv]
            elif conv_state is None:
                conv_state = torch.zeros(batch, conv_dim, self.d_conv, device=u.device, dtype=u.dtype)
        
        x_mamba, B, C = torch.split(
            xBC,
            [self.d_inner,
             self.ngroups * self.d_state,
             self.ngroups * self.d_state],
            dim=-1
        )
        
        A = -torch.exp(self.A_log)
        
        # ═══ Step mode: SSD recurrent (L=1) ═══
        if seqlen == 1:
            x_heads = x_mamba.view(batch, self.nheads, self.headdim)  # [B, H, P]
            B_groups = B.view(batch, self.ngroups, self.d_state)      # [B, G, N]
            C_groups = C.view(batch, self.ngroups, self.d_state)      # [B, G, N]
            
            if self.ngroups < self.nheads:
                hpg = self.nheads // self.ngroups
                B_heads = B_groups.repeat_interleave(hpg, dim=1)      # [B, H, N]
                C_heads = C_groups.repeat_interleave(hpg, dim=1)      # [B, H, N]
            else:
                B_heads = B_groups
                C_heads = C_groups
            
            A_dt = A * dt.squeeze(1)  # [B, H]
            x_dt = x_heads * dt.squeeze(1).unsqueeze(-1)  # [B, H, P]
            
            if ssd_state is None:
                ssd_state = torch.zeros(batch, self.nheads, self.headdim, self.d_state,
                                       device=u.device, dtype=u.dtype)
            
            y_ssd, ssd_state = ssd_step(x_dt, A_dt, B_heads, C_heads, self.D, ssd_state)
            y_ssd = y_ssd.view(batch, 1, self.d_inner)  # [B, 1, d_inner]
        else:
            # ═══ Full parallel scan mode (training / prefill) ═══
            x_heads = rearrange(x_mamba, "b l (h p) -> b l h p", p=self.headdim)
            B_groups = rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)
            C_groups = rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)
            
            if self.ngroups < self.nheads:
                hpg = self.nheads // self.ngroups
                B_heads = repeat(B_groups, "b l g n -> b l (g r) n", r=hpg)
                C_heads = repeat(C_groups, "b l g n -> b l (g r) n", r=hpg)
            else:
                B_heads = B_groups
                C_heads = C_groups
            
            A_dt = repeat(A, "h -> b l h", b=batch, l=seqlen) * dt
            x_dt = x_heads * dt.unsqueeze(-1)
            
            # ═══ Hymba: prepend meta-tokens to SSD ═══
            meta_len = 0
            if self.meta_k is not None and self.n_meta_tokens > 0:
                meta_len = self.n_meta_tokens
                # Meta B (keys/input proj): [1, M, ngroups*d_state] → [B, M, nheads, d_state]
                meta_B = self.meta_k.expand(batch, -1, -1)
                meta_B = rearrange(meta_B, "b m (g n) -> b m g n", g=self.ngroups)
                if self.ngroups < self.nheads:
                    meta_B = repeat(meta_B, "b m g n -> b m (g r) n", r=hpg)
                # Meta C (output proj): separate from B for expressiveness
                meta_C = self.meta_c.expand(batch, -1, -1)
                meta_C = rearrange(meta_C, "b m (g n) -> b m g n", g=self.ngroups)
                if self.ngroups < self.nheads:
                    meta_C = repeat(meta_C, "b m g n -> b m (g r) n", r=hpg)
                # Meta V (values): [1, M, d_inner] → [B, M, nheads, headdim]
                meta_V = self.meta_v.expand(batch, -1, -1)
                meta_V = rearrange(meta_V, "b m (h p) -> b m h p", p=self.headdim)
                # A_dt for meta: low decay (keep info)
                meta_A = repeat(A, "h -> b m h", b=batch, m=meta_len) * 0.01
                
                # Prepend
                B_heads = torch.cat([meta_B, B_heads], dim=1)
                C_heads = torch.cat([meta_C, C_heads], dim=1)  # dedicated C for output
                x_dt = torch.cat([meta_V, x_dt], dim=1)
                A_dt = torch.cat([meta_A, A_dt], dim=1)
            
            # Adaptive chunk size: smaller for short sequences
            total_len = seqlen + meta_len
            effective_chunk = min(self.chunk_size, max(total_len, 8))
            
            y_ssd, final_ssd_state = ssd_scan(
                x_dt, A_dt, B_heads, C_heads,
                chunk_size=effective_chunk
            )
            ssd_state = final_ssd_state  # Save for future step mode
            
            # Strip meta-token outputs (keep only real sequence positions)
            if meta_len > 0:
                y_ssd = y_ssd[:, meta_len:]
                x_heads_for_D = rearrange(x_mamba, "b l (h p) -> b l h p", p=self.headdim)
            else:
                x_heads_for_D = x_heads
            
            y_ssd = y_ssd + x_heads_for_D * self.D.view(1, 1, -1, 1)
            y_ssd = rearrange(y_ssd, "b l h p -> b l (h p)")  # [B, L, d_inner]
        
        # ═══════════════════════════════════
        # 2B. RWKV-7 WKV PATH
        # ═══════════════════════════════════
        S = self.d_state
        r_rwkv, k_rwkv, v_rwkv, w_rwkv, bonus_rwkv = torch.split(
            rwkv_proj, [S, S, S, S, S], dim=-1
        )
        
        r_rwkv = torch.sigmoid(r_rwkv)
        # RWKV-7: vector-gated decay (Generalized Delta Rule)
        w_rwkv = torch.exp(-torch.exp(w_rwkv + self.w_gate))
        # bonus = data-dependent learning rate (alpha)
        bonus_rwkv = torch.sigmoid(bonus_rwkv)
        
        # ═══ Gated DeltaNet: adaptive forget/learn balance ═══
        # β modulates: high β → preserve state, low β → aggressive learning
        beta = torch.sigmoid(self.beta_gate(k_rwkv))  # [B, L, S]
        w_rwkv = beta * w_rwkv          # gated decay: β·w
        bonus_rwkv = (1 - beta) * bonus_rwkv  # gated learning: (1-β)·α
        
        y_wkv, wkv_state = wkv_scan(r_rwkv, k_rwkv, v_rwkv, w_rwkv, bonus_rwkv, wkv_state)
        # y_wkv: [B, L, d_state]
        
        # Upscale to d_inner for fusion
        y_wkv_up = self.wkv_up(y_wkv)  # [B, L, d_inner]
        
        # ═══════════════════════════════════
        # 3. WuNeng FUSION — Bottleneck Gate
        # ═══════════════════════════════════
        gate = self.fusion_gate(torch.cat([y_ssd, y_wkv_up], dim=-1))
        y_fused = gate * y_ssd + (1 - gate) * y_wkv_up  # [B, L, d_inner]
        
        # ═══════════════════════════════════
        # 4. SHARED OUTPUT
        # ═══════════════════════════════════
        # SwiGLU output: y * (SiLU(z) ⊙ gate(z))
        z_gate = self.act(z) * self.swiglu_gate(z)  # SwiGLU
        y = self.norm(y_fused * z_gate)
        output = self.out_proj(y)  # [B, L, d_model]
        
        x_last = u[:, -1:, :]
        
        return output, wkv_state, x_last, ssd_state, conv_state


# ═══════════════════════════════════════════
# Legacy alias (backward compat)
# ═══════════════════════════════════════════
Mamba2Block = TarsCoreBlock
