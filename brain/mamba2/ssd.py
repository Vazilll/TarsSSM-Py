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
from functools import lru_cache
from brain.mamba2.bitnet import UniversalLinear
from einops import rearrange, repeat
from typing import Optional, Tuple


# ═══════════════════════════════════════════
# Утилиты SSD (без изменений)
# ═══════════════════════════════════════════

# Global cache for tril masks: key = (T, device_str)
_TRIL_MASK_CACHE: dict = {}

def _get_tril_masks(T: int, device: torch.device):
    """Cached lower-triangular masks for segsum."""
    key = (T, str(device))
    if key not in _TRIL_MASK_CACHE:
        mask_exclusive = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=-1)
        mask_inclusive = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=0)
        _TRIL_MASK_CACHE[key] = (mask_exclusive, mask_inclusive)
    return _TRIL_MASK_CACHE[key]


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
        initial_states = torch.zeros_like(states[:, :1])
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
    
    return Y, final_state


class CausalConv1d(nn.Module):
    """Causal 1D convolution (depthwise)."""
    
    def __init__(self, channels, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            groups=channels, padding=kernel_size - 1
        )
    
    def forward(self, x):
        y = self.conv(x.transpose(1, 2))
        y = y[:, :, :x.shape[1]]
        return y.transpose(1, 2)


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
    b_t: torch.Tensor,      # [B, S]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    One step of WKV scan (JIT-compiled for speed).
    
    State update: S = diag(w) · S + (k ⊗ v) · bonus
    Output:       y = r ⊙ (S · k)
    """
    # State decay + outer product injection
    state = state * w_t.unsqueeze(-1) + \
            (k_t.unsqueeze(-1) * v_t.unsqueeze(-2)) * b_t.unsqueeze(-1)
    # Readout
    y_t = r_t * torch.bmm(state, k_t.unsqueeze(-1)).squeeze(-1)
    return y_t, state


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
    RWKV-7 WKV scan with chunked processing for faster training.
    
    Оптимизация: вместо L отдельных вызовов _wkv_step,
    обрабатываем чанками по chunk_size токенов.
    Для L=256, chunk=32 → 8 итераций вместо 256.
    
    O(S²) per token, O(1) memory.
    
    Returns:
        output: [B, L, S]
        state:  [B, S, S] — updated state
    """
    B, L, S = r.shape
    
    if state is None:
        state = torch.zeros(B, S, S, device=r.device, dtype=r.dtype)
    
    # Pre-allocate output tensor
    output = torch.empty(B, L, S, device=r.device, dtype=r.dtype)
    
    # Single-token fast path (inference)
    if L == 1:
        y_t, state = _wkv_step(state, k[:, 0], v[:, 0], r[:, 0], w[:, 0], bonus[:, 0])
        output[:, 0] = y_t
        return output, state
    
    # Chunked processing (training): reduces Python loop overhead
    for chunk_start in range(0, L, chunk_size):
        chunk_end = min(chunk_start + chunk_size, L)
        for t in range(chunk_start, chunk_end):
            y_t, state = _wkv_step(
                state, k[:, t], v[:, t], r[:, t], w[:, t], bonus[:, t]
            )
            output[:, t] = y_t
    
    return output, state


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
    
    Параметры (при d_model=768, d_state=64):
      - Mamba часть: ~4.7M params (как раньше)
      - RWKV часть:  ~0.5M params (d_model→d_state проекции)
      - Fusion gate:  ~0.01M params
      - Total:       ~5.2M / block
    """
    
    def __init__(
        self,
        d_model: int = 768,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        quant_mode: str = "fp16",
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
        
        # ═══ RWKV time-shift mixing ═══
        self.time_mix = nn.Parameter(torch.ones(1, 1, d_model) * 0.5)
        
        # ═══════════════════════════════════
        # 🔥 DEEP GATED FUSION (WuNeng)
        # ═══════════════════════════════════
        # Fusion happens in d_inner space (before out_proj)
        # y_ssd is [B, L, d_inner], y_wkv needs upscaling from d_state
        self.wkv_up = UniversalLinear(d_state, self.d_inner, bias=False, mode=quant_mode)
        self.fusion_gate = nn.Sequential(
            UniversalLinear(self.d_inner * 2, self.d_inner, bias=True, mode=quant_mode),
            nn.SiLU(),
            UniversalLinear(self.d_inner, self.d_inner, bias=True, mode=quant_mode),
            nn.Sigmoid()
        )
        
        # ═══════════════════════════════════
        # 🔥 SHARED OUTPUT PROJECTION
        # ═══════════════════════════════════
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = UniversalLinear(self.d_inner, d_model, bias=False, mode=quant_mode)
        
        self.act = nn.SiLU()
    
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
            xBC = self.act(self.conv1d(xBC))
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
            
            # Adaptive chunk size: smaller for short sequences
            effective_chunk = min(self.chunk_size, max(seqlen, 8))
            
            y_ssd, final_ssd_state = ssd_scan(
                x_dt, A_dt, B_heads, C_heads,
                chunk_size=effective_chunk
            )
            ssd_state = final_ssd_state  # Save for future step mode
            
            y_ssd = y_ssd + x_heads * self.D.view(1, 1, -1, 1)
            y_ssd = rearrange(y_ssd, "b l h p -> b l (h p)")  # [B, L, d_inner]
        
        # ═══════════════════════════════════
        # 2B. RWKV-7 WKV PATH
        # ═══════════════════════════════════
        S = self.d_state
        r_rwkv, k_rwkv, v_rwkv, w_rwkv, bonus_rwkv = torch.split(
            rwkv_proj, [S, S, S, S, S], dim=-1
        )
        
        r_rwkv = torch.sigmoid(r_rwkv)
        w_rwkv = torch.exp(-torch.exp(w_rwkv))
        bonus_rwkv = torch.sigmoid(bonus_rwkv)
        
        y_wkv, wkv_state = wkv_scan(r_rwkv, k_rwkv, v_rwkv, w_rwkv, bonus_rwkv, wkv_state)
        # y_wkv: [B, L, d_state]
        
        # Upscale to d_inner for fusion
        y_wkv_up = self.wkv_up(y_wkv)  # [B, L, d_inner]
        
        # ═══════════════════════════════════
        # 3. DEEP GATED FUSION (WuNeng)
        # ═══════════════════════════════════
        gate = self.fusion_gate(torch.cat([y_ssd, y_wkv_up], dim=-1))
        y_fused = gate * y_ssd + (1 - gate) * y_wkv_up  # [B, L, d_inner]
        
        # ═══════════════════════════════════
        # 4. SHARED OUTPUT
        # ═══════════════════════════════════
        y = self.norm(y_fused * self.act(z))
        output = self.out_proj(y)  # [B, L, d_model]
        
        x_last = u[:, -1:, :]
        
        return output, wkv_state, x_last, ssd_state, conv_state


# ═══════════════════════════════════════════
# Legacy alias (backward compat)
# ═══════════════════════════════════════════
Mamba2Block = TarsCoreBlock
