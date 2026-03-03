"""
═══════════════════════════════════════════════════════════════
  Spiking Synapses — Нейроморфные синапсы для спинного мозга ТАРС
═══════════════════════════════════════════════════════════════

Архитектура (SpikingMamba + SI-LIF):

  Input → SI-LIF → SpikingMinGRU → Output
           │           │
           │   {-1,0,+1} спайки (тернарные)
           │           │
           └── совместимы с BitNet SIMD ядрами

SI-LIF (Signed Integer Leaky-Integrate-and-Fire):
  membrane += input
  membrane *= β (decay)
  spike = clip(round(membrane), -D, +D)   # тернарные спайки
  membrane -= spike * θ                    # reset

Surrogate gradient для обучения:
  Forward:  spike = ternary quantize (не дифференцируемо)
  Backward: ∂spike/∂input = α если |input| ≤ D, иначе 0

Circadian coupling:
  Мозг (TemporalEmbedding) модулирует β (decay) синапсов:
    Ночь: β → 0.95 (медленнее, сонный)
    День: β → 0.85 (быстрее, бодрый)

Использование:
  synapse = SpikingMinGRUBlock(dim=256, num_heads=4)
  output, (next_hidden, next_membrane) = synapse(x, prev_state)

  pool = SpikingSynapsePool(dim=256, n_synapses=5)
  results = pool.forward_all(x, task_type="action")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

# BitNet 1.58-bit линейный слой (fallback → nn.Linear)
try:
    from brain.mamba2.bitnet import UniversalLinear as _Linear
except ImportError:
    from torch.nn import Linear as _Linear


# ═══════════════════════════════════════════════════════
# 1. Surrogate Gradient Function
# ═══════════════════════════════════════════════════════

class SurrogateSpikeFunction(torch.autograd.Function):
    """
    Ternary spike с surrogate gradient для backprop.
    
    Forward:  spike = clip(round(x), -D, +D)
    Backward: ∂spike/∂x = α  если |x| ≤ D, иначе 0
    
    Это "rectangular" surrogate из SpikingMamba paper.
    """
    
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, D: int = 1, alpha: float = 1.0):
        # Тернарная квантизация: {-D, ..., 0, ..., +D}
        # Для D=1: {-1, 0, +1} — совместимо с BitNet!
        spike = membrane.round().clamp(-D, D)
        
        # Сохраняем для backward
        ctx.save_for_backward(membrane)
        ctx.D = D
        ctx.alpha = alpha
        return spike
    
    @staticmethod
    def backward(ctx, grad_output):
        membrane, = ctx.saved_tensors
        D = ctx.D
        alpha = ctx.alpha
        
        # Rectangular surrogate: gradient = α внутри [-D, D], 0 снаружи
        mask = (membrane.abs() <= D).float()
        grad_input = grad_output * mask * alpha
        
        return grad_input, None, None


def surrogate_spike(membrane: torch.Tensor, D: int = 1, alpha: float = 1.0) -> torch.Tensor:
    """Convenience wrapper для SurrogateSpikeFunction."""
    return SurrogateSpikeFunction.apply(membrane, D, alpha)


# ═══════════════════════════════════════════════════════
# 2. SI-LIF Neuron (Signed Integer Leaky-Integrate-and-Fire)
# ═══════════════════════════════════════════════════════

class SI_LIF(nn.Module):
    """
    Signed Integer LIF нейрон.
    
    Динамика:
      1. membrane = β * membrane + input     (leak + integrate)
      2. spike = surrogate_spike(membrane, D) (fire)
      3. membrane = membrane - spike * θ      (reset)
    
    Параметры:
      beta:    decay rate мембранного потенциала (0.85-0.95)
      theta:   порог сброса (learnable)
      D:       максимальная амплитуда спайка (1 = тернарный {-1,0,+1})
      alpha:   масштаб surrogate gradient
    
    Circadian coupling:
      temporal_phase ∈ [-1, 1] от TemporalEmbedding → β ± 5%
    """
    
    def __init__(self, dim: int, beta: float = 0.9, D: int = 1, alpha: float = 1.0,
                 learnable_beta: bool = True, learnable_theta: bool = True):
        super().__init__()
        self.dim = dim
        self.D = D
        self.alpha = alpha
        
        # β (decay) — per-neuron learnable
        if learnable_beta:
            # Initialize ~0.9, sigmoid ensures (0, 1)
            self.beta_raw = nn.Parameter(torch.full((dim,), math.log(beta / (1 - beta))))
        else:
            self.register_buffer('beta_raw', torch.full((dim,), math.log(beta / (1 - beta))))
        
        # θ (threshold) — learnable, init = 1.0
        if learnable_theta:
            self.theta = nn.Parameter(torch.ones(dim))
        else:
            self.register_buffer('theta', torch.ones(dim))
        
        # Circadian coupling strength
        self.circadian_strength = nn.Parameter(torch.tensor(0.05))
    
    @property
    def beta(self) -> torch.Tensor:
        """Effective β ∈ (0, 1)."""
        return torch.sigmoid(self.beta_raw)
    
    def init_membrane(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Инициализация мембранного потенциала (нули)."""
        return torch.zeros(batch_size, self.dim, device=device)
    
    def forward(self, x: torch.Tensor, membrane: Optional[torch.Tensor] = None,
                temporal_phase: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, dim] или [B, L, dim] — входной ток
            membrane: [B, dim] — мембранный потенциал (state)
            temporal_phase: [B, 1] или scalar — фаза из TemporalEmbedding (-1..1)
        
        Returns:
            spike: [B, dim] или [B, L, dim] — тернарные спайки {-D,...,+D}
            membrane: [B, dim] — обновлённый мембранный потенциал
        """
        # Handle sequence dimension
        has_seq = x.dim() == 3
        if has_seq:
            B, L, D = x.shape
            spikes = []
            if membrane is None:
                membrane = self.init_membrane(B, x.device)
            for t in range(L):
                spike_t, membrane = self._step(x[:, t], membrane, temporal_phase)
                spikes.append(spike_t)
            return torch.stack(spikes, dim=1), membrane  # [B, L, D], [B, D]
        else:
            if membrane is None:
                membrane = self.init_membrane(x.size(0), x.device)
            return self._step(x, membrane, temporal_phase)
    
    def _step(self, x: torch.Tensor, membrane: torch.Tensor,
              temporal_phase: Optional[torch.Tensor] = None
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Один шаг SI-LIF."""
        # Effective beta (с circadian модуляцией)
        beta = self.beta
        if temporal_phase is not None:
            # temporal_phase ∈ [-1, 1] → β ±5%
            modulation = self.circadian_strength * temporal_phase
            if modulation.dim() == 0:
                modulation = modulation.unsqueeze(0)
            while modulation.dim() < beta.dim():
                modulation = modulation.unsqueeze(-1)
            beta = (beta + modulation).clamp(0.01, 0.99)
        
        # 1. Leak + Integrate
        membrane = beta * membrane + x
        
        # 2. Fire (ternary spike через surrogate gradient)
        spike = surrogate_spike(membrane, self.D, self.alpha)
        
        # 3. Reset (soft reset — вычитаем spike * θ)
        membrane = membrane - spike * self.theta
        
        return spike, membrane


# ═══════════════════════════════════════════════════════
# 3. SpikingLinear — Linear через тернарные спайки
# ═══════════════════════════════════════════════════════

class SpikingLinear(nn.Module):
    """
    Linear layer с SI-LIF на выходе.
    
    x → BitNet Linear → SI-LIF → spike ∈ {-1, 0, +1}
    
    Результат: тернарные веса × тернарные активации = только ADD/SUB.
    """
    
    def __init__(self, d_in: int, d_out: int, beta: float = 0.9, D: int = 1):
        super().__init__()
        self.linear = _Linear(d_in, d_out, bias=False)
        self.lif = SI_LIF(d_out, beta=beta, D=D)
    
    def forward(self, x: torch.Tensor, membrane: Optional[torch.Tensor] = None,
                temporal_phase: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (spike, membrane)."""
        current = self.linear(x)
        return self.lif(current, membrane, temporal_phase)


# ═══════════════════════════════════════════════════════
# 4. SpikingMinGRUBlock — MinGRU с SI-LIF нейронами
# ═══════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMS Normalization."""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)


class SpikingGRUCell(nn.Module):
    """
    Minimal GRU cell с SI-LIF спайковыми активациями.
    
    Стандартный MinGRU:
      z = σ(Linear(x))            ← gate
      h_tilde = Linear(x)          ← candidate
      h = (1-z)*h_prev + z*h_tilde ← update
    
    Spiking MinGRU:
      z = SI-LIF(Linear(x))        ← gate через спайки!
      h_tilde = SI-LIF(Linear(x))  ← candidate через спайки!
      h = lerp(h_prev, h_tilde, z_normalized)
    
    Результат: рекуррентное состояние обновляется через тернарные спайки.
    """
    
    def __init__(self, dim: int, expansion_factor: float = 1.5, beta: float = 0.9):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.dim = dim
        
        # Gate pathway: input → linear → SI-LIF → gate
        self.gate_proj = _Linear(dim, inner_dim, bias=False)
        self.gate_lif = SI_LIF(inner_dim, beta=beta, D=1)
        
        # Candidate pathway: input → linear → SI-LIF → candidate
        self.cand_proj = _Linear(dim, inner_dim, bias=False)
        self.cand_lif = SI_LIF(inner_dim, beta=beta, D=1)
        
        # Output projection: inner_dim → dim
        self.out_proj = _Linear(inner_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor],
                gate_mem: Optional[torch.Tensor] = None,
                cand_mem: Optional[torch.Tensor] = None,
                temporal_phase: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, dim]
            h_prev: [B, inner_dim] recurrent state
            gate_mem, cand_mem: LIF membrane states
            temporal_phase: circadian phase
        
        Returns: (output, h_next, gate_mem, cand_mem)
        """
        inner_dim = self.gate_proj.out_features if hasattr(self.gate_proj, 'out_features') else self.gate_proj.weight.size(0)
        
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), inner_dim, device=x.device)
        
        # Gate: spike-based gating
        gate_current = self.gate_proj(x)
        gate_spike, gate_mem = self.gate_lif(gate_current, gate_mem, temporal_phase)
        # Normalize gate to [0, 1]: spike ∈ {-1,0,+1} → gate ∈ {0, 0.5, 1}
        z = (gate_spike + 1.0) / 2.0  # {-1,0,+1} → {0, 0.5, 1}
        
        # Candidate
        cand_current = self.cand_proj(x)
        cand_spike, cand_mem = self.cand_lif(cand_current, cand_mem, temporal_phase)
        
        # Update hidden state
        h_next = (1 - z) * h_prev + z * cand_spike
        
        # Output
        output = self.out_proj(h_next)
        
        return output, h_next, gate_mem, cand_mem


class SpikingMinGRUBlock(nn.Module):
    """
    Drop-in замена MinGRUBlock с SI-LIF нейронами.
    
    Интерфейс совместим с MinGRUBlock:
      forward(x, prev_hidden) → (output, next_hidden)
    
    Но prev_hidden теперь содержит и GRU state и membrane states.
    """
    
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1,
                 expansion_factor: float = 1.5, beta: float = 0.9,
                 num_layers: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        
        # Pre-norm
        self.input_norm = RMSNorm(dim)
        
        # Multi-head spiking GRU cells
        self.gru_heads = nn.ModuleList([
            SpikingGRUCell(head_dim, expansion_factor=expansion_factor, beta=beta)
            for _ in range(num_heads)
        ])
        
        # SwiGLU FFN
        self.ff_norm = RMSNorm(dim)
        ff_inner = int(dim * 4 * 2 / 3)  # SwiGLU: 2/3 × 4d
        self.ff_gate = _Linear(dim, ff_inner, bias=False)
        self.ff_up = _Linear(dim, ff_inner, bias=False)
        self.ff_down = _Linear(ff_inner, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.res_scale = 1.0 / math.sqrt(num_layers)
        
        # Spike statistics (for monitoring)
        self.register_buffer('_spike_count', torch.tensor(0.0))
        self.register_buffer('_total_count', torch.tensor(0.0))
    
    @property
    def sparsity(self) -> float:
        """Доля нулевых спайков (чем больше — тем эффективнее)."""
        if self._total_count == 0:
            return 0.0
        return 1.0 - (self._spike_count / self._total_count).item()
    
    def _pack_state(self, h_list, gm_list, cm_list) -> torch.Tensor:
        """Упаковать все states в один тензор для совместимости с MinGRU API."""
        # [h_0, h_1, ..., gm_0, gm_1, ..., cm_0, cm_1, ...]
        all_states = h_list + gm_list + cm_list
        return torch.cat(all_states, dim=-1)  # [B, total_dim]
    
    def _unpack_state(self, state: Optional[torch.Tensor]):
        """Распаковать state в per-head (h, gate_mem, cand_mem)."""
        if state is None:
            return [None]*self.num_heads, [None]*self.num_heads, [None]*self.num_heads
        
        # Вычисляем размеры per-head
        inner_dim = int(self.head_dim * 1.5)  # expansion_factor = 1.5
        
        # state layout: [h_0..h_n | gm_0..gm_n | cm_0..cm_n]
        h_total = inner_dim * self.num_heads
        gm_total = inner_dim * self.num_heads
        
        h_cat = state[:, :h_total]
        gm_cat = state[:, h_total:h_total + gm_total]
        cm_cat = state[:, h_total + gm_total:]
        
        h_list = list(h_cat.split(inner_dim, dim=-1))
        gm_list = list(gm_cat.split(inner_dim, dim=-1))
        cm_list = list(cm_cat.split(inner_dim, dim=-1))
        
        return h_list, gm_list, cm_list
    
    def forward(self, x: torch.Tensor, prev_hidden: Optional[torch.Tensor] = None,
                temporal_phase: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        API-совместимо с MinGRUBlock.
        
        Args:
            x: [B, L, dim]
            prev_hidden: packed state tensor (or None)
            temporal_phase: circadian modulation from TemporalEmbedding
        
        Returns:
            (output, next_hidden)
        """
        B, L, D = x.shape
        
        # Pre-norm
        normed = self.input_norm(x)
        
        # Split into heads: [B, L, dim] → list of [B, L, head_dim]
        head_inputs = normed.split(self.head_dim, dim=-1)
        
        # Unpack previous state
        h_list, gm_list, cm_list = self._unpack_state(prev_hidden)
        
        # Process each head
        head_outputs = []
        new_h_list, new_gm_list, new_cm_list = [], [], []
        
        for i, (head, h_in) in enumerate(zip(self.gru_heads, head_inputs)):
            # Process sequence step by step
            h_prev = h_list[i]
            gm = gm_list[i]
            cm = cm_list[i]
            
            step_outputs = []
            for t in range(L):
                out_t, h_prev, gm, cm = head(
                    h_in[:, t], h_prev, gm, cm, temporal_phase
                )
                step_outputs.append(out_t)
            
            head_out = torch.stack(step_outputs, dim=1)  # [B, L, head_dim]
            head_outputs.append(head_out)
            new_h_list.append(h_prev)
            new_gm_list.append(gm)
            new_cm_list.append(cm)
        
        # Concatenate heads
        gru_out = torch.cat(head_outputs, dim=-1)  # [B, L, dim]
        
        # Residual + dropout
        x = self.dropout(gru_out) * self.res_scale + x
        
        # SwiGLU FFN
        ff_in = self.ff_norm(x)
        gate = F.silu(self.ff_gate(ff_in))
        up = self.ff_up(ff_in)
        x = self.ff_down(gate * up) * self.res_scale + x
        
        # Track spike sparsity
        if self.training:
            with torch.no_grad():
                self._spike_count += gru_out.abs().sum()
                self._total_count += gru_out.numel()
        
        # Pack state for return
        next_hidden = self._pack_state(new_h_list, new_gm_list, new_cm_list)
        
        return x, next_hidden


# ═══════════════════════════════════════════════════════
# 5. SpikingSynapsePool — Пул синапсов для Mode 2
# ═══════════════════════════════════════════════════════

class SpikingSynapsePool(nn.Module):
    """
    Пул спайковых синапсов, специализированных по типу задачи.
    
    Каждый синапс — SpikingMinGRUBlock, обученный на своём типе задач:
      0: action  (выполнение команд)
      1: search  (поиск в памяти)
      2: social  (коммуникация)
      3: code    (код и скрипты)
      4: generic (общее)
    
    Маршрутизация через learned gating.
    """
    
    SYNAPSE_TYPES = ["action", "search", "social", "code", "generic"]
    
    def __init__(self, dim: int = 256, n_synapses: int = 5, num_heads: int = 4,
                 beta: float = 0.9):
        super().__init__()
        self.dim = dim
        self.n_synapses = n_synapses
        
        # Synapses
        self.synapses = nn.ModuleList([
            SpikingMinGRUBlock(dim, num_heads=num_heads, beta=beta)
            for _ in range(n_synapses)
        ])
        
        # Router: input → which synapse(s) to use
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, n_synapses),
        )
        
        # Output merger
        self.merge = nn.Linear(dim * n_synapses, dim, bias=False)
    
    def route(self, x: torch.Tensor) -> torch.Tensor:
        """
        Определить веса маршрутизации.
        
        Args:
            x: [B, L, dim]
        Returns:
            weights: [B, n_synapses] — softmax weights
        """
        # Pool over sequence dimension
        pooled = x.mean(dim=1)  # [B, dim]
        logits = self.router(pooled)  # [B, n_synapses]
        return F.softmax(logits, dim=-1)
    
    def forward(self, x: torch.Tensor,
                prev_states: Optional[List[torch.Tensor]] = None,
                temporal_phase: Optional[torch.Tensor] = None,
                task_type: Optional[str] = None,
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: [B, L, dim]
            prev_states: list of packed states per synapse
            temporal_phase: circadian modulation
            task_type: optional forced routing ("action", "search", etc.)
        
        Returns:
            output: [B, L, dim]
            next_states: list of packed states per synapse
        """
        B, L, D = x.shape
        
        if prev_states is None:
            prev_states = [None] * self.n_synapses
        
        # Route
        if task_type and task_type in self.SYNAPSE_TYPES:
            # Hard routing: only use the specified synapse
            idx = self.SYNAPSE_TYPES.index(task_type)
            weights = torch.zeros(B, self.n_synapses, device=x.device)
            weights[:, idx] = 1.0
        else:
            weights = self.route(x)  # [B, n_synapses]
        
        # Forward through top-2 synapses (sparse expert routing)
        topk_vals, topk_idx = weights.topk(2, dim=-1)  # [B, 2]
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        
        output = torch.zeros(B, L, D, device=x.device)
        next_states = [None] * self.n_synapses
        
        for k in range(2):
            for b in range(B):
                synapse_idx = topk_idx[b, k].item()
                synapse = self.synapses[synapse_idx]
                
                out_b, state_b = synapse(
                    x[b:b+1], prev_states[synapse_idx], temporal_phase
                )
                output[b:b+1] += topk_weights[b, k] * out_b
                next_states[synapse_idx] = state_b
        
        return output, next_states
    
    def sparsity_report(self) -> dict:
        """Отчёт о спарсности спайков по синапсам."""
        report = {}
        for i, (name, syn) in enumerate(zip(self.SYNAPSE_TYPES, self.synapses)):
            report[name] = {
                "sparsity": f"{syn.sparsity:.1%}",
                "params": sum(p.numel() for p in syn.parameters()),
            }
        return report
