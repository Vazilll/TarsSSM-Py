"""
═══════════════════════════════════════════════════════════════
  Neural Oscillations — θ-γ Phase Coding for Memory Binding
═══════════════════════════════════════════════════════════════

Нейронаука:
  θ-ритм (4-8 Hz) гиппокампа координирует memory encoding.
  γ-ритм (30-100 Hz) коры — binding problem (связывание признаков).
  θ-γ coupling: γ-пакеты вложены в θ-фазу → temporal code.

Механизм:
  1. θ-генератор создаёт медленную модуляцию по глубине мышления
  2. γ-генератор группирует информацию внутри θ-цикла
  3. Phase-dependent gating: запись в память только на θ↑ (preferred phase)

Математика:
  phase_θ(t) = 2π · (t mod T_θ) / T_θ
  phase_γ(t) = 2π · (t mod T_γ) / T_γ
  gate(t) = σ(α · cos(phase_θ(t)) + β)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class OscillatoryBinding(nn.Module):
    """
    θ-γ Phase Coding Layer.
    
    Модулирует поток данных осциллирующим сигналом:
      - θ-фаза определяет, когда ЗАПИСЫВАТЬ в память (encoding window)
      - γ-фаза определяет, когда ЧИТАТЬ (retrieval window)
    
    Параметры T_θ, T_γ, α, β — обучаемые.
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        
        # Обучаемые периоды осцилляций
        # theta: ~8 шагов (медленный ритм), gamma: ~3 шага (быстрый)
        self.log_theta_period = nn.Parameter(torch.tensor(math.log(8.0)))
        self.log_gamma_period = nn.Parameter(torch.tensor(math.log(3.0)))
        
        # Phase-dependent gate parameters
        self.theta_proj = nn.Linear(2, d_model)  # [cos(θ), sin(θ)] → gate
        self.gamma_proj = nn.Linear(2, d_model)  # [cos(γ), sin(γ)] → modulation
        
        # Mixing strengths
        self.theta_mix = nn.Parameter(torch.tensor(0.1))
        self.gamma_mix = nn.Parameter(torch.tensor(0.05))
    
    def forward(self, x: torch.Tensor, step: int) -> Tuple[torch.Tensor, dict]:
        """
        x: [B, L, d_model]
        step: текущий шаг мышления (0, 1, 2, ...)
        
        Returns:
            x_modulated: [B, L, d_model]
            phase_info: dict с theta_phase, gamma_phase, encoding_gate
        """
        device = x.device
        
        # Вычисляем фазы
        theta_T = torch.exp(self.log_theta_period)  # learned period
        gamma_T = torch.exp(self.log_gamma_period)
        
        theta_phase = 2 * math.pi * (step % theta_T.item()) / theta_T.item()
        gamma_phase = 2 * math.pi * (step % gamma_T.item()) / gamma_T.item()
        
        # Phase vectors
        theta_vec = torch.tensor(
            [math.cos(theta_phase), math.sin(theta_phase)],
            device=device, dtype=x.dtype
        )
        gamma_vec = torch.tensor(
            [math.cos(gamma_phase), math.sin(gamma_phase)],
            device=device, dtype=x.dtype
        )
        
        # Phase-dependent gates (broadcast per-dimension)
        theta_gate = torch.sigmoid(self.theta_proj(theta_vec))  # [d_model]
        gamma_mod = torch.tanh(self.gamma_proj(gamma_vec))      # [d_model]
        
        # Modulate: θ controls overall gating, γ adds fine-grained modulation
        x_mod = x * (1.0 + self.theta_mix * theta_gate.unsqueeze(0).unsqueeze(0))
        x_mod = x_mod + self.gamma_mix * gamma_mod.unsqueeze(0).unsqueeze(0)
        
        # Encoding gate: high when θ ascending → preferred write phase
        encoding_gate = float(math.cos(theta_phase) > 0)
        
        phase_info = {
            "theta_phase": theta_phase / (2 * math.pi),
            "gamma_phase": gamma_phase / (2 * math.pi),
            "encoding_gate": encoding_gate,
            "theta_T": theta_T.item(),
            "gamma_T": gamma_T.item(),
        }
        
        return x_mod, phase_info
