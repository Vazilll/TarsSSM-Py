"""
═══════════════════════════════════════════════════════════════
  Neuromodulator — Глобальная нейромодуляция (§ нейронаука)
═══════════════════════════════════════════════════════════════

Моделирует 4 нейромедиаторных системы мозга:

  Дофамин (DA):    reward prediction → MoLE routing sharpness
  Норадреналин (NA): arousal/novelty → thinking depth (p-threshold)
  Ацетилхолин (ACh): learning mode → self-learn LR modulation
  Серотонин (5-HT): patience → inhibition of premature actions

Каждый нейромедиатор вычисляется из глобального состояния мозга h
и модулирует ВСЕ блоки одновременно (тоническая модуляция).

Математика:
  α_effective = α₀ · (1 + κ · ACh)       # learning rate boost
  p_threshold = p₀ · (1 - λ · NA)         # deeper thinking
  expert_temp = T₀ / (1 + μ · DA)         # sharper routing
  max_depth   = D₀ + floor(γ · 5HT)       # patience bonus
"""

import torch
import torch.nn as nn
import logging
from typing import Dict


class Neuromodulator(nn.Module):
    """
    Глобальная нейромодуляторная система.
    
    Аналог: ствол мозга → VTA (дофамин), LC (норадреналин),
    NBM (ацетилхолин), ядра  шва (серотонин).
    
    Каждый модулятор — скаляр ∈ (0, 1), вычисленный из h_global.
    Используется для глобальной модуляции всех компонентов.
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.d_model = d_model
        self.logger = logging.getLogger("Tars.Neuromod")
        
        # ═══ 4 модулятора из разных проекций ═══
        # Каждый имеет свой MLP для нелинейного маппинга
        
        # Дофамин: reward prediction → sharper MoLE routing
        self.dopamine_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # Норадреналин: arousal → deeper thinking (lower p-threshold)
        self.noradrenaline_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # Ацетилхолин: learning signal → boost LR for self-learning
        self.acetylcholine_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # Серотонин: patience → inhibition (more steps before action)
        self.serotonin_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # ═══ EMA baseline для дофамина (reward prediction) ═══
        # DA = actual_reward - expected_reward (prediction error)
        self.register_buffer('reward_baseline', torch.tensor(0.5))
        self.baseline_momentum = 0.95
        
        # Последние значения (для логирования и внешнего доступа)
        self._last_state = {"DA": 0.5, "NA": 0.5, "ACh": 0.5, "5HT": 0.5}
    
    def forward(self, h_global: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Вычисляет уровни всех 4 нейромедиаторов.
        
        Args:
            h_global: [B, d_model] — глобальное состояние мозга
                      (обычно = mean(last_hidden_states) или output of GW)
        
        Returns:
            dict: DA, NA, ACh, 5HT — каждый [B, 1] ∈ (0, 1)
        """
        DA = self.dopamine_net(h_global)      # [B, 1]
        NA = self.noradrenaline_net(h_global)  # [B, 1]
        ACh = self.acetylcholine_net(h_global) # [B, 1]
        _5HT = self.serotonin_net(h_global)    # [B, 1]
        
        # Сохраняем для логирования
        self._last_state = {
            "DA": DA.mean().item(),
            "NA": NA.mean().item(),
            "ACh": ACh.mean().item(),
            "5HT": _5HT.mean().item(),
        }
        
        return {"DA": DA, "NA": NA, "ACh": ACh, "5HT": _5HT}
    
    def modulate_routing_temperature(self, base_temp: float, DA: torch.Tensor) -> float:
        """
        Дофамин → MoLE routing sharpness.
        
        Высокий DA (positive reward) → lower temperature → sharper choice.
        Низкий DA (confusion) → higher temperature → exploration.
        
        T_eff = T₀ / (1 + μ · DA),  μ = 1.0
        """
        mu = 1.0
        return base_temp / (1.0 + mu * DA.mean().item())
    
    def modulate_p_threshold(self, base_threshold: float, NA: torch.Tensor) -> float:
        """
        Норадреналин → thinking depth.
        
        Высокий NA (arousal, novelty) → lower threshold → deeper thinking.
        Низкий NA (familiar input) → higher threshold → quick exit.
        
        p_eff = p₀ · (1 - λ · NA),  λ = 0.3
        """
        lam = 0.3
        return base_threshold * (1.0 - lam * NA.mean().item())
    
    def modulate_learning_rate(self, base_lr: float, ACh: torch.Tensor) -> float:
        """
        Ацетилхолин → learning rate.
        
        Высокий ACh (attention, novelty) → boost LR для быстрого обучения.
        Низкий ACh (routine) → conservative LR.
        
        lr_eff = lr₀ · (1 + κ · ACh),  κ = 2.0
        """
        kappa = 2.0
        return base_lr * (1.0 + kappa * ACh.mean().item())
    
    def modulate_max_depth(self, base_depth: int, _5HT: torch.Tensor) -> int:
        """
        Серотонин → patience (max thinking steps).
        
        Высокий 5-HT → больше терпения → больше шагов мышления.
        Низкий 5-HT → импульсивный быстрый ответ.
        
        D_eff = D₀ + floor(γ · 5HT),  γ = 3
        """
        gamma = 3
        return base_depth + int(gamma * _5HT.mean().item())
    
    def update_reward_baseline(self, actual_reward: float):
        """
        Дофаминовый baseline: EMA ожидаемой награды.
        DA = reward - baseline → если reward > ожидания, DA↑
        """
        with torch.no_grad():
            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline 
                + (1 - self.baseline_momentum) * actual_reward
            )
    
    def get_state_str(self) -> str:
        """Красивое представление для логирования."""
        s = self._last_state
        return (f"DA={s['DA']:.2f} NA={s['NA']:.2f} "
                f"ACh={s['ACh']:.2f} 5HT={s['5HT']:.2f}")



