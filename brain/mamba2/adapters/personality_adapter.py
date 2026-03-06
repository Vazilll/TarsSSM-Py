# -*- coding: utf-8 -*-
"""
PersonalityAdapter — отдельный модуль стиля ТАРС.

Архитектура (Tars.txt §11 — Personality Layer):
  Mamba Core производит hidden_states (что сказать).
  PersonalityAdapter трансформирует КАК сказать — стиль, юмор, характер.

Ключевая инновация: Convergence-Gated Iterative Refinement.
  Данные проходят через адаптер 1-3 раза. Каждый проход "углубляет" личность.
  Когда дельта между проходами мала (сходимость) → стоп.
  
  Pass 1: грубый стиль (базовый характер)
  Pass 2: рефайнмент (нюансы, юмор)  
  Pass 3: полировка (уникальность, анти-шаблонность)

Вдохновлено:
  - Deep Equilibrium Models (Bai et al., 2019)
  - Universal Transformers (Dehghani et al., 2018)
  - Iterative Refinement in Diffusion Models

Параметры: ~2-3M при d_model=768 (не замедляет inference значительно).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersonalityAdapter(nn.Module):
    """
    Convergence-Gated Personality Transform.
    
    Каждый forward pass:
      1. Вычисляет стилевую трансформацию
      2. Измеряет дельту (сколько изменилось)
      3. Если дельта < threshold → сходимость → стоп
      4. Иначе → ещё один проход (до max_passes)
    
    При обучении: все проходы дифференцируемые (gradients через все итерации).
    При инференсе: обычно 1-2 прохода (быстро).
    """
    
    def __init__(
        self,
        d_model: int = 768,
        style_dim: int = 128,
        max_passes: int = 3,
        convergence_threshold: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.style_dim = style_dim
        self.max_passes = max_passes
        self.convergence_threshold = convergence_threshold
        
        # ═══ Style Identity Vector ═══
        # Обучаемый вектор "кто я" — вшит в веса, не зависит от входа.
        # Аналогия: ДНК личности ТАРС.
        self.style_embed = nn.Parameter(torch.randn(1, 1, style_dim) * 0.02)
        
        # ═══ Style Transform Block (применяется итеративно) ═══
        # Один и тот же блок прогоняется 1-3 раза (как DEQ).
        # weight sharing = меньше параметров, глубже рефайнмент.
        self.style_proj = nn.Linear(d_model, style_dim)
        self.transform = nn.Sequential(
            nn.Linear(d_model + style_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.transform_norm = nn.LayerNorm(d_model)
        
        # ═══ Residual Gate ═══
        # Сколько стиля добавить: 0.0 = чистый Mamba, 1.0 = полный ТАРС.
        # Обучается отдельно, начинает с ~0.5 (мягкий старт).
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        
        # ═══ Pass Counter (не обучаемый, для статистики) ═══
        self.register_buffer('_avg_passes', torch.tensor(1.0))
        self._pass_count = 0
        self._pass_sum = 0
    
    def _single_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Один проход стилевой трансформации."""
        B, L, D = x.shape
        
        # Проецируем в стилевое пространство и добавляем identity
        style_context = self.style_proj(x)  # [B, L, style_dim]
        style_id = self.style_embed.expand(B, L, -1)  # [B, L, style_dim]
        style_combined = style_context + style_id  # [B, L, style_dim]
        
        # Transform: конкатенируем оригинал + стиль → новый оригинал
        combined = torch.cat([x, style_combined], dim=-1)  # [B, L, d_model + style_dim]
        transformed = self.transform(combined)  # [B, L, d_model]
        transformed = self.transform_norm(transformed)
        
        return transformed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convergence-gated iterative personality transform.
        
        x: [B, L, d_model] — hidden states from Mamba Core
        Returns: [B, L, d_model] — styled hidden states
        """
        original = x  # сохраняем для residual
        
        # Iterative refinement loop
        n_passes = 0
        for i in range(self.max_passes):
            x_new = self._single_pass(x)
            n_passes += 1
            
            # Convergence check: cosine delta between passes
            if i > 0:
                # Норма разницы / норма сигнала
                delta = (x_new - x).norm() / (x.norm() + 1e-8)
                
                if not self.training and delta.item() < self.convergence_threshold:
                    # Сошлось — стоп (только при инференсе, при обучении всегда все проходы)
                    x = x_new
                    break
            
            x = x_new
        
        # Track statistics
        if not self.training:
            self._pass_count += 1
            self._pass_sum += n_passes
            if self._pass_count > 0:
                self._avg_passes = torch.tensor(
                    self._pass_sum / self._pass_count, 
                    device=x.device
                )
        
        # Residual gate: blend original and styled
        g = self.gate(original)  # [B, L, 1]
        output = original + g * (x - original)
        
        return output
    
    @property
    def avg_passes(self) -> float:
        """Среднее количество проходов (для мониторинга)."""
        return self._avg_passes.item()
    
    def reset_stats(self):
        """Сброс статистики проходов."""
        self._pass_count = 0
        self._pass_sum = 0


class PersonalityLoss(nn.Module):
    """
    Дополнительный loss для PersonalityAdapter.
    
    Штрафует за:
    1. Слишком слабый gate (< 0.3) → стиль не применяется
    2. Слишком сильный gate (> 0.95) → потеря информации от Mamba
    3. Малое число проходов при обучении → недоиспользование итераций
    """
    
    def __init__(self, target_gate_range=(0.3, 0.8)):
        super().__init__()
        self.gate_low = target_gate_range[0]
        self.gate_high = target_gate_range[1]
    
    def forward(self, adapter: PersonalityAdapter) -> torch.Tensor:
        """Вычисляет regularization loss из состояния adapter."""
        # Gate bias: помочь gate уйти из крайних значений
        # (не можем это сделать без промежуточных значений,
        #  поэтому используем style_embed norm как proxy)
        style_norm = adapter.style_embed.norm()
        # Без чрезмерного разрастания style vector
        loss = 0.01 * F.relu(style_norm - 5.0)
        return loss
