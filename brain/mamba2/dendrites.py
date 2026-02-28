"""
═══════════════════════════════════════════════════════════════
  Active Dendrites — Context-Modulated Computation (Numenta 2021-2025)
═══════════════════════════════════════════════════════════════

Нейронаука (Numenta, Thousand Brains Project):
  Дендриты нейрона — не пассивные провода.
  Каждая дендритная ветвь выполняет НЕЛИНЕЙНЫЕ вычисления
  (NMDA spikes), создавая ~10-50 «подъединиц» внутри нейрона.
  Один нейрон ≈ двухслойная сеть.

  Ключевое свойство: context modulation.
  Разные контексты активируют разные дендритные сегменты →
  один и тот же нейрон специализируется под разные задачи
  БЕЗ catastrophic forgetting.

Математика:
  y = W·x ⊙ σ(max_k(d_k · context))
  
  Где d_k — обучаемые дендритные весы для k-го сегмента,
  max_k — выбирает сегмент с максимальной активацией,
  σ — sigmoid gating.

Источник: Ahmad & Hawkins (2021) "How Can We Be So Dense?
  The Benefits of Using Highly Sparse Representations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DendriticSegment(nn.Module):
    """Один дендритный сегмент — линейная проекция контекста."""
    
    def __init__(self, context_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(context_dim) * 0.01)
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, context_dim] → [B] activation."""
        return (context * self.weight).sum(dim=-1)


class ActiveDendriticLayer(nn.Module):
    """
    Active Dendritic Layer (Numenta, 2021-2025).
    
    Каждый нейрон имеет n_segments дендритных сегментов.
    Контекст выбирает "победивший" сегмент через winner-take-all.
    Этот сегмент модулирует (gate) выход нейрона.
    
    Преимущества:
      - Continual learning без catastrophic forgetting
      - Context-dependent specialization  
      - Один слой ≈ мини mixture-of-experts на уровне нейронов
    
    Args:
        in_features: входная размерность
        out_features: выходная размерность
        context_dim: размерность контекстного вектора
        n_segments: количество дендритных сегментов на нейрон
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        context_dim: int = 768,
        n_segments: int = 7,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_segments = n_segments
        
        # Основные сомативые веса (body of neuron)
        self.weight = nn.Linear(in_features, out_features)
        
        # Дендритные сегменты: [n_segments, context_dim, out_features]
        # Каждый сегмент проецирует контекст → маску gate для out_features
        self.dendritic_weights = nn.Parameter(
            torch.randn(n_segments, context_dim) * 0.01
        )
        
        # Abolutely critical for Numenta: top-k sparsity over segments
        self.top_k = 1  # winner-take-all по дендритам
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, in_features]
        context: [B, context_dim] — контекст задачи/модальности
        
        Returns: [B, out_features]
        """
        # 1. Соматическое вычисление (обычный linear)
        somatic_output = self.weight(x)  # [B, out_features]
        
        if context is None:
            return somatic_output
        
        # 2. Дендритные активации: context → per-segment scores
        # [B, context_dim] × [n_segments, context_dim]^T → [B, n_segments]
        dendritic_activations = torch.matmul(context, self.dendritic_weights.T)
        
        # 3. Winner-take-all: выбираем лучший сегмент
        # (это key insight от Numenta: sparsity prevents forgetting)
        winning_activation, _ = dendritic_activations.max(dim=-1, keepdim=True)
        
        # 4. Gate: sigmoid of winning segment → modulation
        gate = torch.sigmoid(winning_activation)  # [B, 1]
        
        # 5. Модулируем соматический выход
        return somatic_output * gate


class DendriticBlock(nn.Module):
    """
    Блок из 2 Active Dendritic слоёв с residual connection.
    
    Заменяет стандартный MLP в трансформере/SSM, добавляя
    context awareness через дендритную модуляцию.
    """
    
    def __init__(self, d_model: int = 768, context_dim: int = 768, n_segments: int = 7):
        super().__init__()
        self.layer1 = ActiveDendriticLayer(
            d_model, d_model * 2, context_dim, n_segments
        )
        self.layer2 = ActiveDendriticLayer(
            d_model * 2, d_model, context_dim, n_segments
        )
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        self.mix = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, d_model] or [B, L, d_model]
        context: [B, context_dim]
        """
        reshape = False
        if x.dim() == 3:
            B, L, D = x.shape
            reshape = True
            x_flat = x.reshape(B * L, D)
            if context is not None:
                context = context.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
        else:
            x_flat = x
        
        h = self.act(self.layer1(x_flat, context))
        h = self.layer2(h, context)
        out = x_flat + self.mix * h
        
        if reshape:
            out = out.reshape(B, L, D)
        
        return self.norm(out)
