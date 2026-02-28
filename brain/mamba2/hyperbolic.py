"""
═══════════════════════════════════════════════════════════════
  Hyperbolic Geometry — Poincaré Ball for Hierarchical Knowledge
═══════════════════════════════════════════════════════════════

Математика:
  Гиперболическое пространство имеет экспоненциальный рост объёма
  с радиусом (vs линейный в Евклидовом). Идеально для деревьев:
  
  Теорема (Sarkar, 2011): любое дерево с n вершинами вложимо в
  2D гиперболический диск с distortion 1+ε.

  d_P(u, v) = arcosh(1 + 2·||u-v||² / ((1-||u||²)(1-||v||²)))
  
Где u, v ∈ B^n_c = {x ∈ R^n : c·||x||² < 1} (Poincaré ball).

Применение:
  - LEANN: hierarchical similarity вместо cosine
  - MoIRA: tool routing с учётом категорий tools
  - Memory: hierarchical memory organization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


def poincare_distance(u: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Расстояние в модели Пуанкаре гиперболического пространства.
    
    d_P(u, v) = (1/√c) · arcosh(1 + 2c · ||u-v||² / ((1-c·||u||²)(1-c·||v||²)))
    
    Args:
        u, v: [B, D] — точки в Poincaré ball (||x|| < 1/√c)
        c: кривизна (c=1 → unit ball)
    
    Returns:
        [B] — гиперболическое расстояние
    """
    diff_norm_sq = (u - v).pow(2).sum(-1)
    u_norm_sq = u.pow(2).sum(-1).clamp(max=1.0/c - eps)
    v_norm_sq = v.pow(2).sum(-1).clamp(max=1.0/c - eps)
    
    numerator = 2.0 * c * diff_norm_sq
    denominator = (1.0 - c * u_norm_sq) * (1.0 - c * v_norm_sq) + eps
    
    arg = 1.0 + numerator / denominator
    return (1.0 / math.sqrt(c)) * torch.acosh(arg.clamp(min=1.0 + eps))


def poincare_exp_map(v: torch.Tensor, p: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Exponential map: из касательного пространства T_p → Poincaré ball.
    
    Переводит евклидов вектор v (в касательной плоскости к p)
    в точку на гиперболическом многообразии.
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    p_norm_sq = p.pow(2).sum(-1, keepdim=True)
    
    lambda_p = 2.0 / (1.0 - c * p_norm_sq + eps)
    
    tanh_arg = math.sqrt(c) * lambda_p * v_norm / 2.0
    direction = v / v_norm
    
    return mobius_add(p, torch.tanh(tanh_arg) * direction / math.sqrt(c), c=c)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """
    Möbius addition в Poincaré ball — гиперболический аналог сложения.
    """
    x_sq = x.pow(2).sum(-1, keepdim=True)
    y_sq = y.pow(2).sum(-1, keepdim=True)
    xy_dot = (x * y).sum(-1, keepdim=True)
    
    num = (1.0 + 2.0 * c * xy_dot + c * y_sq) * x + (1.0 - c * x_sq) * y
    denom = 1.0 + 2.0 * c * xy_dot + c * c * x_sq * y_sq + eps
    
    return num / denom


class HyperbolicLinear(nn.Module):
    """
    Linear layer в Poincaré ball.
    
    Переводит input в касательное пространство (log map),
    применяет обычный Linear, затем обратно (exp map).
    """
    
    def __init__(self, in_features: int, out_features: int, c: float = 1.0):
        super().__init__()
        self.c = c
        self.linear = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.zeros(out_features) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_features] ∈ Poincaré ball
        Returns: [B, out_features] ∈ Poincaré ball
        """
        # Евклидово преобразование
        h = self.linear(x)
        # Проецируем обратно в ball (clamp norm)
        h_norm = h.norm(dim=-1, keepdim=True)
        max_norm = (1.0 / math.sqrt(self.c)) - 1e-3
        h = h * (max_norm / h_norm.clamp(min=1e-5)).clamp(max=1.0)
        return h


class HyperbolicSimilarity(nn.Module):
    """
    Замена cosine similarity на Poincaré distance.
    
    Для LEANN и MoIRA: вместо cos(u,v) используем
    exp(-d_P(u,v)) ∈ (0, 1] — экспоненциально убывает с расстоянием.
    """
    
    def __init__(self, c: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.c = c
        self.scale = nn.Parameter(torch.tensor(scale))
    
    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        u: [B, D], v: [B, D] or [N, D]
        Returns: similarity ∈ (0, 1]
        """
        dist = poincare_distance(u, v, c=self.c)
        return torch.exp(-self.scale * dist)


def project_to_poincare(x: torch.Tensor, c: float = 1.0, eps: float = 1e-3) -> torch.Tensor:
    """
    Проецирует евклидов вектор в Poincaré ball.
    Clamp ||x|| < 1/√c - eps для числовой стабильности.
    """
    max_norm = (1.0 / math.sqrt(c)) - eps
    x_norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / x_norm.clamp(min=1e-5)).clamp(max=1.0)
