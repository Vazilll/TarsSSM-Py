"""
═══════════════════════════════════════════════════════════════
  Novelty Gate + Hankel SVD — Анти-цикл защита
═══════════════════════════════════════════════════════════════

3 уровня защиты от зацикливания мыслей:
  1. used_mask в MatrixPool (hard — никогда не повторяет матрицу)
  2. Hankel SVD (soft — детекция повторяющихся паттернов)
  3. NoveltyGate (learned — пропуск бесполезных шагов)
"""

import torch
import torch.nn as nn
import logging
from typing import Optional


class NoveltyGate(nn.Module):
    """
    Обучаемый гейт новизны.
    
    Если новый шаг мышления несёт мало информации → пропустить.
    Аналогия: мозг не тратит энергию на бессмысленные мысли.
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        self.threshold = 0.2  # Порог полезности
    
    def forward(self, h_old: torch.Tensor, h_new: torch.Tensor) -> torch.Tensor:
        """
        Решает, принять ли новое состояние.
        
        Returns: обновлённое h (смесь old/new по novelty score)
        """
        delta = h_new - h_old
        novelty = self.gate(torch.cat([h_old, delta], dim=-1))  # [B, 1]
        
        # Пропорциональное обновление
        return h_old + novelty * delta, novelty.squeeze(-1)


class HankelDetector:
    """
    Hankel SVD: детекция зацикливания через ранг матрицы Ганкеля.
    
    Матрица Ганкеля из истории h(t):
        H = [[h(0), h(1), h(2)],
             [h(1), h(2), h(3)],
             [h(2), h(3), h(4)]]
    
    Если σ₂/σ₁ < 0.1 → ранговый коллапс → мысли зациклились.
    """
    
    def __init__(self, window: int = 6, collapse_threshold: float = 0.1):
        self.window = window
        self.collapse_threshold = collapse_threshold
        self.history = []
        self.logger = logging.getLogger("Tars.Hankel")
        self.collapse_count = 0
    
    def reset(self):
        self.history.clear()
        self.collapse_count = 0
    
    def observe(self, h: torch.Tensor) -> dict:
        """
        Добавить наблюдение и проверить на зацикливание.
        
        Returns:
            dict: novelty (float), collapsed (bool), collapse_count (int)
        """
        with torch.no_grad():
            # Сжимаем до вектора
            h_flat = h.float().detach().flatten()[:128]  # Берём первые 128 для скорости
            self.history.append(h_flat)
        
        result = {"novelty": 1.0, "collapsed": False, "collapse_count": 0}
        
        if len(self.history) < self.window:
            return result
        
        # Строим матрицу Ганкеля из последних W наблюдений
        recent = self.history[-self.window:]
        half = self.window // 2
        
        # H[i,j] = h[i+j], размер half × half
        rows = []
        for i in range(half):
            row = torch.stack([recent[i + j] for j in range(half)])
            rows.append(row)
        
        H = torch.stack(rows)  # [half, half, dim]
        H = H.reshape(half, -1)  # [half, half*dim]
        
        # SVD
        try:
            sigma = torch.linalg.svdvals(H)
            if sigma[0] > 1e-8:
                novelty = (sigma[1] / sigma[0]).item() if len(sigma) > 1 else 1.0
            else:
                novelty = 0.0
            
            result["novelty"] = novelty
            
            if novelty < self.collapse_threshold:
                self.collapse_count += 1
                result["collapsed"] = True
                result["collapse_count"] = self.collapse_count
                self.logger.warning(
                    f"Hankel: Ранговый коллапс! novelty={novelty:.4f} "
                    f"(count={self.collapse_count})"
                )
        except Exception:
            pass  # SVD может не сойтись на малых матрицах
        
        return result
    
    def inject_noise(self, h: torch.Tensor, scale: float = 0.05) -> torch.Tensor:
        """Стохастический перезапуск при зацикливании."""
        return h + scale * torch.randn_like(h)
