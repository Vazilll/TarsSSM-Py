"""
═══════════════════════════════════════════════════════════════
  IDME Matrix Pool — Incremental Dynamic Matrix Expansion
═══════════════════════════════════════════════════════════════

Пул матриц = «кора мозга» ТАРС.

Аналогия: думать глубже = рекрутировать НОВЫЕ области коры,
а не прогонять те же нейроны повторно.

- 48 базовых матриц (расширяемо до ∞ через LazyExpand)
- Каждая матрица специализирована (domain embeddings)
- used_mask: матрица использована → исключена (анти-цикл)
- Recirculation: эффективные матрицы → повышенный приоритет
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Tuple


class MiniBlock(nn.Module):
    """
    Одна матрица из пула: mini Mamba-like block.
    
    Легче чем полный Mamba2Block — только linear + gate + norm.
    Стоимость: O(d²) за один проход.
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        self.transform = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()
        
        # Инициализация: identity + шум (начинаем с residual)
        nn.init.eye_(self.transform.weight)
        self.transform.weight.data += 0.01 * torch.randn_like(self.transform.weight)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, d_model] → [B, d_model]"""
        gate = torch.sigmoid(self.gate(h))
        transform = self.act(self.transform(h))
        return self.norm(h * (1 - gate) + transform * gate)


class MatrixPool(nn.Module):
    """
    IDME: пул матриц с динамическим расширением.
    
    Начальный размер: pool_size (48).
    При необходимости: LazyExpand добавляет новые матрицы.
    Каждый запрос: used_mask обнуляется.
    """
    
    def __init__(self, d_model: int = 768, pool_size: int = 48):
        super().__init__()
        self.d_model = d_model
        self.pool_size = pool_size
        self.logger = logging.getLogger("Tars.MatrixPool")
        
        # Пул матриц
        self.matrices = nn.ModuleList([
            MiniBlock(d_model) for _ in range(pool_size)
        ])
        
        # Обучаемые domain embeddings для каждой матрицы
        self.domain_embeddings = nn.Parameter(
            torch.randn(pool_size, d_model) * 0.02
        )
        # Ортогонализируем для максимального разделения доменов
        nn.init.orthogonal_(self.domain_embeddings)
        
        # Маска использования (сбрасывается между запросами)
        self.register_buffer('used_mask', torch.zeros(pool_size, dtype=torch.bool))
        
        # Счётчик эффективности (для Recirculation)
        self.register_buffer('efficiency', torch.zeros(pool_size))
        
        # Динамическое расширение: лениво создаваемые матрицы
        self._expanded: List[MiniBlock] = []
        self._expanded_embeddings: List[torch.Tensor] = []
    
    def reset(self):
        """Сброс между запросами."""
        self.used_mask.zero_()
        self._expanded.clear()
        self._expanded_embeddings.clear()
    
    def select(self, h: torch.Tensor, k: int = 4) -> Tuple[List[nn.Module], List[int]]:
        """
        Выбрать k НОВЫХ матриц, наиболее релевантных состоянию h.
        
        Анти-повтор: used_mask исключает уже использованные.
        Линейная сложность: O(pool_size × d_model)
        
        Args:
            h: [B, d_model] или [d_model] — текущее состояние мысли
            k: сколько матриц рекрутировать
        
        Returns:
            (selected_matrices, selected_indices)
        """
        if h.dim() > 1:
            h = h.mean(0)  # Если batch, берём среднее
        
        # Cosine similarity
        h_norm = F.normalize(h.unsqueeze(0), dim=-1)
        emb_norm = F.normalize(self.domain_embeddings, dim=-1)
        scores = (h_norm @ emb_norm.T).squeeze(0)  # [pool_size]
        
        # Бонус за эффективность (Recirculation)
        scores = scores + 0.1 * torch.tanh(self.efficiency)
        
        # Маскируем уже использованные
        scores[self.used_mask] = -float('inf')
        
        # Проверяем, есть ли достаточно свободных матриц
        available = (~self.used_mask).sum().item()
        
        if available < k:
            # Нужна динамическая экспансия!
            needed = k - available
            self._lazy_expand(needed, h)
            # Используем доступные + новые
            k_base = available
        else:
            k_base = k
        
        selected_matrices = []
        selected_indices = []
        
        # Выбираем из основного пула
        if k_base > 0:
            _, indices = scores.topk(min(k_base, len(scores)))
            for idx in indices.tolist():
                self.used_mask[idx] = True
                selected_matrices.append(self.matrices[idx])
                selected_indices.append(idx)
        
        # Добавляем из расширенного пула
        for exp_block in self._expanded[-max(0, k - k_base):]:
            selected_matrices.append(exp_block)
            selected_indices.append(-1)  # -1 = expanded (без индекса)
        
        self.logger.debug(
            f"MatrixPool: selected {len(selected_matrices)} matrices "
            f"(available: {available}, expanded: {len(self._expanded)})"
        )
        
        return selected_matrices, selected_indices
    
    def _lazy_expand(self, n: int, h: torch.Tensor):
        """
        Динамическое расширение пула (бесконечная глубина).
        
        Создаёт n новых MiniBlock-ов на лету.
        Каждый инициализируется ближе к текущему состоянию h.
        """
        device = h.device
        self.logger.info(f"MatrixPool: IDME расширение +{n} матриц (итого: {self.pool_size + len(self._expanded) + n})")
        
        for _ in range(n):
            block = MiniBlock(self.d_model).to(device)
            self._expanded.append(block)
    
    def recirculate(self, idx: int, delta_p: float):
        """
        Matrix Recirculation: если матрица дала большой вклад
        в сходимость, повышается её приоритет.
        """
        if 0 <= idx < self.pool_size:
            self.efficiency[idx] += delta_p
    
    @property
    def total_available(self) -> int:
        """Всего доступных матриц, включая расширение."""
        return (~self.used_mask).sum().item() + len(self._expanded)
    
    def apply_sequence(self, h: torch.Tensor, matrices: List[nn.Module]) -> torch.Tensor:
        """
        Применить последовательность выбранных матриц к состоянию.
        Каждая матрица = O(d²), итого O(K × d²).
        """
        for matrix in matrices:
            h = matrix(h)
        return h
