"""
═══════════════════════════════════════════════════════════════
  MoLE Router — 8 экспертов с sparse top-2 routing
═══════════════════════════════════════════════════════════════

Каждый эксперт = LoRA адаптер (rank=8, ~16K params).
TopicRouter определяет тему → активирует 2 из 8 экспертов.
Остальные 6 = нулевые вычисления (sparse).

LoRA math: ΔW = B @ A, где B:[d, r], A:[r, d], r=8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, List
from brain.mamba2.bitnet import UniversalLinear


class LoRAAdapter(nn.Module):
    """Один LoRA эксперт: ΔW = B·A (low-rank)."""
    
    def __init__(self, d_model: int = 768, rank: int = 8, alpha: float = 1.0,
                 quant_mode: str = "fp16"):
        super().__init__()
        self.rank = rank
        self.alpha = alpha / rank  # LoRA scaling
        
        self.A = UniversalLinear(d_model, rank, bias=False, mode=quant_mode)
        self.B = UniversalLinear(rank, d_model, bias=False, mode=quant_mode)
        
        # LoRA init: A = normal, B = zeros (начинаем с identity)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d] → ΔW·x: [B, d]"""
        return self.B(self.A(x)) * self.alpha


class TopicRouter(nn.Module):
    """
    Sparse top-k router для MoLE экспертов.
    
    8 экспертов, top-2 активны одновременно.
    Routing по cosine similarity между hidden state и expert embeddings.
    """
    
    EXPERT_NAMES = [
        "general",    # 0: Общая речь
        "analyzer",   # 1: Логический анализ  
        "critic",     # 2: Проверка на ошибки
        "creative",   # 3: Нестандартные решения
        "math",       # 4: Математика и формулы
        "code",       # 5: Программирование
        "memory",     # 6: Работа с воспоминаниями
        "action",     # 7: Планирование действий
    ]
    
    def __init__(self, d_model: int = 768, n_experts: int = 8, top_k: int = 2,
                 quant_mode: str = "fp16"):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.logger = logging.getLogger("Tars.MoLE")
        
        # Обучаемые эмбеддинги доменов
        self.expert_embeddings = nn.Parameter(torch.randn(n_experts, d_model) * 0.02)
        nn.init.orthogonal_(self.expert_embeddings)
        
        # Gate для балансировки нагрузки
        self.gate = UniversalLinear(d_model, n_experts, bias=True, mode=quant_mode)
    
    def route(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Определяет top-k экспертов для текущего состояния.
        
        Args:
            h: [B, d_model]
        Returns:
            indices: [B, top_k] — индексы экспертов
            weights: [B, top_k] — веса (нормализованные)
        """
        # Gate logits
        logits = self.gate(h)  # [B, n_experts]
        
        # Top-k
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        return indices, weights


class MoLELayer(nn.Module):
    """
    MoLE: Mixture of LoRA Experts.
    
    Применяет top-k LoRA адаптеров к входу, взвешивая по routing weights.
    """
    
    def __init__(self, d_model: int = 768, n_experts: int = 8, 
                 rank: int = 8, top_k: int = 2,
                 quant_mode: str = "fp16"):
        super().__init__()
        self.router = TopicRouter(d_model, n_experts, top_k, quant_mode=quant_mode)
        self.experts = nn.ModuleList([
            LoRAAdapter(d_model, rank, quant_mode=quant_mode) for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model] → x + sparse LoRA delta
        """
        # Route по среднему состоянию
        h_mean = x.mean(dim=1)  # [B, d_model]
        indices, weights = self.router.route(h_mean)  # [B, k], [B, k]
        
        # Применяем top-k экспертов
        batch_size = x.shape[0]
        delta = torch.zeros_like(x[:, 0])  # [B, d_model]
        
        for i in range(self.router.top_k):
            expert_idx = indices[:, i]  # [B]
            weight = weights[:, i].unsqueeze(-1)  # [B, 1]
            
            # Собираем вклад каждого эксперта
            for b in range(batch_size):
                eidx = expert_idx[b].item()
                expert_out = self.experts[eidx](h_mean[b:b+1])
                delta[b] += weight[b] * expert_out.squeeze(0)
        
        # Добавляем delta ко всем позициям (broadcast)
        return self.norm(x + delta.unsqueeze(1))
    
    def get_active_experts(self, x: torch.Tensor) -> List[str]:
        """Возвращает имена активных экспертов (для логирования)."""
        h_mean = x.mean(dim=1)
        indices, weights = self.router.route(h_mean)
        names = []
        for i in range(self.router.top_k):
            idx = indices[0, i].item()
            w = weights[0, i].item()
            names.append(f"{TopicRouter.EXPERT_NAMES[idx]}({w:.2f})")
        return names
