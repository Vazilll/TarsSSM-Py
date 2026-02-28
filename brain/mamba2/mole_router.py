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
    
    def route(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Определяет top-k экспертов для текущего состояния.
        
        Dropless Routing (Granite 4.0):
        - Capacity factor ensures no tokens are dropped
        - Jitter noise for exploration during training
        
        Args:
            h: [B, d_model]
        Returns:
            indices: [B, top_k] — индексы экспертов
            weights: [B, top_k] — веса (нормализованные)
            logits:  [B, n_experts] — сырые логиты (для aux loss)
        """
        # Gate logits
        logits = self.gate(h)  # [B, n_experts]
        
        # Jitter noise for exploration (training only)
        if self.training:
            noise = torch.empty_like(logits).uniform_(-0.01, 0.01)
            logits = logits + noise
        
        # Top-k selection
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        return indices, weights, logits


class MoLELayer(nn.Module):
    """
    MoLE: Mixture of LoRA Experts.
    
    Применяет top-k LoRA адаптеров к входу, взвешивая по routing weights.
    
    Auxiliary losses (Switch Transformer / GShard):
      - Load Balancing: штрафует неравномерную нагрузку на экспертов
      - Z-Loss: стабилизирует логиты роутера
    """
    
    def __init__(self, d_model: int = 768, n_experts: int = 8, 
                 rank: int = 8, top_k: int = 2,
                 quant_mode: str = "fp16",
                 balance_coeff: float = 0.01,
                 z_loss_coeff: float = 0.001):
        super().__init__()
        self.router = TopicRouter(d_model, n_experts, top_k, quant_mode=quant_mode)
        self.experts = nn.ModuleList([
            LoRAAdapter(d_model, rank, quant_mode=quant_mode) for _ in range(n_experts)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.balance_coeff = balance_coeff
        self.z_loss_coeff = z_loss_coeff
        self.n_experts = n_experts
    
    def _compute_aux_loss(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисляет вспомогательные потери для обучения MoLE.
        
        1. Load Balancing Loss (Switch Transformer, Fedus et al. 2021):
           L_balance = N × Σ_i (f_i × P_i)
           f_i = fraction of tokens routed to expert i
           P_i = mean router probability for expert i
           
        2. Router Z-Loss (ST-MoE, Zoph et al. 2022):
           L_z = (1/B) × Σ_b (log Σ_j exp(logits_bj))²
        """
        B = logits.shape[0]
        device = logits.device
        
        # ═══ 1. Load Balancing Loss ═══
        # f_i: доля сэмплов, где эксперт i в top-k
        # Считаем сколько раз каждый эксперт был выбран
        expert_counts = torch.zeros(self.n_experts, device=device)
        for i in range(self.router.top_k):
            counts = torch.bincount(indices[:, i], minlength=self.n_experts).float()
            expert_counts += counts
        f = expert_counts / (B * self.router.top_k)  # [n_experts]
        
        # P_i: средняя вероятность роутера для эксперта i
        router_probs = F.softmax(logits, dim=-1)  # [B, n_experts]
        P = router_probs.mean(dim=0)  # [n_experts]
        
        # L_balance = N × Σ(f_i × P_i)
        load_balance_loss = self.n_experts * (f * P).sum()
        
        # ═══ 2. Router Z-Loss ═══
        # Штрафует слишком большие логиты → стабильнее обучение
        log_z = torch.logsumexp(logits, dim=-1)  # [B]
        z_loss = (log_z ** 2).mean()
        
        # ═══ 3. Rényi Diversity Loss (α=2, collision entropy) ═══
        # Нейронаука: мозг максимизирует информационное разнообразие
        # нейронных ансамблей (sparse distributed representations).
        # Rényi H_α = log(Σ pᵢ^α) / (1-α) более чувствительна к
        # доминирующим экспертам, чем Shannon entropy.
        # α=2 → collision entropy: штрафует ситуацию, когда один
        # эксперт получает непропорционально большую долю нагрузки.
        renyi_alpha = 2.0
        expert_probs = (f + 1e-8)  # prevent log(0)
        expert_probs = expert_probs / expert_probs.sum()  # normalize to distribution
        renyi_entropy = torch.log(expert_probs.pow(renyi_alpha).sum()) / (1 - renyi_alpha)
        # Maximize entropy → minimize negative entropy
        renyi_loss = -renyi_entropy
        
        # Итого
        renyi_coeff = 0.01  # conservative coefficient
        aux_loss = (self.balance_coeff * load_balance_loss 
                   + self.z_loss_coeff * z_loss 
                   + renyi_coeff * renyi_loss)
        return aux_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, L, d_model] → (x + sparse LoRA delta, aux_loss)
        
        Returns:
            output: [B, L, d_model] — обогащённый тензор
            aux_loss: scalar — вспомогательная потеря для обучения роутера
        """
        # Route по среднему состоянию
        h_mean = x.mean(dim=1)  # [B, d_model]
        indices, weights, logits = self.router.route(h_mean)  # [B, k], [B, k], [B, N]
        
        # Auxiliary loss для обучения роутера
        aux_loss = self._compute_aux_loss(indices, weights, logits)
        
        # Запоминаем активных экспертов для логирования
        self._last_expert_names = []
        for i in range(self.router.top_k):
            idx = indices[0, i].item()
            w = weights[0, i].item()
            self._last_expert_names.append(f"{TopicRouter.EXPERT_NAMES[idx]}({w:.0%})")
        
        # Применяем top-k экспертов (batched, без Python loop по batch)
        batch_size = x.shape[0]
        delta = torch.zeros_like(h_mean)  # [B, d_model]
        
        for i in range(self.router.top_k):
            expert_idx = indices[:, i]      # [B]
            weight = weights[:, i:i+1]      # [B, 1]
            
            # Группируем по экспертам — один forward на всех samples с общим экспертом
            for eidx in range(len(self.experts)):
                mask = (expert_idx == eidx)  # [B] bool
                if mask.any():
                    expert_out = self.experts[eidx](h_mean[mask])  # [N_selected, d_model]
                    delta[mask] = delta[mask] + weight[mask] * expert_out
        
        # Добавляем delta ко всем позициям (broadcast)
        return self.norm(x + delta.unsqueeze(1)), aux_loss
    
    def get_active_experts(self, x: torch.Tensor) -> List[str]:
        """Возвращает имена активных экспертов (для логирования)."""
        h_mean = x.mean(dim=1)
        indices, weights, _ = self.router.route(h_mean)
        names = []
        for i in range(self.router.top_k):
            idx = indices[0, i].item()
            w = weights[0, i].item()
            names.append(f"{TopicRouter.EXPERT_NAMES[idx]}({w:.2f})")
        return names
