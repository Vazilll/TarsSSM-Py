"""
═══════════════════════════════════════════════════════════════
  QuantBridge — Адаптивный мост между int8 памятью и 1.58-bit MinGRU
═══════════════════════════════════════════════════════════════

Проблема:
  LEANN выдаёт int8[384] + scale  → нужен float32[1024] для MinGRU context_proj
  Прямая деквантизация теряет precision и не масштабирует размерность.

Решение:
  QuantBridge = Dequant → RMSNorm → Project(384→1024) → ActivationQuant

  Это даёт:
  1. Корректную деквантизацию int8 → float32 (с учётом per-vector scale)
  2. Нормализацию (RMSNorm стабилизирует распределение)
  3. Проекцию 384→1024 через UniversalLinear (1.58-bit квантизуемый)
  4. Опциональную реквантизацию активаций до int8 (снижает bandwidth)

Использование:
  bridge = QuantBridge(leann_dim=384, context_dim=1024)
  
  # Из LEANN
  int8_vecs, scales = leann.get_raw_embeddings(query)
  
  # В MinGRU
  context_vec = bridge(int8_vecs, scales)  # [B, 1024] float32
  logits = mingru(tokens, context_vec=context_vec)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union

# UniversalLinear для квантизуемой проекции
try:
    from brain.mamba2.bitnet import UniversalLinear as _Linear, RMSNorm, ActivationQuantizer
except ImportError:
    from torch.nn import Linear as _Linear
    RMSNorm = None
    ActivationQuantizer = None


class QuantBridge(nn.Module):
    """
    Мост между int8 LEANN памятью и 1.58-bit MinGRU.
    
    Поток данных:
      int8[384] + scale → dequant → RMSNorm → Project(384→1024) → context_vec
      
    Всего ~400K параметров, в 1.58-bit ≈ 100 KB.
    """
    
    def __init__(self, leann_dim: int = 384, context_dim: int = 1024, 
                 num_heads: int = 4, use_act_quant: bool = True):
        super().__init__()
        self.leann_dim = leann_dim
        self.context_dim = context_dim
        
        # ═══ 1. Нормализация после деквантизации ═══
        if RMSNorm is not None:
            self.input_norm = RMSNorm(leann_dim)
        else:
            self.input_norm = nn.LayerNorm(leann_dim)
        
        # ═══ 2. Multi-head проекция (384→1024) ═══
        # Несколько "голов" смотрят на разные аспекты LEANN эмбеддинга
        self.num_heads = num_heads
        head_dim = context_dim // num_heads  # 1024/4 = 256
        
        self.proj_heads = nn.ModuleList([
            _Linear(leann_dim, head_dim, bias=False)
            for _ in range(num_heads)
        ])
        
        # ═══ 3. Mixing layer (объединение голов) ═══
        self.mix = _Linear(context_dim, context_dim, bias=False)
        
        # ═══ 4. Выходная нормализация ═══
        if RMSNorm is not None:
            self.output_norm = RMSNorm(context_dim)
        else:
            self.output_norm = nn.LayerNorm(context_dim)
        
        # ═══ 5. Активационный квантизатор (int8 между слоями) ═══
        if ActivationQuantizer is not None and use_act_quant:
            self.act_quant = ActivationQuantizer(enabled=True)
        else:
            self.act_quant = nn.Identity()
    
    def dequantize(self, int8_vecs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        int8 → float32 деквантизация.
        
        Args:
            int8_vecs: [B, 384] int8 LEANN эмбеддинги
            scales: [B] или [B, 1] per-vector scales
        """
        # Ensure float32
        x = int8_vecs.float()
        
        # Apply per-vector scales
        if scales.dim() == 1:
            scales = scales.unsqueeze(-1)  # [B, 1]
        
        return x * scales  # [B, 384]
    
    def forward(self, int8_vecs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        Полный пайплайн: int8[384] → float32[1024].
        
        Args:
            int8_vecs: [B, 384] int8 тензор (или numpy)
            scales: [B] float32 scales
            
        Returns:
            context_vec: [B, 1024] — готов для MinGRU context_proj
        """
        # Numpy → Tensor
        if isinstance(int8_vecs, np.ndarray):
            int8_vecs = torch.from_numpy(int8_vecs)
        if isinstance(scales, np.ndarray):
            scales = torch.from_numpy(scales)
        
        # Move to same device as parameters
        device = next(self.parameters()).device
        int8_vecs = int8_vecs.to(device)
        scales = scales.to(device)
        
        # 1. Деквантизация
        x = self.dequantize(int8_vecs, scales)  # [B, 384]
        
        # 2. Нормализация
        x = self.input_norm(x)  # [B, 384]
        
        # 3. Multi-head проекция
        heads = [head(x) for head in self.proj_heads]  # 4 × [B, 256]
        x = torch.cat(heads, dim=-1)  # [B, 1024]
        
        # 4. Mixing
        x = self.mix(x) + x  # residual, [B, 1024]
        
        # 5. Выходная нормализация
        x = self.output_norm(x)
        
        # 6. Activation quantization (int8 на выходе для снижения bandwidth)
        x = self.act_quant(x)
        
        return x
    
    def from_numpy(self, emb_int8: np.ndarray, scale: np.ndarray) -> torch.Tensor:
        """
        Удобный метод: принимает numpy из LEANN, возвращает torch.
        
        Пример:
            int8_vec, scale = leann._get_embedding("query text")
            context = bridge.from_numpy(int8_vec, scale)
        """
        if emb_int8.ndim == 1:
            emb_int8 = emb_int8[np.newaxis, :]  # [384] → [1, 384]
            scale = np.array([scale])
        return self.forward(
            torch.from_numpy(emb_int8),
            torch.from_numpy(scale.astype(np.float32))
        )

    def extra_repr(self):
        return (f"leann_dim={self.leann_dim}, context_dim={self.context_dim}, "
                f"heads={self.num_heads}")
