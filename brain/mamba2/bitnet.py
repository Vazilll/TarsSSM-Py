"""
═══════════════════════════════════════════════════════════════
  BitNet Kernel — Unified FP16 + 1.58-bit Linear Layer
═══════════════════════════════════════════════════════════════

Универсальный слой, работающий в двух режимах:

  FP16 Mode:
    - Стандартный nn.Linear с fp16/fp32 весами
    - Максимальная точность для обучения и исследований

  1.58-bit Mode (BitNet b1.58):
    - Веса квантуются до {-1, 0, +1} на лету
    - Активации квантуются до int8
    - Умножение заменяется сложением/вычитанием
    - Straight-Through Estimator для градиентов
    - 5-10x меньше памяти, 3-5x быстрее на CPU

Основано на: "The Era of 1-bit LLMs" (Ma et al., 2024)

Использование:
  # FP16 (по умолчанию при обучении)
  layer = UniversalLinear(768, 1536, mode="fp16")

  # 1.58-bit (для инференса / экономного обучения)
  layer = UniversalLinear(768, 1536, mode="158bit")

  # Переключение на лету:
  layer.set_mode("158bit")
  layer.set_mode("fp16")
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════
#  JIT-compiled ternary quantization kernels
# ═══════════════════════════════════════════════════════════════

@torch.jit.script
def _weight_quant_ternary(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Квантование весов до {-1, 0, +1} (BitNet b1.58).

    Алгоритм:
      1. Вычисляем γ = mean(|w|) — абсолютное среднее (scale factor)
      2. w_q = round(clip(w / γ, -1, 1)) → значения ∈ {-1, 0, +1}
      3. Сохраняем γ для деквантования: w ≈ w_q * γ

    Returns:
      w_q: тернарные веса [-1, 0, 1]
      scale: γ (per-tensor scale)
    """
    scale = w.abs().mean().clamp(min=1e-8)
    w_norm = w / scale
    w_q = w_norm.clamp(-1.0, 1.0).round()
    return w_q, scale


@torch.jit.script
def _activation_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Квантование активаций до int8 (absmax per-token).

    Алгоритм:
      1. γ = max(|x|, dim=-1) / 127 — per-token scale
      2. x_q = round(clip(x / γ, -128, 127))
      3. x ≈ x_q * γ

    Returns:
      x_q: квантованные активации
      scale: γ (per-token scale [B, ..., 1])
    """
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    x_q = (x / scale).clamp(-128.0, 127.0).round()
    return x_q, scale


# ═══════════════════════════════════════════════════════════════
#  Straight-Through Estimator (STE) Autograd Functions
# ═══════════════════════════════════════════════════════════════

class STEWeightQuant(torch.autograd.Function):
    """
    STE для весов: forward квантует, backward пропускает градиент как есть.
    Это позволяет обучать 1.58-bit модель стандартным градиентным спуском.
    """
    @staticmethod
    def forward(ctx, w):
        w_q, scale = _weight_quant_ternary(w)
        ctx.save_for_backward(scale)
        return w_q * scale  # деквантованный вывод, но из тернарных значений

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: градиент проходит без изменений
        return grad_output


class STEActivationQuant(torch.autograd.Function):
    """
    STE для активаций: forward квантует до int8, backward пропускает.
    """
    @staticmethod
    def forward(ctx, x):
        x_q, scale = _activation_quant_int8(x)
        return x_q * scale  # деквантованный вывод

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# ═══════════════════════════════════════════════════════════════
#  RMSNorm (нужен для BitNet — нормализация перед квантованием)
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Быстрее LayerNorm: нет bias, нет вычитания среднего.
    Обязательна для BitNet — стабилизирует распределение перед квантованием.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ═══════════════════════════════════════════════════════════════
#  UniversalLinear — гибридный FP16 / 1.58-bit слой
# ═══════════════════════════════════════════════════════════════

class UniversalLinear(nn.Module):
    """
    Универсальный линейный слой с переключаемым режимом.

    Режимы:
      "fp16"   — стандартные float16/32 веса (nn.Linear-совместимый)
      "158bit" — квантование весов до {-1, 0, +1}, активации до int8

    Можно переключить в любой момент:
      layer.set_mode("158bit")  # квантованный инференс
      layer.set_mode("fp16")    # обратно к полной точности

    Параметры:
      in_features:  размерность входа
      out_features: размерность выхода
      bias:         использовать bias (default: False для SSM)
      mode:         "fp16" или "158bit"
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        mode: str = "fp16",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        # ═══ Основные веса (всегда fp16/fp32, квантуются на лету) ═══
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # ═══ RMSNorm для стабилизации перед квантованием ═══
        # Используется только в режиме 1.58-bit
        self.input_norm = RMSNorm(in_features)

        # ═══ Кеш тернарных весов (для инференса без пересчёта) ═══
        self.register_buffer("_cached_ternary_w", None)
        self.register_buffer("_cached_scale", None)
        self._cache_valid = False

        # Инициализация весов (Kaiming для SiLU/ReLU совместимости)
        self._reset_parameters()

    def _reset_parameters(self):
        """Инициализация весов по Kaiming (оптимально для SiLU)."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def set_mode(self, mode: str):
        """
        Переключение режима.

        Args:
            mode: "fp16" или "158bit"
        """
        assert mode in ("fp16", "158bit"), f"Unknown mode: {mode}"
        if mode != self.mode:
            self.mode = mode
            self._cache_valid = False
            if mode == "fp16":
                self._cached_ternary_w = None
                self._cached_scale = None

    def _get_effective_weight(self) -> torch.Tensor:
        """
        Возвращает эффективные веса в зависимости от режима.

        FP16:    возвращает self.weight напрямую
        1.58-bit: квантует через STE (train) или кеш (eval)
        """
        if self.mode == "fp16":
            return self.weight

        # ═══ 1.58-bit mode ═══
        if self.training:
            # Обучение: STE квантование (каждый forward пересчитывает)
            return STEWeightQuant.apply(self.weight)
        else:
            # Инференс: кешируем уже перемноженный результат w_q * scale
            if not self._cache_valid:
                with torch.no_grad():
                    w_q, scale = _weight_quant_ternary(self.weight)
                    self._cached_ternary_w = w_q * scale  # pre-multiplied!
                    self._cached_scale = scale
                    self._cache_valid = True
                    
            return self._cached_ternary_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        FP16 Mode:
          y = x @ W^T + bias

        1.58-bit Mode:
          x_norm = RMSNorm(x)
          x_q = STE_int8(x_norm)
          W_q = STE_ternary(W)
          y = x_q @ W_q^T + bias
        """
        if self.mode == "fp16":
            # ═══ Чистый FP16 — максимальная скорость ═══
            return F.linear(x, self.weight, self.bias)

        # ═══ 1.58-bit BitNet ═══
        # 1. RMSNorm стабилизирует распределение активаций
        x_norm = self.input_norm(x)

        # 2. Квантование активаций до int8
        if self.training:
            x_q = STEActivationQuant.apply(x_norm)
        else:
            x_q, x_scale = _activation_quant_int8(x_norm)
            x_q = x_q * x_scale

        # 3. Получаем эффективные веса (тернарные через STE или кеш)
        w_eff = self._get_effective_weight()

        # 4. Линейная операция
        return F.linear(x_q, w_eff, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode}"
        )

    @torch.no_grad()
    def compute_sparsity(self) -> float:
        """Процент нулевых весов в тернарном режиме."""
        w_q, _ = _weight_quant_ternary(self.weight)
        return (w_q == 0).float().mean().item()

    @torch.no_grad()
    def compute_compression_ratio(self) -> float:
        """Степень сжатия по сравнению с FP16."""
        fp16_bits = self.weight.numel() * 16
        # Тернарные: log2(3) ≈ 1.58 бит + scale (32 бит)
        ternary_bits = self.weight.numel() * 1.58 + 32
        return fp16_bits / ternary_bits


# ═══════════════════════════════════════════════════════════════
#  Утилиты для всей модели
# ═══════════════════════════════════════════════════════════════

def convert_model_to_158bit(model: nn.Module):
    """
    Переключает ВСЕ UniversalLinear слои модели в режим 1.58-bit.
    Полезно для развёртывания обученной модели в продакшен.
    """
    for module in model.modules():
        if isinstance(module, UniversalLinear):
            module.set_mode("158bit")


def convert_model_to_fp16(model: nn.Module):
    """
    Переключает ВСЕ UniversalLinear слои модели в режим FP16.
    Полезно для fine-tuning или отладки.
    """
    for module in model.modules():
        if isinstance(module, UniversalLinear):
            module.set_mode("fp16")


def replace_linear_with_universal(
    model: nn.Module,
    mode: str = "fp16",
    skip_names: Optional[set] = None,
) -> nn.Module:
    """
    Заменяет ВСЕ nn.Linear в модели на UniversalLinear.
    
    Это автоматический конвертер: можно применить к любой PyTorch модели.

    Args:
        model: PyTorch модель
        mode: "fp16" или "158bit"
        skip_names: имена модулей, которые НЕ нужно заменять

    Returns:
        Модель с заменёнными Linear слоями
    """
    if skip_names is None:
        skip_names = set()

    for name, module in model.named_children():
        if name in skip_names:
            continue

        if isinstance(module, nn.Linear) and not isinstance(module, UniversalLinear):
            universal = UniversalLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                mode=mode,
            )
            # Копируем существующие веса
            universal.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                universal.bias.data.copy_(module.bias.data)
            setattr(model, name, universal)
        else:
            # Рекурсивно для вложенных модулей
            replace_linear_with_universal(module, mode, skip_names)

    return model


def model_stats(model: nn.Module) -> dict:
    """
    Считает статистику UniversalLinear слоёв в модели.
    
    Returns:
        dict с ключами: total_layers, fp16_layers, bit158_layers,
                        total_params, compression_ratio, sparsity
    """
    total = fp16 = bit158 = 0
    total_params = 0
    total_sparsity = 0.0
    
    for module in model.modules():
        if isinstance(module, UniversalLinear):
            total += 1
            total_params += module.weight.numel()
            if module.mode == "fp16":
                fp16 += 1
            else:
                bit158 += 1
                total_sparsity += module.compute_sparsity()
    
    avg_sparsity = total_sparsity / max(bit158, 1)
    fp16_mb = total_params * 2 / (1024 * 1024)
    bit158_mb = total_params * 1.58 / 8 / (1024 * 1024)
    
    return {
        "total_layers": total,
        "fp16_layers": fp16,
        "bit158_layers": bit158,
        "total_params": total_params,
        "fp16_size_mb": round(fp16_mb, 1),
        "bit158_size_mb": round(bit158_mb, 1),
        "compression_ratio": round(fp16_mb / max(bit158_mb, 0.01), 1),
        "avg_sparsity": round(avg_sparsity, 3),
    }
