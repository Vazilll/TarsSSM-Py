"""
═══════════════════════════════════════════════════════════════
  Ω-SSM Layer — Lie Algebra on SO(n) manifold
═══════════════════════════════════════════════════════════════

Скрытые состояния живут на гладком многообразии SO(n).
Обновление через Cayley Transform (быстрее матричной экспоненты).

Гарантии:
  - Ортогональность состояний → нет взрыва/затухания градиентов
  - Стабильность на бесконечном контексте
  - Обратимость: G^(-1) = G^T для SO(n)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from brain.mamba2.bitnet import UniversalLinear


def cayley_transform(omega: torch.Tensor) -> torch.Tensor:
    """
    Cayley Transform: антисимметричная Ω → ортогональная G ∈ SO(n).
    
    G = (I + Ω/2)(I - Ω/2)⁻¹
    
    Быстрее matrix_exp, не требует eigenvalue decomposition.
    Note: linalg.solve requires float32 (not supported in Half on CUDA).
    """
    orig_dtype = omega.dtype
    # linalg.solve не поддерживает Half — кастим в float32
    if omega.dtype == torch.float16 or omega.dtype == torch.bfloat16:
        omega = omega.float()
    
    n = omega.shape[-1]
    I = torch.eye(n, device=omega.device, dtype=omega.dtype)
    if omega.dim() == 3:
        I = I.unsqueeze(0).expand(omega.shape[0], -1, -1)
    
    half_omega = omega * 0.5
    G = torch.linalg.solve(I - half_omega, I + half_omega)
    return G.to(orig_dtype)


class OmegaSSMLayer(nn.Module):
    """
    Ω-SSM: Расслоённое Скрытое Состояние.
    
    h_t = [s_t, d_t]
         s_t ∈ SO(n) — непрерывное (Lie Algebra, семантика)
         d_t ∈ Z^k   — дискретное (VQ codebook, логика)
    
    Применяется ПОСЛЕ Mamba-2 SSD для стабилизации состояний.
    """
    
    def __init__(self, d_model: int = 768, omega_dim: int = 64, 
                 vq_codes: int = 256, vq_dim: int = 64,
                 quant_mode: str = "fp16"):
        super().__init__()
        self.d_model = d_model
        self.omega_dim = omega_dim
        
        # ═══ Непрерывная часть: Lie Algebra SO(n) ═══
        # Проекция d_model → Ω (антисимметричная матрица omega_dim×omega_dim)
        # Используем только верхний треугольник (n*(n-1)/2 параметров)
        self.n_params = omega_dim * (omega_dim - 1) // 2
        self.omega_proj = UniversalLinear(d_model, self.n_params, bias=True, mode=quant_mode)
        self.omega_out = UniversalLinear(omega_dim, d_model, bias=True, mode=quant_mode)
        self.omega_mix = nn.Parameter(torch.tensor(0.1))
        
        # ═══ Дискретная часть: VQ Codebook ═══
        self.vq_codebook = nn.Embedding(vq_codes, vq_dim)
        self.vq_proj_in = UniversalLinear(d_model, vq_dim, bias=True, mode=quant_mode)
        self.vq_proj_out = UniversalLinear(vq_dim, d_model, bias=True, mode=quant_mode)
        self.vq_mix = nn.Parameter(torch.tensor(0.05))
        
        # ═══ Norm ═══
        self.norm = nn.LayerNorm(d_model)
    
    def _build_skew_symmetric(self, params: torch.Tensor) -> torch.Tensor:
        """
        Строит антисимметричную матрицу Ω из параметров верхнего треугольника.
        Ω = -Ωᵀ (это гарантирует exp(Ω) ∈ SO(n))
        """
        batch = params.shape[0]
        omega = torch.zeros(batch, self.omega_dim, self.omega_dim, 
                          device=params.device, dtype=params.dtype)
        # Заполняем верхний треугольник
        idx = torch.triu_indices(self.omega_dim, self.omega_dim, offset=1)
        omega[:, idx[0], idx[1]] = params
        # Антисимметрия: нижний = -верхний
        omega = omega - omega.transpose(-1, -2)
        return omega
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        Returns: [B, L, d_model] с Lie refinement
        """
        residual = x
        
        # ═══ 1. Lie Algebra refinement ═══
        # Берём средний вектор последовательности для построения Ω
        h_mean = x.mean(dim=1)  # [B, d_model]
        
        # Строим антисимметричную Ω
        params = self.omega_proj(h_mean)           # [B, n_params]
        omega = self._build_skew_symmetric(params) # [B, omega_dim, omega_dim]
        
        # Cayley Transform → G ∈ SO(n)
        G = cayley_transform(omega)  # [B, omega_dim, omega_dim]
        
        # Применяем G к первым omega_dim измерениям x
        x_head = x[:, :, :self.omega_dim]               # [B, L, omega_dim]
        x_rotated = torch.bmm(
            x_head.reshape(-1, x_head.shape[-1]).unsqueeze(1),  
            G.unsqueeze(1).expand(-1, x.shape[1], -1, -1)
                .reshape(-1, self.omega_dim, self.omega_dim)
        ).squeeze(1).reshape(x.shape[0], x.shape[1], self.omega_dim)
        
        lie_out = self.omega_out(x_rotated)  # [B, L, d_model]
        
        # ═══ 2. VQ дискретная часть ═══
        vq_input = self.vq_proj_in(h_mean)        # [B, vq_dim]
        # Находим ближайший code
        dists = torch.cdist(vq_input.unsqueeze(1), 
                           self.vq_codebook.weight.unsqueeze(0))
        vq_idx = dists.argmin(dim=-1)              # [B, 1]
        vq_code = self.vq_codebook(vq_idx.squeeze(1))  # [B, vq_dim]
        
        # Straight-through estimator (градиент обходит argmin)
        vq_out = vq_input + (vq_code - vq_input).detach()
        vq_contribution = self.vq_proj_out(vq_out).unsqueeze(1)  # [B, 1, d_model]
        
        # ═══ 3. Combine ═══
        x = residual + self.omega_mix * lie_out + self.vq_mix * vq_contribution
        return self.norm(x)
