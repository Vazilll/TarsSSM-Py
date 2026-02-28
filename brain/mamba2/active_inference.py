"""
═══════════════════════════════════════════════════════════════
  Active Inference — Free Energy Principle (Friston, 2006-2026)
═══════════════════════════════════════════════════════════════

Нейронаука:
  Free Energy Principle (Friston): мозг — байесовская машина,
  которая минимизирует variational free energy (= surprise).
  
  Active Inference: агент ДЕЙСТВУЕТ, чтобы уменьшить неопределённость.
  Не просто реагирует — активно ищет информацию.

Математика:
  F = E_q[log q(s) - log p(o,s)]     # variational free energy
    = D_KL[q(s) || p(s|o)] - log p(o) # = complexity - accuracy
  
  Агент минимизирует F через:
    1. Perception: обновление q(s) → уменьшение D_KL (belief update)
    2. Action: изменение o → уменьшение surprise (active sensing)

Применение в TARS:
  - GIE agent использует expected free energy для выбора действий
  - Belief state обновляется после каждого взаимодействия
  - Epistemic curiosity: предпочтение действий с высоким information gain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Tuple, Optional


class BeliefState(nn.Module):
    """
    Внутреннее состояние убеждений агента (posterior beliefs).
    
    Параметризовано как μ (mean) и log_σ (log-variance)
    мультивариатного гауссиана q(s) = N(μ, σ²I).
    
    Обновление через байесовское правило:
      q(s|o) ∝ p(o|s) · q(s)    # posterior ∝ likelihood × prior
    """
    
    def __init__(self, d_state: int = 128):
        super().__init__()
        self.d_state = d_state
        
        # Prior beliefs: p(s) = N(0, I)
        self.register_buffer('prior_mean', torch.zeros(d_state))
        self.register_buffer('prior_log_var', torch.zeros(d_state))
        
        # Posterior beliefs: q(s) — updated after each observation
        self.posterior_mean = nn.Parameter(torch.zeros(d_state))
        self.posterior_log_var = nn.Parameter(torch.zeros(d_state))
        
        # Observation encoder: o → likelihood params
        self.obs_encoder = nn.Sequential(
            nn.Linear(768, d_state * 2),  # assumes d_model=768
            nn.SiLU(),
            nn.Linear(d_state * 2, d_state * 2),
        )
        
        self.logger = logging.getLogger("Tars.Belief")
    
    def update(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Байесовское обновление убеждений.
        
        observation: [B, d_model] — текущее наблюдение (h_global)
        
        Returns:
            dict: posterior_mean, posterior_var, kl_divergence, surprise
        """
        # Encode observation → likelihood params
        likelihood_params = self.obs_encoder(observation)  # [B, 2*d_state]
        obs_mean = likelihood_params[:, :self.d_state]
        obs_log_var = likelihood_params[:, self.d_state:]
        
        # Bayesian update (Gaussian conjugacy):
        # q(s|o) ∝ N(obs_mean, obs_var) · N(prior_mean, prior_var)
        prior_var = torch.exp(self.posterior_log_var)  # use current posterior as prior
        obs_var = torch.exp(obs_log_var)
        
        # Precision-weighted mean
        new_precision = 1.0 / prior_var + 1.0 / (obs_var + 1e-8)
        new_var = 1.0 / (new_precision + 1e-8)
        new_mean = new_var * (self.posterior_mean / prior_var + obs_mean / (obs_var + 1e-8))
        
        # Update posterior (in-place, no gradient through update itself)
        with torch.no_grad():
            # EMA update (soft, not hard replacement)
            momentum = 0.8
            self.posterior_mean.data = (
                momentum * self.posterior_mean.data 
                + (1 - momentum) * new_mean.mean(dim=0)
            )
            self.posterior_log_var.data = (
                momentum * self.posterior_log_var.data
                + (1 - momentum) * torch.log(new_var.mean(dim=0) + 1e-8)
            )
        
        # KL divergence: D_KL[q(s) || p(s)] — complexity cost
        kl = 0.5 * (
            torch.exp(self.posterior_log_var) 
            + self.posterior_mean.pow(2) 
            - 1.0 
            - self.posterior_log_var
        ).sum()
        
        # Surprise: -log p(o) ≈ reconstruction error
        surprise = (obs_mean - self.posterior_mean).pow(2).sum()
        
        return {
            "posterior_mean": new_mean,
            "posterior_var": new_var,
            "kl_divergence": kl,
            "surprise": surprise,
            "free_energy": kl + surprise,  # F = complexity + surprise
        }
    
    def sample(self, n_samples: int = 1) -> torch.Tensor:
        """Сэмплирование из posterior q(s) через reparameterization trick."""
        std = torch.exp(0.5 * self.posterior_log_var)
        eps = torch.randn(n_samples, self.d_state, device=std.device)
        return self.posterior_mean + eps * std


class ExpectedFreeEnergy(nn.Module):
    """
    Expected Free Energy (EFE) для выбора действий.
    
    G(π) = E_q[log q(s|π) - log p(o,s|π)]
         = -Epistemic Value - Pragmatic Value
    
    Epistemic: информационный выигрыш (curiosity)
    Pragmatic: достижение цели (exploitation)
    """
    
    def __init__(self, d_action: int = 16, d_state: int = 128):
        super().__init__()
        
        # Transition model: q(s'|s,a) = N(μ_trans, σ_trans)
        self.transition = nn.Sequential(
            nn.Linear(d_state + d_action, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state * 2),  # mean + log_var
        )
        
        # Observation model: q(o|s') → predicted observation
        self.observation_model = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.SiLU(),
            nn.Linear(d_state, d_state),
        )
        
        # Preference prior: p(o) — desired outcomes
        self.preference = nn.Parameter(torch.zeros(d_state))
    
    def compute_efe(
        self, 
        belief: BeliefState,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Вычисляет Expected Free Energy для действия.
        
        Args:
            belief: текущее состояние убеждений
            action: [B, d_action] — кандидат действие
        
        Returns:
            G: [B] — EFE score (чем ниже, тем лучше действие)
        """
        # State prediction: q(s'|s, a)
        state_action = torch.cat([
            belief.posterior_mean.unsqueeze(0).expand(action.shape[0], -1),
            action
        ], dim=-1)
        trans_params = self.transition(state_action)
        pred_mean = trans_params[:, :belief.d_state]
        pred_log_var = trans_params[:, belief.d_state:]
        pred_var = torch.exp(pred_log_var)
        
        # Epistemic value: информационный выигрыш
        # = H[q(o|π)] - E_q[H[q(o|s',π)]]
        # Аппроксимация: entropy of predicted state
        epistemic = 0.5 * pred_log_var.sum(dim=-1)  # higher entropy = more to learn
        
        # Pragmatic value: proximity to preference
        pred_obs = self.observation_model(pred_mean)
        pragmatic = -(pred_obs - self.preference).pow(2).sum(dim=-1)
        
        # G = -epistemic - pragmatic
        # Минимизация G = максимизация epistemic + pragmatic
        G = -epistemic - pragmatic
        
        return G
    
    def select_action(
        self,
        belief: BeliefState,
        candidate_actions: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """
        Выбирает лучшее действие по минимальному EFE.
        
        candidate_actions: [N, d_action]
        Returns: (best_idx, G_values)
        """
        G = self.compute_efe(belief, candidate_actions)
        best_idx = G.argmin().item()
        return best_idx, G
