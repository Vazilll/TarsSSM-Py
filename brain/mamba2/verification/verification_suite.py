"""
═══════════════════════════════════════════════════════════════
  VerificationSuite — Unified Verification Pipeline
═══════════════════════════════════════════════════════════════

Извлечён из TarsMamba2LM (model.py) при декомпозиции T07.
Содержит:
  - VerificationSuite (nn.Module) — объединяет все системы верификации
  - CriticHead, WaveCritic (из critic.py)
  - IntegralAuditor, MetaAuditor, MetaCortex (из integral_auditor.py)
  - HankelDetector (из novelty.py)
  - DreamEngine, Neuromodulator и др. lazy-init модули
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Any

from brain.mamba2.integral_auditor import (
    IntegralAuditor, MetaAuditor,
    TemporalEmbedding, MetaCortex,
)
from brain.mamba2.critic import CriticHead, WaveCritic, TaskSpec
from brain.mamba2.novelty import HankelDetector


logger = logging.getLogger("Tars.VerificationSuite")


class VerificationSuite(nn.Module):
    """
    VerificationSuite — все верификационные подсистемы ТАРС.
    
    Содержит:
      1. IntegralAuditor — p-convergence tracking (сходимость мысли)
      2. MetaAuditor — тип задачи → адаптивный порог
      3. MetaCortex — метакогниция: P(error) → глубина обработки
      4. TemporalEmbedding — нейронные часы (4 частоты)
      5. CriticHead + WaveCritic — мозжечок (ТЗ-проверка)
      6. HankelDetector — анти-цикл детекция (lazy-init)
    
    Lazy-initialized:
      - HankelDetector
      - DreamEngine, Neuromodulator, OscillatoryBinding, etc.
    """
    
    def __init__(self, d_model: int = 768, n_criteria: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # ═══ IntegralAuditor (p-convergence) ═══
        self.integral_auditor = IntegralAuditor(window=8, default_threshold=1.1)
        self.meta_auditor = MetaAuditor()
        
        # ═══ TemporalEmbedding (нейронные часы) ═══
        self.temporal_embedding = TemporalEmbedding(d_model, n_frequencies=4)
        
        # ═══ MetaCortex (метакогниция) ═══
        self.meta_cortex = MetaCortex(d_model=d_model, history_size=100)
        
        # ═══ CriticHead + WaveCritic ═══
        self.critic_head = CriticHead(d_model, n_criteria=n_criteria)
        self.wave_critic = WaveCritic(self.critic_head, d_model)
        
        # ═══ Lazy-init inference-only modules ═══
        self._hankel = None
        self._dream_engine = None
        self._neuromodulator = None
        self._oscillatory = None
        self._dendritic_block = None
        self._hyper_sim = None
        self._belief_state = None
    
    @property
    def hankel(self):
        if self._hankel is None:
            self._hankel = HankelDetector(window=6)
        return self._hankel
    
    @property
    def dream_engine(self):
        if self._dream_engine is None:
            try:
                from brain.mamba2.model import DreamEngine
                self._dream_engine = DreamEngine(noise_scale=0.1, recombine_k=20)
            except (ImportError, AttributeError):
                pass
        return self._dream_engine
    
    @property
    def belief_state(self):
        if self._belief_state is None:
            try:
                from brain.mamba2.model import BeliefState
                device = next(self.parameters()).device
                self._belief_state = BeliefState(d_state=128).to(device)
            except (ImportError, AttributeError):
                pass
        return self._belief_state
    
    def ensure_inference_modules(self):
        """Lazy-initialize all inference-only modules."""
        device = next(self.parameters()).device
        if self._hankel is None:
            self._hankel = HankelDetector(window=6)
        # Other modules initialized via properties when accessed
    
    def reset(self):
        """Reset all auditors for a new query."""
        self.integral_auditor.reset()
        if self._hankel is not None:
            self._hankel.reset()
    
    def classify_task(self, query_text: str):
        """Classify task type and get convergence threshold."""
        return self.meta_auditor.classify_task(query_text)
    
    def adapt_threshold(self, p_threshold: float, task_type: str,
                        error_pred: float) -> float:
        """Adapt threshold using MetaCortex."""
        return self.meta_cortex.adapt_threshold(p_threshold, task_type, error_pred)
    
    def predict_error(self, query_emb: torch.Tensor) -> float:
        """Predict error probability for a query."""
        return self.meta_cortex.predict_error(query_emb).item()
    
    def observe(self, h_curr: torch.Tensor, h_prev: torch.Tensor) -> dict:
        """Observe convergence via IntegralAuditor."""
        return self.integral_auditor.observe(h_curr, h_prev)
    
    def set_task_spec(self, task_spec):
        """Set task spec for WaveCritic."""
        self.wave_critic.set_task_spec(task_spec)
    
    def evaluate_wave(self, h_curr, h_query_mean, wave_idx, memory_vec=None):
        """Evaluate wave quality via CriticHead."""
        return self.wave_critic.evaluate_wave(
            h_curr, h_query_mean, wave_idx, memory_vec
        )
    
    def verify_rag_completion(self, **kwargs):
        """Verify RAG completion via WaveCritic."""
        return self.wave_critic.verify_rag_completion(**kwargs)
    
    def observe_hankel(self, h_curr):
        """Observe Hankel detector for cycle detection."""
        return self.hankel.observe(h_curr)
