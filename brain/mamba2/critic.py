"""
═══════════════════════════════════════════════════════════════
  ТАРС CriticHead (Мозжечок) — Межслойная проверка по ТЗ
═══════════════════════════════════════════════════════════════

Встраивается между WaveConsolidation и RAG (ThinkingChain):

  [Block_L || Block_R] → Consolidation → ★CriticHead★ → RAG → next wave

CriticHead проверяет: "соответствует ли текущее состояние мышления ТЗ?"
Если нет — запросить данные через синапсы, инжектировать, продолжить.

TaskSpec — автогенерируемое ТЗ, которое пользователь подтверждает.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("Tars.Critic")


# ═══════════════════════════════════════════
# TaskSpec — Техническое задание
# ═══════════════════════════════════════════

@dataclass
class TaskSpec:
    """
    ТЗ (техническое задание) для текущей задачи.
    
    Автоматически генерируется из запроса, пользователь
    может подтвердить, дополнить или изменить.
    """
    query: str                                # Исходный запрос
    points: List[str] = field(default_factory=list)  # Пункты ТЗ
    approved: bool = False                    # Подтверждено пользователем?
    scores: Dict[int, float] = field(default_factory=dict)  # Score по каждому пункту
    
    @property
    def all_passed(self) -> bool:
        """Все пункты ТЗ выполнены (score > 0.8)?"""
        if not self.points:
            return True
        return all(self.scores.get(i, 0.0) > 0.8 for i in range(len(self.points)))
    
    @property
    def overall_score(self) -> float:
        """Средний score по всем пунктам."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / max(len(self.points), 1)
    
    @property
    def weak_points(self) -> List[str]:
        """Пункты с низким score."""
        return [
            self.points[i] for i in range(len(self.points))
            if self.scores.get(i, 0.0) < 0.7
        ]
    
    def summary(self) -> str:
        """Текстовый отчёт по ТЗ."""
        lines = [f"ТЗ: {self.query[:60]}"]
        for i, point in enumerate(self.points):
            score = self.scores.get(i, 0.0)
            mark = "✅" if score > 0.8 else "⚠️" if score > 0.5 else "❌"
            lines.append(f"  {mark} [{score:.0%}] {point}")
        lines.append(f"  Итого: {self.overall_score:.0%}")
        return "\n".join(lines)


# ═══════════════════════════════════════════
# CriticHead — Мозжечок
# ═══════════════════════════════════════════

class CriticHead(nn.Module):
    """
    Мозжечок: оценивает соответствие текущего hidden state задаче.
    
    Встраивается МЕЖДУ волнами (после Consolidation, перед RAG):
      wave_output [B, L, d_model] → CriticHead → score [B, 1]
    
    Если score < threshold:
      - Feedback инжектируется в memory_vec
      - Синапсы могут загрузить доп. данные
      - Следующая волна учитывает ошибки
    """
    
    # Порог для "всё ок, можно двигаться дальше"
    PASS_THRESHOLD = 0.80
    # Порог для "критически плохо, нужны внешние данные"
    FETCH_THRESHOLD = 0.40
    
    def __init__(self, d_model: int, n_criteria: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_criteria = n_criteria
        
        # Проекция: текущее состояние → оценка по критериям
        # query_aware: используем и текущий state и исходный query state
        self.query_proj = nn.Linear(d_model, d_model // 2, bias=False)
        self.state_proj = nn.Linear(d_model, d_model // 2, bias=False)
        
        # Per-criterion scoring
        self.criteria_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_criteria),
        )
        
        # Aggregation: criteria → single score
        self.aggregate = nn.Sequential(
            nn.Linear(n_criteria, n_criteria),
            nn.GELU(),
            nn.Linear(n_criteria, 1),
        )
        
        # Feedback projection: low score → correction vector
        self.feedback_proj = nn.Sequential(
            nn.Linear(n_criteria + d_model, d_model),
            nn.Tanh(),  # bounded correction
        )
        
        # Wave progress embedding (знает номер волны)
        self.wave_embed = nn.Embedding(16, d_model // 2)
        
        # ТЗ encoder: текстовые пункты → embedding
        self.tz_proj = nn.Linear(d_model, d_model // 2, bias=False)
    
    def forward(
        self,
        h_current: torch.Tensor,       # [B, d_model] — текущее состояние после consolidation
        h_query: torch.Tensor,          # [B, d_model] — исходный query embedding
        wave_idx: int = 0,              # номер текущей волны
        tz_embedding: torch.Tensor = None,  # [B, d_model] — закодированное ТЗ (опционально)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            score: [B, 1] — общая оценка (0=плохо, 1=отлично)
            criteria: [B, n_criteria] — оценки по критериям
            feedback: [B, d_model] — вектор коррекции для memory_vec
        """
        # Combine query + current state
        q = self.query_proj(h_query)          # [B, d/2]
        s = self.state_proj(h_current)        # [B, d/2]
        
        # Add wave progress
        w = self.wave_embed(
            torch.tensor([min(wave_idx, 15)], device=h_current.device)
        ).expand(h_current.shape[0], -1)     # [B, d/2]
        
        # Combined representation
        combined = q * s + w * s  # [B, d/2]
        
        # Add TZ awareness if provided
        if tz_embedding is not None:
            tz = self.tz_proj(tz_embedding)   # [B, d/2]
            combined = combined + tz
        
        # Pad to full d_model for criteria_net
        combined_full = torch.cat([combined, combined], dim=-1)  # [B, d_model]
        
        # Per-criterion scores
        criteria = torch.sigmoid(self.criteria_net(combined_full))  # [B, 8]
        
        # Aggregate score
        score = torch.sigmoid(self.aggregate(criteria))  # [B, 1]
        
        # Feedback vector (for memory_vec correction)
        feedback_input = torch.cat([criteria, h_current], dim=-1)  # [B, 8 + d_model]
        feedback = self.feedback_proj(feedback_input)  # [B, d_model]
        
        # Scale feedback by (1 - score): stronger correction when score is low
        feedback = feedback * (1.0 - score)
        
        return score, criteria, feedback
    
    def should_fetch_data(self, score: torch.Tensor) -> bool:
        """Нужно ли загружать внешние данные?"""
        return score.mean().item() < self.FETCH_THRESHOLD
    
    def should_continue(self, score: torch.Tensor) -> bool:
        """Продолжать вычисления (score не достаточен)?"""
        return score.mean().item() < self.PASS_THRESHOLD


# ═══════════════════════════════════════════
# WaveCritic — обёртка для wave loop
# ═══════════════════════════════════════════

class WaveCritic:
    """
    Обёртка CriticHead для использования в wave loop think().
    
    Между волнами:
      1. CriticHead оценивает текущее состояние
      2. Если плохо → генерирует feedback → inject в memory_vec
      3. Если критически плохо → вызывает синапсы для данных
      4. Логирует прогресс по ТЗ
    """
    
    def __init__(self, critic_head: CriticHead, d_model: int):
        self.critic = critic_head
        self.d_model = d_model
        self.task_spec: Optional[TaskSpec] = None
        self.wave_scores: List[float] = []
        self.total_fetches = 0
        self._synapse_pool = None
    
    def set_task_spec(self, spec: TaskSpec):
        """Установить ТЗ для текущей задачи."""
        self.task_spec = spec
        self.wave_scores = []
        self.total_fetches = 0
    
    @torch.no_grad()
    def evaluate_wave(
        self,
        h_current: torch.Tensor,   # [B, d_model]
        h_query: torch.Tensor,     # [B, d_model]
        wave_idx: int,
        memory_vec: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Оценить состояние после волны.
        
        Returns:
            {
                "score": float,
                "criteria": list[float],
                "feedback_vec": Tensor or None,  # inject в memory_vec
                "needs_data": bool,               # нужны синапсы?
                "should_stop": bool,               # можно остановиться?
                "report": str,
            }
        """
        score, criteria, feedback = self.critic(
            h_current, h_query, wave_idx
        )
        
        score_val = score.mean().item()
        criteria_vals = criteria[0].tolist() if criteria.dim() > 1 else criteria.tolist()
        self.wave_scores.append(score_val)
        
        # Update TaskSpec scores if available
        if self.task_spec and self.task_spec.points:
            for i in range(min(len(self.task_spec.points), len(criteria_vals))):
                self.task_spec.scores[i] = criteria_vals[i]
        
        needs_data = self.critic.should_fetch_data(score)
        should_stop = not self.critic.should_continue(score)
        
        # Feedback vector → inject in memory_vec
        feedback_vec = None
        if not should_stop and memory_vec is not None:
            feedback_vec = feedback.detach()
        
        # Progress report
        trend = ""
        if len(self.wave_scores) >= 2:
            delta = self.wave_scores[-1] - self.wave_scores[-2]
            trend = f" (↑{delta:+.0%})" if delta > 0 else f" (↓{delta:+.0%})"
        
        report = (
            f"Critic wave {wave_idx}: {score_val:.0%}{trend}"
            f" | criteria={[f'{c:.0%}' for c in criteria_vals[:4]]}"
        )
        
        if needs_data:
            report += " | 🔍 FETCH NEEDED"
            self.total_fetches += 1
        
        logger.info(report)
        
        return {
            "score": score_val,
            "criteria": criteria_vals,
            "feedback_vec": feedback_vec,
            "needs_data": needs_data,
            "should_stop": should_stop,
            "report": report,
        }
    
    def get_final_report(self) -> str:
        """Финальный отчёт по всем волнам."""
        if not self.wave_scores:
            return "No waves evaluated"
        
        lines = [
            f"Critic: {len(self.wave_scores)} waves, "
            f"final={self.wave_scores[-1]:.0%}, "
            f"fetches={self.total_fetches}",
        ]
        
        if self.task_spec:
            lines.append(self.task_spec.summary())
        
        return "\n".join(lines)
