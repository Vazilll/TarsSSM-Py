"""
═══════════════════════════════════════════════════════════════
  Self-Learning Module — Онлайн дообучение ТАРС
═══════════════════════════════════════════════════════════════

ТАРС учится на собственном опыте:
  1. Логирует каждый цикл мышления (ThinkingLogger)
  2. Пользователь даёт feedback (quality 0-1)
  3. Self-learner находит паттерны:
     - Какие MoLE эксперты лучше для каких задач
     - Какие матрицы из пула наиболее эффективны
     - Оптимальные пороги p для разных типов задач
  4. Периодически дообучает модель на лучших сессиях

Три режима:
  - Passive: только собирает логи
  - Active: корректирует routing weights
  - Full: gradient update на параметрах модели (требует GPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import json
import shutil
import numpy as np
from typing import Optional, List, Dict
from datetime import datetime, timedelta

from brain.tokenizer import TarsTokenizer


class QualityFilter:
    """
    Quality gate for SelfLearner data.
    
    Prevents low-quality or duplicate data from entering training:
      1. CriticHead score > 0.8
      2. DoubtEngine coherence > 0.6 (if available)
      3. Diversity: cosine_similarity to last 10 < 0.9
      4. Quarantine: 24h delay before data is usable
    """
    
    def __init__(self, quarantine_dir: str = "data/quarantine",
                 quarantine_hours: float = 24.0,
                 critic_threshold: float = 0.8,
                 coherence_threshold: float = 0.6,
                 diversity_threshold: float = 0.9):
        self.quarantine_dir = quarantine_dir
        self.quarantine_hours = quarantine_hours
        self.critic_threshold = critic_threshold
        self.coherence_threshold = coherence_threshold
        self.diversity_threshold = diversity_threshold
        self.logger = logging.getLogger("Tars.QualityFilter")
        
        # Metrics
        self.total_checked = 0
        self.total_rejected = 0
        self.reject_reasons: Dict[str, int] = {
            "low_critic": 0,
            "low_coherence": 0,
            "low_diversity": 0,
            "quarantined": 0,
        }
    
    def check_quality(self, session: dict) -> bool:
        """Check if session passes critic + coherence thresholds."""
        self.total_checked += 1
        
        critic_score = session.get("critic_score", session.get("response_quality", 0.0))
        if critic_score < self.critic_threshold:
            self.total_rejected += 1
            self.reject_reasons["low_critic"] += 1
            return False
        
        doubt_coherence = session.get("doubt_coherence", 1.0)  # default pass if no DoubtEngine
        if doubt_coherence < self.coherence_threshold:
            self.total_rejected += 1
            self.reject_reasons["low_coherence"] += 1
            return False
        
        return True
    
    def check_diversity(self, session: dict, recent_sessions: List[dict]) -> bool:
        """Check Jaccard similarity to last N sessions (proxy for cosine)."""
        if not recent_sessions:
            return True
        
        new_text = session.get("response", "")
        if not new_text:
            return True
        
        new_words = set(new_text.lower().split())
        if not new_words:
            return True
        
        for prev in recent_sessions[-10:]:
            prev_text = prev.get("response", "")
            if not prev_text:
                continue
            prev_words = set(prev_text.lower().split())
            if not prev_words:
                continue
            intersection = len(new_words & prev_words)
            union = len(new_words | prev_words)
            if union > 0 and (intersection / union) > self.diversity_threshold:
                self.total_rejected += 1
                self.reject_reasons["low_diversity"] += 1
                return False
        
        return True
    
    def quarantine_session(self, session: dict):
        """Move session to quarantine (usable after quarantine_hours)."""
        os.makedirs(self.quarantine_dir, exist_ok=True)
        ts = datetime.now().isoformat().replace(":", "-")
        path = os.path.join(self.quarantine_dir, f"session_{ts}.json")
        session["quarantine_time"] = datetime.now().isoformat()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(session, f, ensure_ascii=False, indent=2)
    
    def load_quarantined(self) -> List[dict]:
        """Load sessions that have passed quarantine period."""
        released = []
        if not os.path.exists(self.quarantine_dir):
            return released
        
        cutoff = datetime.now() - timedelta(hours=self.quarantine_hours)
        
        for fname in os.listdir(self.quarantine_dir):
            if not fname.endswith('.json'):
                continue
            path = os.path.join(self.quarantine_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                qt = session.get("quarantine_time", "")
                if qt:
                    q_time = datetime.fromisoformat(qt)
                    if q_time < cutoff:
                        released.append(session)
                        os.remove(path)
            except Exception:
                continue
        
        return released
    
    @property
    def rejection_rate(self) -> float:
        if self.total_checked == 0:
            return 0.0
        return self.total_rejected / self.total_checked
    
    def get_metrics(self) -> dict:
        return {
            "total_checked": self.total_checked,
            "total_rejected": self.total_rejected,
            "rejection_rate": self.rejection_rate,
            "reject_reasons": dict(self.reject_reasons),
        }


class SelfLearner:
    """
    Модуль самообучения ТАРС.
    
    Использует данные из ThinkingLogger для:
      - Коррекции routing weights в MoLE
      - Обновления efficiency в MatrixPool
      - Адаптации порогов MetaAuditor
      - (Full mode) Gradient update на модели
    """
    
    def __init__(self, model=None, lr: float = 1e-5, 
                 log_dir: str = "data/thinking_logs"):
        self.model = model
        self.lr = lr
        self.log_dir = log_dir
        self.logger = logging.getLogger("Tars.SelfLearn")
        
        # Quality filter
        self.quality_filter = QualityFilter()
        
        # Статистика
        self.session_count = 0
        self.positive_sessions = 0
        self.negative_sessions = 0
        
        # Оптимизатор (lazy init)
        self._optimizer = None
        
        # Кешированный токенизатор (не создавать в каждом цикле)
        self._tokenizer = TarsTokenizer(mode="auto")
    
    def record_feedback(self, quality: float):
        """
        Обратная связь от пользователя.
        quality: 0.0 (плохо) → 1.0 (отлично)
        """
        self.session_count += 1
        if quality >= 0.7:
            self.positive_sessions += 1
        else:
            self.negative_sessions += 1
        
        # Сохраняем feedback в лог
        if self.model and hasattr(self.model, 'thinking_logger'):
            self.model.thinking_logger.record_feedback(quality)
        
        self.logger.info(
            f"SelfLearn: feedback={quality:.2f} "
            f"(pos={self.positive_sessions}, neg={self.negative_sessions})"
        )
    
    def adapt_routing(self):
        """
        Passive learning: адаптирует MoLE routing на основе логов.
        
        Анализирует, какие эксперты активировались в хороших/плохих сессиях,
        и корректирует routing weights.
        """
        if self.model is None:
            return
        
        good_sessions = self._load_sessions(min_quality=0.8)
        bad_sessions = self._load_sessions(max_quality=0.4)
        
        if len(good_sessions) < 3:
            self.logger.debug("SelfLearn: Недостаточно данных для адаптации")
            return
        
        # Анализ экспертов
        good_experts = {}
        bad_experts = {}
        
        for session in good_sessions:
            for step in session.get("steps", []):
                for expert in step.get("experts", []):
                    name = expert.split("(")[0]
                    good_experts[name] = good_experts.get(name, 0) + 1
        
        for session in bad_sessions:
            for step in session.get("steps", []):
                for expert in step.get("experts", []):
                    name = expert.split("(")[0]
                    bad_experts[name] = bad_experts.get(name, 0) + 1
        
        self.logger.info(f"SelfLearn: Good experts: {good_experts}")
        self.logger.info(f"SelfLearn: Bad experts: {bad_experts}")
    
    def adapt_thresholds(self):
        """
        Адаптирует пороги p для разных типов задач
        на основе реальных данных.
        """
        if self.model is None or not hasattr(self.model, 'meta_auditor'):
            return
        
        sessions = self._load_sessions(min_quality=0.7)
        
        # Группируем по task_type
        type_stats = {}
        for session in sessions:
            task_type = session.get("task_type", "chat")
            final_p = session.get("final_p", 0)
            converged = session.get("converged", False)
            
            if task_type not in type_stats:
                type_stats[task_type] = {"p_values": [], "converged": 0, "total": 0}
            
            type_stats[task_type]["p_values"].append(final_p)
            type_stats[task_type]["total"] += 1
            if converged:
                type_stats[task_type]["converged"] += 1
        
        # Адаптируем пороги
        for task_type, stats in type_stats.items():
            if stats["total"] >= 5:
                avg_p = sum(stats["p_values"]) / len(stats["p_values"])
                convergence_rate = stats["converged"] / stats["total"]
                
                # Если сходимость > 90% → порог можно снизить немного
                if convergence_rate > 0.9 and task_type in self.model.meta_auditor.THRESHOLDS:
                    old_threshold = self.model.meta_auditor.THRESHOLDS[task_type]
                    # Медленная адаптация
                    new_threshold = 0.9 * old_threshold + 0.1 * avg_p
                    self.model.meta_auditor.THRESHOLDS[task_type] = new_threshold
                    
                    self.logger.info(
                        f"SelfLearn: {task_type} threshold: "
                        f"{old_threshold:.2f} → {new_threshold:.2f}"
                    )
    
    def fine_tune_step(self, input_ids: torch.Tensor, 
                       labels: torch.Tensor) -> float:
        """
        Full mode: один шаг gradient update.
        
        Замораживает TarsCoreBlock (SSD+WKV), учит только:
          - MoLE routing weights (per block)
          - Ω-SSM projections (per block)  
          - NoveltyGate (per block)
          - MatrixPool domain embeddings (model-level)
        """
        if self.model is None:
            return 0.0
        
        if self._optimizer is None:
            # Замораживаем TarsCoreBlock (SSD + WKV + Fusion)
            for block in self.model.blocks:
                for param in block.core.parameters():
                    param.requires_grad = False
            
            # Учим только адаптивные модули
            trainable = []
            for block in self.model.blocks:
                trainable.extend(block.omega.parameters())
                trainable.extend(block.mole.parameters())
                trainable.extend(block.novelty_gate.parameters())
                trainable.extend(block.rag_query.parameters())
                trainable.extend(block.rag_out.parameters())
                trainable.extend(block.mem_proj.parameters())
            
            # Model-level modules
            if hasattr(self.model, 'matrix_pool'):
                trainable.extend(self.model.matrix_pool.parameters())
            
            self._optimizer = torch.optim.AdamW(
                trainable, lr=self.lr, weight_decay=0.01
            )
        
        self.model.train()
        self._optimizer.zero_grad()
        
        logits, loss = self.model(input_ids, labels=labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self._optimizer.step()
        
        self.model.eval()
        return loss.item()
    
    def sleep_phase(self):
        """
        Фаза сна: периодическая консолидация знаний.
        
        Вызывается например каждые 50 взаимодействий или по таймеру.
        """
        self.logger.info("SelfLearn: 💤 Фаза сна — консолидация знаний...")
        
        # 1. Адаптация routing
        self.adapt_routing()
        
        # 2. Адаптация порогов
        self.adapt_thresholds()
        
        # 3. Full fine-tune на лучших данных (если есть GPU)
        good_sessions = self._load_sessions(min_quality=0.8)
        
        # Apply quality filter
        filtered = []
        for s in good_sessions:
            if self.quality_filter.check_quality(s):
                if self.quality_filter.check_diversity(s, filtered):
                    filtered.append(s)
        
        # Include released quarantine sessions
        released = self.quality_filter.load_quarantined()
        for s in released:
            if self.quality_filter.check_quality(s):
                filtered.append(s)
        
        self.logger.info(
            f"SelfLearn: Quality filter: {len(filtered)}/{len(good_sessions)} passed "
            f"(rejection_rate={self.quality_filter.rejection_rate:.1%})"
        )
        
        if len(filtered) >= 10 and self.model is not None:
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                self.logger.info(
                    f"SelfLearn: Fine-tune на {len(filtered)} сессиях..."
                )
                total_loss = 0.0
                n_steps = 0
                for session in filtered[-20:]:  # Последние 20 хороших
                    input_text = session.get("input", "")
                    response_text = session.get("response", "")
                    if not input_text or not response_text:
                        continue
                    # Токенизация через TarsTokenizer (BPE or byte fallback)
                    combined = f"{input_text}\n{response_text}"
                    tokens = self._tokenizer.encode(combined)
                    if len(tokens) < 8:
                        continue
                    token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
                    input_ids = token_tensor[:-1].unsqueeze(0)
                    labels = token_tensor[1:].unsqueeze(0)
                    loss = self.fine_tune_step(input_ids, labels)
                    total_loss += loss
                    n_steps += 1
                if n_steps > 0:
                    self.logger.info(
                        f"SelfLearn: Fine-tune done: {n_steps} steps, "
                        f"avg_loss={total_loss / n_steps:.4f}"
                    )
        
        # 4. Synaptic Homeostasis (Tononi, 2003)
        # Нейронаука: во время бодрствования синапсы усиливаются (LTP).
        # Во время сна — глобальное ослабление (downscaling), сохраняющее
        # относительные пропорции. Это улучшает SNR и освобождает ресурсы.
        self._synaptic_downscaling()
        
        # 5. Hippocampal Replay (гиппокампальный реплей)
        # Нейронаука: гиппокамп прокручивает дневной опыт во сне
        # в 5-20× ускоренном темпе (sharp-wave ripples),
        # приоритизируя удивительные/важные эпизоды.
        self._hippocampal_replay()
        
        self.logger.info("SelfLearn: 💤 Фаза сна завершена")
    
    def _hippocampal_replay(self, n_replays: int = 15, compression: int = 5):
        """
        Hippocampal Replay (Wilson & McNaughton, 1994).
        
        Нейронаука: во сне гиппокамп прокручивает эпизоды дня
        в 5-20× ускоренном темпе. Приоритетные эпизоды
        (удивительные, эмоциональные) воспроизводятся чаще.
        
        Математика (Prioritized Experience Replay):
          P(i) = p_i^α / Σ_k p_k^α       # priority sampling
          p_i = surprise_i * recency_i     # priority = surprise × recency
          tokens_replay = tokens[::compression]  # temporal compression
        """
        if self.model is None:
            return
        
        device = next(self.model.parameters()).device
        if device.type != 'cuda':
            return  # Only on GPU
        
        all_sessions = self._load_sessions(min_quality=0.0)
        if len(all_sessions) < 5:
            return
        
        # Compute priorities: surprise × recency
        import numpy as np
        priorities = []
        for i, s in enumerate(all_sessions):
            surprise = s.get("quality", 0.5)  # quality as proxy for importance
            recency = 1.0 / (len(all_sessions) - i + 1)
            priorities.append((surprise + 0.1) * recency)
        
        # Normalize to probability distribution
        priorities = np.array(priorities)
        probs = priorities / (priorities.sum() + 1e-8)
        
        # Sample sessions proportionally to priority
        n_replay = min(n_replays, len(all_sessions))
        indices = np.random.choice(len(all_sessions), size=n_replay, p=probs, replace=False)
        
        total_loss = 0.0
        replayed = 0
        
        for idx in indices:
            session = all_sessions[idx]
            input_text = session.get("input", "")
            response_text = session.get("response", "")
            if not input_text or not response_text:
                continue
            
            combined = f"{input_text}\n{response_text}"
            tokens = self._tokenizer.encode(combined)
            
            # Temporal compression: skip every N-th token (5× faster)
            tokens = tokens[::compression]
            
            if len(tokens) < 4:
                continue
            
            token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
            input_ids = token_tensor[:-1].unsqueeze(0)
            labels = token_tensor[1:].unsqueeze(0)
            
            loss = self.fine_tune_step(input_ids, labels)
            total_loss += loss
            replayed += 1
        
        if replayed > 0:
            self.logger.info(
                f"SelfLearn: 🌙 Hippocampal replay: {replayed} episodes "
                f"(×{compression} compression), avg_loss={total_loss / replayed:.4f}"
            )
    
    def _synaptic_downscaling(self, factor: float = 0.92):
        """
        Synaptic Homeostasis (Tononi, 2003).
        
        Во время бодрствования синапсы усиливаются (LTP/fine-tune).
        Во время сна — глобальное ослабление:
          w_sleep = w_wake × λ,  λ ∈ (0.8, 0.95)
        
        Свойства:
          - Сохраняет w₁/w₂ = const (относительную важность)
          - Уменьшает ||w|| (энергозатраты, saturation risk)
          - Улучшает SNR (signal-to-noise ratio)
        
        Трогаем ТОЛЬКО адаптивные компоненты (MoLE, Ω-SSM, NoveltyGate,
        Memory projections), НЕ трогаем core SSD/WKV — они обучены за
        дорогую фазу full pretrain.
        """
        if self.model is None:
            return
        
        downscaled = 0
        with torch.no_grad():
            for block in self.model.blocks:
                # MoLE expert LoRA weights (самые адаптивные)
                if hasattr(block, 'mole'):
                    for expert in block.mole.experts:
                        if hasattr(expert, 'A'):
                            expert.A.weight.data *= factor
                        if hasattr(expert, 'B'):
                            expert.B.weight.data *= factor
                        downscaled += 1
                
                # Ω-SSM projections (Lie algebra)
                if hasattr(block, 'omega'):
                    if hasattr(block.omega, 'omega_proj'):
                        block.omega.omega_proj.weight.data *= factor
                    if hasattr(block.omega, 'omega_mix'):
                        block.omega.omega_mix.data *= factor
                    downscaled += 1
                
                # NoveltyGate
                if hasattr(block, 'novelty_gate'):
                    for p in block.novelty_gate.parameters():
                        p.data *= factor
                    downscaled += 1
                
                # Memory injection gates
                if hasattr(block, 'mem_gate'):
                    for p in block.mem_gate.parameters():
                        p.data *= factor
                    downscaled += 1
            
            # MatrixPool efficiency scores (recirculation)
            if hasattr(self.model, 'matrix_pool'):
                self.model.matrix_pool.efficiency *= factor
                downscaled += 1
        
        self.logger.info(
            f"SelfLearn: 🧬 Synaptic downscaling (λ={factor}): "
            f"{downscaled} components renormalized"
        )
    
    def _load_sessions(self, min_quality: float = 0.0, 
                       max_quality: float = 1.0) -> List[Dict]:
        """Загрузка сессий из логов."""
        sessions = []
        try:
            for fname in os.listdir(self.log_dir):
                if not fname.endswith('.json'):
                    continue
                filepath = os.path.join(self.log_dir, fname)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                quality = data.get("response_quality")
                if quality is not None:
                    if min_quality <= quality <= max_quality:
                        sessions.append(data)
        except Exception:
            pass
        return sessions
    
    def save_checkpoint(self, path: str):
        """Сохранение модели."""
        if self.model is None:
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'd_model': self.model.d_model,
            'n_layers': self.model.n_layers,
            'vocab_size': self.model.vocab_size,
            'session_count': self.session_count,
            'positive_sessions': self.positive_sessions,
            'timestamp': datetime.now().isoformat(),
        }, path)
        self.logger.info(f"SelfLearn: Checkpoint saved to {path}")
