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
from typing import Optional, List, Dict
from datetime import datetime


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
        
        # Статистика
        self.session_count = 0
        self.positive_sessions = 0
        self.negative_sessions = 0
        
        # Оптимизатор (lazy init)
        self._optimizer = None
    
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
        if len(good_sessions) >= 10 and self.model is not None:
            device = next(self.model.parameters()).device
            if device.type == 'cuda':
                self.logger.info(
                    f"SelfLearn: Fine-tune на {len(good_sessions)} сессиях..."
                )
                # TODO: подготовка данных из сессий и fine-tune
        
        self.logger.info("SelfLearn: 💤 Фаза сна завершена")
    
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
