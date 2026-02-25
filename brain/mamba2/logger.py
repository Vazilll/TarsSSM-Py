"""
═══════════════════════════════════════════════════════════════
  ТАРС Logger — Структурированное логирование мышления
═══════════════════════════════════════════════════════════════

Записывает каждый шаг рассуждений:
  - p-сходимость (Integral Auditor)
  - Активные MoLE эксперты 
  - Рекрутированные матрицы (IDME)
  - Hankel novelty
  - Итоговое время мышления

Данные используются для self-learning.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional


class ThinkingLogger:
    """
    Структурированный лог одного цикла мышления.
    Сохраняется в data/thinking_logs/ для анализа и дообучения.
    """
    
    def __init__(self, log_dir: str = "data/thinking_logs"):
        self.log_dir = log_dir
        self.logger = logging.getLogger("Tars.ThinkingLog")
        self.current_session = None
        os.makedirs(log_dir, exist_ok=True)
    
    def start_session(self, query: str, task_type: str, p_threshold: float):
        """Начало нового цикла мышления."""
        self.current_session = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:200],  # обрезаем для экономии
            "task_type": task_type,
            "p_threshold": p_threshold,
            "steps": [],
            "matrices_used": 0,
            "matrices_expanded": 0,
            "experts_activated": [],
            "final_p": 0.0,
            "converged": False,
            "total_ms": 0.0,
            "hankel_collapses": 0,
            "response_quality": None,  # будет заполнено после feedback
        }
    
    def log_step(self, step: int, data: Dict[str, Any]):
        """Логирование одного шага мышления."""
        if self.current_session is None:
            return
        
        entry = {
            "step": step,
            "p": data.get("p", 0.0),
            "r_squared": data.get("r_squared", 0.0),
            "f_t": data.get("f_t", 0.0),
            "novelty": data.get("novelty", 1.0),
            "experts": data.get("experts", []),
            "matrices_recruited": data.get("matrices_recruited", 0),
            "converged": data.get("converged", False),
        }
        self.current_session["steps"].append(entry)
        
        # Обновляем агрегаты
        self.current_session["final_p"] = entry["p"]
        self.current_session["converged"] = entry["converged"]
    
    def end_session(self, total_ms: float, response: str):
        """Завершение цикла мышления."""
        if self.current_session is None:
            return
        
        self.current_session["total_ms"] = total_ms
        self.current_session["response_preview"] = response[:100]
        self.current_session["matrices_used"] = sum(
            s.get("matrices_recruited", 0) for s in self.current_session["steps"]
        ) + 12  # 12 основных блоков
        
        # Сохраняем
        filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        filepath = os.path.join(self.log_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save thinking log: {e}")
        
        self.current_session = None
    
    def record_feedback(self, quality: float):
        """
        Запись обратной связи (quality 0-1).
        Используется self-learning модулем.
        """
        # Обновляем последний лог
        logs = sorted(os.listdir(self.log_dir))
        if not logs:
            return
        
        latest = os.path.join(self.log_dir, logs[-1])
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data["response_quality"] = quality
            with open(latest, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    
    def get_training_data(self, min_quality: float = 0.7) -> List[Dict]:
        """
        Извлекает высококачественные сессии для self-learning.
        
        Returns:
            Список сессий с quality >= min_quality
        """
        good_sessions = []
        
        try:
            for fname in os.listdir(self.log_dir):
                if not fname.endswith('.json'):
                    continue
                filepath = os.path.join(self.log_dir, fname)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                quality = data.get("response_quality")
                if quality is not None and quality >= min_quality:
                    good_sessions.append(data)
        except Exception as e:
            self.logger.warning(f"Error reading training data: {e}")
        
        return good_sessions
