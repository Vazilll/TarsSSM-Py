"""
═══════════════════════════════════════════════════════════════
  routine_detector.py — Proactive Routine Detector для TARS v3
═══════════════════════════════════════════════════════════════

Наблюдает за паттернами пользователя и САМА предлагает автоматизацию.

Как это работает:
  1. Логирует ВСЕ действия пользователя (время, команда, контекст)
  2. Обнаруживает повторяющиеся паттерны (≥3 раза за неделю)
  3. Предлагает автоматизацию или выполняет сама (если разрешено)

Примеры:
  - "Ты каждое утро в 9:00 открываешь Chrome → Gmail → Slack. Автоматизировать?"
  - "За последнюю неделю ты 5 раз искал ошибки в логах. Создать скрипт?"
  - "Ты всегда делаешь бэкап проекта перед коммитом. Настроить авто-бэкап?"
"""

import json
import os
import logging
import time
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("Tars.RoutineDetector")

_ROOT = Path(__file__).parent.parent
_ROUTINE_DB = _ROOT / "data" / "routines.json"


class ActionLog:
    """Единичное действие пользователя."""
    def __init__(self, action: str, context: str = "", timestamp: str = None):
        self.action = action
        self.context = context
        self.timestamp = timestamp or datetime.now().isoformat()
        self.hour = datetime.fromisoformat(self.timestamp).hour
        self.weekday = datetime.fromisoformat(self.timestamp).weekday()
    
    def to_dict(self):
        return {"action": self.action, "context": self.context, "time": self.timestamp}
    
    @staticmethod
    def from_dict(d):
        return ActionLog(d["action"], d.get("context", ""), d.get("time"))


class RoutinePattern:
    """Обнаруженный паттерн рутины."""
    def __init__(self, actions: List[str], frequency: int, 
                 time_pattern: Optional[str] = None,
                 confidence: float = 0.0):
        self.actions = actions          # Последовательность действий
        self.frequency = frequency      # Сколько раз обнаружено
        self.time_pattern = time_pattern  # "утро 9:00" / "вечер" / None
        self.confidence = confidence    # 0-1
        self.automation_script = None   # Предложенный скрипт
        self.approved = False           # Пользователь одобрил?
    
    def describe(self) -> str:
        actions_str = " → ".join(self.actions[:5])
        time_str = f" ({self.time_pattern})" if self.time_pattern else ""
        return f"Паттерн{time_str}: {actions_str} [×{self.frequency}, уверенность {self.confidence:.0%}]"


class RoutineDetector:
    """
    Детектор рутин — наблюдает за действиями и находит паттерны.
    
    Использует:
      - N-gram анализ действий (bigrams, trigrams)
      - Временные паттерны (утро/вечер, день недели)
      - Частотный анализ с confidence scoring
    """
    
    def __init__(self, min_frequency: int = 3, lookback_days: int = 7):
        self.min_frequency = min_frequency
        self.lookback_days = lookback_days
        self.action_log: List[ActionLog] = []
        self.patterns: List[RoutinePattern] = []
        self.approved_automations: List[Dict] = []
        
        self._load()
    
    def log_action(self, action: str, context: str = ""):
        """Логирует действие пользователя."""
        entry = ActionLog(action, context)
        self.action_log.append(entry)
        
        # Каждые 10 действий проверяем паттерны
        if len(self.action_log) % 10 == 0:
            self._detect_patterns()
        
        self._save()
    
    def log_conversation(self, user_msg: str, tars_response: str, tier: str = "brain"):
        """Логирует диалог как действие для паттерн-анализа."""
        # Извлекаем ключевое действие из сообщения
        action = self._extract_action(user_msg)
        context = f"tier={tier}, response_len={len(tars_response)}"
        self.log_action(action, context)
    
    def _extract_action(self, text: str) -> str:
        """Извлекает ключевое действие из текста."""
        text_lower = text.lower().strip()
        
        # Категоризация по ключевым словам
        categories = {
            "search": ["найди", "поиск", "ищи", "search", "google"],
            "open": ["открой", "запусти", "open", "launch"],
            "code": ["напиши", "код", "python", "script", "функцию", "скрипт"],
            "explain": ["объясни", "расскажи", "что такое", "как работает"],
            "file": ["файл", "папка", "folder", "directory", "создай файл"],
            "remember": ["запомни", "напомни", "remember", "remind"],
            "automate": ["автоматизируй", "сделай так чтобы", "настрой"],
            "debug": ["ошибка", "баг", "error", "не работает", "fix"],
            "translate": ["переведи", "translate", "на английский", "на русский"],
        }
        
        for category, keywords in categories.items():
            for kw in keywords:
                if kw in text_lower:
                    return f"{category}:{text_lower[:50]}"
        
        return f"other:{text_lower[:50]}"
    
    def _detect_patterns(self):
        """Обнаружение повторяющихся паттернов."""
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        recent = [a for a in self.action_log 
                  if datetime.fromisoformat(a.timestamp) > cutoff]
        
        if len(recent) < self.min_frequency:
            return
        
        self.patterns = []
        
        # 1. Простая частота действий
        action_counts = Counter(a.action.split(":")[0] for a in recent)
        for action, count in action_counts.most_common(10):
            if count >= self.min_frequency:
                # Временной паттерн
                hours = [a.hour for a in recent if a.action.startswith(action)]
                time_pattern = self._detect_time_pattern(hours)
                
                self.patterns.append(RoutinePattern(
                    actions=[action],
                    frequency=count,
                    time_pattern=time_pattern,
                    confidence=min(1.0, count / (self.min_frequency * 3))
                ))
        
        # 2. Bigrams (пары действий)
        if len(recent) >= 2:
            bigrams = [(recent[i].action.split(":")[0], recent[i+1].action.split(":")[0]) 
                       for i in range(len(recent) - 1)]
            bigram_counts = Counter(bigrams)
            for (a1, a2), count in bigram_counts.most_common(5):
                if count >= self.min_frequency and a1 != a2:
                    self.patterns.append(RoutinePattern(
                        actions=[a1, a2],
                        frequency=count,
                        confidence=min(1.0, count / (self.min_frequency * 2))
                    ))
        
        # 3. Trigrams
        if len(recent) >= 3:
            trigrams = [(recent[i].action.split(":")[0], 
                        recent[i+1].action.split(":")[0],
                        recent[i+2].action.split(":")[0]) 
                       for i in range(len(recent) - 2)]
            trigram_counts = Counter(trigrams)
            for (a1, a2, a3), count in trigram_counts.most_common(3):
                if count >= self.min_frequency:
                    self.patterns.append(RoutinePattern(
                        actions=[a1, a2, a3],
                        frequency=count,
                        confidence=min(1.0, count / (self.min_frequency * 1.5))
                    ))
        
        if self.patterns:
            logger.info(f"RoutineDetector: обнаружено {len(self.patterns)} паттернов")
    
    def _detect_time_pattern(self, hours: List[int]) -> Optional[str]:
        """Определяет временной паттерн из списка часов."""
        if not hours:
            return None
        
        avg_hour = sum(hours) / len(hours)
        std_hour = (sum((h - avg_hour)**2 for h in hours) / len(hours)) ** 0.5
        
        # Если стандартное отклонение < 2 часов → стабильный паттерн
        if std_hour < 2.0:
            h = int(avg_hour)
            if 5 <= h < 12:
                return f"утро ~{h}:00"
            elif 12 <= h < 17:
                return f"день ~{h}:00"
            elif 17 <= h < 22:
                return f"вечер ~{h}:00"
            else:
                return f"ночь ~{h}:00"
        return None
    
    def get_suggestions(self) -> List[str]:
        """Возвращает предложения по автоматизации."""
        suggestions = []
        for p in self.patterns:
            if p.confidence >= 0.5 and not p.approved:
                suggestions.append(p.describe())
        return suggestions
    
    def get_proactive_message(self) -> Optional[str]:
        """
        Генерирует проактивное сообщение если обнаружен сильный паттерн.
        Вызывается при каждом обращении к ТАРС.
        """
        high_conf = [p for p in self.patterns if p.confidence >= 0.7 and not p.approved]
        
        if not high_conf:
            return None
        
        best = max(high_conf, key=lambda p: p.confidence)
        
        actions_str = " → ".join(best.actions)
        time_str = f" {best.time_pattern}" if best.time_pattern else ""
        
        return (
            f"💡 Я заметил: ты часто делаешь {actions_str}{time_str} "
            f"(уже {best.frequency} раз). Хочешь чтобы я автоматизировал это?"
        )
    
    def approve_pattern(self, index: int):
        """Одобрить паттерн для автоматизации."""
        if 0 <= index < len(self.patterns):
            self.patterns[index].approved = True
            self.approved_automations.append({
                "actions": self.patterns[index].actions,
                "time": self.patterns[index].time_pattern,
                "approved_at": datetime.now().isoformat(),
            })
            self._save()
    
    def _save(self):
        """Сохранение в файл."""
        _ROUTINE_DB.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "log": [a.to_dict() for a in self.action_log[-5000:]],
            "approved": self.approved_automations,
        }
        try:
            with open(_ROUTINE_DB, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"RoutineDetector save error: {e}")
    
    def _load(self):
        """Загрузка из файла."""
        if _ROUTINE_DB.exists():
            try:
                with open(_ROUTINE_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.action_log = [ActionLog.from_dict(d) for d in data.get("log", [])]
                self.approved_automations = data.get("approved", [])
                logger.info(f"RoutineDetector: загружено {len(self.action_log)} действий")
                self._detect_patterns()
            except Exception as e:
                logger.warning(f"RoutineDetector load error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    rd = RoutineDetector(min_frequency=2)
    
    # Симуляция паттерна
    for _ in range(5):
        rd.log_action("open:chrome")
        rd.log_action("search:python tutorial")
        rd.log_action("code:write function")
    
    print("\nОбнаруженные паттерны:")
    for s in rd.get_suggestions():
        print(f"  {s}")
    
    msg = rd.get_proactive_message()
    if msg:
        print(f"\n{msg}")
