"""
═══════════════════════════════════════════════════════════════
  learning_helper.py — Помощник в обучении для TARS v3
═══════════════════════════════════════════════════════════════

Помогает пользователю учиться, запоминать и закреплять знания:
  1. Adaptive Flashcards — карточки с интервальным повторением  
  2. Concept Tracker — отслеживание изученных тем
  3. Quiz Generator — генерация вопросов для самопроверки
  4. Progress Reporter — отчёт о прогрессе обучения
"""

import json
import os
import math
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("Tars.LearningHelper")

_ROOT = Path(__file__).parent.parent
_LEARNING_DB = _ROOT / "data" / "learning.json"


class Flashcard:
    """Карточка для интервального повторения (Spaced Repetition — SM-2 алгоритм)."""
    
    def __init__(self, question: str, answer: str, topic: str = "general"):
        self.question = question
        self.answer = answer
        self.topic = topic
        
        # SM-2 параметры
        self.easiness = 2.5      # Лёгкость (2.5 = начальное)
        self.interval = 1        # Интервал повторения (дни)
        self.repetitions = 0     # Количество успешных повторений
        self.next_review = datetime.now().isoformat()
        self.created = datetime.now().isoformat()
        self.last_reviewed = None
    
    def review(self, quality: int):
        """
        Обновляет интервал на основе качества ответа (SM-2).
        
        quality: 0-5 (0=полный провал, 5=идеально)
        """
        self.last_reviewed = datetime.now().isoformat()
        
        if quality >= 3:  # Успешно
            if self.repetitions == 0:
                self.interval = 1
            elif self.repetitions == 1:
                self.interval = 6
            else:
                self.interval = int(self.interval * self.easiness)
            self.repetitions += 1
        else:  # Провал — reset
            self.repetitions = 0
            self.interval = 1
        
        # Обновление easiness (E-Factor)
        self.easiness = max(1.3, self.easiness + 0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        
        # Следующее повторение
        self.next_review = (datetime.now() + timedelta(days=self.interval)).isoformat()
    
    def is_due(self) -> bool:
        """Пора ли повторять?"""
        return datetime.now() >= datetime.fromisoformat(self.next_review)
    
    def to_dict(self):
        return {
            "question": self.question, "answer": self.answer,
            "topic": self.topic, "easiness": self.easiness,
            "interval": self.interval, "repetitions": self.repetitions,
            "next_review": self.next_review, "created": self.created,
            "last_reviewed": self.last_reviewed,
        }
    
    @staticmethod
    def from_dict(d):
        fc = Flashcard(d["question"], d["answer"], d.get("topic", "general"))
        fc.easiness = d.get("easiness", 2.5)
        fc.interval = d.get("interval", 1)
        fc.repetitions = d.get("repetitions", 0)
        fc.next_review = d.get("next_review", datetime.now().isoformat())
        fc.created = d.get("created", datetime.now().isoformat())
        fc.last_reviewed = d.get("last_reviewed")
        return fc


class ConceptTracker:
    """Отслеживание изученных тем и уровня уверенности."""
    
    def __init__(self):
        self.concepts: Dict[str, Dict] = {}  # topic → {level, last_seen, times_asked, ...}
    
    def track(self, topic: str, success: bool = True):
        """Обновить прогресс по теме."""
        if topic not in self.concepts:
            self.concepts[topic] = {
                "level": 0.0,         # 0-1 mastery level
                "times_asked": 0,
                "times_correct": 0,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
            }
        
        c = self.concepts[topic]
        c["times_asked"] += 1
        if success:
            c["times_correct"] += 1
        c["level"] = c["times_correct"] / c["times_asked"]
        c["last_seen"] = datetime.now().isoformat()
    
    def get_weak_topics(self, threshold: float = 0.6) -> List[str]:
        """Темы в которых пользователь слаб."""
        return [t for t, c in self.concepts.items() if c["level"] < threshold]
    
    def get_strong_topics(self, threshold: float = 0.8) -> List[str]:
        """Темы которые пользователь хорошо знает."""
        return [t for t, c in self.concepts.items() if c["level"] >= threshold]
    
    def get_report(self) -> str:
        """Текстовый отчёт о прогрессе."""
        if not self.concepts:
            return "Пока нет данных об обучении."
        
        lines = ["📊 Прогресс обучения:\n"]
        
        sorted_topics = sorted(self.concepts.items(), key=lambda x: x[1]["level"])
        
        for topic, data in sorted_topics:
            level = data["level"]
            bar_len = int(level * 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            emoji = "🟢" if level >= 0.8 else "🟡" if level >= 0.5 else "🔴"
            lines.append(f"  {emoji} {topic}: [{bar}] {level:.0%}")
        
        weak = self.get_weak_topics()
        if weak:
            lines.append(f"\n⚠️ Нужно подтянуть: {', '.join(weak)}")
        
        return "\n".join(lines)
    
    def to_dict(self):
        return self.concepts
    
    def from_dict(self, d):
        self.concepts = d or {}


class LearningHelper:
    """
    Основной класс помощника в обучении.
    
    Интегрируется с GIE:
      - Ловит вопросы "объясни...", "что такое..." → создаёт flashcards
      - Предлагает повторение по расписанию
      - Генерирует мини-квизы  
      - Отслеживает прогресс по темам
    """
    
    def __init__(self):
        self.flashcards: List[Flashcard] = []
        self.tracker = ConceptTracker()
        self._load()
    
    def add_card(self, question: str, answer: str, topic: str = "general") -> str:
        """Добавить карточку для повторения."""
        # Проверка дубликатов
        for fc in self.flashcards:
            if fc.question.lower() == question.lower():
                return f"Карточка уже существует: {question[:50]}"
        
        fc = Flashcard(question, answer, topic)
        self.flashcards.append(fc)
        self.tracker.track(topic, success=True)
        self._save()
        return f"✅ Карточка добавлена ({topic}). Всего: {len(self.flashcards)}"
    
    def auto_create_card(self, user_question: str, tars_answer: str):
        """
        Автоматически создаёт flashcard из диалога.
        Вызывается когда пользователь спрашивает "объясни...", "что такое..." и т.д.
        """
        # Определяем тему из вопроса
        topic = self._extract_topic(user_question)
        
        # Обрезаем ответ до ключевого
        short_answer = tars_answer[:300]
        if len(tars_answer) > 300:
            short_answer += "..."
        
        self.add_card(user_question, short_answer, topic)
        self.tracker.track(topic)
        logger.info(f"LearningHelper: авто-карточка [{topic}]: {user_question[:40]}...")
    
    def get_due_cards(self) -> List[Flashcard]:
        """Карточки готовые к повторению."""
        return [fc for fc in self.flashcards if fc.is_due()]
    
    def review_card(self, card_index: int, quality: int) -> str:
        """
        Отметить повторение карточки.
        quality: 0-5 (0=забыл, 3=вспомнил с трудом, 5=идеально)
        """
        if 0 <= card_index < len(self.flashcards):
            fc = self.flashcards[card_index]
            fc.review(quality)
            self.tracker.track(fc.topic, success=quality >= 3)
            self._save()
            
            next_days = fc.interval
            return f"Следующее повторение через {next_days} {'день' if next_days == 1 else 'дней'}"
        return "Карточка не найдена"
    
    def get_review_prompt(self) -> Optional[str]:
        """
        Генерирует промпт для повторения (вызывается при взаимодействии с ТАРС).
        """
        due = self.get_due_cards()
        if not due:
            return None
        
        card = due[0]
        return (
            f"📝 Время для повторения! ({len(due)} карточек ждут)\n\n"
            f"Вопрос: **{card.question}**\n\n"
            f"Попробуй ответить, а потом скажи «покажи ответ»."
        )
    
    def _extract_topic(self, text: str) -> str:
        """Извлекает тему из вопроса."""
        text_lower = text.lower()
        
        topic_keywords = {
            "python": ["python", "питон", "пайтон", "pip"],
            "math": ["математика", "формула", "уравнение", "производная", "интеграл"],
            "algorithms": ["алгоритм", "сортировка", "поиск", "граф", "структура данных"],
            "networks": ["сеть", "tcp", "ip", "http", "dns", "протокол"],
            "ml": ["нейросеть", "обучение", "модель", "gradient", "loss", "optimizer"],
            "linux": ["linux", "bash", "terminal", "команда"],
            "git": ["git", "commit", "push", "branch", "merge"],
            "web": ["html", "css", "javascript", "react", "frontend"],
            "db": ["sql", "database", "база данных", "таблица", "запрос"],
        }
        
        for topic, keywords in topic_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return topic
        
        return "general"
    
    def should_create_card(self, user_msg: str) -> bool:
        """Нужно ли создавать карточку из этого вопроса?"""
        triggers = [
            "объясни", "что такое", "как работает", "расскажи про",
            "в чём разница", "зачем нужен", "как сделать",
            "explain", "what is", "how does", "how to",
        ]
        text_lower = user_msg.lower()
        return any(t in text_lower for t in triggers)
    
    def get_progress(self) -> str:
        """Полный отчёт о прогрессе."""
        report = self.tracker.get_report()
        
        due = len(self.get_due_cards())
        total = len(self.flashcards)
        mastered = sum(1 for fc in self.flashcards if fc.repetitions >= 5)
        
        report += f"\n\n📚 Карточки: {total} всего, {mastered} освоено, {due} к повторению"
        
        return report
    
    def _save(self):
        """Сохранение состояния."""
        _LEARNING_DB.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "flashcards": [fc.to_dict() for fc in self.flashcards],
            "concepts": self.tracker.to_dict(),
        }
        try:
            with open(_LEARNING_DB, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"LearningHelper save error: {e}")
    
    def _load(self):
        """Загрузка состояния."""
        if _LEARNING_DB.exists():
            try:
                with open(_LEARNING_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.flashcards = [Flashcard.from_dict(d) for d in data.get("flashcards", [])]
                self.tracker.from_dict(data.get("concepts", {}))
                logger.info(f"LearningHelper: {len(self.flashcards)} карточек, "
                           f"{len(self.tracker.concepts)} тем загружено")
            except Exception as e:
                logger.warning(f"LearningHelper load error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    helper = LearningHelper()
    
    # Тест: авто-создание карточек
    helper.auto_create_card(
        "Объясни что такое рекурсия",
        "Рекурсия — когда функция вызывает саму себя..."
    )
    helper.auto_create_card(
        "Как работает сортировка пузырьком?",
        "Сортировка пузырьком проходит по массиву и меняет соседние элементы..."
    )
    
    # Повторение
    due = helper.get_due_cards()
    print(f"\nКарточек к повторению: {len(due)}")
    for fc in due:
        print(f"  Q: {fc.question}")
        print(f"  A: {fc.answer[:80]}...")
    
    # Прогресс
    print(f"\n{helper.get_progress()}")
