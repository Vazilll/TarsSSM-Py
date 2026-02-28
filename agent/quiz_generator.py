"""
═══════════════════════════════════════════════════════════════
  quiz_generator.py — Генератор тестов и квизов TARS v3
═══════════════════════════════════════════════════════════════

Генерирует тесты из:
  - Flashcards (learning_helper)
  - Конспектов лекций (lecture_summarizer)
  - Графа знаний (knowledge_graph)
  - Произвольного текста

Типы вопросов:
  - Multiple Choice (4 варианта)
  - True/False
  - Fill-in-blank (заполни пропуск)
  - Open question (свободный ответ)

Адаптивная сложность: больше вопросов по слабым темам.
"""

import json
import random
import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger("Tars.QuizGenerator")

_ROOT = Path(__file__).parent.parent
_QUIZ_DB = _ROOT / "data" / "quiz_history.json"


class Question:
    """Один вопрос теста."""
    def __init__(self, text: str, answer: str, q_type: str = "open",
                 options: List[str] = None, topic: str = "general",
                 difficulty: int = 1):
        self.text = text
        self.answer = answer
        self.q_type = q_type  # multiple_choice, true_false, fill_blank, open
        self.options = options or []
        self.topic = topic
        self.difficulty = difficulty  # 1-3
    
    def to_dict(self):
        return {
            "text": self.text, "answer": self.answer,
            "type": self.q_type, "options": self.options,
            "topic": self.topic, "difficulty": self.difficulty,
        }
    
    def format(self, show_answer: bool = False) -> str:
        lines = [f"❓ {self.text}"]
        if self.q_type == "multiple_choice" and self.options:
            for i, opt in enumerate(self.options):
                letter = chr(65 + i)  # A, B, C, D
                lines.append(f"  {letter}) {opt}")
        if show_answer:
            lines.append(f"  ✅ Ответ: {self.answer}")
        return "\n".join(lines)


class QuizResult:
    """Результат теста."""
    def __init__(self, topic: str, total: int, correct: int):
        self.topic = topic
        self.total = total
        self.correct = correct
        self.date = datetime.now().isoformat()
        self.score = correct / total * 100 if total > 0 else 0
        self.details: List[Dict] = []


class QuizGenerator:
    """
    Генератор адаптивных тестов.
    
    Интеграции:
      - LearningHelper → берёт flashcards по теме
      - LectureSummarizer → извлекает факты из конспектов
      - KnowledgeGraph → ищет связанные узлы
      - ConceptTracker → адаптивная сложность по слабым темам
    """
    
    def __init__(self, learning_helper=None, knowledge_graph=None):
        self.learning_helper = learning_helper
        self.knowledge_graph = knowledge_graph
        self.history: List[Dict] = []
        self.current_quiz: Optional[List[Question]] = []
        self.current_index = 0
        self.current_score = 0
        self._load()
    
    def generate_quiz(self, topic: str = None, n_questions: int = 10,
                      source: str = "auto") -> str:
        """
        Сгенерировать тест.
        
        topic: тема (None = все темы)
        n_questions: количество вопросов
        source: "flashcards", "knowledge", "auto"
        """
        questions = []
        
        # 1. Из flashcards 
        if self.learning_helper and source in ("flashcards", "auto"):
            cards = self.learning_helper.flashcards
            if topic:
                cards = [c for c in cards if topic.lower() in c.topic.lower() 
                        or topic.lower() in c.question.lower()]
            
            for card in cards:
                # Multiple choice из flashcard
                q = self._card_to_question(card, cards)
                if q:
                    questions.append(q)
                
                # Fill-in-blank
                q2 = self._card_to_fill_blank(card)
                if q2:
                    questions.append(q2)
        
        # 2. Из графа знаний
        if self.knowledge_graph and source in ("knowledge", "auto"):
            kg_questions = self._from_knowledge_graph(topic, n_questions // 2)
            questions.extend(kg_questions)
        
        if not questions:
            return (
                f"❌ Недостаточно материала для теста"
                f"{f' по теме «{topic}»' if topic else ''}.\n"
                f"Подсказка: сначала добавь карточки или заметки в граф знаний."
            )
        
        # Перемешать и ограничить
        random.shuffle(questions)
        self.current_quiz = questions[:n_questions]
        self.current_index = 0
        self.current_score = 0
        
        topic_str = f" по теме «{topic}»" if topic else ""
        return (
            f"📝 Тест{topic_str}: {len(self.current_quiz)} вопросов\n\n"
            f"{self.current_quiz[0].format()}\n\n"
            f"Ответь на вопрос (1/{len(self.current_quiz)})"
        )
    
    def answer(self, user_answer: str) -> str:
        """Ответ на текущий вопрос."""
        if not self.current_quiz or self.current_index >= len(self.current_quiz):
            return "⚠️ Нет активного теста. Скажи «тест [тема]» чтобы начать."
        
        q = self.current_quiz[self.current_index]
        is_correct = self._check_answer(q, user_answer)
        
        if is_correct:
            self.current_score += 1
            feedback = "✅ Правильно!"
        else:
            feedback = f"❌ Неправильно. Ответ: {q.answer}"
        
        # Обновить ConceptTracker
        if self.learning_helper:
            self.learning_helper.tracker.track(q.topic, success=is_correct)
        
        self.current_index += 1
        
        # Следующий вопрос или итоги
        if self.current_index >= len(self.current_quiz):
            result = self._finish_quiz()
            return f"{feedback}\n\n{result}"
        else:
            next_q = self.current_quiz[self.current_index]
            return (
                f"{feedback}\n\n"
                f"{next_q.format()}\n"
                f"({self.current_index + 1}/{len(self.current_quiz)})"
            )
    
    def get_weak_topics_quiz(self, n: int = 10) -> str:
        """Тест по слабым темам (адаптивный)."""
        if not self.learning_helper:
            return self.generate_quiz(n_questions=n)
        
        weak = self.learning_helper.tracker.get_weak_topics(threshold=0.7)
        if not weak:
            return "🎉 Нет слабых тем! Попробуй тест по всему материалу."
        
        topic = random.choice(weak)
        return self.generate_quiz(topic=topic, n_questions=n)
    
    def _card_to_question(self, card, all_cards) -> Optional[Question]:
        """Flashcard → Multiple Choice вопрос."""
        correct = card.answer[:100]
        
        # Генерируем неправильные ответы из других карточек
        wrong_answers = []
        for other in all_cards:
            if other.question != card.question and len(wrong_answers) < 3:
                wrong_answers.append(other.answer[:100])
        
        if len(wrong_answers) < 2:
            return None
        
        wrong_answers = wrong_answers[:3]
        options = wrong_answers + [correct]
        random.shuffle(options)
        
        correct_letter = chr(65 + options.index(correct))
        
        return Question(
            text=card.question,
            answer=correct_letter,
            q_type="multiple_choice",
            options=options,
            topic=card.topic,
        )
    
    def _card_to_fill_blank(self, card) -> Optional[Question]:
        """Flashcard → Fill-in-blank."""
        answer_text = card.answer
        words = answer_text.split()
        
        if len(words) < 3:
            return None
        
        # Убираем ключевое слово
        key_words = [w for w in words if len(w) > 4 and w.isalpha()]
        if not key_words:
            return None
        
        blank_word = random.choice(key_words[:3])
        blanked = answer_text.replace(blank_word, "______", 1)
        
        return Question(
            text=f"{card.question}\n  Заполни пропуск: {blanked}",
            answer=blank_word,
            q_type="fill_blank",
            topic=card.topic,
        )
    
    def _from_knowledge_graph(self, topic: str, n: int) -> List[Question]:
        """Генерация вопросов из графа знаний."""
        questions = []
        
        try:
            if topic:
                node_ids = self.knowledge_graph._find_by_keyword(topic)
            else:
                nodes = self.knowledge_graph._load_all_nodes(limit=50)
                node_ids = [n.id for n in nodes]
            
            for node_id in node_ids[:n * 2]:
                node = self.knowledge_graph._load_node(node_id)
                if node and len(node.content) > 30:
                    # True/False вопрос
                    sentences = [s.strip() for s in node.content.split('.') 
                                if len(s.strip()) > 20]
                    if sentences:
                        sent = random.choice(sentences[:3])
                        questions.append(Question(
                            text=f"Верно ли: «{sent[:120]}»?",
                            answer="Да",
                            q_type="true_false",
                            options=["Да", "Нет"],
                            topic=node.node_type,
                        ))
                    
                    # Open question
                    if len(node.title) > 5:
                        questions.append(Question(
                            text=f"Расскажи что ты знаешь про: {node.title}",
                            answer=node.content[:200],
                            q_type="open",
                            topic=node.node_type,
                        ))
        except Exception as e:
            logger.debug(f"KG quiz error: {e}")
        
        return questions
    
    def _check_answer(self, question: Question, user_answer: str) -> bool:
        """Проверка ответа."""
        answer = question.answer.lower().strip()
        user = user_answer.lower().strip()
        
        if question.q_type == "multiple_choice":
            return user in answer or user == answer
        
        if question.q_type == "true_false":
            yes_words = {"да", "верно", "true", "yes", "правда"}
            no_words = {"нет", "неверно", "false", "no", "ложь"}
            if answer in ("да", "true"):
                return user in yes_words
            return user in no_words
        
        if question.q_type == "fill_blank":
            return answer in user or user in answer
        
        # Open: хотя бы 30% слов содержится
        answer_words = set(answer.split())
        user_words = set(user.split())
        if not answer_words:
            return False
        overlap = len(answer_words & user_words) / len(answer_words)
        return overlap >= 0.3
    
    def _finish_quiz(self) -> str:
        """Итоги теста."""
        total = len(self.current_quiz)
        correct = self.current_score
        pct = correct / total * 100 if total > 0 else 0
        
        # Оценка
        if pct >= 90: grade, emoji = "Отлично!", "🏆"
        elif pct >= 75: grade, emoji = "Хорошо", "👍"
        elif pct >= 60: grade, emoji = "Удовлетворительно", "📚"
        else: grade, emoji = "Нужно подтянуть", "💪"
        
        # Сохранить результат
        result = {
            "date": datetime.now().isoformat(),
            "total": total, "correct": correct, "score": pct,
        }
        self.history.append(result)
        self._save()
        
        lines = [
            f"\n{'='*40}",
            f"{emoji} Результат: {correct}/{total} ({pct:.0f}%) — {grade}",
            f"{'='*40}",
        ]
        
        # Прогресс
        if len(self.history) > 1:
            prev = self.history[-2]["score"]
            diff = pct - prev
            if diff > 0:
                lines.append(f"📈 Прогресс: +{diff:.0f}% по сравнению с прошлым тестом")
            elif diff < 0:
                lines.append(f"📉 Регресс: {diff:.0f}% — нужно повторить материал")
        
        self.current_quiz = []
        return "\n".join(lines)
    
    def _save(self):
        _QUIZ_DB.parent.mkdir(parents=True, exist_ok=True)
        with open(_QUIZ_DB, "w", encoding="utf-8") as f:
            json.dump(self.history[-500:], f, ensure_ascii=False, indent=2)
    
    def _load(self):
        if _QUIZ_DB.exists():
            try:
                with open(_QUIZ_DB, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception:
                pass
