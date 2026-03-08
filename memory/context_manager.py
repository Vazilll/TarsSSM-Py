"""
context_manager.py — Sliding Window + Auto-Summary.

Управляет контекстом разговора в пределах токен-бюджета (8K max):
  - Хранит полную историю диалога
  - Sliding window: последние N реплик в полном виде
  - Auto-summary: при >75% бюджета — сжимает старые реплики
  - Сохраняет сжатые реплики в SDM (через store.py)

Связь с пайплайном:
  ContextManager → подаёт контекст в brain/omega_core/model.py
  Старые реплики → SDM (memory/store.py) для долговременного хранения

Не на hot path: вызывается 1 раз при каждом новом сообщении.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable

logger = logging.getLogger("Tars.ContextManager")


# ═══════════════════════════════════════
# Типы данных
# ═══════════════════════════════════════

@dataclass
class Turn:
    """Одна реплика в диалоге."""
    role: str       # "user", "assistant", "system"
    text: str       # содержимое реплики
    timestamp: float = 0.0
    token_count: int = 0   # оценочное кол-во токенов
    is_summary: bool = False  # True если это сжатая версия

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.token_count == 0:
            self.token_count = self._estimate_tokens(self.text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Грубая оценка токенов: ~1.3 символа/токен для RU, ~4 символа/токен для EN.
        Средний коэффициент: len / 2.5 (conservative для Qwen BPE).
        """
        if not text:
            return 0
        return max(1, len(text) // 3)


@dataclass
class ContextWindow:
    """Результат get_context(): набор реплик с метаданными."""
    turns: List[Turn]
    total_tokens: int
    budget_tokens: int
    usage_ratio: float       # total / budget
    summarized_count: int    # сколько реплик было суммаризовано


class ContextManager:
    """
    Управление контекстным окном для TARS.

    Алгоритм:
      1. Все новые реплики добавляются в self.history
      2. get_context() собирает контекст в пределах бюджета:
         - Системный промпт (всегда первый)
         - Последние recent_turns реплик полностью
         - Старые реплики — суммаризованы (если есть summarizer)
      3. Когда usage > summary_threshold → автоматическое сжатие

    Attributes:
        max_tokens: максимальный токен-бюджет (default=8192)
        recent_turns: кол-во последних реплик без сжатия (default=10)
        summary_threshold: порог для auto-summary (default=0.75)
        system_prompt: системный промпт (всегда в начале)
    """

    def __init__(self,
                 max_tokens: int = 8192,
                 recent_turns: int = 10,
                 summary_threshold: float = 0.75,
                 system_prompt: str = "",
                 summarizer: Optional[Callable[[List[Turn]], str]] = None,
                 sdm_callback: Optional[Callable[[str], None]] = None):
        self.max_tokens = max_tokens
        self.recent_turns = recent_turns
        self.summary_threshold = summary_threshold
        self.system_prompt = system_prompt
        self._summarizer = summarizer
        self._sdm_callback = sdm_callback  # callback для сохранения в SDM

        # Полная история
        self.history: List[Turn] = []

        # Хранилище суммари (сжатые блоки старых реплик)
        self._summaries: List[Turn] = []

        # Статистика
        self.total_turns_added = 0
        self.total_summarized = 0
        self.total_archived = 0

    def add_turn(self, role: str, text: str, timestamp: float = 0.0):
        """
        Добавить новую реплику в историю.

        Args:
            role: "user", "assistant", "system"
            text: текст реплики
            timestamp: Unix timestamp (0 = auto)
        """
        turn = Turn(role=role, text=text, timestamp=timestamp)
        self.history.append(turn)
        self.total_turns_added += 1

        # Проверяем, нужна ли автоматическая суммаризация
        usage = self._current_usage_ratio()
        if usage > self.summary_threshold:
            self._auto_summarize()

    def get_context(self, budget_tokens: Optional[int] = None) -> ContextWindow:
        """
        Собрать контекст для передачи в модель.

        Возвращает реплики в пределах бюджета:
          1. System prompt (если есть)
          2. Суммари старых реплик
          3. Последние recent_turns реплик полностью

        Args:
            budget_tokens: кастомный бюджет (default = self.max_tokens)

        Returns:
            ContextWindow с отобранными репликами
        """
        budget = budget_tokens or self.max_tokens
        turns_out: List[Turn] = []
        total = 0

        # 1. System prompt (всегда)
        if self.system_prompt:
            sys_turn = Turn(role="system", text=self.system_prompt)
            turns_out.append(sys_turn)
            total += sys_turn.token_count

        # 2. Суммари старых реплик
        for summary in self._summaries:
            if total + summary.token_count <= budget * 0.3:  # не более 30% бюджета
                turns_out.append(summary)
                total += summary.token_count

        # 3. Последние реплики (в полном виде)
        recent = self.history[-self.recent_turns:]
        for turn in recent:
            if total + turn.token_count <= budget:
                turns_out.append(turn)
                total += turn.token_count
            else:
                break

        return ContextWindow(
            turns=turns_out,
            total_tokens=total,
            budget_tokens=budget,
            usage_ratio=total / budget if budget > 0 else 0.0,
            summarized_count=self.total_summarized,
        )

    def _current_usage_ratio(self) -> float:
        """Текущий % использования бюджета."""
        total = sum(t.token_count for t in self.history)
        if self.system_prompt:
            total += Turn._estimate_tokens(self.system_prompt)
        return total / self.max_tokens if self.max_tokens > 0 else 0.0

    def _auto_summarize(self):
        """
        Автоматическая суммаризация старых реплик.

        Берёт первые (len - recent_turns) реплик из history,
        суммаризует их в 1-2 предложения, сохраняет в _summaries,
        архивирует в SDM, и удаляет из history.
        """
        if len(self.history) <= self.recent_turns:
            return  # нечего суммаризовать

        # Определяем, сколько старых реплик сжимать
        n_old = len(self.history) - self.recent_turns
        old_turns = self.history[:n_old]

        if not old_turns:
            return

        # Суммаризация
        summary_text = self._summarize_turns(old_turns)
        summary_turn = Turn(
            role="system",
            text=f"[Сводка предыдущих {len(old_turns)} реплик]: {summary_text}",
            is_summary=True,
        )
        self._summaries.append(summary_turn)
        self.total_summarized += len(old_turns)

        # Архивирование в SDM (if callback set)
        if self._sdm_callback:
            try:
                for turn in old_turns:
                    self._sdm_callback(f"[{turn.role}]: {turn.text}")
                    self.total_archived += 1
            except Exception as e:
                logger.warning(f"SDM archive failed: {e}")

        # Удаляем сжатые реплики из history
        self.history = self.history[n_old:]

        logger.info(
            f"ContextManager: summarized {len(old_turns)} turns → "
            f"{summary_turn.token_count} tokens. "
            f"History: {len(self.history)} turns remain."
        )

    def _summarize_turns(self, turns: List[Turn]) -> str:
        """
        Суммаризация списка реплик.

        Если есть neural summarizer — используем его.
        Иначе — extractive: первое предложение каждой реплики.
        """
        if self._summarizer:
            try:
                return self._summarizer(turns)
            except Exception as e:
                logger.warning(f"Neural summarizer failed, using extractive: {e}")

        # Extractive fallback: первые слова каждой реплики
        parts = []
        for turn in turns:
            snippet = turn.text[:100].strip()
            first_sentence = snippet.split('.')[0].strip()
            if first_sentence:
                parts.append(f"{turn.role}: {first_sentence}")

        # Ограничиваем суммари
        summary = "; ".join(parts[:8])
        if len(parts) > 8:
            summary += f" (+{len(parts) - 8} ещё)"
        return summary

    def clear(self):
        """Полная очистка контекста."""
        self.history.clear()
        self._summaries.clear()

    def get_stats(self) -> dict:
        """Статистика контекст-менеджера."""
        return {
            "history_turns": len(self.history),
            "summaries": len(self._summaries),
            "total_turns_added": self.total_turns_added,
            "total_summarized": self.total_summarized,
            "total_archived": self.total_archived,
            "current_tokens": sum(t.token_count for t in self.history),
            "max_tokens": self.max_tokens,
            "usage_ratio": self._current_usage_ratio(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cm = ContextManager(
        max_tokens=200,       # низкий для теста
        recent_turns=3,
        summary_threshold=0.75,
        system_prompt="Ты — TARS, AI ассистент.",
    )

    # Имитация диалога
    turns = [
        ("user", "Привет, как дела?"),
        ("assistant", "Отлично! Чем могу помочь?"),
        ("user", "Напиши функцию сортировки на Python"),
        ("assistant", "Конечно! def sort_list(lst): return sorted(lst)"),
        ("user", "А как оптимизировать её для больших данных?"),
        ("assistant", "Используй Timsort — он встроен в Python! O(n log n) worst case."),
        ("user", "Спасибо, а есть ещё советы?"),
        ("assistant", "Да, numpy.sort() для числовых данных в 10× быстрее."),
    ]

    for role, text in turns:
        cm.add_turn(role, text)
        ctx = cm.get_context()
        print(f"After '{text[:30]}...' → {ctx.total_tokens} tokens, "
              f"usage={ctx.usage_ratio:.0%}, turns={len(ctx.turns)}")

    print(f"\nStats: {cm.get_stats()}")
