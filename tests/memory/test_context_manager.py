"""Tests for ContextManager (sliding window + auto-summary)."""
import pytest
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestContextManager:
    """Sliding window + auto-summary tests."""

    @pytest.fixture
    def cm(self):
        from memory.context_manager import ContextManager
        return ContextManager(
            max_tokens=500,
            recent_turns=3,
            summary_threshold=0.75,
            system_prompt="Ты — TARS.",
        )

    def test_add_turn(self, cm):
        cm.add_turn("user", "Привет!")
        assert len(cm.history) == 1
        assert cm.history[0].role == "user"
        assert cm.history[0].text == "Привет!"

    def test_get_context_includes_system_prompt(self, cm):
        cm.add_turn("user", "Привет!")
        ctx = cm.get_context()
        assert any(t.role == "system" and "TARS" in t.text for t in ctx.turns)

    def test_get_context_returns_context_window(self, cm):
        from memory.context_manager import ContextWindow
        cm.add_turn("user", "Тест")
        ctx = cm.get_context()
        assert isinstance(ctx, ContextWindow)
        assert ctx.total_tokens > 0
        assert ctx.budget_tokens == 500

    def test_context_respects_budget(self, cm):
        for i in range(50):
            cm.add_turn("user", f"Длинное сообщение номер {i} " * 10)
        ctx = cm.get_context()
        assert ctx.total_tokens <= ctx.budget_tokens

    def test_auto_summary_triggers(self):
        from memory.context_manager import ContextManager
        cm = ContextManager(
            max_tokens=100,  # очень низкий бюджет
            recent_turns=3,
            summary_threshold=0.5,
        )
        for i in range(20):
            cm.add_turn("user", f"Сообщение {i}: " + "слово " * 20)
        # После множества сообщений должна быть суммаризация
        assert cm.total_summarized > 0

    def test_recent_turns_preserved(self, cm):
        for i in range(10):
            cm.add_turn("user", f"Сообщение {i}")
        # Последние 3 должны быть в истории
        ctx = cm.get_context()
        texts = [t.text for t in ctx.turns if t.role == "user"]
        assert len(texts) <= cm.recent_turns + 1

    def test_clear(self, cm):
        cm.add_turn("user", "Тест")
        cm.clear()
        assert len(cm.history) == 0
        assert len(cm._summaries) == 0

    def test_get_stats(self, cm):
        cm.add_turn("user", "Тест")
        stats = cm.get_stats()
        assert "history_turns" in stats
        assert "total_turns_added" in stats
        assert stats["total_turns_added"] == 1
        assert stats["max_tokens"] == 500

    def test_sdm_callback_called(self):
        from memory.context_manager import ContextManager
        archived = []
        cm = ContextManager(
            max_tokens=100,
            recent_turns=2,
            summary_threshold=0.5,
            sdm_callback=lambda text: archived.append(text),
        )
        for i in range(20):
            cm.add_turn("user", f"Сообщение {i}: " + "текст " * 15)
        assert len(archived) > 0

    def test_turn_token_estimate(self):
        from memory.context_manager import Turn
        t = Turn(role="user", text="Hello world test")
        assert t.token_count > 0

    def test_usage_ratio_increases(self, cm):
        ratio_before = cm._current_usage_ratio()
        cm.add_turn("user", "довольно длинный текст " * 20)
        ratio_after = cm._current_usage_ratio()
        assert ratio_after > ratio_before

    def test_custom_budget_in_get_context(self, cm):
        cm.add_turn("user", "Тест")
        ctx = cm.get_context(budget_tokens=200)
        assert ctx.budget_tokens == 200
