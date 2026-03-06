"""Intent classification accuracy tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestIntentClassification:
    """Тесты классификации намерений."""

    def _classify(self, query):
        from agent.tars_agent import classify_intent
        return classify_intent(query)

    @pytest.mark.parametrize("query,expected", [
        ("найди информацию о Python", "search"),
        ("что такое нейросеть", "search"),
        ("запусти скрипт test.py", "execute"),
        ("установи numpy", "execute"),
        ("напиши функцию сортировки", "code"),
        ("создай класс для обработки данных", "code"),
        ("загугли последние новости", "web_search"),
        ("поищи в интернете рецепт", "web_search"),
        ("научись решать уравнения", "learn"),
        ("изучи этот документ", "learn"),
        ("запомни мой номер телефона", "remember"),
        ("не забудь купить молоко", "remember"),
        ("прочитай этот pdf файл", "doc_work"),
        ("создай excel таблицу", "doc_work"),
        ("проанализируй этот код", "analyze"),
        ("сравни два подхода", "analyze"),
        ("покажи статус системы", "status"),
        ("диагностика здоровья", "status"),
        ("открой файл config.json", "file_op"),
        ("скопируй папку backup", "file_op"),
        ("привет как дела", "chat"),
        ("расскажи анекдот", "chat"),
    ])
    def test_intent_accuracy(self, query, expected):
        result = self._classify(query)
        assert result == expected, f"'{query}' → got '{result}', expected '{expected}'"

    def test_doc_priority_over_search(self):
        assert self._classify("прочитай pdf документ") == "doc_work"
