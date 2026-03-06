"""Auto-TZ generation tests."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAutoTZ:
    """Тесты генерации ТЗ."""

    def _generate_tz(self, query):
        from agent.tars_agent import _auto_generate_tz
        return _auto_generate_tz(query)

    def test_file_entities_extracted(self):
        combined = " ".join(self._generate_tz("прочитай файл data.csv и config.json"))
        assert "data.csv" in combined and "config.json" in combined

    def test_code_queries_have_edge_cases(self):
        combined = " ".join(self._generate_tz("напиши функцию для сортировки списка"))
        assert "граничн" in combined.lower() or "edge" in combined.lower()

    def test_min_3_points(self):
        assert len(self._generate_tz("привет")) >= 3
