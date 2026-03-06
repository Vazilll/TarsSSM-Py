"""T13: AST hardening — f-string, starred, global/nonlocal blocking."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestASTHardening:
    """T13: f-string, starred, global/nonlocal AST blocking."""

    @pytest.fixture
    def engine(self):
        from agent.executor import ActionEngine
        return ActionEngine()

    def test_fstring_blocked(self, engine):
        result = engine._safe_execute_script({"code": "x = f'{1+1}'"})
        assert "Error" in result or "not allowed" in result

    def test_starred_blocked(self, engine):
        result = engine._safe_execute_script({"code": "a, *b = [1,2,3]"})
        assert "Error" in result or "not allowed" in result

    def test_global_blocked(self, engine):
        result = engine._safe_execute_script({"code": "def f():\n  global x\n  x = 1"})
        assert "Error" in result or "not allowed" in result

    def test_nonlocal_blocked(self, engine):
        result = engine._safe_execute_script(
            {"code": "def f():\n  x = 1\n  def g():\n    nonlocal x\n    x = 2"}
        )
        assert "Error" in result or "not allowed" in result
