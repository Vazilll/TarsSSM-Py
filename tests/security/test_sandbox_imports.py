"""Sandbox import/eval/exec/dunder blocking tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestSandboxBlocking:
    """ActionEngine: import/eval/exec/dunder blocking in sandbox."""

    @pytest.fixture
    def engine(self):
        from agent.executor import ActionEngine
        return ActionEngine()

    def test_blocks_os_import(self, engine):
        result = engine._safe_execute_script({"code": "import os\nos.system('echo hacked')"})
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_blocks_subprocess_import(self, engine):
        result = engine._safe_execute_script({"code": "import subprocess\nsubprocess.run(['ls'])"})
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_blocks_eval(self, engine):
        result = engine._safe_execute_script({"code": "eval('1+1')"})
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_blocks_exec(self, engine):
        result = engine._safe_execute_script({"code": "exec('print(1)')"})
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_blocks_open(self, engine):
        result = engine._safe_execute_script({"code": "f = open('test.txt', 'w')"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_getattr(self, engine):
        result = engine._safe_execute_script({"code": "x = getattr(int, '__class__')"})
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_blocks_dunder_access(self, engine):
        result = engine._safe_execute_script({"code": "x = ().__class__.__bases__[0]"})
        assert "not allowed" in result.lower() or "error" in result.lower()

    def test_blocks_string_dunder_subscript(self, engine):
        result = engine._safe_execute_script({"code": "d = {}; x = d['__class__']"})
        assert "blocked" in result.lower() or "error" in result.lower()

    def test_allows_safe_math(self, engine):
        result = engine._safe_execute_script({"code": "import math\nprint(math.pi)"})
        assert "3.14" in result

    def test_allows_print(self, engine):
        result = engine._safe_execute_script({"code": "print('hello world')"})
        assert "hello world" in result

    def test_empty_script_error(self, engine):
        result = engine._safe_execute_script({"code": ""})
        assert "Error" in result

    def test_syntax_error(self, engine):
        result = engine._safe_execute_script({"code": "def ("})
        assert "SyntaxError" in result or "Error" in result
