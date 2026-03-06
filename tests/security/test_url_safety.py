"""URL scheme and injection validation tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestURLSafety:
    """URL scheme and injection validation."""

    @pytest.fixture
    def engine(self):
        from agent.executor import ActionEngine
        return ActionEngine()

    def test_blocks_javascript_url(self, engine):
        result = engine._safe_open_url({"url": "javascript:alert(1)"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_file_url(self, engine):
        result = engine._safe_open_url({"url": "file:///etc/passwd"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_url_with_shell_chars(self, engine):
        result = engine._safe_open_url({"url": "http://evil.com;rm -rf /"})
        assert "Error" in result or "forbidden" in result

    def test_allows_https_url(self, engine):
        result = engine._safe_open_url({"url": "https://example.com"})
        assert "not allowed" not in result
        assert "forbidden" not in result
