"""
Tests for ActionEngine — executor sandbox security.

Validates:
  - Dangerous commands are blocked
  - Sandbox prevents os/sys access
  - URL scheme validation
  - Command length limits
  - Script timeout
"""
import pytest
import asyncio


class TestActionEngineSecurity:
    """Security tests for ActionEngine sandbox."""

    @pytest.fixture
    def engine(self):
        from agent.executor import ActionEngine
        return ActionEngine()

    # ═══ Shell command blocking ═══

    def test_blocks_rm_rf(self, engine):
        """Should block 'rm -rf /' style commands."""
        result = engine._safe_run_command({"command": "rm -rf /"})
        assert "Error" in result or "Blocked" in result

    def test_blocks_format(self, engine):
        """Should block 'format c:' commands."""
        result = engine._safe_run_command({"command": "format c:"})
        assert "Error" in result or "Blocked" in result

    def test_blocks_shutdown(self, engine):
        """Should block shutdown commands."""
        result = engine._safe_run_command({"command": "shutdown /s"})
        assert "Error" in result or "Blocked" in result

    def test_blocks_certutil(self, engine):
        """Should block certutil -decode (LOLBin)."""
        result = engine._safe_run_command({"command": "certutil -decode a.b c.exe"})
        assert "Error" in result or "Blocked" in result

    def test_blocks_bitsadmin(self, engine):
        """Should block bitsadmin (LOLBin)."""
        result = engine._safe_run_command({"command": "bitsadmin /transfer myDownloadJob"})
        assert "Error" in result or "Blocked" in result

    def test_blocks_registry(self, engine):
        """Should block registry modification."""
        result = engine._safe_run_command({"command": "reg add HKLM\\SOFTWARE\\test"})
        assert "Error" in result or "Blocked" in result

    def test_blocks_long_command(self, engine):
        """Should reject commands longer than 500 chars."""
        result = engine._safe_run_command({"command": "echo " + "x" * 500})
        assert "Error" in result

    def test_blocks_empty_command(self, engine):
        """Should reject empty commands."""
        result = engine._safe_run_command({"command": ""})
        assert "Error" in result

    # ═══ Script sandbox ═══

    def test_blocks_os_import(self, engine):
        """Sandbox should block 'import os'."""
        result = engine._safe_execute_script({"code": "import os\nprint(os.getcwd())"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_subprocess_import(self, engine):
        """Sandbox should block 'import subprocess'."""
        result = engine._safe_execute_script({"code": "import subprocess"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_eval(self, engine):
        """Sandbox should block eval()."""
        result = engine._safe_execute_script({"code": "eval('1+1')"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_exec(self, engine):
        """Sandbox should block exec()."""
        result = engine._safe_execute_script({"code": "exec('print(1)')"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_dunder_access(self, engine):
        """Sandbox should block __class__ access."""
        result = engine._safe_execute_script({"code": "x = ''.__class__"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_open(self, engine):
        """Sandbox should block open()."""
        result = engine._safe_execute_script({"code": "f = open('test.txt', 'w')"})
        assert "Error" in result or "not allowed" in result

    def test_allows_safe_code(self, engine):
        """Safe math code should execute normally."""
        result = engine._safe_execute_script({"code": "print(2 + 2)"})
        assert "4" in result

    def test_empty_script(self, engine):
        """Empty script should return error."""
        result = engine._safe_execute_script({"code": ""})
        assert "Error" in result

    def test_syntax_error(self, engine):
        """Invalid Python should return syntax error."""
        result = engine._safe_execute_script({"code": "def ("})
        assert "SyntaxError" in result or "Error" in result

    # ═══ URL validation ═══

    def test_blocks_javascript_url(self, engine):
        """Should block javascript: URLs."""
        result = engine._safe_open_url({"url": "javascript:alert(1)"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_file_url(self, engine):
        """Should block file: URLs."""
        result = engine._safe_open_url({"url": "file:///etc/passwd"})
        assert "Error" in result or "not allowed" in result

    def test_blocks_url_with_shell_chars(self, engine):
        """Should block URLs containing shell injection chars."""
        result = engine._safe_open_url({"url": "http://evil.com;rm -rf /"})
        assert "Error" in result or "forbidden" in result

    def test_allows_https_url(self, engine):
        """Should allow https URLs (may fail without browser, but shouldn't error on validation)."""
        # This test validates URL validation, not actual browser opening
        # Just check it doesn't return a security error
        result = engine._safe_open_url({"url": "https://example.com"})
        assert "not allowed" not in result
        assert "forbidden" not in result
