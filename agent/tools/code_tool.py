"""
═══════════════════════════════════════════════════════════════
  CodeTool — Sandboxed Python Execution (Agent 5)
═══════════════════════════════════════════════════════════════

Runs Python code in isolated subprocess with AST validation.
Refactored from agent/core/executor.py:_safe_execute_script.

Owner: Agent 5 (EXCLUSIVE)
"""

import os
import sys
import ast
import subprocess
import tempfile
import logging
from typing import Dict, Any

from agent.tools.tool_registry import Tool, ToolResult

logger = logging.getLogger("Tars.CodeTool")


class CodeTool(Tool):
    """
    Sandboxed Python code execution.

    Safety layers:
      1. AST validation (block imports, dunders, eval/exec/open)
      2. Subprocess isolation (separate process)
      3. Timeout (10s default)
      4. Output truncation (10KB)
      5. Safe env whitelist
    """

    SAFE_MODULES = {'math', 'json', 'datetime', 'collections', 're', 'itertools', 'functools'}
    BLOCKED_CALLS = {
        '__import__', 'eval', 'exec', 'compile', 'open',
        'getattr', 'setattr', 'delattr', 'globals', 'locals',
        'breakpoint', 'input',
    }
    BLOCKED_DUNDERS = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__globals__', '__builtins__', '__code__', '__func__',
        '__self__', '__module__', '__dict__', '__init__',
        '__new__', '__del__', '__reduce__', '__reduce_ex__',
    }

    SCRIPT_TIMEOUT = 10
    MAX_OUTPUT = 10240
    SAFE_ENV_VARS = {
        'PATH', 'TEMP', 'TMP', 'SystemRoot', 'USERPROFILE',
        'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'COMSPEC',
        'PATHEXT', 'SYSTEMDRIVE', 'HOME',
    }

    def __init__(self, workspace: str = "."):
        self.workspace = os.path.abspath(workspace)

    def name(self) -> str:
        return "code_execute"

    def description(self) -> str:
        return (
            "Execute Python code in a sandbox. "
            f"Safe modules: {', '.join(sorted(self.SAFE_MODULES))}. "
            f"Timeout: {self.SCRIPT_TIMEOUT}s."
        )

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
            },
            "required": ["code"],
        }

    def _validate_ast(self, code: str) -> str:
        """
        AST validation — returns error message or empty string if safe.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError: {e}"

        for node in ast.walk(tree):
            # Block unauthorized imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, 'module', None) or \
                         (node.names[0].name if node.names else '')
                if module not in self.SAFE_MODULES:
                    return f"Import of '{module}' not allowed. Safe: {self.SAFE_MODULES}"

            # Block dangerous function calls
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in self.BLOCKED_CALLS:
                    return f"Call to '{func.id}' not allowed"

            # Block dunder attribute access
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                return f"Access to '{node.attr}' not allowed"

            # Block f-string eval tricks
            if isinstance(node, ast.FormattedValue):
                return "F-string expressions not allowed"

            # Block global/nonlocal
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return "global/nonlocal not allowed"

            # Block subscript dunder access
            if isinstance(node, ast.Subscript):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    if sl.value in self.BLOCKED_DUNDERS:
                        return f"Subscript access to '{sl.value}' not allowed"

        return ""

    def _safe_env(self) -> dict:
        env = {}
        for key in self.SAFE_ENV_VARS:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return env

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        code = args.get("code", "").strip()
        if not code:
            return ToolResult.error("No code provided")

        # AST validation
        error = self._validate_ast(code)
        if error:
            return ToolResult.error(f"Validation: {error}")

        # Subprocess execution
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', dir=self.workspace,
                delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                tmp_path = f.name

            creation_flags = 0
            if os.name == 'nt':
                creation_flags = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(
                [sys.executable, '-u', tmp_path],
                capture_output=True, text=True,
                timeout=self.SCRIPT_TIMEOUT,
                cwd=self.workspace,
                creationflags=creation_flags,
                env=self._safe_env(),
            )

            output = result.stdout[:self.MAX_OUTPUT]
            errors = result.stderr[:2000]

            if result.returncode != 0:
                return ToolResult.error(
                    f"Execution error (code {result.returncode}):\n{errors}\n{output}"
                )

            if not output.strip():
                return ToolResult.success("Script executed successfully (no output)")
            return ToolResult.success(f"Output:\n{output}")

        except subprocess.TimeoutExpired:
            return ToolResult.error(f"Script timed out after {self.SCRIPT_TIMEOUT}s")
        except Exception as e:
            return ToolResult.error(f"Error running script: {e}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
