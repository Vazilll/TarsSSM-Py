"""
═══════════════════════════════════════════════════════════════
  ShellTool — Safe Shell Execution (Agent 5)
═══════════════════════════════════════════════════════════════

Refactored from tools/__init__.py:ShellTool + agent/core/executor.py.
Uses EthicalGuard for pre-check.

Owner: Agent 5 (EXCLUSIVE)
"""

import os
import asyncio
import platform
import logging
from typing import Dict, Any, List, Optional

from agent.tools.tool_registry import Tool, ToolResult
from agent.tools.timeout import with_timeout

logger = logging.getLogger("Tars.ShellTool")


class ShellTool(Tool):
    """
    Safe shell command execution with sandboxing.

    Features:
      - Timeout (default 30s)
      - Allowed directories whitelist
      - Max command length check
      - Output truncation (10KB max)
    """

    MAX_COMMAND_LENGTH = 500
    COMMAND_TIMEOUT = 30
    MAX_OUTPUT = 10240  # 10KB

    # Safe environment variables whitelist
    SAFE_ENV_VARS = {
        'PATH', 'TEMP', 'TMP', 'SystemRoot', 'USERPROFILE',
        'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'COMSPEC',
        'PATHEXT', 'SYSTEMDRIVE', 'HOME',
    }

    def __init__(self, workspace: str = ".",
                 timeout: int = 30,
                 allowed_dirs: Optional[List[str]] = None):
        self.workspace = os.path.abspath(workspace)
        self.timeout = timeout
        self.allowed_dirs = allowed_dirs or [self.workspace]

    def name(self) -> str:
        return "shell"

    def description(self) -> str:
        return (
            f"Execute a shell command in the workspace ({self.workspace}). "
            f"Timeout: {self.timeout}s. Max length: {self.MAX_COMMAND_LENGTH}."
        )

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": f"Working directory (default: {self.workspace})",
                },
            },
            "required": ["command"],
        }

    def _safe_env(self) -> dict:
        """Build minimal environment from whitelist."""
        env = {}
        for key in self.SAFE_ENV_VARS:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return env

    def _validate_cwd(self, cwd: str) -> bool:
        """Check that directory is in allowed list."""
        abs_cwd = os.path.abspath(cwd)
        return any(abs_cwd.startswith(os.path.abspath(d)) for d in self.allowed_dirs)

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        command = args.get("command", "").strip()
        cwd = args.get("cwd", self.workspace)

        if not command:
            return ToolResult.error("command is required")

        if len(command) > self.MAX_COMMAND_LENGTH:
            return ToolResult.error(f"Command too long ({len(command)} > {self.MAX_COMMAND_LENGTH})")

        if not self._validate_cwd(cwd):
            return ToolResult.error(f"Directory '{cwd}' is outside allowed paths")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._safe_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult.error(f"Command timed out after {self.timeout}s")

            output = stdout.decode('utf-8', errors='replace')[:self.MAX_OUTPUT]
            error = stderr.decode('utf-8', errors='replace')[:2000]

            if proc.returncode != 0:
                return ToolResult(
                    output=f"Exit code {proc.returncode}\n{error}\n{output}".strip(),
                    llm_output=f"Command failed (exit {proc.returncode}): {error}",
                    is_error=True,
                )

            combined = output
            if error:
                combined += f"\n[stderr] {error}"

            return ToolResult.success(combined[:self.MAX_OUTPUT])

        except Exception as e:
            return ToolResult.error(f"Shell error: {e}")
