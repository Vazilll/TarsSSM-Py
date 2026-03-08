"""
═══════════════════════════════════════════════════════════════
  ToolRegistry — Dynamic Tool Registry for TARS (Agent 5)
═══════════════════════════════════════════════════════════════

Manages tool registration, discovery, and execution.
Each tool: name, description, parameters schema, execute().
Integrates with AuditLogger and EthicalGuard.

Refactored from tools/__init__.py (ToolRegistry, Tool, ToolResult).
Backward-compatible API.

Owner: Agent 5 (EXCLUSIVE)
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger("Tars.ToolRegistry")


# ═══════════════════════════════════════════════════════════════
# ToolResult — execution result
# ═══════════════════════════════════════════════════════════════

@dataclass
class ToolResult:
    """Result of tool execution."""
    output: str
    llm_output: str
    is_error: bool = False
    silent: bool = False
    execution_time: float = 0.0

    @staticmethod
    def success(output: str, llm_output: str = "") -> 'ToolResult':
        return ToolResult(output=output, llm_output=llm_output or output)

    @staticmethod
    def error(message: str) -> 'ToolResult':
        return ToolResult(output=message, llm_output=message, is_error=True)


# ═══════════════════════════════════════════════════════════════
# Tool — abstract base class
# ═══════════════════════════════════════════════════════════════

class Tool(ABC):
    """Base class for TARS tools."""

    @abstractmethod
    def name(self) -> str:
        """Unique tool name."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Description for LLM function calling."""
        ...

    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for parameters."""
        ...

    @abstractmethod
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Execute the tool."""
        ...


# ═══════════════════════════════════════════════════════════════
# ToolRegistry — dynamic registration & dispatch
# ═══════════════════════════════════════════════════════════════

class ToolRegistry:
    """
    Dynamic tool registry with safety and audit integration.

    Features:
      - Register/unregister tools
      - Execute by name with timeout
      - JSON schema export for LLM function calling
      - EthicalGuard pre-check before execution
      - AuditLogger for all invocations
    """

    def __init__(self, ethical_guard=None, audit_logger=None):
        self._tools: Dict[str, Tool] = {}
        self._guard = ethical_guard
        self._audit = audit_logger

    def register(self, tool: Tool):
        """Register a tool."""
        name = tool.name()
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")
        self._tools[name] = tool
        logger.info(f"Tool registered: {name}")

    def unregister(self, name: str):
        """Remove a tool."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.

        1. Check if tool exists
        2. (Optional) EthicalGuard pre-check
        3. Execute with timing
        4. (Optional) AuditLogger log
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult.error(f"Unknown tool: {tool_name}")

        # Pre-check with EthicalGuard
        if self._guard is not None:
            try:
                verdict = self._guard.check(tool_name, args)
                if verdict.is_blocked:
                    if self._audit:
                        self._audit.log_tool(tool_name, args, error=f"BLOCKED: {verdict.reason}")
                    return ToolResult.error(f"⛔ Blocked by EthicalGuard: {verdict.reason}")
            except Exception as e:
                # Fail-closed: block on guard error for action tools
                logger.warning(f"EthicalGuard error for '{tool_name}': {e}")
                return ToolResult.error(f"⛔ Safety check failed: {e}")

        start = time.time()
        try:
            result = await tool.execute(args)
            result.execution_time = time.time() - start

            # Audit log
            if self._audit:
                self._audit.log_tool(
                    tool_name, args,
                    result=result.output[:200],
                    duration=result.execution_time,
                    error=result.output if result.is_error else "",
                )

            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"Tool '{tool_name}' error: {e}")
            if self._audit:
                self._audit.log_tool(tool_name, args, error=str(e), duration=duration)
            return ToolResult.error(f"Tool error: {e}")

    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        JSON schemas for all tools (OpenAI function calling format).
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters(),
                }
            })
        return schemas

    def create_default(self, workspace: str = "."):
        """Register all default tools."""
        from agent.tools.shell_tool import ShellTool
        from agent.tools.file_tool import FileTool
        from agent.tools.code_tool import CodeTool
        from agent.tools.memory_tool import MemoryTool

        self.register(ShellTool(workspace))
        self.register(FileTool(workspace))
        self.register(CodeTool(workspace))
        self.register(MemoryTool())

        # Optional tools
        try:
            from agent.tools.web_tool import WebTool
            self.register(WebTool())
        except ImportError:
            logger.debug("WebTool not available (duckduckgo_search not installed)")

        logger.info(f"Default tools registered: {self.list_tools()}")
