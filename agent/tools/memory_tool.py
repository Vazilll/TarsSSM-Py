"""
═══════════════════════════════════════════════════════════════
  MemoryTool — LEANN/SDM Memory Access (Agent 5)
═══════════════════════════════════════════════════════════════

Tool wrapper for memory read/write operations.
Uses memory/ subsystem from Agent 2.

Owner: Agent 5 (EXCLUSIVE)
"""

import logging
from typing import Dict, Any, Optional

from agent.tools.tool_registry import Tool, ToolResult

logger = logging.getLogger("Tars.MemoryTool")


class MemoryTool(Tool):
    """
    Read/write/search TARS memory (LEANN + SDM).

    Operations:
      - search: semantic search in memory
      - store: add text to memory
      - status: memory statistics
    """

    def __init__(self, memory=None):
        """
        Args:
            memory: memory.leann.LeannIndex or compatible object
        """
        self._memory = memory

    def set_memory(self, memory):
        """Late-bind memory object (for lazy init)."""
        self._memory = memory

    def name(self) -> str:
        return "memory"

    def description(self) -> str:
        return "Search, store, or query TARS memory (semantic LEANN + SDM)."

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["search", "store", "status"],
                    "description": "Memory operation",
                },
                "query": {
                    "type": "string",
                    "description": "Search query or text to store",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results for search (default 5)",
                },
            },
            "required": ["operation"],
        }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        operation = args.get("operation", "search")
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        if self._memory is None:
            return ToolResult.error("Memory not initialized. Call set_memory() first.")

        try:
            if operation == "search":
                if not query:
                    return ToolResult.error("'query' required for search")
                results = self._memory.search(query, top_k=top_k)
                if not results:
                    return ToolResult.success("No results found in memory")
                parts = []
                for i, doc in enumerate(results[:top_k], 1):
                    snippet = str(doc)[:200]
                    parts.append(f"{i}. {snippet}")
                return ToolResult.success("\n".join(parts))

            elif operation == "store":
                if not query:
                    return ToolResult.error("'query' required for store (text to remember)")
                self._memory.add_document(query)
                return ToolResult.success(f"Stored in memory: {query[:100]}...")

            elif operation == "status":
                info = {}
                if hasattr(self._memory, 'count'):
                    info['documents'] = self._memory.count()
                elif hasattr(self._memory, '__len__'):
                    info['documents'] = len(self._memory)
                if hasattr(self._memory, 'dim'):
                    info['dimension'] = self._memory.dim
                return ToolResult.success(f"Memory status: {info}")

            else:
                return ToolResult.error(f"Unknown operation: {operation}")

        except Exception as e:
            return ToolResult.error(f"Memory error: {e}")
