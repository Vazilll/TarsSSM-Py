"""
═══════════════════════════════════════════════════════════════
  WebTool — Web Search (Agent 5)
═══════════════════════════════════════════════════════════════

Wraps tools/communication/web_search.py as a Tool.

Owner: Agent 5 (EXCLUSIVE)
"""

import logging
from typing import Dict, Any

from agent.tools.tool_registry import Tool, ToolResult

logger = logging.getLogger("Tars.WebTool")


class WebTool(Tool):
    """Web search via DuckDuckGo."""

    MAX_RESULTS = 5

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def name(self) -> str:
        return "web_search"

    def description(self) -> str:
        return "Search the web using DuckDuckGo. Returns top results with snippets."

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Max results (default {self.max_results})",
                },
            },
            "required": ["query"],
        }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        query = args.get("query", "").strip()
        max_results = args.get("max_results", self.max_results)

        if not query:
            return ToolResult.error("query is required")

        try:
            from tools.communication.web_search import search_duckduckgo
            results = search_duckduckgo(query, max_results=max_results)

            if not results:
                return ToolResult.success("No results found")

            parts = []
            for i, r in enumerate(results[:max_results], 1):
                if isinstance(r, dict):
                    title = r.get("title", "")
                    body = r.get("body", r.get("snippet", ""))
                    href = r.get("href", r.get("url", ""))
                    parts.append(f"{i}. {title}\n   {href}\n   {body[:200]}")
                else:
                    parts.append(f"{i}. {str(r)[:300]}")

            return ToolResult.success("\n\n".join(parts))

        except ImportError:
            return ToolResult.error("duckduckgo_search not installed: pip install duckduckgo-search")
        except Exception as e:
            return ToolResult.error(f"Web search error: {e}")
