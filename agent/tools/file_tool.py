"""
═══════════════════════════════════════════════════════════════
  FileTool — File Search & Operations (Agent 5)
═══════════════════════════════════════════════════════════════

Refactored from tools/__init__.py:FileSearchTool.

Owner: Agent 5 (EXCLUSIVE)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

from agent.tools.tool_registry import Tool, ToolResult

logger = logging.getLogger("Tars.FileTool")


class FileTool(Tool):
    """Search for files by name pattern or content in workspace."""

    MAX_RESULTS = 20
    MAX_FILE_SIZE = 1_000_000  # 1MB max for content search

    def __init__(self, workspace: str = "."):
        self.workspace = os.path.abspath(workspace)

    def name(self) -> str:
        return "file_search"

    def description(self) -> str:
        return "Search for files by name pattern (glob) or content text."

    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Filename pattern (glob, e.g. '*.py')",
                },
                "content": {
                    "type": "string",
                    "description": "Search for this text inside files",
                },
                "max_results": {
                    "type": "integer",
                    "description": f"Max results (default {self.MAX_RESULTS})",
                },
            },
        }

    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern", "")
        content = args.get("content", "")
        max_results = min(args.get("max_results", self.MAX_RESULTS), 50)

        if not pattern and not content:
            return ToolResult.error("Either 'pattern' or 'content' is required")

        results = []
        workspace = Path(self.workspace)

        try:
            if pattern:
                for f in workspace.rglob(pattern):
                    if f.is_file() and not any(p.startswith('.') for p in f.parts):
                        rel = f.relative_to(workspace)
                        size_kb = f.stat().st_size / 1024
                        results.append(f"{rel} ({size_kb:.1f} KB)")
                        if len(results) >= max_results:
                            break

            if content:
                for f in workspace.rglob("*"):
                    if not f.is_file() or f.stat().st_size > self.MAX_FILE_SIZE:
                        continue
                    if any(p.startswith('.') for p in f.parts):
                        continue
                    if f.suffix in ('.pyc', '.pyo', '.so', '.dll', '.exe', '.bin',
                                    '.whl', '.egg', '.tar', '.gz', '.zip'):
                        continue
                    try:
                        text = f.read_text(encoding='utf-8', errors='ignore')
                        if content.lower() in text.lower():
                            rel = f.relative_to(workspace)
                            for i, line in enumerate(text.split('\n'), 1):
                                if content.lower() in line.lower():
                                    results.append(f"{rel}:{i}: {line.strip()[:100]}")
                                    break
                            if len(results) >= max_results:
                                break
                    except Exception:
                        pass

            if not results:
                return ToolResult.success("No results found")

            return ToolResult.success("\n".join(results))

        except Exception as e:
            return ToolResult.error(f"Search error: {e}")
