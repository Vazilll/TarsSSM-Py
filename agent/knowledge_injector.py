"""
knowledge_injector.py — TARS v3 RAG (Retrieval-Augmented Generation).

Handles:
  - Web search (DuckDuckGo)
  - Local file reading
  - LEANN vector recall
  - Injection into active Mamba-2 generation stream

Usage:
    injector = KnowledgeInjector(leann=leann_index, titans=titans_mem)
    result = injector.handle_tool("search_web", "квантовые компьютеры 2026")
"""
import os
import logging
from typing import Optional

logger = logging.getLogger("KnowledgeInjector")


class KnowledgeInjector:
    """RAG handler for TARS tools: search_web, read_file, recall."""

    def __init__(self, leann=None, titans=None, max_chunk_len=2000):
        self.leann = leann
        self.titans = titans
        self.max_chunk_len = max_chunk_len

    def handle_tool(self, action: str, args: str) -> Optional[str]:
        """Route tool calls to appropriate handler."""
        action = action.strip().lower()
        args = args.strip()

        handlers = {
            "search_web": self._search_web,
            "read_file": self._read_file,
            "recall": self._recall,
        }

        handler = handlers.get(action)
        if handler:
            try:
                result = handler(args)
                if result:
                    return result[:self.max_chunk_len]
                return None
            except Exception as e:
                logger.error(f"Tool '{action}' failed: {e}")
                return f"[Ошибка инструмента: {e}]"
        else:
            logger.warning(f"Unknown tool: {action}")
            return None

    def _search_web(self, query: str) -> Optional[str]:
        """Search the web via DuckDuckGo (no API key needed)."""
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
            if not results:
                return "[Поиск: результатов не найдено]"
            texts = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                texts.append(f"# {title}\n{body}")
            return "\n---\n".join(texts)
        except ImportError:
            logger.warning("duckduckgo_search not installed. pip install duckduckgo-search")
            return "[Поиск: duckduckgo_search не установлен]"
        except Exception as e:
            return f"[Поиск: ошибка — {e}]"

    def _read_file(self, filepath: str) -> Optional[str]:
        """Read a local file."""
        # Expand ~ and relative paths
        filepath = os.path.expanduser(filepath)
        if not os.path.isabs(filepath):
            filepath = os.path.abspath(filepath)

        if not os.path.exists(filepath):
            return f"[Файл не найден: {filepath}]"

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(self.max_chunk_len * 2)
            return content[:self.max_chunk_len]
        except Exception as e:
            return f"[Ошибка чтения файла: {e}]"

    def _recall(self, query: str) -> Optional[str]:
        """Recall from LEANN vector index (semantic memory)."""
        if self.leann is None:
            return "[Recall: LEANN не подключен]"
        try:
            results = self.leann.search(query, top_k=3)
            if not results:
                return "[Recall: ничего не найдено]"
            texts = [r.get("text", r.get("content", str(r))) for r in results]
            return "\n---\n".join(texts)
        except Exception as e:
            return f"[Recall: ошибка — {e}]"
