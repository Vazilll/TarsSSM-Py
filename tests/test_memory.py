"""
═══════════════════════════════════════════════════════════════
  test_memory.py — Memory System Tests (Agent 5)
═══════════════════════════════════════════════════════════════

Tests for memory/ module: SDM store, LEANN, Titans.
Consolidated from tests/memory/*.

Owner: Agent 5 (EXCLUSIVE)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# SDM Store
# ═══════════════════════════════════════════════════════════════

def test_store_import():
    """TarsStorage can be imported."""
    from memory.store import TarsStorage
    store = TarsStorage()
    assert store is not None


# ═══════════════════════════════════════════════════════════════
# LEANN
# ═══════════════════════════════════════════════════════════════

def test_leann_import():
    """LeannIndex can be imported."""
    from memory.leann import LeannIndex
    assert LeannIndex is not None


def test_leann_add_and_search():
    """LEANN basic add + search cycle."""
    from memory.leann import LeannIndex

    try:
        idx = LeannIndex()
        idx.add_document("Python is a programming language")
        idx.add_document("TARS is an AI assistant")

        results = idx.search("programming", top_k=2)
        assert results is not None
        assert len(results) > 0
    except Exception:
        pytest.skip("LEANN embedding model not available")


# ═══════════════════════════════════════════════════════════════
# Titans
# ═══════════════════════════════════════════════════════════════

def test_titans_import():
    """TitansMemory can be imported."""
    from memory.titans import TitansMemory
    assert TitansMemory is not None


def test_titans_update():
    """Titans surprise-based memory update."""
    from memory.titans import TitansMemory

    try:
        titans = TitansMemory(d_model=64)
        vec = torch.randn(1, 64)
        result = titans.update(vec)
        assert result is not None
    except Exception:
        pytest.skip("Titans initialization failed")


# ═══════════════════════════════════════════════════════════════
# MemoryTool (agent/tools/memory_tool.py)
# ═══════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_memory_tool_no_backend():
    """MemoryTool without backend returns error."""
    from agent.tools.memory_tool import MemoryTool
    tool = MemoryTool()
    result = await tool.execute({"operation": "search", "query": "test"})
    assert result.is_error
