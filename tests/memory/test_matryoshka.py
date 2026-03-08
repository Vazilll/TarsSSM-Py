"""Tests for MatryoshkaRetriever (2-stage retrieval)."""
import pytest
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestMatryoshkaRetriever:
    """2-stage Matryoshka retrieval tests."""

    @pytest.fixture
    def leann(self):
        from memory.leann import LeannIndex
        idx = LeannIndex(model_path='models/embeddings', index_path=None)
        # Добавляем 20 документов (ниже порога min_docs=500 → brute force)
        for i in range(20):
            idx.add_document(f"Тестовый документ номер {i} с уникальным текстом")
        return idx

    @pytest.fixture
    def retriever(self, leann):
        from memory.matryoshka import MatryoshkaRetriever
        return MatryoshkaRetriever(leann, min_docs=10)  # low threshold for tests

    @pytest.mark.asyncio
    async def test_search_returns_results(self, retriever):
        results = await retriever.search("тестовый документ", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_search_result_has_text(self, retriever):
        results = await retriever.search("документ", top_k=1)
        assert len(results) >= 1
        assert isinstance(results[0].text, str)
        assert len(results[0].text) > 0

    @pytest.mark.asyncio
    async def test_search_result_has_score(self, retriever):
        results = await retriever.search("тестовый", top_k=1)
        assert len(results) >= 1
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_search_empty_index(self):
        from memory.leann import LeannIndex
        from memory.matryoshka import MatryoshkaRetriever
        idx = LeannIndex(model_path='models/embeddings', index_path=None)
        ret = MatryoshkaRetriever(idx)
        results = await ret.search("anything", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_matryoshka_activates_for_large_index(self, leann):
        """When min_docs threshold met, Matryoshka is used."""
        from memory.matryoshka import MatryoshkaRetriever
        ret = MatryoshkaRetriever(leann, min_docs=10)
        assert ret._should_use_matryoshka() is True

    def test_matryoshka_inactive_small_index(self, leann):
        from memory.matryoshka import MatryoshkaRetriever
        ret = MatryoshkaRetriever(leann, min_docs=1000)
        assert ret._should_use_matryoshka() is False

    def test_get_stats(self, retriever):
        stats = retriever.get_stats()
        assert "total_docs" in stats
        assert "coarse_dim" in stats
        assert stats["coarse_dim"] == 64
        assert "matryoshka_active" in stats

    @pytest.mark.asyncio
    async def test_skips_deleted_documents(self, leann, retriever):
        """Tombstoned documents should not appear in results."""
        leann.deleted[0] = True
        results = await retriever.search("тестовый документ номер 0", top_k=10)
        for r in results:
            assert r.index != 0
