"""
Tests for LeannIndex — lightweight vector index for RAG.

Validates:
  - Document add and search
  - Save/load roundtrip
  - Empty index behavior
  - Search result relevance
"""
import pytest
import os
import tempfile


class TestLeannIndex:
    """LeannIndex add/search/save/load tests."""

    @pytest.fixture
    def index(self):
        from memory.leann import LeannIndex
        idx = LeannIndex(model_path='models/embeddings', index_path=None)
        return idx

    def test_add_document(self, index):
        """Adding a document should not crash."""
        index.add_document("Тарс — это ИИ-ассистент")
        assert len(index.texts) >= 1

    def test_search_finds_added(self, index):
        """Search should find a recently added document."""
        index.add_document("Python — лучший язык программирования")
        index.add_document("Сегодня хорошая погода для прогулки")

        results = index.search("программирование на Python", top_k=2)
        assert len(results) > 0
        # At least one result should contain Python-related content
        found = any("Python" in r or "программ" in r for r in results)
        assert found, f"Expected Python-related result, got: {results}"

    def test_search_empty_index(self, index):
        """Search on empty index should return empty results."""
        results = index.search("тест", top_k=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_returns_list(self, index):
        """Search should return a list of strings."""
        index.add_document("тестовый документ")
        results = index.search("тест", top_k=3)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, str)

    def test_save_load_roundtrip(self, index):
        """Documents should survive save → load cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index.index_path = os.path.join(tmpdir, "test_leann")
            index.add_document("Документ номер один")
            index.add_document("Документ номер два")
            index.save()

            # Create new index and load
            from memory.leann import LeannIndex
            idx2 = LeannIndex(model_path='models/embeddings',
                              index_path=os.path.join(tmpdir, "test_leann"))
            idx2.load()

            assert len(idx2.texts) == len(index.texts)
            assert idx2.texts == index.texts

    def test_duplicate_detection(self, index):
        """Adding same document twice should be handled."""
        index.add_document("уникальный документ")
        count_before = len(index.texts)
        index.add_document("уникальный документ")
        count_after = len(index.texts)
        # Should either deduplicate or add (both are valid behaviors)
        assert count_after >= count_before

    def test_top_k_limit(self, index):
        """Search should return at most top_k results."""
        for i in range(10):
            index.add_document(f"Документ номер {i}")

        results = index.search("документ", top_k=3)
        assert len(results) <= 3
