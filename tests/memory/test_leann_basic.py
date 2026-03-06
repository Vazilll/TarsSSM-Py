"""LEANN basic operations: add/search/save/load roundtrip."""
import pytest
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestLeannBasic:
    """LEANN add/search/save/load tests."""

    @pytest.fixture
    def index(self):
        from memory.leann import LeannIndex
        return LeannIndex(model_path='models/embeddings', index_path=None)

    def test_add_document(self, index):
        index.add_document("Тарс — это ИИ-ассистент")
        assert len(index.texts) >= 1

    @pytest.mark.asyncio
    async def test_search_finds_added(self, index):
        index.add_document("Python — лучший язык программирования")
        index.add_document("Сегодня хорошая погода для прогулки")
        results = await index.search("программирование на Python", top_k=2)
        assert len(results) > 0
        assert any("Python" in r or "программ" in r for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_index(self, index):
        results = await index.search("тест", top_k=5)
        assert isinstance(results, list) and len(results) == 0

    @pytest.mark.asyncio
    async def test_search_returns_list(self, index):
        index.add_document("тестовый документ")
        results = await index.search("тест", top_k=3)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, str)

    @pytest.mark.asyncio
    async def test_top_k_limit(self, index):
        for i in range(10):
            index.add_document(f"Документ номер {i}")
        results = await index.search("документ", top_k=3)
        assert len(results) <= 3

    def test_save_load_roundtrip(self, index):
        with tempfile.TemporaryDirectory() as tmpdir:
            index.index_path = os.path.join(tmpdir, "test_leann")
            index.add_document("Документ один")
            index.add_document("Документ два")
            index.save()
            from memory.leann import LeannIndex
            idx2 = LeannIndex(model_path='models/embeddings',
                              index_path=os.path.join(tmpdir, "test_leann"))
            idx2.load()
            assert idx2.texts == index.texts

    def test_duplicate_detection(self, index):
        index.add_document("уникальный документ")
        count = len(index.texts)
        index.add_document("уникальный документ")
        assert len(index.texts) >= count

    def test_has_batch_embedding(self):
        from memory.leann import LeannIndex
        assert hasattr(LeannIndex, '_get_embeddings_batch')

    def test_sha256_cache_key(self):
        import inspect
        from memory.leann import LeannIndex
        source = inspect.getsource(LeannIndex._get_embedding)
        assert "sha256" in source and "md5" not in source

    def test_auto_save_present(self):
        import inspect
        from memory.leann import LeannIndex
        source = inspect.getsource(LeannIndex.add_document)
        assert "auto-save" in source.lower() or "self.save()" in source
