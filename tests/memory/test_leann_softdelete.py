"""LEANN soft-delete + purge tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestLeannSoftDelete:
    """LEANN soft-delete + purge_deleted tests."""

    @pytest.fixture
    def index(self):
        from memory.leann import LeannIndex
        return LeannIndex(model_path='models/embeddings', index_path=None)

    def test_soft_delete(self, index):
        index.add_document("Document A")
        index.add_document("Document B")
        index.remove_document(0)
        assert len(index.texts) == 2
        assert index.deleted[0] == True
        assert index.deleted[1] == False

    def test_purge_deleted(self, index):
        index.add_document("Keep me")
        index.add_document("Delete me")
        index.add_document("Keep me too")
        index.remove_document(1)
        purged = index.purge_deleted()
        assert purged == 1
        assert len(index.texts) == 2
        assert index.texts == ["Keep me", "Keep me too"]
        assert all(d == False for d in index.deleted)
        assert len(index.access_count) == 2
        assert len(index.last_accessed) == 2
        assert len(index.importance) == 2

    def test_has_remove_document(self):
        from memory.leann import LeannIndex
        assert hasattr(LeannIndex, 'remove_document')
