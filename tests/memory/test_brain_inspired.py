"""Brain-inspired memory: Ebbinghaus decay, synaptic strength, importance."""
import pytest
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestBrainInspired:
    """Ebbinghaus decay, synaptic strength, reconsolidation, importance."""

    @pytest.fixture
    def index(self):
        from memory.leann import LeannIndex
        return LeannIndex(model_path='models/embeddings', index_path=None)

    def test_access_count_initialized(self, index):
        index.add_document("Test doc")
        assert index.access_count[0] == 0

    def test_importance_stored(self, index):
        index.add_document("Low", importance=0.2)
        index.add_document("High", importance=0.9)
        assert index.importance[0] == pytest.approx(0.2)
        assert index.importance[1] == pytest.approx(0.9)

    def test_importance_default(self, index):
        index.add_document("No importance specified")
        assert index.importance[0] == pytest.approx(0.5)

    def test_last_accessed_initialized(self, index):
        before = time.time()
        index.add_document("Fresh doc")
        after = time.time()
        assert before <= index.last_accessed[0] <= after

    def test_purge_preserves_brain_fields(self, index):
        index.add_document("A", importance=0.1)
        index.add_document("B", importance=0.9)
        index.add_document("C", importance=0.5)
        index.remove_document(1)
        index.purge_deleted()
        assert len(index.importance) == 2
        assert index.importance[0] == pytest.approx(0.1)
        assert index.importance[1] == pytest.approx(0.5)

    def test_save_load_brain_fields(self, index, tmp_path):
        index.index_path = str(tmp_path / "test_brain.index")
        index.add_document("Doc A", importance=0.8)
        index.add_document("Doc B", importance=0.3)
        index.access_count[0] = 5
        index.last_accessed[0] = 1000.0
        index.save()

        from memory.leann import LeannIndex
        idx2 = LeannIndex(model_path='models/embeddings',
                          index_path=str(tmp_path / "test_brain.index"))
        assert idx2.access_count[0] == 5
        assert idx2.last_accessed[0] == pytest.approx(1000.0)
        assert idx2.importance[0] == pytest.approx(0.8, abs=0.01)
        assert idx2.importance[1] == pytest.approx(0.3, abs=0.01)
