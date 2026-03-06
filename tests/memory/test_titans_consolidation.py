"""Titans → LEANN hippocampal consolidation tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTitansConsolidation:
    """Hippocampal → neocortical memory migration."""

    @pytest.fixture
    def titans(self, tmp_path):
        from memory.titans import TitansMemory
        return TitansMemory(snapshot_dir=str(tmp_path))

    @pytest.fixture
    def leann(self):
        from memory.leann import LeannIndex
        return LeannIndex(model_path='models/embeddings', index_path=None)

    @pytest.mark.asyncio
    async def test_consolidation(self, titans, leann):
        titans.total_updates = 10
        titans.total_surprises = 5
        recent = [("Important fact", None, 0.8), ("Boring info", None, 0.1)]
        count = await titans.consolidate(leann, recent)
        assert count == 1
        assert len(leann.texts) == 1
        assert leann.importance[0] > 0.5

    @pytest.mark.asyncio
    async def test_no_consolidation_low_rate(self, titans, leann):
        titans.total_updates = 100
        titans.total_surprises = 1
        count = await titans.consolidate(leann, [("Data", None, 0.9)])
        assert count == 0
