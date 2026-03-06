"""Titans snapshot/restore/rotation tests."""
import pytest
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestTitansSnapshot:
    """Titans snapshot/restore/rotation tests."""

    @pytest.fixture
    def titans(self, tmp_path):
        from memory.titans import TitansMemory
        return TitansMemory(snapshot_dir=str(tmp_path), max_snapshots=3)

    def test_snapshot_created(self, titans):
        path = titans.snapshot()
        assert os.path.exists(path) and path.endswith(".pt")

    def test_restore_snapshot(self, titans):
        titans.total_updates = 42
        titans.total_surprises = 7
        snap = titans.snapshot()
        titans.total_updates = 999
        success = titans.restore_snapshot(snap)
        assert success and titans.total_updates == 42

    def test_rotation(self, titans):
        for _ in range(5):
            titans.snapshot()
            time.sleep(0.05)
        assert len(titans._list_snapshots()) <= 3

    def test_restore_latest(self, titans):
        titans.total_updates = 10; titans.snapshot(); time.sleep(0.05)
        titans.total_updates = 20; titans.snapshot()
        titans.total_updates = 999
        assert titans.restore_snapshot() and titans.total_updates == 20

    def test_stats_includes_snapshots(self, titans):
        titans.snapshot()
        assert titans.get_stats().get("snapshots", 0) >= 1
