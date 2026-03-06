"""T12: SelfLearner quality filter tests."""
import pytest
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestQualityFilter:
    """Quality filter for SelfLearner feedback loop."""

    @pytest.fixture
    def qf(self):
        from brain.mamba2.self_learn import QualityFilter
        return QualityFilter(quarantine_hours=0.001)

    def test_rejects_low_critic(self, qf):
        assert not qf.check_quality({"critic_score": 0.3, "response": "test"})
        assert qf.reject_reasons["low_critic"] > 0

    def test_rejects_low_coherence(self, qf):
        assert not qf.check_quality({"critic_score": 0.9, "doubt_coherence": 0.2, "response": "test"})

    def test_accepts_good_session(self, qf):
        assert qf.check_quality({"critic_score": 0.9, "doubt_coherence": 0.8, "response": "test"})

    def test_diversity_rejects_duplicate(self, qf):
        s1 = {"response": "the quick brown fox jumps over the lazy dog"}
        assert not qf.check_diversity({"response": s1["response"]}, [s1])

    def test_diversity_accepts_different(self, qf):
        s1 = {"response": "the quick brown fox jumps"}
        s2 = {"response": "machine learning is interesting field"}
        assert qf.check_diversity(s2, [s1])

    def test_quarantine_roundtrip(self, qf, tmp_path):
        qf.quarantine_dir = str(tmp_path / "quarantine")
        qf.quarantine_session({"critic_score": 0.9, "response": "quarantined data"})
        time.sleep(0.01)
        released = qf.load_quarantined()
        assert len(released) >= 1 and released[0]["response"] == "quarantined data"

    def test_metrics(self, qf):
        qf.check_quality({"critic_score": 0.3})
        qf.check_quality({"critic_score": 0.9, "doubt_coherence": 0.8})
        m = qf.get_metrics()
        assert m["total_checked"] == 2 and m["total_rejected"] == 1
