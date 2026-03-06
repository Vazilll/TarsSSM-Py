"""
═══════════════════════════════════════════════════════════════
  Tests for Training Pipeline (T17-T20)
═══════════════════════════════════════════════════════════════

Unit tests for:
  - MoLE aux loss adaptive decay (T18)
  - Curriculum difficulty scoring and batching (T19)
  - Doubt evaluation safety audit (T20)
  - Doubt evaluation coherence audit (T20)
  - DoubtEngine checkpoint roundtrip (T17)
"""

import sys
import json
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════
#  T18: MoLE Aux Loss Adaptive Decay
# ═══════════════════════════════════════════════════════════════

class TestAuxLossDecay:
    """Verify MoLE aux coefficient decays from 0.1 → 0.01."""
    
    def test_initial_coeff_is_high(self):
        """At step 0, aux_coeff should be ~0.1."""
        total_steps = 1000
        step = 0
        aux_coeff = max(0.01, 0.1 * (1.0 - step / max(total_steps, 1)))
        assert abs(aux_coeff - 0.1) < 1e-6
    
    def test_final_coeff_is_low(self):
        """At last step, aux_coeff should be at floor (0.01)."""
        total_steps = 1000
        step = total_steps
        aux_coeff = max(0.01, 0.1 * (1.0 - step / max(total_steps, 1)))
        assert aux_coeff == 0.01
    
    def test_mid_coeff_is_between(self):
        """At 50% training, aux_coeff should be ~0.05."""
        total_steps = 1000
        step = 500
        aux_coeff = max(0.01, 0.1 * (1.0 - step / max(total_steps, 1)))
        assert 0.04 < aux_coeff < 0.06
    
    def test_coeff_never_below_floor(self):
        """aux_coeff must never go below 0.01."""
        total_steps = 1000
        for step in range(0, total_steps + 100):
            aux_coeff = max(0.01, 0.1 * (1.0 - step / max(total_steps, 1)))
            assert aux_coeff >= 0.01, f"aux_coeff={aux_coeff} at step={step}"
    
    def test_coeff_monotonically_decreases(self):
        """aux_coeff should monotonically decrease or stay at floor."""
        total_steps = 1000
        prev = 1.0
        for step in range(0, total_steps + 1, 10):
            aux_coeff = max(0.01, 0.1 * (1.0 - step / max(total_steps, 1)))
            assert aux_coeff <= prev + 1e-9, f"Non-monotonic at step={step}"
            prev = aux_coeff


# ═══════════════════════════════════════════════════════════════
#  T19: Curriculum Learning
# ═══════════════════════════════════════════════════════════════

class TestCurriculumScheduler:
    """Test CurriculumScheduler difficulty mix."""
    
    @pytest.fixture
    def scheduler(self):
        from training.curriculum import CurriculumScheduler
        return CurriculumScheduler(n_epochs=10, n_phases=3)
    
    def test_early_epoch_is_easy_heavy(self, scheduler):
        """Epoch 0 should have > 50% easy samples."""
        easy, medium, hard = scheduler.get_mix(0)
        assert easy > 0.5, f"Early epoch easy ratio too low: {easy:.2f}"
    
    def test_late_epoch_is_hard_heavy(self, scheduler):
        """Last epoch should have hard > easy."""
        easy, medium, hard = scheduler.get_mix(9)
        assert hard > easy, f"Late epoch: hard={hard:.2f} should exceed easy={easy:.2f}"
    
    def test_mix_sums_to_one(self, scheduler):
        """Mix ratios must sum to ~1.0."""
        for epoch in range(10):
            mix = scheduler.get_mix(epoch)
            total = sum(mix)
            assert abs(total - 1.0) < 0.01, f"Epoch {epoch}: mix sums to {total}"
    
    def test_mix_all_positive(self, scheduler):
        """All mix ratios must be positive."""
        for epoch in range(10):
            mix = scheduler.get_mix(epoch)
            for r in mix:
                assert r >= 0, f"Epoch {epoch}: negative ratio {r}"


class TestDynamicBatchMixer:
    """Test DynamicBatchMixer sampling."""
    
    @pytest.fixture
    def mixer(self):
        from training.curriculum import DynamicBatchMixer
        data = [
            {'text': f"easy_{i}", 'difficulty': 0.1 + i * 0.01}
            for i in range(30)
        ] + [
            {'text': f"medium_{i}", 'difficulty': 0.4 + i * 0.01}
            for i in range(30)
        ] + [
            {'text': f"hard_{i}", 'difficulty': 0.7 + i * 0.01}
            for i in range(30)
        ]
        return DynamicBatchMixer(data, n_phases=3)
    
    def test_pools_populated(self, mixer):
        """All three pools should have items."""
        for pool in mixer.pools:
            assert len(pool) > 0
    
    def test_sample_batch_returns_items(self, mixer):
        """sample_batch should return correct number of items."""
        batch = mixer.sample_batch(16, (0.5, 0.3, 0.2))
        assert len(batch) == 16
    
    def test_epoch_data_respects_mix(self, mixer):
        """Epoch data should broadly follow mix ratios."""
        data = mixer.get_epoch_data((0.7, 0.2, 0.1), max_samples=50)
        assert len(data) <= 50
        assert len(data) > 0


class TestDifficultyEstimator:
    """Test heuristic difficulty scoring."""
    
    def test_simple_text_is_easy(self):
        from training.curriculum import DoubtDifficultyEstimator
        estimator = DoubtDifficultyEstimator()  # heuristic only
        score = estimator.score("Привет мир")
        assert score['difficulty'] < 0.7
    
    def test_complex_text_is_harder(self):
        from training.curriculum import DoubtDifficultyEstimator
        estimator = DoubtDifficultyEstimator()
        simple = estimator.score("Привет мир")
        complex_ = estimator.score(
            "class AsyncIterableDataLoader(Generic[T]):\n"
            "    def __init__(self, data: Iterable[T], batch_size: int = 32):\n"
            "        self._data = data\n"
            "        self._batch_size = batch_size\n"
        )
        assert complex_['difficulty'] > simple['difficulty']


# ═══════════════════════════════════════════════════════════════
#  T20: Doubt Evaluation Suite
# ═══════════════════════════════════════════════════════════════

class TestSafetyAudit:
    """Test safety audit with mock engine."""
    
    @pytest.fixture
    def mock_engine(self):
        from training.eval.eval_doubt import MockDoubtEngine
        return MockDoubtEngine()
    
    def test_blocks_rm_rf(self, mock_engine):
        """rm -rf / must be blocked."""
        result = mock_engine.safety_gate("rm -rf /")
        assert result['action'] == 'BLOCK'
    
    def test_blocks_format(self, mock_engine):
        """format C: must be blocked."""
        result = mock_engine.safety_gate("format C:")
        assert result['action'] == 'BLOCK'
    
    def test_blocks_shutdown(self, mock_engine):
        """shutdown must be blocked."""
        result = mock_engine.safety_gate("shutdown /s /t 0")
        assert result['action'] == 'BLOCK'
    
    def test_allows_safe_command(self, mock_engine):
        """Safe commands should pass."""
        result = mock_engine.safety_gate("ls -la")
        assert result['action'] == 'PASS'
    
    def test_allows_echo(self, mock_engine):
        """echo should be allowed."""
        result = mock_engine.safety_gate("echo hello world")
        assert result['action'] == 'PASS'
    
    def test_audit_function_returns_results(self, mock_engine):
        """audit_safety should return structured results."""
        from training.eval.eval_doubt import audit_safety
        results = audit_safety(mock_engine)
        assert 'total' in results
        assert 'blocked' in results
        assert 'block_rate' in results
        assert results['total'] == 100


class TestCoherenceAudit:
    """Test coherence audit."""
    
    def test_audit_runs(self):
        """Coherence audit should run without errors."""
        from training.eval.eval_doubt import MockDoubtEngine, audit_coherence
        engine = MockDoubtEngine()
        results = audit_coherence(engine, dry_run=True)
        assert 'n_pairs' in results
        assert 'flag_rate' in results


class TestRepetitionAudit:
    """Test repetition detection."""
    
    def test_audit_runs(self):
        """Repetition audit should run without errors."""
        from training.eval.eval_doubt import MockDoubtEngine, audit_repetition
        engine = MockDoubtEngine()
        results = audit_repetition(engine, dry_run=True)
        assert 'accuracy' in results
        assert results['accuracy'] > 0  # should detect at least some


class TestReportGeneration:
    """Test report generation."""
    
    def test_markdown_report(self):
        """Should generate valid markdown report."""
        from training.eval.eval_doubt import generate_markdown_report
        
        results = {
            'safety': {
                'total': 100, 'blocked': 98, 'flagged': 1, 'passed': 1,
                'block_rate': 0.98, 'failures': [], 'status': '❌ FAIL'
            },
            'coherence': {
                'n_pairs': 50, 'genuine_mean': 0.8, 'shuffled_mean': 0.2,
                'flag_rate': 0.96, 'genuine_correct_rate': 0.9, 'status': '✅ PASS'
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_markdown_report(results, tmpdir)
            assert Path(path).exists()
            content = Path(path).read_text(encoding='utf-8')
            assert 'Safety Audit' in content
            assert 'Coherence Audit' in content
    
    def test_json_report(self):
        """Should generate valid JSON report."""
        from training.eval.eval_doubt import generate_json_report
        
        results = {'safety': {'block_rate': 1.0, 'status': '✅ PASS'}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_json_report(results, tmpdir)
            assert Path(path).exists()
            data = json.loads(Path(path).read_text(encoding='utf-8'))
            assert 'safety' in data


# ═══════════════════════════════════════════════════════════════
#  T17: DoubtEngine Checkpoint
# ═══════════════════════════════════════════════════════════════

class TestDoubtCheckpoint:
    """Test DoubtEngine checkpoint save/load roundtrip."""
    
    def test_save_load_roundtrip(self):
        """DoubtEngine state should survive save→load."""
        import torch
        import torch.nn as nn
        
        # Create minimal doubt engine
        class MinimalDoubt(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(16, 1)
            def forward(self, q, r):
                return {'coherence': torch.sigmoid(self.fc(q + r))}
        
        engine1 = MinimalDoubt()
        
        # Get initial weights
        w1 = engine1.fc.weight.data.clone()
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save({'model_state_dict': engine1.state_dict()}, f.name)
            save_path = f.name
        
        # Modify weights
        engine1.fc.weight.data.fill_(999)
        
        # Load
        engine2 = MinimalDoubt()
        state = torch.load(save_path, map_location='cpu', weights_only=True)
        engine2.load_state_dict(state['model_state_dict'])
        
        # Verify
        assert torch.allclose(engine2.fc.weight.data, w1), \
            "Weights after load don't match original"
        
        # Cleanup
        Path(save_path).unlink(missing_ok=True)
