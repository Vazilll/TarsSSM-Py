"""Tests for GaussianEmbedding (uncertainty embeddings)."""
import pytest
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestGaussianEmbedding:
    """Gaussian N(μ,σ²) embedding tests."""

    @pytest.fixture
    def ge(self):
        from memory.gaussian_embed import GaussianEmbedding
        return GaussianEmbedding()

    def test_encode_returns_gaussian_vector(self, ge):
        from memory.gaussian_embed import GaussianVector
        gv = ge.encode("Тест текст для эмбеддинга")
        assert isinstance(gv, GaussianVector)
        assert gv.mu.shape == (384,)
        assert gv.sigma.shape == (384,)

    def test_encode_mu_is_float32(self, ge):
        gv = ge.encode("hello world")
        assert gv.mu.dtype == np.float32
        assert gv.sigma.dtype == np.float32

    def test_sigma_within_bounds(self, ge):
        gv = ge.encode("любой текст")
        assert np.all(gv.sigma >= 1e-4)
        assert np.all(gv.sigma <= 2.0)

    def test_similarity_identical_high(self, ge):
        from memory.gaussian_embed import GaussianEmbedding as GE
        a = ge.encode("Москва столица России")
        b = ge.encode("Москва столица России")
        sim = GE.similarity(a, b)
        assert sim > 0.9  # identical → very high

    def test_similarity_range(self, ge):
        from memory.gaussian_embed import GaussianEmbedding as GE
        a = ge.encode("Python программирование")
        b = ge.encode("кулинария рецепты борщ")
        sim = GE.similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_wasserstein_zero_for_identical(self, ge):
        from memory.gaussian_embed import GaussianEmbedding as GE
        a = ge.encode("тест")
        dist = GE.wasserstein2_distance(a, a)
        assert dist < 1e-5

    def test_wasserstein_positive_for_different(self, ge):
        from memory.gaussian_embed import GaussianEmbedding as GE
        a = ge.encode("Python")
        b = ge.encode("шоколадный торт")
        dist = GE.wasserstein2_distance(a, b)
        assert dist > 0.0

    def test_sigma_decay_on_recall(self, ge):
        from memory.gaussian_embed import GaussianEmbedding as GE
        gv = ge.encode("запоминаемый факт")
        original_sigma = gv.sigma.mean()
        recalled = GE.update_sigma_on_recall(gv)
        assert recalled.sigma.mean() < original_sigma

    def test_sigma_boost_on_conflict(self, ge):
        from memory.gaussian_embed import GaussianEmbedding as GE
        gv = ge.encode("обновляемый факт")
        original_sigma = gv.sigma.mean()
        conflicted = GE.update_sigma_on_conflict(gv)
        assert conflicted.sigma.mean() > original_sigma

    def test_to_dict_from_dict_roundtrip(self, ge):
        from memory.gaussian_embed import GaussianVector
        gv = ge.encode("roundtrip test")
        d = gv.to_dict()
        gv2 = GaussianVector.from_dict(d)
        np.testing.assert_array_almost_equal(gv.mu, gv2.mu)
        np.testing.assert_array_almost_equal(gv.sigma, gv2.sigma)

    def test_encode_batch(self, ge):
        from memory.gaussian_embed import GaussianVector
        texts = ["один", "два", "три"]
        results = ge.encode_batch(texts)
        assert len(results) == 3
        for gv in results:
            assert isinstance(gv, GaussianVector)

    def test_elk_kernel_positive(self, ge):
        from memory.gaussian_embed import GaussianEmbedding as GE
        a = ge.encode("тест A")
        b = ge.encode("тест B")
        elk = GE.expected_likelihood_kernel(a, b)
        assert elk >= 0.0

    def test_dim_property(self, ge):
        gv = ge.encode("dim test")
        assert gv.dim == 384
