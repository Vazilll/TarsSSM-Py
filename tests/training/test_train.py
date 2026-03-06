"""
═══════════════════════════════════════════════════════════════
  Training Tests — Corpus Quality, Checkpoint, Config
═══════════════════════════════════════════════════════════════

Покрытие:
  1. Corpus quality filter (SHA256 dedup, length, alpha ratio)
  2. No auto-download of Wikipedia
  3. Configurable repeat multipliers
  4. Training config validation

Запуск:
  python -m pytest tests/training/test_train.py -v
"""
import sys
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════
# Corpus Quality Filter
# ═══════════════════════════════════════════

class TestCorpusQuality:
    """Training corpus quality filter tests."""

    def test_sha256_dedup(self):
        """filter_corpus_quality должен использовать SHA256."""
        import inspect
        from training.train.train_mamba2 import filter_corpus_quality
        source = inspect.getsource(filter_corpus_quality)
        assert "sha256" in source
        assert ".md5(" not in source

    def test_quality_filter_works(self):
        """Should remove short, duplicate, and low-alpha paragraphs."""
        from training.train.train_mamba2 import filter_corpus_quality
        corpus = (
            "Это нормальный параграф длиной более пятидесяти символов для тестирования.\n\n"
            "Это нормальный параграф длиной более пятидесяти символов для тестирования.\n\n"
            "ab\n\n"
            "123456789012345678901234567890123456789@#$%^&*()!@#%$^\n\n"
            "Уникальный валидный параграф длиной более пятидесяти символов для проверки.\n\n"
        )
        result = filter_corpus_quality(corpus)
        paragraphs = [p for p in result.split('\n\n') if p.strip()]
        assert len(paragraphs) == 2


# ═══════════════════════════════════════════
# Load Corpus Config
# ═══════════════════════════════════════════

class TestCorpusConfig:
    """Training corpus loading configuration."""

    def test_no_wiki_auto_download(self):
        """load_corpus не должен автоматически скачивать Wikipedia."""
        import inspect
        from training.train.train_mamba2 import load_corpus
        source = inspect.getsource(load_corpus)
        assert "download_corpus" not in source

    def test_configurable_repeat_multipliers(self):
        """load_corpus должен принимать repeat multiplier parameters."""
        import inspect
        from training.train.train_mamba2 import load_corpus
        sig = inspect.signature(load_corpus)
        assert "identity_repeat" in sig.parameters
        assert "personality_repeat" in sig.parameters
        assert "mega_repeat" in sig.parameters
