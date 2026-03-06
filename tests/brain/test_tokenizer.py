"""
Tests for TarsTokenizer — BPE mode and byte-level fallback.

Validates:
  - Encode/decode roundtrip in both modes
  - Empty string handling
  - Special characters and Unicode
  - BPE compression (tokens < bytes for Russian)
  - Vocab size ranges
  - BPE training from corpus
"""
import pytest
import os
import tempfile


class TestTarsTokenizerByte:
    """TarsTokenizer byte-mode tests (CP1251 fallback)."""

    @pytest.fixture
    def tokenizer(self):
        from brain.tokenizer import TarsTokenizer
        return TarsTokenizer(mode="byte")

    def test_encode_returns_list_of_ints(self, tokenizer):
        result = tokenizer.encode("привет")
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_decode_returns_string(self, tokenizer):
        tokens = tokenizer.encode("hello")
        result = tokenizer.decode(tokens)
        assert isinstance(result, str)

    def test_encode_decode_roundtrip_ascii(self, tokenizer):
        text = "Hello, world! 123"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_decode_roundtrip_cyrillic(self, tokenizer):
        text = "Привет, мир!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        assert "Привет" in decoded or "мир" in decoded

    def test_empty_string(self, tokenizer):
        tokens = tokenizer.encode("")
        assert tokens == []
        decoded = tokenizer.decode([])
        assert decoded == "" or isinstance(decoded, str)

    def test_tokens_in_vocab_range(self, tokenizer):
        text = "Test 123 тест !@#$"
        tokens = tokenizer.encode(text)
        assert tokenizer.vocab_size == 256
        for t in tokens:
            assert 0 <= t < 256, f"Token {t} out of range [0, 256)"

    def test_different_inputs_different_tokens(self, tokenizer):
        t1 = tokenizer.encode("hello")
        t2 = tokenizer.encode("world")
        assert t1 != t2

    def test_long_text(self, tokenizer):
        text = "x" * 10000
        tokens = tokenizer.encode(text)
        assert len(tokens) > 0
        decoded = tokenizer.decode(tokens)
        assert len(decoded) > 0

    def test_special_chars(self, tokenizer):
        for text in ["\n\t\r", "\x00\xff", "¿¡"]:
            tokens = tokenizer.encode(text)
            assert isinstance(tokens, list)

    def test_is_not_bpe(self, tokenizer):
        assert not tokenizer.is_bpe
        assert tokenizer._mode == "byte"


class TestTarsTokenizerAuto:
    """TarsTokenizer auto-mode tests."""

    def test_auto_mode_without_model(self):
        """Without a trained model, auto should fall back to byte."""
        from brain.tokenizer import TarsTokenizer
        t = TarsTokenizer(mode="auto", model_path="/nonexistent/path/model.model")
        assert t._mode == "byte"
        assert t.vocab_size == 256

    def test_repr(self):
        from brain.tokenizer import TarsTokenizer
        t = TarsTokenizer(mode="byte")
        r = repr(t)
        assert "byte" in r
        assert "256" in r


class TestTarsTokenizerBPE:
    """TarsTokenizer BPE training and usage tests."""

    @pytest.fixture
    def trained_tokenizer(self, tmp_path):
        """Train a small BPE tokenizer on sample corpus."""
        from brain.tokenizer import TarsTokenizer

        corpus = (
            "Вопрос: Привет\nОтвет: Привет! Я ТАРС.\n\n"
            "Вопрос: Кто ты?\nОтвет: Я автономная нейронная система.\n\n"
            "Вопрос: Как дела?\nОтвет: Всё хорошо, готов к работе.\n\n"
        ) * 50  # Repeat for sufficient training data

        model_path = str(tmp_path / "test_bpe.model")
        tokenizer = TarsTokenizer.train(
            corpus_text=corpus,
            vocab_size=256,  # Small vocab for test speed
            model_path=model_path,
        )
        return tokenizer

    def test_bpe_mode(self, trained_tokenizer):
        assert trained_tokenizer.is_bpe
        assert trained_tokenizer._mode == "bpe"

    def test_bpe_vocab_size(self, trained_tokenizer):
        # vocab_size should be close to requested (SentencePiece may round)
        assert trained_tokenizer.vocab_size >= 100
        assert trained_tokenizer.vocab_size <= 512

    def test_bpe_encode_decode_roundtrip(self, trained_tokenizer):
        text = "Привет! Я ТАРС."
        tokens = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(tokens)
        # SentencePiece may normalize whitespace slightly
        assert "Привет" in decoded
        assert "ТАРС" in decoded

    def test_bpe_compression(self, trained_tokenizer):
        """BPE should compress Russian text (fewer tokens than bytes)."""
        text = "Привет, как дела? Я автономная нейронная система."
        tokens = trained_tokenizer.encode(text)
        byte_count = len(text.encode('utf-8'))
        # BPE should produce fewer tokens than raw bytes
        assert len(tokens) < byte_count, (
            f"No compression: {len(tokens)} tokens >= {byte_count} bytes"
        )

    def test_bpe_tokens_in_range(self, trained_tokenizer):
        text = "Тест кодирования на русском языке"
        tokens = trained_tokenizer.encode(text)
        for t in tokens:
            assert 0 <= t < trained_tokenizer.vocab_size

    def test_bpe_empty_string(self, trained_tokenizer):
        tokens = trained_tokenizer.encode("")
        assert isinstance(tokens, list)
        # SentencePiece may return empty list for empty string
        decoded = trained_tokenizer.decode(tokens)
        assert isinstance(decoded, str)

    def test_bpe_special_tokens(self, trained_tokenizer):
        assert trained_tokenizer.pad_token_id >= 0
        assert trained_tokenizer.eos_token_id >= 0
        assert trained_tokenizer.bos_token_id >= 0
