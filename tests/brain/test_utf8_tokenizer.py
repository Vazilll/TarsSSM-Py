"""T14: UTF-8 byte-level tokenizer roundtrip tests."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestUTF8Tokenizer:
    """UTF-8 byte-level tokenizer roundtrip tests."""

    @pytest.fixture
    def tokenizer(self):
        from brain.tokenizer import TarsTokenizer
        return TarsTokenizer(mode="utf8")

    def test_cyrillic(self, tokenizer):
        text = "Привет, мир! Это ТАРС."
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_cjk(self, tokenizer):
        text = "你好世界"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_emoji(self, tokenizer):
        text = "Hello 🌍🚀"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_arabic(self, tokenizer):
        text = "مرحبا بالعالم"
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_cp1251_backward_compat(self):
        from brain.tokenizer import TarsTokenizer
        t = TarsTokenizer(mode="byte")
        assert t._mode == "byte"
        assert t.decode(t.encode("Hello")) == "Hello"
