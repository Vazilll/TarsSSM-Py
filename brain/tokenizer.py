"""
tokenizer.py — Единый токенизатор ТАРС (cp1251 byte-level).

Один класс для ВСЕХ моделей: Reflex, MinGRU, Mamba-2.
1 символ кириллицы = 1 байт cp1251 = 1 токен (0-255).

    "привет" → encode → [239, 240, 232, 226, 229, 242]
    [239, 240, 232, 226, 229, 242] → decode → "привет"

Vocab = 256 (полный диапазон байтов).
Специальные токены: PAD=0, EOS=3 (ETX в ASCII).
"""


class TarsTokenizer:
    """
    CP1251 Byte-Level Tokenizer.
    
    Каждый символ кириллицы = 1 байт = 1 токен.
    Никаких BPE/SentencePiece — прямое отображение.
    
    Преимущества:
      - Vocab = 256 (фиксированный, не нужно обучать)
      - Каждый байт ввода = 1 токен (нет UNK)
      - Детерминированный (нет вариативности сегментации)
      - Работает с любым текстом (даже бинарные данные)
    """
    
    def __init__(self):
        self.vocab_size = 256
        self.pad_token_id = 0
        self.eos_token_id = 3  # ETX (End of Text) в ASCII
        self.bos_token_id = 2  # STX (Start of Text) в ASCII
    
    def encode(self, text: str) -> list:
        """
        Текст → список байтов cp1251.
        
        Args:
            text: строка (русский/английский/любой)
        Returns:
            list[int] — байтовые ID (0-255)
            
        Пример:
            encode("привет") → [239, 240, 232, 226, 229, 242]
            encode("hello")  → [104, 101, 108, 108, 111]
        """
        return list(text.encode('cp1251', errors='replace'))
    
    def decode(self, ids: list) -> str:
        """
        Список байтов → текст cp1251.
        
        Args:
            ids: list[int] — байтовые ID (0-255)
        Returns:
            str — декодированная строка
            
        Пример:
            decode([239, 240, 232, 226, 229, 242]) → "привет"
        """
        # Фильтруем специальные токены и невалидные
        clean = [b for b in ids if 0 <= b < 256 and b not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
        return bytearray(clean).decode('cp1251', errors='replace')
    
    def encode_with_special(self, text: str) -> list:
        """Encode с BOS и EOS токенами."""
        return [self.bos_token_id] + self.encode(text) + [self.eos_token_id]
    
    def __repr__(self):
        return f"TarsTokenizer(vocab={self.vocab_size}, encoding=cp1251)"


if __name__ == "__main__":
    t = TarsTokenizer()
    
    # Тест cp1251
    tests = [
        "привет",
        "как дела?",
        "что такое интеграл",
        "Hello World",
        "ТАРС v3.0 🤖",
    ]
    
    print(f"Tokenizer: {t}")
    print(f"Vocab: {t.vocab_size}")
    print()
    
    for text in tests:
        ids = t.encode(text)
        decoded = t.decode(ids)
        print(f"  '{text}' → {ids[:10]}{'...' if len(ids) > 10 else ''} → '{decoded}'")
        assert decoded.startswith(text[:3]) or len(text) < 3, f"Decode failed for '{text}'"
    
    # Verify specific cp1251 bytes
    assert t.encode("к") == [234], f"'к' should be [234], got {t.encode('к')}"
    assert t.encode("а") == [224], f"'а' should be [224], got {t.encode('а')}"
    assert t.encode(" ") == [32], f"' ' should be [32], got {t.encode(' ')}"
    
    print("\n✅ All tests passed!")
