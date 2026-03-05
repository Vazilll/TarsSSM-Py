"""
tokenizer.py — Единый токенизатор ТАРС (SentencePiece BPE + CP1251 fallback).

Один класс для ВСЕХ моделей: Reflex, MinGRU, Mamba-2.

Режимы работы:
  1. SentencePiece BPE (vocab ~4096) — если обучена модель
  2. CP1251 byte-level (vocab 256)  — fallback если нет модели

    t = TarsTokenizer()                      # auto-detect mode
    t = TarsTokenizer(mode="bpe")            # force BPE
    t = TarsTokenizer(mode="byte")           # force byte-level

    t.train("корпус текста...", vocab_size=4096)  # обучить BPE
    ids = t.encode("привет")                      # [1423, 87]
    text = t.decode(ids)                           # "привет"
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

_logger = logging.getLogger("Tars.Tokenizer")

# Путь к модели SentencePiece по умолчанию
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "tokenizer"
_DEFAULT_MODEL_PATH = _DEFAULT_MODEL_DIR / "tars_bpe.model"


class TarsTokenizer:
    """
    Универсальный токенизатор ТАРС.

    Приоритет:
      1. Если есть обученная SentencePiece модель → BPE (vocab ~4096)
      2. Иначе → CP1251 byte-level (vocab 256)

    BPE преимущества:
      - "привет" → 1-2 токена вместо 6 байтов
      - Модель быстрее учит семантику
      - Меньше последовательности → больше контекста

    CP1251 fallback:
      - Vocab = 256 (фиксированный)
      - Каждый байт ввода = 1 токен
      - Не требует обучения
    """

    def __init__(self, mode: str = "auto", model_path: Optional[str] = None):
        """
        Args:
            mode: "auto" | "bpe" | "byte"
            model_path: путь к .model файлу SentencePiece (None = default)
        """
        self._model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self._sp = None  # SentencePieceProcessor
        self._mode = "byte"  # actual mode after init

        if mode == "byte":
            self._init_byte()
        elif mode == "bpe":
            self._init_bpe()
        elif mode == "auto":
            # Try BPE first, fall back to byte
            if self._model_path.exists():
                try:
                    self._init_bpe()
                except Exception as e:
                    _logger.warning(f"SentencePiece load failed ({e}), fallback to byte")
                    self._init_byte()
            else:
                self._init_byte()
        else:
            raise ValueError(f"Unknown tokenizer mode: {mode!r}")

    def _init_byte(self):
        """CP1251 byte-level fallback."""
        self._mode = "byte"
        self.vocab_size = 256
        self.pad_token_id = 0
        self.eos_token_id = 3   # ETX
        self.bos_token_id = 2   # STX
        _logger.info(f"TarsTokenizer: byte mode (vocab={self.vocab_size})")

    def _init_bpe(self):
        """SentencePiece BPE mode."""
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(
                "sentencepiece not installed. Run: pip install sentencepiece\n"
                "Or use TarsTokenizer(mode='byte') for CP1251 fallback."
            )

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"SentencePiece model not found: {self._model_path}\n"
                "Train it first: TarsTokenizer.train(corpus, vocab_size=4096)"
            )

        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(str(self._model_path))

        self._mode = "bpe"
        self.vocab_size = self._sp.GetPieceSize()
        self.pad_token_id = self._sp.pad_id() if self._sp.pad_id() >= 0 else 0
        self.eos_token_id = self._sp.eos_id() if self._sp.eos_id() >= 0 else 3
        self.bos_token_id = self._sp.bos_id() if self._sp.bos_id() >= 0 else 2
        _logger.info(f"TarsTokenizer: BPE mode (vocab={self.vocab_size}, model={self._model_path.name})")

    @property
    def is_bpe(self) -> bool:
        return self._mode == "bpe"

    # ─── Encode / Decode ───

    def encode(self, text: str) -> List[int]:
        """
        Текст → список token IDs.

        BPE:   "привет" → [1423, 87]  (2 подслова)
        Byte:  "привет" → [239, 240, 232, 226, 229, 242]  (6 байтов)
        """
        if self._mode == "bpe":
            return self._sp.EncodeAsIds(text)
        else:
            return list(text.encode('cp1251', errors='replace'))

    def decode(self, ids: List[int]) -> str:
        """
        Список token IDs → текст.
        """
        if self._mode == "bpe":
            # Filter invalid IDs
            valid = [i for i in ids if 0 <= i < self.vocab_size]
            return self._sp.DecodeIds(valid)
        else:
            clean = [b for b in ids if 0 <= b < 256
                     and b not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]
            return bytearray(clean).decode('cp1251', errors='replace')

    def encode_with_special(self, text: str) -> List[int]:
        """Encode с BOS и EOS."""
        return [self.bos_token_id] + self.encode(text) + [self.eos_token_id]

    # ─── Training ───

    @staticmethod
    def train(
        corpus_text: str,
        vocab_size: int = 4096,
        model_path: Optional[str] = None,
        model_type: str = "bpe",
        character_coverage: float = 0.9995,
        num_threads: int = 4,
    ) -> "TarsTokenizer":
        """
        Обучить SentencePiece BPE модель на корпусе текста.

        Args:
            corpus_text: полный обучающий текст
            vocab_size: размер словаря (4096 рекомендуется для малых моделей)
            model_path: куда сохранить .model (None = default)
            model_type: "bpe" или "unigram"
            character_coverage: покрытие символов (0.9995 для кириллицы)
            num_threads: число потоков для обучения

        Returns:
            TarsTokenizer с загруженной моделью
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("pip install sentencepiece")

        save_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # SentencePiece работает с файлами
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                          encoding='utf-8') as f:
            f.write(corpus_text)
            tmp_corpus = f.name

        try:
            prefix = str(save_path).replace('.model', '')
            spm.SentencePieceTrainer.Train(
                input=tmp_corpus,
                model_prefix=prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                num_threads=num_threads,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                # Нормализации отключены для точного roundtrip
                normalization_rule_name="identity",
                # Специальные символы для формата Q&A
                user_defined_symbols=["\\n", "Вопрос:", "Ответ:"],
                byte_fallback=True,  # UTF-8 byte fallback для неизвестных символов
                max_sentence_length=16384,
            )
            _logger.info(f"SentencePiece BPE trained: vocab={vocab_size}, saved to {save_path}")
        finally:
            os.unlink(tmp_corpus)

        return TarsTokenizer(mode="bpe", model_path=str(save_path))

    def __repr__(self):
        return f"TarsTokenizer(mode={self._mode}, vocab={self.vocab_size})"


# ─── Singleton for quick access ───

_global_tokenizer: Optional[TarsTokenizer] = None


def get_tokenizer(mode: str = "auto", model_path: Optional[str] = None) -> TarsTokenizer:
    """Get or create the global TarsTokenizer singleton."""
    global _global_tokenizer
    if _global_tokenizer is None:
        _global_tokenizer = TarsTokenizer(mode=mode, model_path=model_path)
    return _global_tokenizer


def reset_tokenizer():
    """Reset the global tokenizer singleton (e.g. after training a new model)."""
    global _global_tokenizer
    _global_tokenizer = None


if __name__ == "__main__":
    t = TarsTokenizer()

    tests = [
        "привет",
        "как дела?",
        "что такое интеграл",
        "Hello World",
        "ТАРС v3.0",
    ]

    print(f"Tokenizer: {t}")
    print(f"Vocab: {t.vocab_size}")
    print(f"Mode: {t._mode}")
    print()

    for text in tests:
        ids = t.encode(text)
        decoded = t.decode(ids)
        byte_count = len(text.encode('utf-8'))
        compression = byte_count / max(len(ids), 1)
        print(f"  '{text}' → {ids[:10]}{'...' if len(ids) > 10 else ''} "
              f"→ '{decoded}' (tokens={len(ids)}, bytes={byte_count}, "
              f"compression={compression:.1f}x)")

    print(f"\n✅ Tokenizer OK ({t._mode} mode)")
