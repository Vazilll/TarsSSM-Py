"""
MinGRU utility functions — tokenizer, parameter counting, etc.

tokenize_text / decode_tokens use UTF-8 byte-level encoding
(MinGRU has num_tokens=256, so we must stay in byte range 0-255).
"""


def tokenize_text(text):
    """Преобразует текст в байтовые токены UTF-8 (для MinGRU, num_tokens=256)."""
    return list(text.encode('utf-8', errors='replace'))


def decode_tokens(tokens):
    """Декодирует байтовые токены обратно в строку (UTF-8)."""
    clean = [b for b in tokens if 0 <= b < 256]
    return bytes(clean).decode('utf-8', errors='replace')


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_params = f"{total_params:,}"
    print(f"Total trainable parameters: {formatted_params}")
    return total_params


def default(v, d):
    return v if exists(v) else d


def exists(v):
    return v is not None
