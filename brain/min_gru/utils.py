def decode_tokens(tokens):
    """Декодирует CP1251 байты обратно в строку (1-байтовая кириллица)."""
    byte_array = bytearray([t for t in tokens if 0 <= t < 256])
    return byte_array.decode('cp1251', errors='replace')

def tokenize_text(text):
    """Преобразует строку в байты CP1251 (1 символ кириллицы = 1 токен 0-255)."""
    return list(text.encode('cp1251', errors='replace'))
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_params = f"{total_params:,}"
    print(f"Total trainable parameters: {formatted_params}")
    return total_params

def default(v, d):
    return v if exists(v) else d
    
def exists(v):
    return v is not None
