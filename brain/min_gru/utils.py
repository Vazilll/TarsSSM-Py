def decode_tokens(tokens):
    """Декодирует UTF-8 байты обратно в строку, игнорируя ошибки кодировки, чтобы кириллица работала на vocab=256."""
    byte_array = bytearray([t for t in tokens if 0 <= t < 256])
    return byte_array.decode('utf-8', errors='ignore')

def tokenize_text(text):
    """Преобразует строку в байты UTF-8, чтобы кириллица мапилась на vocab=256."""
    return list(text.encode('utf-8', errors='ignore'))
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    formatted_params = f"{total_params:,}"
    print(f"Total trainable parameters: {formatted_params}")
    return total_params

def default(v, d):
    return v if exists(v) else d
    
def exists(v):
    return v is not None
