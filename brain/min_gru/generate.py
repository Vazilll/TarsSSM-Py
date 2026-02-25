import torch
from .utils import decode_tokens, tokenize_text

def generate_text(model, start_text="Привет", max_length=128, temperature=0.7, device='cuda', context_vec=None):
    """
    Генерация текста через MinGRU_LM с поддержкой CP1251 кириллицы.
    
    Args:
        model: MinGRU_LM модель
        start_text: затравка (промпт)
        max_length: макс. количество генерируемых байтов
        temperature: температура сэмплирования (0.1 = детерминированно, 1.0 = креативно)
        device: устройство
        context_vec: [1, 1024] вектор из Ω-SSM/RRN для кондиционирования (опционально)
    """
    model.eval()
    
    tokens = tokenize_text(start_text)
    if not tokens:
        tokens = [32] 
    
    if not isinstance(device, torch.device):
        device = next(model.parameters()).device
        
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated_tokens = tokens.copy()
    
    # Подготовка контекста (инжекция Ω-SSM вектора)
    ctx = None
    if context_vec is not None:
        if not isinstance(context_vec, torch.Tensor):
            context_vec = torch.tensor(context_vec, dtype=torch.float32)
        ctx = context_vec.unsqueeze(0).to(device) if context_vec.dim() == 1 else context_vec.to(device)
    
    with torch.no_grad():
        for step in range(max_length):
            # Контекст инжектируется только на первом шаге
            logits = model(input_tensor, labels=None, context_vec=ctx if step == 0 else None)
            
            last_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token < 256:  
                generated_tokens.append(next_token)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)
            
            # Стоп: конец строки или макс длина
            if len(generated_tokens) >= len(tokens) + max_length:
                break
            # Стоп на двойном переводе строки (конец ответа в Q&A формате)
            if len(generated_tokens) > len(tokens) + 5 and next_token == 10:
                break

    return decode_tokens(generated_tokens)
