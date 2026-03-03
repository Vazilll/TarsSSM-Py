import torch
from .utils import decode_tokens, tokenize_text

def generate_text(model, start_text="Привет", max_length=128, temperature=0.7, device='cuda', context_vec=None, top_k=20):
    """
    Генерация текста через MinGRU_LM с инкрементальным декодированием.
    
    Phase 1: Prompt обрабатывается в FULL SEQUENCE mode (с Conv1d).
    Phase 2: Инкрементальная генерация — по одному токену.
    
    Args:
        model: MinGRU_LM модель
        start_text: затравка (промпт)
        max_length: макс. количество генерируемых байтов
        temperature: температура сэмплирования (0.1 = детерминированно, 1.0 = креативно)
        device: устройство
        context_vec: [1, context_dim] вектор из Ω-SSM/RRN для кондиционирования (опционально)
    """
    model.eval()
    
    tokens = tokenize_text(start_text)
    if not tokens:
        tokens = [32] 
    
    if not isinstance(device, torch.device):
        device = next(model.parameters()).device
    
    # Подготовка контекста (инжекция Ω-SSM вектора)
    ctx = None
    if context_vec is not None:
        if not isinstance(context_vec, torch.Tensor):
            context_vec = torch.tensor(context_vec, dtype=torch.float32)
        ctx = context_vec.unsqueeze(0).to(device) if context_vec.dim() == 1 else context_vec.to(device)
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
        
        # Phase 1: Process prompt in FULL SEQUENCE mode (with Conv1d!)
        # This matches training mode where prev_hiddens is not passed.
        logits = model(input_tensor, context_vec=ctx)
        
        # Re-run to get hidden states for incremental continuation
        _, prev_hiddens = model(
            input_tensor, context_vec=ctx,
            return_hiddens=True
        )
        
        # Sample from last position with top-k filtering
        last_logits = logits[0, -1, :] / max(temperature, 0.01)
        if top_k > 0 and top_k < last_logits.size(-1):
            topk_vals, _ = torch.topk(last_logits, top_k)
            last_logits[last_logits < topk_vals[-1]] = float('-inf')
        probs = torch.softmax(last_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        if next_token < 256:
            generated_tokens.append(next_token)
        
        # Phase 2: Incremental decoding — only process last token each step
        for step in range(1, max_length):
            # Feed only the last generated token
            input_single = torch.tensor([[next_token]], device=device, dtype=torch.long)
            logits, prev_hiddens = model(
                input_single, prev_hiddens=prev_hiddens
            )
            
            last_logits = logits[0, -1, :] / max(temperature, 0.01)
            if top_k > 0 and top_k < last_logits.size(-1):
                topk_vals, _ = torch.topk(last_logits, top_k)
                last_logits[last_logits < topk_vals[-1]] = float('-inf')
            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token < 256:
                generated_tokens.append(next_token)
            
            # Стоп на двойном переводе строки (конец ответа в Q&A формате)
            if len(generated_tokens) > len(tokens) + 5 and next_token == 10:
                break

    return decode_tokens(generated_tokens)
