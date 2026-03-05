"""
generate.py — MinGRU text generation with incremental decoding.

Uses prev_hiddens cache for O(n) generation instead of O(n²).
"""
import torch
from .utils import decode_tokens, tokenize_text


def generate_text(
    model,
    start_text="Привет",
    max_length=128,
    temperature=0.7,
    device='cuda',
    context_vec=None,
    top_k=20,
    top_p=0.9,
    repetition_penalty=1.3,
    tokenizer=None,
):
    """
    Генерация текста через MinGRU_LM с инкрементальным декодированием.

    Prompt обрабатывается целиком (full-context), затем каждый новый
    токен генерируется за O(1) через кэшированные hidden states.

    Args:
        model: MinGRU_LM модель
        start_text: затравка (промпт)
        max_length: макс. количество генерируемых токенов
        temperature: температура сэмплирования
        device: устройство
        context_vec: [1, context_dim] вектор из Ω-SSM (опционально)
        top_k: кол-во лучших токенов для сэмплирования (0 = off)
        top_p: nucleus sampling — кумулятивный порог (0 = off)
        repetition_penalty: штраф за повтор токенов
        tokenizer: TarsTokenizer instance (None = use global)
    """
    model.eval()

    # ─── Tokenize prompt ───
    if tokenizer is not None:
        tokens = tokenizer.encode(start_text)
        actual_vocab = tokenizer.vocab_size
        eos_id = tokenizer.eos_token_id
    else:
        tokens = tokenize_text(start_text)
        actual_vocab = 256  # fallback assumption
        eos_id = 10  # newline as EOS for byte-level
    if not tokens:
        tokens = [32]

    if not isinstance(device, torch.device):
        device = next(model.parameters()).device

    # ─── Context preparation ───
    ctx = None
    if context_vec is not None:
        if not isinstance(context_vec, torch.Tensor):
            context_vec = torch.tensor(context_vec, dtype=torch.float32)
        ctx = context_vec.unsqueeze(0).to(device) if context_vec.dim() == 1 else context_vec.to(device)

    generated_tokens = tokens.copy()

    with torch.no_grad():
        # ═══ Phase 1: Process entire prompt (full-context) ═══
        input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, hiddens = model(
            input_tensor,
            context_vec=ctx,
            return_hiddens=True,
        )

        # ═══ Phase 2: Generate token-by-token (incremental) ═══
        for step in range(max_length):
            # Логиты последней позиции
            next_logits = logits[0, -1, :actual_vocab].float()

            # Repetition penalty
            if repetition_penalty > 1.0:
                for prev_token in set(generated_tokens[-30:]):
                    if prev_token < next_logits.size(0):
                        if next_logits[prev_token] > 0:
                            next_logits[prev_token] /= repetition_penalty
                        else:
                            next_logits[prev_token] *= repetition_penalty

            # Temperature
            next_logits = next_logits / max(temperature, 0.01)

            # Top-k
            if top_k > 0 and top_k < next_logits.size(-1):
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[-1]] = float('-inf')

            # Top-p (nucleus)
            if 0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[0] = False  # keep at least 1
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token < actual_vocab:
                generated_tokens.append(next_token)

            # Stop conditions
            if next_token == eos_id:
                break
            # Byte-mode: stop on double newline
            if actual_vocab == 256 and len(generated_tokens) > len(tokens) + 5 and next_token == 10:
                break

            # ═══ Incremental step: feed only the new token ═══
            new_token = torch.tensor([[next_token]], dtype=torch.long, device=device)
            logits, hiddens = model(
                new_token,
                prev_hiddens=hiddens,
                return_hiddens=True,
            )

    if tokenizer is not None:
        return tokenizer.decode(generated_tokens)
    return decode_tokens(generated_tokens)
