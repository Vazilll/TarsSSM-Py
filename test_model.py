# -*- coding: utf-8 -*-
"""Quick test of trained TARS v3 model."""
import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from brain.mamba2.model import TarsMamba2LM

print("=" * 60)
print("  TARS v3 - Model Test")
print("=" * 60)
print()

# Load
state = torch.load('models/tars_v3/mamba2.pt', map_location='cpu', weights_only=False)
cfg = state.get('config', {})
print(f"Config: d_model={cfg.get('d_model')}, n_layers={cfg.get('n_layers')}, vocab={cfg.get('vocab_size')}")

model = TarsMamba2LM(**cfg)
loaded = model.load_state_dict(state['model_state_dict'], strict=False)
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Params: {total_params:,}")
print(f"Missing keys: {len(loaded.missing_keys)}, Unexpected: {len(loaded.unexpected_keys)}")
print()

# Generation
prompts = [
    "Привет",
    "Кто ты?",
    "Что ты умеешь?",
    "Помоги мне написать код",
    "Как работает нейросеть?",
    "Расскажи о Python",
    "ТАРС, ты здесь?",
]

print("-" * 60)
print("  Generation (greedy, 100 tokens)")
print("-" * 60)
print()

for prompt in prompts:
    tokens = torch.tensor([list(prompt.encode('cp1251', errors='replace'))]).long()
    prompt_len = tokens.shape[1]
    
    with torch.no_grad():
        for _ in range(100):
            logits = model(tokens)
            next_token = logits[0, -1].argmax().item()
            tokens = torch.cat([tokens, torch.tensor([[next_token]])], dim=1)
            if tokens.shape[1] > prompt_len + 5 and next_token == 10:
                break
    
    generated = bytes(tokens[0].tolist()[prompt_len:]).decode('cp1251', errors='replace')
    print(f"  Q: {prompt}")
    print(f"  A: {generated.strip()[:200]}")
    print()

# Top-5
print("-" * 60)
print("  Top-5 next token predictions")
print("-" * 60)
print()

for prompt in ["привет", "кто ты", "помоги"]:
    tokens = torch.tensor([list(prompt.encode('cp1251', errors='replace'))]).long()
    with torch.no_grad():
        logits = model(tokens)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top5 = probs.topk(5)
    preds = []
    for i in range(5):
        idx = top5.indices[i].item()
        prob = top5.values[i].item()
        try:
            char = bytes([idx]).decode('cp1251')
        except:
            char = f'[{idx}]'
        preds.append(f"'{char}'({prob:.1%})")
    print(f"  \"{prompt}\" -> {', '.join(preds)}")

print()
print("=" * 60)
print("  Done!")
print("=" * 60)
