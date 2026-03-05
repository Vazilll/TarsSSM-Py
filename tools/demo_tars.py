"""
═══════════════════════════════════════════════════
  ТАРС — Demo: проверка обученных моделей
═══════════════════════════════════════════════════
Загружает все обученные модели и показывает выхлоп.
"""
import sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # tools/ → project root
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception: pass

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'═'*60}")
print(f"  🤖 ТАРС — Live Demo ({device})")
print(f"{'═'*60}\n")

# ═══════════════════════════════════════
# 1. Reflex Classifier
# ═══════════════════════════════════════
print("─── 🔁 Reflex Classifier ───")
try:
    from brain.reflexes.reflex_classifier import ReflexClassifier
    from brain.tokenizer import TarsTokenizer
    
    tokenizer = TarsTokenizer()
    model_reflex = ReflexClassifier(vocab_size=256, embed_dim=64, hidden_dim=64, n_intents=6)
    
    reflex_path = ROOT / "models" / "reflex" / "reflex_classifier.pt"
    if reflex_path.exists():
        model_reflex.load_state_dict(torch.load(str(reflex_path), map_location='cpu', weights_only=True))
        model_reflex.eval()
        print(f"  ✅ Loaded ({sum(p.numel() for p in model_reflex.parameters()):,} params)")
        
        test_phrases = [
            "привет", "пока", "как дела", "который час",
            "открой браузер", "объясни трансформеры",
        ]
        intent_names = ["greeting", "farewell", "status", "time", "quick_action", "complex"]
        
        with torch.no_grad():
            for phrase in test_phrases:
                tokens = tokenizer.encode(phrase.lower())[:64]
                tokens += [0] * (64 - len(tokens))
                x = torch.tensor([tokens], dtype=torch.long)
                
                emb = model_reflex.embed(x)
                h = torch.zeros(1, model_reflex.hidden_dim)
                for t in range(emb.shape[1]):
                    gate = torch.sigmoid(model_reflex.mingru_gate(emb[:, t, :]))
                    h_tilde = model_reflex.mingru_hidden(emb[:, t, :])
                    h = (1 - gate) * h + gate * h_tilde
                
                intent_logits = model_reflex.intent_head(h)
                conf = torch.sigmoid(model_reflex.confidence_head(h)).item()
                intent_id = intent_logits.argmax(-1).item()
                intent = intent_names[intent_id] if intent_id < len(intent_names) else f"#{intent_id}"
                
                is_simple = conf > 0.5
                mode = "⚡ Reflex" if is_simple else "🧠 Brain"
                print(f"  \"{phrase:25s}\" → {intent:15s} conf={conf:.0%} [{mode}]")
    else:
        print(f"  ⚠ Файл не найден: {reflex_path}")
except Exception as e:
    print(f"  ❌ {e}")

# ═══════════════════════════════════════
# 2. MinGRU — Text Generation
# ═══════════════════════════════════════
print(f"\n─── 🧪 MinGRU Text Generation ───")
try:
    from brain.min_gru.mingru_lm import MinGRU_LM
    
    # Load config
    import json
    config_path = ROOT / "models" / "tars_v3" / "config.json"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        d_model = cfg.get("models", {}).get("mamba2", {}).get("params", {}).get("d_model", 768)
    else:
        d_model = 768
    
    mingru_path = ROOT / "models" / "tars_v3" / "mingru.pt"
    if mingru_path.exists():
        ckpt = torch.load(str(mingru_path), map_location='cpu', weights_only=True)
        
        # Detect model config from checkpoint
        state = ckpt if isinstance(ckpt, dict) and 'model_state_dict' not in ckpt else ckpt.get('model_state_dict', ckpt)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
        
        # Get dim from embedding weight
        for k, v in state.items():
            if 'embedding' in k or 'token_emb' in k:
                dim = v.shape[1] if len(v.shape) > 1 else v.shape[0]
                break
        else:
            dim = 512
        
        # Count layers
        layer_ids = set()
        for k in state.keys():
            parts = k.split('.')
            for i, p in enumerate(parts):
                if p == 'blocks' and i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_ids.add(int(parts[i+1]))
        n_layers = max(layer_ids) + 1 if layer_ids else 4
        
        # Detect vocab_size from checkpoint
        ckpt_vocab = 256
        if isinstance(ckpt, dict):
            ckpt_vocab = ckpt.get('vocab_size', ckpt.get('config', {}).get('vocab_size', 256))
        
        print(f"  Config: dim={dim}, layers={n_layers}, vocab={ckpt_vocab}")
        
        model = MinGRU_LM(dim=dim, num_tokens=ckpt_vocab, num_layers=n_layers).to(device)
        model.load_state_dict(state, strict=False)
        model.eval()
        
        params = sum(p.numel() for p in model.parameters())
        mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        print(f"  ✅ Loaded ({params:,} params, {mb:.1f} MB)")
        
        # Generate text
        prompts = ["Привет, меня зовут", "Сегодня я хочу", "Как работает"]
        
        from brain.tokenizer import TarsTokenizer
        demo_tokenizer = TarsTokenizer(mode="auto")
        
        for prompt in prompts:
            tokens = demo_tokenizer.encode(prompt)
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            
            generated = list(tokens)
            with torch.no_grad():
                hidden = None
                for _ in range(80):
                    logits, hidden = model(x, prev_hiddens=hidden, return_hiddens=True)
                    logits_last = logits[0, -1] / 0.8
                    top_k = 40
                    vals, idxs = logits_last.topk(top_k)
                    probs = torch.softmax(vals, dim=0)
                    idx = idxs[torch.multinomial(probs, 1)].item()
                    generated.append(idx)
                    x = torch.tensor([[idx]], dtype=torch.long, device=device)
            
            text = demo_tokenizer.decode(generated)
            
            print(f"\n  📝 \"{prompt}\":")
            print(f"     {text[:120]}...")
    else:
        print(f"  ⚠ Файл не найден: {mingru_path}")
except Exception as e:
    import traceback
    print(f"  ❌ {e}")
    traceback.print_exc()

# ═══════════════════════════════════════
# 3. SNN Spiking Synapses
# ═══════════════════════════════════════
print(f"\n─── ⚡ SNN Spiking Synapses ───")
try:
    from brain.spiking.spiking_synapse import SpikingMinGRUBlock, SI_LIF
    
    block = SpikingMinGRUBlock(dim=256, num_heads=4)
    x = torch.randn(1, 5, 256)
    out, state = block(x)
    
    sparsity = block.sparsity
    params = sum(p.numel() for p in block.parameters())
    
    print(f"  ✅ SpikingMinGRUBlock: {params:,} params")
    print(f"     Input:    {list(x.shape)}")
    print(f"     Output:   {list(out.shape)}")
    print(f"     Sparsity: {sparsity:.1%} (zero spikes)")
    print(f"     Ternary:  {{-1, 0, +1}} — BitNet compatible")
    
    # Show spike distribution
    lif = SI_LIF(dim=64)
    test = torch.randn(1, 100, 64)
    spikes, _ = lif(test)
    neg = (spikes == -1).float().mean().item()
    zero = (spikes == 0).float().mean().item()
    pos = (spikes == 1).float().mean().item()
    print(f"     Spike dist: -1={neg:.1%}, 0={zero:.1%}, +1={pos:.1%}")
except Exception as e:
    print(f"  ❌ {e}")

# ═══════════════════════════════════════
# 4. Mamba-2 Brain (if available)
# ═══════════════════════════════════════
print(f"\n─── 🧠 Mamba-2 Brain ───")
try:
    from brain.mamba2.model import TarsMamba2LM
    
    mamba_path = ROOT / "models" / "tars_v3" / "mamba2.pt"
    if mamba_path.exists():
        ckpt = torch.load(str(mamba_path), map_location='cpu', weights_only=True)
        state = ckpt.get('model_state_dict', ckpt)
        
        # Auto-detect d_model from checkpoint embedding weight
        ckpt_d_model = 256
        for k, v in state.items():
            if 'embedding.weight' in k and len(v.shape) == 2:
                ckpt_d_model = v.shape[1]
                break
        
        # Auto-detect n_layers from checkpoint block indices
        ckpt_layer_ids = set()
        for k in state.keys():
            parts = k.split('.')
            for i, p in enumerate(parts):
                if p == 'blocks' and i + 1 < len(parts) and parts[i+1].isdigit():
                    ckpt_layer_ids.add(int(parts[i+1]))
        ckpt_n_layers = max(ckpt_layer_ids) + 1 if ckpt_layer_ids else 4
        
        print(f"  Config (from checkpoint): d_model={ckpt_d_model}, n_layers={ckpt_n_layers}")
        
        # Detect vocab_size
        ckpt_vocab = ckpt.get('vocab_size', ckpt.get('config', {}).get('vocab_size', 256))
        
        model = TarsMamba2LM(d_model=ckpt_d_model, n_layers=ckpt_n_layers, vocab_size=ckpt_vocab)
        model.load_state_dict(state, strict=False)
        model.eval()
        
        params = sum(p.numel() for p in model.parameters())
        mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        print(f"  ✅ Loaded ({params:,} params, {mb:.1f} MB, vocab={ckpt_vocab})")
        
        # Quick inference
        from brain.tokenizer import TarsTokenizer
        mamba_tokenizer = TarsTokenizer(mode="auto")
        
        prompt = "Я — ТАРС"
        tokens = mamba_tokenizer.encode(prompt)
        x = torch.tensor([tokens], dtype=torch.long)
        
        with torch.no_grad():
            logits = model(x)
        
        print(f"  Logits: {logits.shape}")
        
        # Generate
        generated = list(tokens)
        with torch.no_grad():
            for _ in range(60):
                x_in = torch.tensor([generated[-128:]], dtype=torch.long)
                logits = model(x_in)
                logits_last = logits[0, -1, :ckpt_vocab] / 0.7
                vals, idxs = logits_last.topk(30)
                probs = torch.softmax(vals, dim=0)
                idx = idxs[torch.multinomial(probs, 1)].item()
                generated.append(idx)
        
        text = mamba_tokenizer.decode(generated)
        print(f"\n  📝 \"{prompt}\":")
        print(f"     {text[:150]}...")
    else:
        print(f"  ⚠ mamba2.pt не найден (будет после Phase 4 обучения)")
except Exception as e:
    print(f"  ❌ {e}")

# ═══════════════════════════════════════
# Summary
# ═══════════════════════════════════════
print(f"\n{'═'*60}")
print(f"  🤖 ТАРС — Model Summary")
print(f"{'═'*60}")

model_dirs = {
    "models/tars_v3": "🧠 Brain (Mamba-2 + MinGRU)",
    "models/reflex": "🔁 Reflex Classifier",
    "models/mingru": "🧪 MinGRU Checkpoints", 
    "models/spiking": "⚡ SNN Synapses",
    "models/voice": "🎙 Voice (Whisper)",
}

total_size = 0
for path, name in model_dirs.items():
    p = ROOT / path
    if p.exists():
        files = [f for f in p.glob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        total_size += size
        mb = size / 1024 / 1024
        print(f"  {name:40s} {len(files)} files, {mb:.1f} MB")
    else:
        print(f"  {name:40s} — не обучено")

print(f"\n  Total: {total_size / 1024 / 1024:.0f} MB")
print(f"{'═'*60}\n")
