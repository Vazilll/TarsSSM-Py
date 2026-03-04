"""
═══════════════════════════════════════════════════
  TARS Model Server — HTTP API для Rust Hub
═══════════════════════════════════════════════════

Запуск:
  python serve_model.py --model models/mamba2/brain_best.pt 
                        --port 8090

API:
  POST /generate  { "prompt": "...", "max_tokens": 256 }
  → { "text": "...", "tokens_per_sec": 42.5, "converged": true }

  POST /generate_stream  { "prompt": "...", "max_tokens": 256 }
  → SSE stream: data: {"token": "...", "done": false}

  GET /health
  → { "status": "ok", "model": "TarsMamba2LM", "d_model": 2048 }
"""

import os
import sys
import time
import json
import argparse
import logging
import asyncio
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tars.Server")

ROOT = Path(__file__).resolve().parent.parent  # hub/ → project root
sys.path.insert(0, str(ROOT))

# Lazy load torch + model
_model = None
_tokenizer = None
_device = None


def _load_model(model_path: str, device: str = "auto"):
    """Load TarsMamba2LM from checkpoint."""
    global _model, _tokenizer, _device
    
    import torch
    from brain.mamba2.model import TarsMamba2LM
    
    if device == "auto":
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _device = torch.device(device)
    
    logger.info(f"Loading model from {model_path} on {_device}...")
    
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract config
    config = ckpt.get('config', {})
    d_model = config.get('d_model', 2048)
    n_layers = config.get('n_layers', 24)
    vocab_size = config.get('vocab_size', 256)
    
    _model = TarsMamba2LM(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
    ).to(_device)
    
    # Load weights
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    _model.load_state_dict(state, strict=False)
    _model.eval()
    
    logger.info(f"Model loaded: d={d_model}, L={n_layers}, V={vocab_size}")
    logger.info(f"Parameters: {sum(p.numel() for p in _model.parameters()):,}")
    
    # Byte-level tokenizer (trivial)
    class ByteTokenizer:
        def encode(self, text: str):
            return list(text.encode('utf-8', errors='replace'))
        
        def decode(self, tokens):
            return bytes(tokens).decode('utf-8', errors='replace')
    
    _tokenizer = ByteTokenizer()
    return _model


def _generate(prompt: str, max_tokens: int = 256, temperature: float = 0.7,
              task_type: str = "chat"):
    """Generate text from prompt."""
    import torch
    
    tokens = _tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=_device)
    
    start = time.time()
    generated = []
    
    _model.reset_cache()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = _model.think(input_ids, task_type=task_type)
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            token_id = next_token.item()
            
            # EOS check (byte-level: 0 = null)
            if token_id == 0:
                break
            
            generated.append(token_id)
            input_ids = next_token
    
    elapsed = time.time() - start
    text = _tokenizer.decode(generated)
    tps = len(generated) / max(elapsed, 1e-6)
    
    return {
        "text": text,
        "tokens": len(generated),
        "tokens_per_sec": round(tps, 1),
        "elapsed_ms": round(elapsed * 1000, 1),
    }


async def _generate_stream(prompt: str, max_tokens: int = 256,
                           temperature: float = 0.7, task_type: str = "chat"):
    """Generate tokens one by one (for SSE streaming)."""
    import torch

    tokens = _tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=_device)

    _model.reset_cache()

    with torch.no_grad():
        for i in range(max_tokens):
            logits = _model.think(input_ids, task_type=task_type)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            token_id = next_token.item()
            if token_id == 0:
                break

            char = _tokenizer.decode([token_id])
            yield {"token": char, "id": token_id, "pos": i, "done": False}
            input_ids = next_token

    yield {"token": "", "id": 0, "pos": -1, "done": True}


# ─── HTTP Server (aiohttp) ───

async def handle_generate(request):
    """POST /generate — полная генерация."""
    from aiohttp import web
    
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.7)
    task_type = data.get("task_type", "chat")
    
    if not prompt:
        return web.json_response({"error": "prompt required"}, status=400)
    
    result = _generate(prompt, max_tokens, temperature, task_type)
    return web.json_response(result)


async def handle_stream(request):
    """POST /generate_stream — SSE streaming."""
    from aiohttp import web
    
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.7)
    task_type = data.get("task_type", "chat")
    
    if not prompt:
        return web.json_response({"error": "prompt required"}, status=400)
    
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
        }
    )
    await response.prepare(request)
    
    async for chunk in _generate_stream(prompt, max_tokens, temperature, task_type):
        line = f"data: {json.dumps(chunk)}\n\n"
        await response.write(line.encode('utf-8'))
    
    return response


async def handle_health(request):
    """GET /health — проверка состояния."""
    from aiohttp import web
    
    info = {
        "status": "ok",
        "model": "TarsMamba2LM",
        "compiled": getattr(_model, '_compiled', False),
        "device": str(_device),
        "params": sum(p.numel() for p in _model.parameters()),
    }
    return web.json_response(info)


def main():
    parser = argparse.ArgumentParser(description="TARS Model Server")
    parser.add_argument("--model", type=str, default="models/mamba2/brain_best.pt")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    from aiohttp import web
    
    _load_model(args.model, args.device)
    
    app = web.Application()
    app.router.add_post('/generate', handle_generate)
    app.router.add_post('/generate_stream', handle_stream)
    app.router.add_get('/health', handle_health)
    
    logger.info(f"TARS Model Server starting on {args.host}:{args.port}")
    logger.info(f"  POST /generate         — full generation")
    logger.info(f"  POST /generate_stream  — SSE streaming")
    logger.info(f"  GET  /health           — health check")
    
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
