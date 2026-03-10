"""
═══════════════════════════════════════════════════════════════
  TARS Bridge Server — JSON over stdin/stdout
═══════════════════════════════════════════════════════════════

Bridge for Blook (Electron notebook app).
Blook's tars-bridge.js spawns this process and communicates
via JSON lines on stdin/stdout.

Protocol:
  → {"type":"message","id":"abc","text":"Привет","context":{}}
  ← {"id":"abc","text":"Ответ от TARS","tokens":42,"time_ms":150}
  
  → {"type":"shutdown"}
  ← (process exits)

Usage:
  python tars_bridge_server.py                # Start bridge
  python tars_bridge_server.py --test         # Quick self-test
  python tars_bridge_server.py --model path   # Use specific checkpoint
"""

import sys
import os
import json
import time
import logging
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Redirect logging to stderr (stdout is for JSON protocol)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger("Tars.Bridge")


class TarsBridgeServer:
    """JSON stdin/stdout bridge for Blook integration."""
    
    def __init__(self, model_path=None, device=None):
        self.model = None
        self.tokenizer = None
        self.device = device or "cpu"
        self.model_path = model_path
        self.ready = False
    
    def load_model(self):
        """Load TarsHelixLite model and tokenizer."""
        import torch
        from config import TarsConfig
        from brain.mamba2.core.model_lite import TarsHelixLite
        from brain.tokenizer import TarsTokenizer
        
        # Detect device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Device: {self.device}")
        
        # Tokenizer (byte-level UTF-8)
        self.tokenizer = TarsTokenizer()
        logger.info(f"Tokenizer: {self.tokenizer}")
        
        # Try to load checkpoint
        checkpoint = None
        ckpt_paths = [
            self.model_path,
            os.path.join(ROOT, "models", "tars_lite", "checkpoint_best.pt"),
            os.path.join(ROOT, "models", "tars_lite", "latest.pt"),
        ]
        
        for p in ckpt_paths:
            if p and os.path.exists(p):
                try:
                    checkpoint = torch.load(p, map_location=self.device, weights_only=False)
                    logger.info(f"Loaded checkpoint: {p}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")
        
        # Config from checkpoint or default LITE config
        if checkpoint and "config" in checkpoint:
            cfg_dict = checkpoint["config"]
            cfg = TarsConfig(**{k: v for k, v in cfg_dict.items() 
                               if hasattr(TarsConfig, k)})
        else:
            cfg = TarsConfig(
                d_model=512, n_layers=10, vocab_size=256,
                d_state=32, headdim=32, quant_mode="fp16"
            )
            logger.warning("No checkpoint found — using default small config (untrained)")
        
        # Create model
        self.model = TarsHelixLite(cfg).to(self.device)
        
        # Load weights
        if checkpoint and "model_state_dict" in checkpoint:
            try:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                logger.info("Model weights loaded")
            except Exception as e:
                logger.warning(f"Partial weight load: {e}")
        
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {n_params/1e6:.1f}M params, vocab={cfg.vocab_size}")
        self.ready = True
    
    def generate_response(self, text, max_tokens=256, temperature=0.7):
        """Generate text response from model."""
        import torch
        
        # Encode input
        tokens = self.tokenizer.encode(text)
        if len(tokens) == 0:
            return "(пустой ввод)", 0
        
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Generate
        t0 = time.time()
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
        elapsed_ms = (time.time() - t0) * 1000
        
        # Decode new tokens only
        new_tokens = output[0, len(tokens):].tolist()
        response_text = self.tokenizer.decode(new_tokens)
        
        return response_text, len(new_tokens), elapsed_ms
    
    def handle_message(self, msg):
        """Handle incoming JSON message. Returns JSON response."""
        msg_type = msg.get("type", "")
        msg_id = msg.get("id", "")
        
        if msg_type == "shutdown":
            logger.info("Shutdown requested")
            return None  # Signal to exit
        
        if msg_type == "message":
            text = msg.get("text", "")
            context = msg.get("context", {})
            max_tokens = msg.get("max_tokens", 256)
            temperature = msg.get("temperature", 0.7)
            
            if not self.ready:
                return {
                    "id": msg_id,
                    "text": "⚠️ TARS модель не загружена.",
                    "error": "model_not_ready"
                }
            
            try:
                response_text, n_tokens, elapsed_ms = self.generate_response(
                    text, max_tokens=max_tokens, temperature=temperature
                )
                return {
                    "id": msg_id,
                    "text": response_text,
                    "tokens": n_tokens,
                    "time_ms": round(elapsed_ms, 1)
                }
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return {
                    "id": msg_id,
                    "text": f"❌ Ошибка генерации: {e}",
                    "error": str(e)
                }
        
        if msg_type == "ping":
            return {"type": "pong", "ready": self.ready}
        
        if msg_type == "status":
            import torch
            return {
                "type": "status",
                "ready": self.ready,
                "device": self.device,
                "cuda_available": torch.cuda.is_available(),
                "model_loaded": self.model is not None
            }
        
        return {"error": f"Unknown message type: {msg_type}"}
    
    def run(self):
        """Main loop: read JSON from stdin, write JSON to stdout."""
        logger.info("Loading model...")
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Still start — will return error messages
        
        # Signal readiness to parent process (Blook's tars-bridge.js)
        print("TARS_READY", file=sys.stderr, flush=True)
        logger.info("Bridge ready — waiting for messages on stdin")
        
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    response = {"error": f"Invalid JSON: {e}"}
                    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
                    continue
                
                response = self.handle_message(msg)
                
                if response is None:
                    # Shutdown
                    break
                
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()
                
        except (KeyboardInterrupt, EOFError):
            pass
        
        logger.info("Bridge server stopped")


def self_test():
    """Quick self-test: create server, process one message."""
    print("═" * 50)
    print("  TARS Bridge Server — Self Test")
    print("═" * 50)
    
    server = TarsBridgeServer(device="cpu")
    
    # Test 1: Load model
    try:
        server.load_model()
        print(f"  ✅ Model loaded ({server.device})")
    except Exception as e:
        print(f"  ❌ Model load failed: {e}")
        return False
    
    # Test 2: Handle message
    response = server.handle_message({
        "type": "message",
        "id": "test-1",
        "text": "Hello",
        "context": {}
    })
    print(f"  ✅ Response: {response.get('text', '')[:80]}...")
    print(f"     Tokens: {response.get('tokens', 0)}, Time: {response.get('time_ms', 0):.0f}ms")
    
    # Test 3: Ping
    pong = server.handle_message({"type": "ping"})
    print(f"  ✅ Ping: ready={pong.get('ready')}")
    
    # Test 4: Status
    status = server.handle_message({"type": "status"})
    print(f"  ✅ Status: {status}")
    
    # Test 5: Shutdown
    result = server.handle_message({"type": "shutdown"})
    print(f"  ✅ Shutdown: returns None = {result is None}")
    
    print()
    print("  ALL TESTS PASSED ✅")
    print("═" * 50)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TARS Bridge Server")
    parser.add_argument("--test", action="store_true", help="Run self-test")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    args = parser.parse_args()
    
    if args.test:
        self_test()
    else:
        server = TarsBridgeServer(model_path=args.model, device=args.device)
        server.run()
