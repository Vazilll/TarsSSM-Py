"""
═══════════════════════════════════════════════════════════════
  TARS Hub — FastAPI Server (Hardened)
═══════════════════════════════════════════════════════════════

Security features:
  - API key authentication (TARS_API_KEY env variable)
  - Rate limiting (per-IP, sliding window)
  - Safe file uploads (sanitized filename, size limit)
  - Sanitized telemetry (no internal model details)
  - CORS configuration
  - Healthcheck endpoint
  - Binds to 127.0.0.1 by default (use --host to override)
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, UploadFile, File, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import logging
import asyncio
import os
import re
import time
import uuid
import queue
import secrets
import tempfile
from collections import defaultdict
from pathlib import Path

from agent.gie import GieAgent
from agent.moira import MoIRA
from memory.leann import TarsMemory
from memory.titans import TitansMemory
from memory.store import TarsStorage
try:
    from brain.mamba2.model import TarsMamba2LM as TarsBrain
except ImportError:
    TarsBrain = None
from brain.reflexes.reflex_dispatcher import ReflexDispatcher
from sensory.vision import TarsVision
from sensory.voice import TarsVoice

logger = logging.getLogger("Tars.Hub")


# ═══════════════════════════════════════════
# Security: API Key Authentication
# ═══════════════════════════════════════════

# Set via environment: TARS_API_KEY=your-secret-key
# If not set, a random key is generated and printed at startup.
_API_KEY = os.environ.get("TARS_API_KEY", "")


def _verify_api_key(request: Request):
    """FastAPI dependency: verify X-API-Key header."""
    if not _API_KEY:
        return  # No key configured — allow (dev mode)
    
    provided = request.headers.get("X-API-Key", "")
    if not secrets.compare_digest(provided, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ═══════════════════════════════════════════
# Security: Rate Limiter (per-IP sliding window)
# ═══════════════════════════════════════════

class _RateLimiter:
    """Simple in-memory rate limiter. 
    Limits: 30 requests per 60 seconds per IP.
    """
    def __init__(self, max_requests: int = 30, window_sec: int = 60):
        self.max_requests = max_requests
        self.window_sec = window_sec
        self._requests: dict = defaultdict(list)  # ip -> [timestamps]
    
    def check(self, client_ip: str) -> bool:
        """Returns True if request is allowed."""
        now = time.time()
        cutoff = now - self.window_sec
        # Prune old entries
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > cutoff
        ]
        if len(self._requests[client_ip]) >= self.max_requests:
            return False
        self._requests[client_ip].append(now)
        return True

_rate_limiter = _RateLimiter(max_requests=30, window_sec=60)


async def _check_rate_limit(request: Request):
    """FastAPI dependency: rate limit check."""
    client_ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.check(client_ip):
        raise HTTPException(
            status_code=429, 
            detail="Rate limit exceeded. Try again later."
        )


# ═══════════════════════════════════════════
# Lifespan (replaces deprecated @app.on_event)
# ═══════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global _API_KEY
    if not _API_KEY:
        _API_KEY = secrets.token_urlsafe(32)
        logger.warning(f"TARS Hub: No TARS_API_KEY set. Generated: {_API_KEY}")
        logger.warning("Set TARS_API_KEY env variable for production use.")
    else:
        logger.info("TARS Hub: API key authentication enabled.")
    
    logger.info("Tars Hub: Системы активированы. Режим максимального функционала.")
    yield
    logger.info("Tars Hub: Shutting down.")


# ═══════════════════════════════════════════
# App Setup
# ═══════════════════════════════════════════

app = FastAPI(
    title="Tars Python Hub - Ultimate Stack",
    lifespan=lifespan,
)

# CORS: allow only localhost by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # dev frontend
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Singleton Services
memory = TarsMemory()
titans = TitansMemory()
try:
    brain = TarsBrain.load_pretrained()[0] if TarsBrain else None
except Exception as e:
    logger.warning(f"Brain load failed: {e}")
    brain = None
moira = MoIRA()
storage = TarsStorage()
vision = TarsVision()
voice = TarsVoice()
dispatcher = ReflexDispatcher(memory=memory)
gie = GieAgent(brain=brain, moira=moira, memory=memory, titans=titans)


# ═══════════════════════════════════════════
# Safe File Upload Helper
# ═══════════════════════════════════════════

_ALLOWED_AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".webm", ".m4a"}
_MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


def _safe_temp_path(filename: str) -> str:
    """Generate a safe temporary file path, preventing path traversal.
    
    Strips directory components, validates extension, and uses 
    a random UUID prefix to prevent collisions.
    """
    # Strip all path components — only keep the basename
    safe_name = Path(filename).name
    # Remove any remaining suspicious characters
    safe_name = re.sub(r'[^\w\-.]', '_', safe_name)
    # Validate extension
    ext = Path(safe_name).suffix.lower()
    if ext not in _ALLOWED_AUDIO_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported audio format: {ext}")
    # Use system temp directory with UUID
    return os.path.join(tempfile.gettempdir(), f"tars_{uuid.uuid4().hex[:8]}{ext}")


# ═══════════════════════════════════════════
# Endpoints  
# ═══════════════════════════════════════════

@app.get("/health")
async def health():
    """Healthcheck for monitoring and load balancers."""
    return {
        "status": "ok",
        "brain_loaded": brain is not None,
        "memory_docs": len(memory.leann.texts) if hasattr(memory, 'leann') else 0,
    }


@app.post("/voice_interaction", dependencies=[Depends(_verify_api_key), Depends(_check_rate_limit)])
async def voice_interaction(file: UploadFile = File(...)):
    """
    Обработка голоса через волновую архитектуру:
      Audio → Whisper + IntonationSensor → текст + эмоция
      → ReflexDispatcher (7 сенсоров) → ReflexContext
      → GIE (brain.think с контекстом) → ответ
      → Piper TTS (адаптивный к эмоции) → озвучка
    """
    # Safe file handling — prevents path traversal
    temp_path = _safe_temp_path(file.filename or "audio.wav")
    
    try:
        # Read with size limit
        content = await file.read()
        if len(content) > _MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large (max {_MAX_UPLOAD_BYTES // (1024*1024)} MB)")
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # 1. Транскрипция + Интонация (параллельно)
        text, intonation_data = await voice.transcribe_with_intonation(temp_path)
    finally:
        # Always cleanup temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass
    
    if not text:
        return {"status": "ignored", "reason": "No speech detected (Silero VAD)"}
    
    # 2. ReflexDispatcher: 7 сенсоров с данными интонации
    ctx = dispatcher.dispatch(text, intonation_data=intonation_data)
    
    if ctx.can_handle_fast and ctx.fast_response:
        response = ctx.fast_response
        emotion = ctx.voice_emotion if ctx.has_voice_data else "neutral"
    else:
        response = await gie.execute_goal(text)
        emotion = ctx.voice_emotion if ctx.has_voice_data else ctx.dominant_emotion
    
    # 3. Ответ голосом (адаптивный Piper)
    await voice.speak(response, emotion=emotion)
    
    # 4. История для ContextSensor
    dispatcher.add_to_history(text, response, ctx.intent)
    
    return {
        "input": text,
        "response": response,
        "reflex": ctx.summary_line(),
        "intonation": intonation_data,
        "supplement": ctx.is_supplement,
    }


@app.post("/execute", dependencies=[Depends(_verify_api_key), Depends(_check_rate_limit)])
async def execute(goal: str):
    """
    Текстовое управление через ReflexDispatcher.
    """
    # Input validation
    if len(goal) > 4096:
        raise HTTPException(status_code=400, detail="Goal too long (max 4096 chars)")
    if not goal.strip():
        raise HTTPException(status_code=400, detail="Goal cannot be empty")
    
    ctx = dispatcher.dispatch(goal)
    
    if ctx.can_handle_fast and ctx.fast_response:
        response = ctx.fast_response
    else:
        response = await gie.execute_goal(goal)
    
    # Озвучиваем ответ
    await voice.speak(response, emotion=ctx.dominant_emotion)
    
    dispatcher.add_to_history(goal, response, ctx.intent)
    return {"status": "done", "goal": goal, "response": response, "reflex": ctx.summary_line()}


@app.websocket("/ws/voice_stream")
async def voice_stream(websocket: WebSocket):
    """
    WebSocket: потоковый голос с supplement injection.
    Requires API key in query param: ws://host/ws/voice_stream?api_key=<key>
    """
    # WebSocket auth via query param
    if _API_KEY:
        key = websocket.query_params.get("api_key", "")
        if not secrets.compare_digest(key, _API_KEY):
            await websocket.close(code=4001, reason="Invalid API key")
            return
    
    await websocket.accept()
    sup_queue = queue.Queue()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio_segment":
                text = data.get("text", "")
                intonation = data.get("intonation", {})
                is_first = data.get("is_first", True)
                
                # Input length validation
                if len(text) > 4096:
                    text = text[:4096]
                
                if is_first:
                    ctx = dispatcher.dispatch(text, intonation_data=intonation)
                    response = await gie.execute_goal(text)
                    await websocket.send_json({
                        "type": "response",
                        "text": response,
                        "reflex": ctx.summary_line(),
                    })
                else:
                    sup_queue.put({
                        "text": text,
                        "tokens": None,
                        "intonation": intonation,
                    })
                    await websocket.send_json({
                        "type": "supplement_received",
                        "text": text,
                    })
            
            elif data.get("type") == "stop":
                break
    except Exception as e:
        logger.error(f"Voice stream error: {e}")
    finally:
        await websocket.close()


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket):
    """Sanitized telemetry — no internal model details."""
    # WebSocket auth
    if _API_KEY:
        key = websocket.query_params.get("api_key", "")
        if not secrets.compare_digest(key, _API_KEY):
            await websocket.close(code=4001, reason="Invalid API key")
            return
    
    await websocket.accept()
    try:
        while True:
            v_data = await vision.analyze_workspace()
            await websocket.send_json({
                "vision": {"status": v_data.get("status", "unknown")},
                "brain": {
                    "model_loaded": brain is not None,
                    # Removed: n_layers, d_model, params (sensitive architecture info)
                },
                "memory": {
                    "leann_docs": len(memory.leann.texts) if hasattr(memory, 'leann') else 0,
                    # Removed: storage_facts (accesses private _hub._fact_log)
                },
                "session": {
                    "goals_processed": gie.state.get("total_processed", 0),
                },
                "dispatcher": dispatcher.get_stats(),
            })
            await asyncio.sleep(1.0)
    except Exception as e:
        logger.debug(f"Telemetry disconnected: {e}")


# Static files mount (after all API routes)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TARS Hub Server")
    parser.add_argument("--host", default="127.0.0.1", 
                        help="Bind host (default: 127.0.0.1, use 0.0.0.0 with caution)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.host == "0.0.0.0":
        logger.warning("⚠ Binding to 0.0.0.0 — server accessible from any network device!")
        logger.warning("  Ensure TARS_API_KEY is set for production use.")
    
    uvicorn.run(app, host=args.host, port=args.port)
