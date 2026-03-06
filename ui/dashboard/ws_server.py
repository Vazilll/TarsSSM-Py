"""
═══════════════════════════════════════════════════════════════
  TARS Thinking Dashboard — WebSocket Server
═══════════════════════════════════════════════════════════════

FastAPI WebSocket endpoint for real-time thinking visualization.
Streams wave updates, IDME rounds, session start/end events
from ThinkingLogger to the Thinking Dashboard.

Usage:
  python -m ui.ws_server          # standalone (port 7861)
  
  # or integrate into existing app:
  from ui.ws_server import ThinkingBroadcaster, get_broadcaster
  broadcaster = get_broadcaster()
  model.thinking_logger.set_broadcast_hook(broadcaster.broadcast)
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import Set, Optional, Dict, Any

# Добавляем корень проекта в path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse
except ImportError:
    print("pip install fastapi uvicorn websockets")
    sys.exit(1)

logger = logging.getLogger("Tars.WS")

# ═══════════════════════════════════════════════════════════════
#  ThinkingBroadcaster — Singleton for WebSocket broadcast
# ═══════════════════════════════════════════════════════════════

class ThinkingBroadcaster:
    """
    Manages connected WebSocket clients and broadcasts
    thinking events to all of them.
    
    Thread-safe: can be called from model's think() thread.
    Uses asyncio.run_coroutine_threadsafe for cross-thread delivery.
    """
    
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for thread-safe broadcasting."""
        self._loop = loop
    
    async def connect(self, ws: WebSocket):
        """Accept and register a new WebSocket client."""
        await ws.accept()
        self.clients.add(ws)
        logger.info(f"Dashboard client connected ({len(self.clients)} total)")
    
    async def disconnect(self, ws: WebSocket):
        """Remove a disconnected client."""
        self.clients.discard(ws)
        logger.info(f"Dashboard client disconnected ({len(self.clients)} remaining)")
    
    async def _send_to_all(self, data: dict):
        """Send JSON data to all connected clients."""
        if not self.clients:
            return
        
        payload = json.dumps(data, ensure_ascii=False, default=str)
        disconnected = set()
        
        for ws in self.clients.copy():
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.add(ws)
        
        # Cleanup disconnected
        for ws in disconnected:
            self.clients.discard(ws)
    
    def broadcast(self, data: dict):
        """
        Thread-safe broadcast.
        
        Called from ThinkingLogger's hook (potentially from a different thread).
        If we're already in an async context, schedule directly.
        Otherwise, use run_coroutine_threadsafe.
        """
        if not self.clients:
            return
        
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._send_to_all(data), self._loop
            )
        else:
            # Fallback: try to get running loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_to_all(data))
            except RuntimeError:
                # No event loop — can't send
                pass
    
    def broadcast_session_start(self, query: str, task_type: str, p_threshold: float,
                                error_pred: float = 0.0, estimated_waves: int = 6,
                                estimated_depth: int = 12, total_blocks: int = 12,
                                source_weights: dict = None, rag_skipped: bool = False,
                                cache_hit: bool = False, cache_similarity: float = 0.0):
        """Broadcast session start event with predictive complexity."""
        self.broadcast({
            "type": "session_start",
            "query": query[:200],
            "task_type": task_type,
            "p_threshold": p_threshold,
            "error_pred": error_pred,
            "estimated_waves": estimated_waves,
            "estimated_depth": estimated_depth,
            "total_blocks": total_blocks,
            "source_weights": source_weights or {},
            "rag_skipped": rag_skipped,
            "cache_hit": cache_hit,
            "cache_similarity": cache_similarity,
        })
    
    def broadcast_wave_update(
        self, wave: int,
        block_l: dict, block_r: dict,
        merge_alpha: float,
        ia: dict, critic: dict,
        thinking: dict,
        doubt: Optional[dict] = None,
    ):
        """Broadcast a wave update event."""
        self.broadcast({
            "type": "wave_update",
            "wave": wave,
            "block_l": block_l,
            "block_r": block_r,
            "merge_alpha": merge_alpha,
            "ia": ia,
            "critic": critic,
            "thinking": thinking,
            "doubt": doubt or {},
        })
    
    def broadcast_idme_round(self, round_num: int, matrices_recruited: int, best_p_delta: float):
        """Broadcast an IDME expansion round event."""
        self.broadcast({
            "type": "idme_round",
            "round": round_num,
            "matrices_recruited": matrices_recruited,
            "best_p_delta": best_p_delta,
        })
    
    def broadcast_session_end(self, stats: dict):
        """Broadcast session end with final stats + brain system status."""
        doubt_data = stats.get("doubt_scores", {})
        sleep_data = stats.get("sleep", {})
        
        self.broadcast({
            "type": "session_end",
            "total_ms": stats.get("total_ms", 0),
            "waves": stats.get("waves", 0),
            "blocks": stats.get("blocks_executed", 0),
            "final_p": stats.get("final_p", 0),
            "converged": stats.get("converged", False),
            "confidence": stats.get("tc_confidence", 0),
            "matrices_recruited": stats.get("matrices_recruited", 0),
            "rag_complete": stats.get("rag_all_loaded", True),
            "doubt": doubt_data,
            "sleep": sleep_data,
            "self_verify": stats.get("self_verify_consistency", None),
            "spec_accept_rate": stats.get("spec_accept_rate", None),
        })
    
    @property
    def client_count(self) -> int:
        return len(self.clients)


# ═══ Singleton ═══
_broadcaster: Optional[ThinkingBroadcaster] = None

def get_broadcaster() -> ThinkingBroadcaster:
    """Get or create the global ThinkingBroadcaster singleton."""
    global _broadcaster
    if _broadcaster is None:
        _broadcaster = ThinkingBroadcaster()
    return _broadcaster


# ═══════════════════════════════════════════════════════════════
#  FastAPI Application
# ═══════════════════════════════════════════════════════════════

app = FastAPI(title="TARS Thinking Dashboard WS", version="1.0")

# CORS for localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    """Store the event loop for thread-safe broadcasting."""
    loop = asyncio.get_running_loop()
    get_broadcaster().set_loop(loop)
    logger.info("ThinkingBroadcaster event loop set")


@app.get("/")
async def dashboard_page():
    """Serve the thinking dashboard HTML."""
    html_path = Path(__file__).parent / "thinking_dashboard.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    return {"error": "thinking_dashboard.html not found"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    b = get_broadcaster()
    return {
        "status": "ok",
        "clients": b.client_count,
    }


@app.websocket("/ws/thinking")
async def thinking_ws(ws: WebSocket):
    """
    WebSocket endpoint for real-time thinking data.
    
    Protocol:
      Server → Client (JSON):
        {type: "session_start", query, task_type, p_threshold}
        {type: "wave_update", wave, block_l, block_r, merge_alpha, ia, critic, thinking, doubt}
        {type: "idme_round", round, matrices_recruited, best_p_delta}
        {type: "session_end", total_ms, waves, blocks, final_p, converged, confidence, doubt}
    """
    broadcaster = get_broadcaster()
    await broadcaster.connect(ws)
    
    try:
        # Keep connection alive — listen for client pings
        while True:
            try:
                data = await ws.receive_text()
                # Client can send ping/pong or commands
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"WS error: {e}")
    finally:
        await broadcaster.disconnect(ws)


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

def main():
    """Run the WebSocket server standalone."""
    import uvicorn
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    print("=" * 55)
    print("  TARS Thinking Dashboard — WebSocket Server")
    print("  WS:   ws://localhost:7861/ws/thinking")
    print("  HTTP: http://localhost:7861  (dashboard)")
    print("  Health: http://localhost:7861/health")
    print("=" * 55)
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=7861,
        log_level="info",
    )


if __name__ == "__main__":
    main()
