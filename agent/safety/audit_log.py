"""
═══════════════════════════════════════════════════════════════
  AuditLogger — Append-only JSON-lines Audit Log (Agent 5)
═══════════════════════════════════════════════════════════════

Immutable log of every safety decision, tool invocation, and
user interaction. JSON-lines format for easy parsing.

Owner: Agent 5 (EXCLUSIVE)
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("Tars.Audit")


class AuditLogger:
    """
    Append-only JSON-lines audit logger.

    Each line is a self-contained JSON object with:
      - timestamp (ISO 8601)
      - event_type: "safety", "tool", "user", "system"
      - details: event-specific data
      - session_id: unique session identifier

    File rotation: new file per day, max 10MB per file.
    """

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    DEFAULT_DIR = "data/audit"

    def __init__(self, audit_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.audit_dir = Path(audit_dir or self.DEFAULT_DIR)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or f"s_{int(time.time())}"
        self._lock = threading.Lock()
        self._current_file: Optional[str] = None
        self._event_count = 0

    def _get_file_path(self) -> Path:
        """Get current audit file path (date-based)."""
        date_str = time.strftime("%Y-%m-%d")
        base = self.audit_dir / f"audit_{date_str}.jsonl"

        # Rotate if file too large
        if base.exists() and base.stat().st_size > self.MAX_FILE_SIZE:
            i = 1
            while True:
                rotated = self.audit_dir / f"audit_{date_str}_{i}.jsonl"
                if not rotated.exists() or rotated.stat().st_size < self.MAX_FILE_SIZE:
                    return rotated
                i += 1

        return base

    def _write_event(self, event: Dict[str, Any]):
        """Write single event to audit file (thread-safe, append-only)."""
        with self._lock:
            try:
                path = self._get_file_path()
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                self._event_count += 1
            except Exception as e:
                logger.error(f"AuditLogger write error: {e}")

    # ─────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────

    def log_safety(self, action_type: str, content: str, verdict, context: str = ""):
        """Log a safety check decision."""
        self._write_event({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "type": "safety",
            "session": self.session_id,
            "action_type": action_type,
            "content": content[:500],
            "verdict": verdict.action if hasattr(verdict, "action") else str(verdict),
            "reason": getattr(verdict, "reason", ""),
            "rule_id": getattr(verdict, "rule_id", ""),
            "scores": getattr(verdict, "scores", {}),
            "context": context,
        })

    def log_tool(self, tool_name: str, params: Dict[str, Any],
                 result: str = "", duration: float = 0.0, error: str = ""):
        """Log a tool invocation."""
        # Sanitize params — don't log full code/command content
        safe_params = {}
        for k, v in params.items():
            sv = str(v)
            safe_params[k] = sv[:200] if len(sv) > 200 else sv

        self._write_event({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "type": "tool",
            "session": self.session_id,
            "tool": tool_name,
            "params": safe_params,
            "result": result[:300],
            "duration_ms": round(duration * 1000, 1),
            "error": error,
        })

    def log_user(self, query: str, intent: str = "", response_preview: str = ""):
        """Log a user interaction."""
        self._write_event({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "type": "user",
            "session": self.session_id,
            "query": query[:500],
            "intent": intent,
            "response_preview": response_preview[:200],
        })

    def log_system(self, event: str, details: Dict[str, Any] = None):
        """Log a system event (startup, shutdown, error, etc.)."""
        self._write_event({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "type": "system",
            "session": self.session_id,
            "event": event,
            "details": details or {},
        })

    @property
    def event_count(self) -> int:
        """Total events logged in this session."""
        return self._event_count

    def read_recent(self, n: int = 50) -> list:
        """Read last N events from current audit file."""
        try:
            path = self._get_file_path()
            if not path.exists():
                return []
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            events = []
            for line in lines[-n:]:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return events
        except Exception as e:
            logger.error(f"AuditLogger read error: {e}")
            return []
