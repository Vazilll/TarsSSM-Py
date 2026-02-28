"""
═══════════════════════════════════════════════════════════════
  clipboard_manager.py — История буфера обмена TARS v3
═══════════════════════════════════════════════════════════════

Автоматически логирует всё что пользователь копирует.
"Что я копировал вчера?" → история
"Найди URL который я копировал" → поиск
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger("Tars.Clipboard")

_ROOT = Path(__file__).parent.parent
_CLIP_DB = _ROOT / "data" / "clipboard.json"


class ClipboardManager:
    """
    Менеджер буфера обмена — логирует всё что копируется.
    Фоновый поток мониторит clipboard каждые 2 секунды.
    """
    
    def __init__(self, max_history: int = 5000):
        self.history: List[Dict] = []
        self.max_history = max_history
        self._last_content = ""
        self._running = True
        self._load()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def get_history(self, n: int = 10) -> str:
        """Последние N записей из буфера."""
        recent = self.history[-n:][::-1]
        if not recent:
            return "📋 История буфера пуста."
        
        lines = [f"📋 Буфер обмена (последние {len(recent)}):\n"]
        for i, entry in enumerate(recent, 1):
            t = entry.get("time", "")[:16].replace("T", " ")
            text = entry["text"][:80]
            if len(entry["text"]) > 80:
                text += "..."
            lines.append(f"  {i}. [{t}] {text}")
        return "\n".join(lines)
    
    def search(self, query: str) -> str:
        """Поиск по истории буфера."""
        query_lower = query.lower()
        results = [
            e for e in self.history 
            if query_lower in e["text"].lower()
        ]
        
        if not results:
            return f"🔍 Не найдено в буфере: «{query}»"
        
        lines = [f"🔍 Найдено в буфере ({len(results)}):\n"]
        for e in results[-5:][::-1]:
            t = e.get("time", "")[:16].replace("T", " ")
            text = e["text"][:100]
            lines.append(f"  [{t}] {text}")
        return "\n".join(lines)
    
    def get_today(self) -> str:
        """Что копировалось сегодня."""
        today = datetime.now().date().isoformat()
        todays = [e for e in self.history if e.get("time", "")[:10] == today]
        
        if not todays:
            return "📋 Сегодня ничего не копировалось."
        
        lines = [f"📋 Скопировано сегодня ({len(todays)} записей):\n"]
        for e in todays[-10:][::-1]:
            t = e["time"][11:16]
            text = e["text"][:80]
            lines.append(f"  {t} — {text}")
        return "\n".join(lines)
    
    def _monitor_loop(self):
        """Фоновый мониторинг буфера обмена."""
        while self._running:
            try:
                import ctypes
                
                if not ctypes.windll.user32.OpenClipboard(0):
                    time.sleep(2)
                    continue
                
                try:
                    # CF_UNICODETEXT = 13
                    handle = ctypes.windll.user32.GetClipboardData(13)
                    if handle:
                        ctypes.windll.kernel32.GlobalLock.restype = ctypes.c_wchar_p
                        text = ctypes.windll.kernel32.GlobalLock(handle)
                        if text and text != self._last_content:
                            self._last_content = text
                            self._add_entry(text)
                        if text:
                            ctypes.windll.kernel32.GlobalUnlock(handle)
                finally:
                    ctypes.windll.user32.CloseClipboard()
            
            except Exception as e:
                logger.debug(f"Clipboard error: {e}")
            
            time.sleep(2)
    
    def _add_entry(self, text: str):
        """Добавить запись."""
        if not text or len(text.strip()) < 2:
            return
        
        entry = {
            "text": text[:2000],
            "time": datetime.now().isoformat(),
        }
        self.history.append(entry)
        
        # Лимит
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        self._save()
    
    def stop(self):
        self._running = False
    
    def _save(self):
        _CLIP_DB.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CLIP_DB, "w", encoding="utf-8") as f:
                json.dump(self.history[-self.max_history:], f, ensure_ascii=False)
        except Exception:
            pass
    
    def _load(self):
        if _CLIP_DB.exists():
            try:
                with open(_CLIP_DB, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
                logger.info(f"Clipboard: {len(self.history)} records loaded")
            except Exception:
                pass
