"""
═══════════════════════════════════════════════════════════════
  TARS Chat UI — Web Interface (Agent 5)
═══════════════════════════════════════════════════════════════

Refactored from ui/chat/app.py.
Flask-based web chat interface with TarsOrchestrator backend.

Owner: Agent 5 (EXCLUSIVE)
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Optional

logger = logging.getLogger("Tars.ChatUI")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class ChatUI:
    """
    Web-based chat interface for TARS.

    Uses Flask for HTTP + optional WebSocket for streaming.
    Backend: TarsOrchestrator.

    Usage:
        ui = ChatUI(workspace=".")
        ui.run(host="127.0.0.1", port=7860)
    """

    def __init__(self, workspace: str = ".", verbose: bool = True):
        self.workspace = workspace
        self.verbose = verbose
        self._orchestrator = None

    def _get_orchestrator(self):
        """Lazy-init orchestrator."""
        if self._orchestrator is None:
            from agent.orchestrator import TarsOrchestrator
            self._orchestrator = TarsOrchestrator(
                workspace=self.workspace,
                verbose=self.verbose,
            )
        return self._orchestrator

    def run(self, host: str = "127.0.0.1", port: int = 7860):
        """Start Flask web server."""
        try:
            from flask import Flask, request, jsonify, render_template_string
        except ImportError:
            print("❌ Flask not installed: pip install flask")
            return

        app = Flask(__name__)

        HTML = self._build_html()

        @app.route("/")
        def index():
            return render_template_string(HTML)

        @app.route("/api/chat", methods=["POST"])
        def chat():
            data = request.get_json()
            query = data.get("query", "").strip()
            if not query:
                return jsonify({"error": "Empty query"}), 400

            orch = self._get_orchestrator()
            result = asyncio.run(orch.process(query))
            return jsonify({
                "response": result.response,
                "mode": result.mode,
                "engine": result.engine_used,
                "time_ms": round(result.total_time_ms, 1),
                "tokens": result.tokens_generated,
                "safety": result.safety_verdict,
            })

        @app.route("/api/status")
        def status():
            orch = self._get_orchestrator()
            return jsonify(orch.status())

        print(f"\n  🌐 TARS Chat UI: http://{host}:{port}")
        print(f"  Engine: {self._get_orchestrator().status()['engine']}\n")
        app.run(host=host, port=port, debug=False)

    @staticmethod
    def _build_html() -> str:
        """Build embedded chat HTML."""
        return '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TARS Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 16px 24px;
            border-bottom: 1px solid #2a2a4a;
        }
        .header h1 {
            font-size: 20px;
            color: #a78bfa;
        }
        .header .status {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .msg {
            max-width: 80%;
            margin-bottom: 16px;
            padding: 12px 16px;
            border-radius: 12px;
            line-height: 1.5;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .msg.user {
            background: #1e3a5f;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .msg.tars {
            background: #1a1a2e;
            border: 1px solid #2a2a4a;
            border-bottom-left-radius: 4px;
        }
        .msg .meta {
            font-size: 11px;
            color: #666;
            margin-top: 8px;
        }
        .input-area {
            padding: 16px 24px;
            background: #111118;
            border-top: 1px solid #2a2a4a;
            display: flex;
            gap: 12px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #2a2a4a;
            border-radius: 8px;
            background: #1a1a2e;
            color: #e0e0e0;
            font-size: 15px;
            outline: none;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            border-color: #a78bfa;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #7c3aed, #a78bfa);
            color: white;
            font-size: 15px;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        button:hover { opacity: 0.9; }
        button:disabled { opacity: 0.4; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 TARS v3</h1>
        <div class="status" id="status">Loading...</div>
    </div>
    <div class="messages" id="messages"></div>
    <div class="input-area">
        <input type="text" id="input" placeholder="Введите сообщение..."
               onkeypress="if(event.key==='Enter')send()">
        <button onclick="send()" id="sendBtn">Отправить</button>
    </div>
    <script>
        const msgs = document.getElementById('messages');
        const input = document.getElementById('input');
        const btn = document.getElementById('sendBtn');

        fetch('/api/status').then(r=>r.json()).then(s=>{
            document.getElementById('status').textContent =
                `Engine: ${s.engine} | Tools: ${s.tools.join(', ')}`;
        });

        function addMsg(text, cls, meta='') {
            const div = document.createElement('div');
            div.className = 'msg ' + cls;
            div.textContent = text;
            if (meta) {
                const m = document.createElement('div');
                m.className = 'meta';
                m.textContent = meta;
                div.appendChild(m);
            }
            msgs.appendChild(div);
            msgs.scrollTop = msgs.scrollHeight;
        }

        async function send() {
            const q = input.value.trim();
            if (!q) return;
            input.value = '';
            btn.disabled = true;
            addMsg(q, 'user');

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: q})
                });
                const data = await res.json();
                const meta = `${data.engine} | ${data.time_ms}ms | ${data.tokens} tok | ${data.mode}`;
                addMsg(data.response, 'tars', meta);
            } catch(e) {
                addMsg('Ошибка: ' + e.message, 'tars');
            }
            btn.disabled = false;
            input.focus();
        }
        input.focus();
    </script>
</body>
</html>'''


if __name__ == "__main__":
    ui = ChatUI()
    ui.run()
