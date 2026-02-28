"""
═══════════════════════════════════════════════════════════════
  TARS v3 — Web UI Server
═══════════════════════════════════════════════════════════════

Flask-сервер для веб-интерфейса ТАРС.
Подключается к GIE и даёт доступ ко всем 23 подсистемам.

Запуск: python ui/app.py
Открыть: http://localhost:7860
"""

import os
import sys
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime

# Добавляем корень проекта в path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from flask import Flask, request, jsonify, send_from_directory
except ImportError:
    print("pip install flask")
    sys.exit(1)

logger = logging.getLogger("Tars.UI")

app = Flask(__name__, static_folder="static", static_url_path="/static")


# ═══ Ленивая инициализация GIE ═══
_gie = None

def get_gie():
    global _gie
    if _gie is None:
        try:
            from agent.gie import GieAgent
            _gie = GieAgent(brain=None, reflex=None, rrn=None, moira=None, titans=None)
            logger.info("GIE initialized for UI")
        except Exception as e:
            logger.error(f"GIE init failed: {e}")
    return _gie


# ═══ Статические файлы ═══
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ═══ Main Chat API ═══
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    
    if not message:
        return jsonify({"error": "empty message"})
    
    gie = get_gie()
    thinking = []
    
    # Step 1: Analyze intent
    m = message.lower().strip()
    thinking.append("parsing input...")
    thinking.append(f"intent: \"{m[:40]}\"")
    
    # Step 2: Route to subsystem
    subsystem = detect_subsystem(m)
    if subsystem:
        thinking.append(f"subsystem: {subsystem}")
    
    # Step 3: Check available modules
    available = []
    if gie:
        for attr in ['schedule', 'pomodoro', 'habits', 'quiz', 'expenses', 
                     'knowledge_graph', 'clipboard', 'file_helper', 'system_monitor',
                     'learning_helper', 'reminders', 'dashboard', 'meeting_scribe']:
            if getattr(gie, attr, None):
                available.append(attr)
    thinking.append(f"modules active: {len(available)}")
    
    # Step 4: Process
    thinking.append("processing...")
    response = handle_command(message, gie)
    
    if response:
        thinking.append(f"routed -> {subsystem or 'handler'}")
        thinking.append("response ready")
    else:
        thinking.append("no local handler found")
        thinking.append("routing to brain...")
        try:
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(gie.execute_goal(message))
            loop.close()
            response = result.get("text", "")
            if response:
                tier = result.get("tier", "brain")
                thinking.append(f"tier: {tier}")
                thinking.append(f"tokens: {result.get('tokens', '?')}")
                thinking.append("response ready")
            else:
                response = "Не понял. Попробуй ещё раз."
        except Exception:
            thinking.append("brain offline")
            thinking.append("standalone mode")
            response = f"standalone mode\n\n> {message}\n\ncommands: /dashboard /schedule /habits /pomodoro /help"
    
    return jsonify({
        "response": response,
        "thinking": thinking,
        "timestamp": datetime.now().strftime("%H:%M"),
    })


def detect_subsystem(m: str) -> str:
    """Determine which subsystem handles this input."""
    if any(w in m for w in ('дашборд', 'мой день', 'привет', 'доброе утро', '/dashboard')):
        return 'dashboard'
    if any(w in m for w in ('расписание', '/schedule', '/week', 'следующая пара', '/next')):
        return 'schedule'
    if any(w in m for w in ('помодоро', '/pomodoro', 'сколько я учился', '/studystats')):
        return 'pomodoro'
    if any(w in m for w in ('привычк', '/habits', 'отметь', '/check')):
        return 'habits'
    if any(w in m for w in ('тест', '/quiz')):
        return 'quiz'
    if any(w in m for w in ('напомни', '/remind', 'напоминания')):
        return 'reminders'
    if any(w in m for w in ('потратил', 'расходы', 'бюджет', '/expense', '/budget')):
        return 'expenses'
    if any(w in m for w in ('запомни', 'что я знаю', 'граф знаний', 'дневник', '/note', '/graph')):
        return 'knowledge_graph'
    if any(w in m for w in ('система', '/system', 'тормозит')):
        return 'system_monitor'
    if any(w in m for w in ('разбери', 'найди файл', 'дубликаты', '/sort')):
        return 'file_helper'
    if any(w in m for w in ('буфер', '/clipboard')):
        return 'clipboard'
    if any(w in m for w in ('запиши встречу', 'останови запись', '/record')):
        return 'meeting_scribe'
    if any(w in m for w in ('помощь', 'help', '/help', '?')):
        return 'help'
    return None


def handle_command(msg: str, gie) -> str:
    """Обработка команд подсистем."""
    m = msg.lower().strip()
    
    # ═══ Dashboard ═══
    if m in ("дашборд", "мой день", "привет", "доброе утро", "/dashboard"):
        if gie and gie.dashboard:
            return gie.dashboard.render()
        return "Дашборд недоступен"
    
    # ═══ Schedule ═══
    if m in ("расписание", "расписание на сегодня", "/schedule"):
        if gie and gie.schedule:
            return gie.schedule.get_today()
    if m in ("расписание на неделю", "/week"):
        if gie and gie.schedule:
            return gie.schedule.get_week()
    if m in ("следующая пара", "какая следующая пара", "/next"):
        if gie and gie.schedule:
            return gie.schedule.next_class()
    if m.startswith("добавь пару ") or m.startswith("/addclass "):
        if gie and gie.schedule:
            parts = msg.split(maxsplit=5)
            if len(parts) >= 5:
                return gie.schedule.add_class(parts[2], parts[3], parts[4],
                                              parts[5] if len(parts) > 5 else "")
            return "Формат: добавь пару Математика понедельник 9:00 ауд.301"
    
    # ═══ Pomodoro ═══
    if m.startswith("помодоро") or m.startswith("/pomodoro"):
        if gie and gie.pomodoro:
            parts = msg.split(maxsplit=2)
            subject = parts[1] if len(parts) > 1 else "общее"
            return gie.pomodoro.start(subject)
    if m in ("стоп помодоро", "остановить помодоро", "/stoppomodoro"):
        if gie and gie.pomodoro:
            return gie.pomodoro.stop()
    if m in ("сколько я учился", "статистика учёбы", "/studystats"):
        if gie and gie.pomodoro:
            return gie.pomodoro.stats_today()
    if m in ("статистика за неделю", "/weekstats"):
        if gie and gie.pomodoro:
            return gie.pomodoro.stats_week()
    
    # ═══ Habits ═══
    if m in ("привычки", "мои привычки", "/habits"):
        if gie and gie.habits:
            return gie.habits.get_overview()
    if m.startswith("привычка:") or m.startswith("/addhabit "):
        if gie and gie.habits:
            name = msg.split(":", 1)[-1].strip() if ":" in msg else msg.split(maxsplit=1)[-1]
            return gie.habits.add_habit(name)
    if m.startswith("отметь ") or m.startswith("/check "):
        if gie and gie.habits:
            name = msg.split(maxsplit=1)[-1]
            return gie.habits.check_habit(name)
    
    # ═══ Quiz ═══
    if m.startswith("тест") or m.startswith("/quiz"):
        if gie and gie.quiz:
            parts = msg.split(maxsplit=1)
            topic = parts[1] if len(parts) > 1 else None
            return gie.quiz.generate_quiz(topic=topic)
    
    # ═══ Reminders ═══
    if m.startswith("напомни") or m.startswith("/remind "):
        if gie and gie.reminders:
            text = msg.split(maxsplit=1)[-1] if len(msg.split()) > 1 else ""
            return gie.reminders.add(text)
    if m in ("мои напоминания", "напоминания", "/reminders"):
        if gie and gie.reminders:
            return gie.reminders.list_all()
    
    # ═══ Expenses ═══
    if m.startswith("потратил") or m.startswith("/expense "):
        if gie and gie.expenses:
            parts = msg.split()
            for p in parts:
                try:
                    amount = float(p.replace("р", "").replace("₽", ""))
                    desc = msg.replace(p, "").replace("потратил", "").strip()
                    return gie.expenses.add(amount, desc or "без описания")
                except ValueError:
                    continue
    if m in ("расходы", "траты за неделю", "/expenses"):
        if gie and gie.expenses:
            return gie.expenses.stats_week()
    if m in ("бюджет", "расходы за месяц", "/budget"):
        if gie and gie.expenses:
            return gie.expenses.stats_month()
    
    # ═══ Knowledge Graph ═══
    if m.startswith("запомни:") or m.startswith("/note "):
        if gie and gie.knowledge_graph:
            text = msg.split(":", 1)[-1].strip() if ":" in msg else msg.split(maxsplit=1)[-1]
            return gie.knowledge_graph.add_note(text[:80], text)
    if m.startswith("что я знаю про") or m.startswith("/search "):
        if gie and gie.knowledge_graph:
            query = msg.replace("что я знаю про", "").strip()
            if not query:
                query = msg.split(maxsplit=1)[-1]
            return gie.knowledge_graph.search(query)
    if m in ("граф знаний", "/graph"):
        if gie and gie.knowledge_graph:
            return gie.knowledge_graph.get_graph_ascii()
    if m == "дневник" or m.startswith("дневник:"):
        if gie and gie.knowledge_graph:
            content = msg.split(":", 1)[-1].strip() if ":" in msg else None
            return gie.knowledge_graph.daily_note(content)
    
    # ═══ System ═══
    if m in ("система", "статус системы", "/system"):
        if gie and gie.system_monitor:
            return gie.system_monitor.full_report()
    if m in ("почему тормозит", "/whyslow"):
        if gie and gie.system_monitor:
            return gie.system_monitor.why_slow()
    
    # ═══ Files ═══
    if m in ("разбери downloads", "сортировка файлов", "/sort"):
        if gie and gie.file_helper:
            return gie.file_helper.sort_folder()
    if m.startswith("найди файл ") or m.startswith("/findfile "):
        if gie and gie.file_helper:
            query = msg.split(maxsplit=2)[-1]
            return gie.file_helper.search(query)
    if m in ("дубликаты", "/duplicates"):
        if gie and gie.file_helper:
            return gie.file_helper.find_duplicates()
    
    # ═══ Clipboard ═══
    if m in ("буфер", "история буфера", "/clipboard"):
        if gie and gie.clipboard:
            return gie.clipboard.get_history()
    
    # ═══ Meeting ═══
    if m in ("запиши встречу", "начни запись", "/recordmeeting"):
        if gie and gie.meeting_scribe:
            return gie.meeting_scribe.start_recording()
    if m in ("останови запись", "/stoprecording"):
        if gie and gie.meeting_scribe:
            return gie.meeting_scribe.stop_recording()
    
    # ═══ Help ═══
    if m in ("помощь", "help", "/help", "команды", "?"):
        return HELP_TEXT
    
    return None


HELP_TEXT = """🤖 ТАРС — Команды:

📅 Расписание:
  • расписание / расписание на неделю
  • следующая пара
  • добавь пару [предмет] [день] [время] [ауд]

🍅 Учёба:
  • помодоро [предмет] / стоп помодоро
  • сколько я учился / статистика за неделю
  • тест [тема] — генератор квизов

🔄 Привычки:
  • привычки — обзор
  • привычка: [название] — добавить
  • отметь [название] — отметить выполнение

📝 Знания:
  • запомни: [текст] — новая заметка
  • что я знаю про [тема] — поиск
  • граф знаний — визуализация
  • дневник: [текст] — daily note

💰 Финансы:
  • потратил [сумма] на [описание]
  • расходы / бюджет

📁 Файлы:
  • разбери downloads / найди файл [запрос]
  • дубликаты

💻 Система:
  • система / почему тормозит

🔔 Напоминания:
  • напомни [текст] / мои напоминания

🎙 Встречи:
  • запиши встречу / останови запись

📊 Общее:
  • мой день / дашборд
  • буфер — история clipboard
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 50)
    print("  TARS v3 - Web Interface")
    print("  http://localhost:7860")
    print("=" * 50)
    app.run(host="0.0.0.0", port=7860, debug=False)
