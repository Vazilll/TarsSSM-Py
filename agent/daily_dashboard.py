"""
═══════════════════════════════════════════════════════════════
  daily_dashboard.py — Дашборд «Мой день» TARS v3
═══════════════════════════════════════════════════════════════

Единый экран со всеми данными за день.
Собирает информацию из ВСЕХ подсистем ТАРС.
"""

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("Tars.Dashboard")


class DailyDashboard:
    """
    Агрегатор — собирает данные со всех подсистем в один отчёт.
    
    «Доброе утро!» / «Мой день» / «Дашборд» → полный обзор.
    """
    
    def __init__(self, schedule=None, reminders=None, pomodoro=None,
                 learning_helper=None, habit_tracker=None,
                 expenses=None, knowledge_graph=None,
                 system_monitor=None):
        self.schedule = schedule
        self.reminders = reminders
        self.pomodoro = pomodoro
        self.learning_helper = learning_helper
        self.habit_tracker = habit_tracker
        self.expenses = expenses
        self.knowledge_graph = knowledge_graph
        self.system_monitor = system_monitor
    
    def render(self) -> str:
        """Полный дашборд."""
        now = datetime.now()
        hour = now.hour
        
        if hour < 12: greeting = "🌅 Доброе утро"
        elif hour < 18: greeting = "☀️ Добрый день"
        else: greeting = "🌙 Добрый вечер"
        
        lines = [
            f"{greeting}! {now.strftime('%d.%m.%Y, %A')}",
            "═" * 45,
        ]
        
        # 📅 Расписание
        if self.schedule:
            try:
                today = self.schedule.get_today()
                lines.append(f"\n{today}")
                
                next_cls = self.schedule.next_class()
                if "Нет пар" not in next_cls:
                    lines.append(f"  {next_cls}")
            except Exception:
                pass
        
        # 🔔 Напоминания
        if self.reminders:
            try:
                today_rem = self.reminders.list_today()
                if "ничего" not in today_rem:
                    lines.append(f"\n{today_rem}")
            except Exception:
                pass
        
        # 🍅 Учёба  
        if self.pomodoro:
            try:
                stats = self.pomodoro.stats_today()
                if "не учился" not in stats:
                    lines.append(f"\n{stats}")
                else:
                    lines.append("\n🍅 Учёба: пока 0 мин. Скажи «помодоро [предмет]»!")
            except Exception:
                pass
        
        # 📝 Flashcards
        if self.learning_helper:
            try:
                due = self.learning_helper.get_due_cards()
                total = len(self.learning_helper.flashcards)
                if due:
                    lines.append(f"\n📝 Карточки: {len(due)} из {total} ждут повторения")
                elif total > 0:
                    lines.append(f"\n📝 Все {total} карточек повторены ✅")
            except Exception:
                pass
        
        # 🔄 Привычки
        if self.habit_tracker:
            try:
                if self.habit_tracker.habits:
                    lines.append(f"\n🔄 Привычки:")
                    for h in self.habit_tracker.habits:
                        week = h.week_visual()
                        streak = h.get_streak()
                        fire = f" 🔥{streak}" if streak >= 3 else ""
                        lines.append(f"  {week} {h.name}{fire}")
                    
                    motivation = self.habit_tracker.get_motivation()
                    if motivation:
                        lines.append(f"  {motivation}")
            except Exception:
                pass
        
        # 💰 Бюджет
        if self.expenses:
            try:
                if self.expenses.budget_monthly > 0:
                    spent = self.expenses._month_total()
                    remaining = self.expenses.budget_monthly - spent
                    pct = spent / self.expenses.budget_monthly * 100
                    lines.append(f"\n💰 Бюджет: {spent:.0f}/{self.expenses.budget_monthly:.0f}р ({pct:.0f}%) | осталось {remaining:.0f}р")
            except Exception:
                pass
        
        # 🕸 Граф знаний
        if self.knowledge_graph:
            try:
                from datetime import date
                today_str = date.today().isoformat()
                import sqlite3
                from pathlib import Path
                _KG_DB = Path(__file__).parent.parent / "data" / "knowledge" / "graph.db"
                conn = sqlite3.connect(str(_KG_DB))
                c = conn.cursor()
                c.execute("SELECT COUNT(*) FROM nodes")
                total = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM nodes WHERE created > ?",
                          (datetime.now().replace(hour=0, minute=0).isoformat(),))
                today_count = c.fetchone()[0]
                conn.close()
                if total > 0:
                    lines.append(f"\n🕸 Граф знаний: {total} узлов (+{today_count} сегодня)")
            except Exception:
                pass
        
        # 💻 Система
        if self.system_monitor:
            try:
                ram = self.system_monitor._get_ram()
                battery = self.system_monitor._get_battery()
                parts = []
                if ram.get("percent", 0) > 75:
                    parts.append(f"RAM {ram['percent']:.0f}%")
                if battery and not battery["plugged"] and battery["percent"] < 30:
                    parts.append(f"🔋 {battery['percent']}%")
                if parts:
                    lines.append(f"\n💻 Система: {' | '.join(parts)}")
            except Exception:
                pass
        
        lines.append(f"\n{'═' * 45}")
        lines.append("Скажи «помодоро», «тест», «привычки» или «расписание»")
        
        return "\n".join(lines)
    
    def render_compact(self) -> str:
        """Компактная версия для уведомлений."""
        parts = []
        
        if self.schedule:
            try:
                next_cls = self.schedule.next_class()
                if "Нет пар" not in next_cls:
                    parts.append(next_cls)
            except Exception:
                pass
        
        if self.learning_helper:
            try:
                due = self.learning_helper.get_due_cards()
                if due:
                    parts.append(f"📝 {len(due)} карточек ждут")
            except Exception:
                pass
        
        if self.habit_tracker:
            try:
                motivation = self.habit_tracker.get_motivation()
                if motivation:
                    parts.append(motivation)
            except Exception:
                pass
        
        return " | ".join(parts) if parts else ""
