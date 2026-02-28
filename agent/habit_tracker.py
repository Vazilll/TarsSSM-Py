"""
═══════════════════════════════════════════════════════════════
  habit_tracker.py — Трекер привычек и целей TARS v3
═══════════════════════════════════════════════════════════════

"Привычка: читать 30 мин перед сном"
"Цель: учиться 3 часа в день"
"Отметь привычку чтение"
"Покажи мои streaks"
"""

import json
import logging
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger("Tars.HabitTracker")

_ROOT = Path(__file__).parent.parent
_HABITS_DB = _ROOT / "data" / "habits.json"


class Habit:
    """Одна привычка."""
    def __init__(self, name: str, frequency: str = "daily",
                 target_value: float = 1, unit: str = "раз"):
        self.name = name
        self.frequency = frequency  # daily, weekly
        self.target_value = target_value
        self.unit = unit
        self.created = datetime.now().isoformat()
        self.check_log: Dict[str, float] = {}  # "2026-02-28" → value
    
    def check(self, value: float = 1, day: str = None) -> str:
        """Отметить выполнение привычки."""
        day = day or date.today().isoformat()
        self.check_log[day] = self.check_log.get(day, 0) + value
        
        streak = self.get_streak()
        msg = f"✅ {self.name}: отмечено"
        if streak >= 3:
            msg += f" | 🔥 Streak: {streak} дней подряд!"
        return msg
    
    def get_streak(self) -> int:
        """Текущая серия дней подряд."""
        streak = 0
        check_day = date.today()
        while True:
            day_str = check_day.isoformat()
            if day_str in self.check_log and self.check_log[day_str] >= self.target_value:
                streak += 1
                check_day -= timedelta(days=1)
            else:
                break
        return streak
    
    def get_best_streak(self) -> int:
        """Лучшая серия."""
        if not self.check_log:
            return 0
        
        sorted_days = sorted(self.check_log.keys())
        best = 0
        current = 0
        prev_date = None
        
        for day_str in sorted_days:
            if self.check_log[day_str] >= self.target_value:
                d = date.fromisoformat(day_str)
                if prev_date and (d - prev_date).days == 1:
                    current += 1
                else:
                    current = 1
                best = max(best, current)
                prev_date = d
            else:
                current = 0
                prev_date = None
        
        return best
    
    def completion_rate(self, days: int = 30) -> float:
        """Процент выполнения за последние N дней."""
        completed = 0
        for i in range(days):
            day = (date.today() - timedelta(days=i)).isoformat()
            if day in self.check_log and self.check_log[day] >= self.target_value:
                completed += 1
        return completed / days * 100
    
    def week_visual(self) -> str:
        """Визуализация недели."""
        days = []
        for i in range(6, -1, -1):
            d = (date.today() - timedelta(days=i)).isoformat()
            if d in self.check_log and self.check_log[d] >= self.target_value:
                days.append("🟢")
            elif d in self.check_log:
                days.append("🟡")
            else:
                days.append("🔴")
        return "".join(days)
    
    def to_dict(self):
        return {
            "name": self.name, "frequency": self.frequency,
            "target_value": self.target_value, "unit": self.unit,
            "created": self.created, "check_log": self.check_log,
        }
    
    @staticmethod
    def from_dict(d):
        h = Habit(d["name"], d.get("frequency", "daily"),
                  d.get("target_value", 1), d.get("unit", "раз"))
        h.created = d.get("created", "")
        h.check_log = d.get("check_log", {})
        return h


class HabitTracker:
    """
    Трекер привычек с streak-мотивацией.
    """
    
    def __init__(self):
        self.habits: List[Habit] = []
        self.goals: List[Dict] = []
        self._load()
    
    def add_habit(self, name: str, target: float = 1, 
                  unit: str = "раз", frequency: str = "daily") -> str:
        """Добавить привычку."""
        # Проверка дубликатов
        for h in self.habits:
            if h.name.lower() == name.lower():
                return f"⚠️ Привычка «{name}» уже существует."
        
        habit = Habit(name, frequency, target, unit)
        self.habits.append(habit)
        self._save()
        return f"✅ Привычка добавлена: «{name}» ({target} {unit}/день)"
    
    def check_habit(self, name: str, value: float = 1) -> str:
        """Отметить привычку."""
        for h in self.habits:
            if name.lower() in h.name.lower():
                result = h.check(value)
                self._save()
                return result
        return f"❌ Привычка «{name}» не найдена."
    
    def add_goal(self, name: str, target: str, deadline: str = None) -> str:
        """Добавить цель."""
        goal = {
            "name": name, "target": target, "deadline": deadline,
            "created": datetime.now().isoformat(), "done": False,
            "progress": 0,
        }
        self.goals.append(goal)
        self._save()
        dl = f" (до {deadline})" if deadline else ""
        return f"🎯 Цель добавлена: «{name}» — {target}{dl}"
    
    def update_goal(self, name: str, progress: int) -> str:
        """Обновить прогресс цели."""
        for g in self.goals:
            if name.lower() in g["name"].lower() and not g["done"]:
                g["progress"] = min(progress, 100)
                if progress >= 100:
                    g["done"] = True
                    self._save()
                    return f"🏆 Цель достигнута: «{g['name']}»! Поздравляю!"
                self._save()
                bar_len = int(progress / 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                return f"🎯 {g['name']}: [{bar}] {progress}%"
        return f"❌ Цель «{name}» не найдена."
    
    def get_overview(self) -> str:
        """Обзор привычек и целей."""
        lines = ["📊 Привычки и цели:\n"]
        
        if self.habits:
            lines.append("🔄 Привычки:")
            for h in self.habits:
                streak = h.get_streak()
                week = h.week_visual()
                rate = h.completion_rate(30)
                fire = f" 🔥{streak}" if streak >= 3 else ""
                lines.append(f"  {week} {h.name} ({rate:.0f}% за месяц){fire}")
        
        active_goals = [g for g in self.goals if not g["done"]]
        if active_goals:
            lines.append("\n🎯 Цели:")
            for g in active_goals:
                p = g["progress"]
                bar_len = int(p / 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)
                dl = f" (до {g['deadline']})" if g.get("deadline") else ""
                lines.append(f"  [{bar}] {g['name']}: {p}%{dl}")
        
        done_goals = [g for g in self.goals if g["done"]]
        if done_goals:
            lines.append(f"\n🏆 Достигнуто: {len(done_goals)} целей")
        
        if not self.habits and not self.goals:
            lines.append("  Пусто! Скажи «привычка: читать 30 мин» чтобы начать.")
        
        return "\n".join(lines)
    
    def get_motivation(self) -> Optional[str]:
        """Мотивационное сообщение (вызывается утром)."""
        if not self.habits:
            return None
        
        # Лучший streak
        best_habit = max(self.habits, key=lambda h: h.get_streak())
        streak = best_habit.get_streak()
        
        if streak >= 7:
            return f"🔥 {streak} дней подряд «{best_habit.name}»! Не ломай streak!"
        elif streak >= 3:
            return f"💪 Серия {streak} дней «{best_habit.name}». Продолжай!"
        
        # Проверяем не отмечалось ли вчера
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        unchecked = [h for h in self.habits 
                    if yesterday not in h.check_log or h.check_log[yesterday] < h.target_value]
        if unchecked:
            names = ", ".join(h.name for h in unchecked[:2])
            return f"⚠️ Вчера пропущено: {names}. Сегодня наверстаем?"
        
        return None
    
    def remove_habit(self, name: str) -> str:
        """Удалить привычку."""
        before = len(self.habits)
        self.habits = [h for h in self.habits if name.lower() not in h.name.lower()]
        self._save()
        if len(self.habits) < before:
            return f"❌ Привычка «{name}» удалена."
        return f"Не найдено: {name}"
    
    def _save(self):
        _HABITS_DB.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "habits": [h.to_dict() for h in self.habits],
            "goals": self.goals,
        }
        with open(_HABITS_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        if _HABITS_DB.exists():
            try:
                with open(_HABITS_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.habits = [Habit.from_dict(d) for d in data.get("habits", [])]
                self.goals = data.get("goals", [])
                logger.info(f"Habits: {len(self.habits)} habits, {len(self.goals)} goals")
            except Exception:
                pass
