"""
═══════════════════════════════════════════════════════════════
  schedule.py — Расписание пар / задач TARS v3
═══════════════════════════════════════════════════════════════

"Какая следующая пара?"
"Расписание на среду"
"Добавь математику в понедельник в 9:00, аудитория 301"
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("Tars.Schedule")

_ROOT = Path(__file__).parent.parent
_SCHEDULE_DB = _ROOT / "data" / "schedule.json"

WEEKDAYS_RU = {
    0: "Понедельник", 1: "Вторник", 2: "Среда",
    3: "Четверг", 4: "Пятница", 5: "Суббота", 6: "Воскресенье",
}

WEEKDAY_PARSE = {
    "понедельник": 0, "пн": 0, "вторник": 1, "вт": 1,
    "среда": 2, "среду": 2, "ср": 2, "четверг": 3, "чт": 3,
    "пятница": 4, "пятницу": 4, "пт": 4,
    "суббота": 5, "субботу": 5, "сб": 5,
    "воскресенье": 6, "воскресенье": 6, "вс": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}


class ScheduleEntry:
    """Одна запись в расписании."""
    def __init__(self, name: str, weekday: int, hour: int, minute: int = 0,
                 duration_min: int = 90, location: str = "", notes: str = ""):
        self.name = name
        self.weekday = weekday  # 0=пн
        self.hour = hour
        self.minute = minute
        self.duration_min = duration_min
        self.location = location
        self.notes = notes
    
    def time_str(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d}"
    
    def end_time_str(self) -> str:
        end = datetime.now().replace(hour=self.hour, minute=self.minute) + timedelta(minutes=self.duration_min)
        return f"{end.hour:02d}:{end.minute:02d}"
    
    def to_dict(self):
        return {
            "name": self.name, "weekday": self.weekday,
            "hour": self.hour, "minute": self.minute,
            "duration_min": self.duration_min,
            "location": self.location, "notes": self.notes,
        }
    
    @staticmethod
    def from_dict(d):
        return ScheduleEntry(
            d["name"], d["weekday"], d["hour"], d.get("minute", 0),
            d.get("duration_min", 90), d.get("location", ""), d.get("notes", "")
        )


class StudentSchedule:
    """
    Расписание студента — пары, дедлайны, задачи.
    """
    
    def __init__(self):
        self.entries: List[ScheduleEntry] = []
        self.deadlines: List[Dict] = []
        self._load()
    
    def add_class(self, name: str, weekday_str: str, 
                  time_str: str, location: str = "", 
                  duration: int = 90) -> str:
        """
        Добавить пару в расписание.
        
        "Математика", "понедельник", "9:00", "ауд. 301"
        """
        weekday = WEEKDAY_PARSE.get(weekday_str.lower())
        if weekday is None:
            return f"❌ Не понял день: «{weekday_str}». Попробуй: понедельник, вторник, ..."
        
        # Парсинг времени
        try:
            parts = time_str.replace(".", ":").split(":")
            hour = int(parts[0])
            minute = int(parts[1]) if len(parts) > 1 else 0
        except (ValueError, IndexError):
            return f"❌ Не понял время: «{time_str}». Формат: 9:00 или 14:30"
        
        entry = ScheduleEntry(name, weekday, hour, minute, duration, location)
        self.entries.append(entry)
        self._save()
        
        day_name = WEEKDAYS_RU[weekday]
        loc = f", {location}" if location else ""
        return f"✅ Добавлено: {name} — {day_name} {time_str}{loc}"
    
    def add_deadline(self, name: str, date_str: str, notes: str = "") -> str:
        """
        Добавить дедлайн.
        "Курсовая", "15.03"
        """
        try:
            parts = date_str.split(".")
            day = int(parts[0])
            month = int(parts[1])
            year = datetime.now().year
            if len(parts) > 2:
                year = int(parts[2])
            deadline_date = datetime(year, month, day)
        except (ValueError, IndexError):
            return f"❌ Не понял дату: «{date_str}». Формат: 15.03 или 15.03.2026"
        
        self.deadlines.append({
            "name": name, "date": deadline_date.isoformat(),
            "notes": notes, "done": False,
        })
        self._save()
        
        days_left = (deadline_date - datetime.now()).days
        return f"✅ Дедлайн: {name} — {date_str} (через {days_left} дней)"
    
    def get_today(self) -> str:
        """Расписание на сегодня."""
        today = datetime.now().weekday()
        return self._format_day(today, "Сегодня")
    
    def get_tomorrow(self) -> str:
        """Расписание на завтра."""
        tomorrow = (datetime.now().weekday() + 1) % 7
        return self._format_day(tomorrow, "Завтра")
    
    def get_day(self, weekday_str: str) -> str:
        """Расписание на конкретный день."""
        weekday = WEEKDAY_PARSE.get(weekday_str.lower())
        if weekday is None:
            return f"❌ Не понял день: «{weekday_str}»"
        return self._format_day(weekday, WEEKDAYS_RU[weekday])
    
    def get_week(self) -> str:
        """Расписание на всю неделю."""
        lines = ["📅 Расписание на неделю:\n"]
        for day_num in range(7):
            day_entries = sorted(
                [e for e in self.entries if e.weekday == day_num],
                key=lambda e: (e.hour, e.minute)
            )
            if day_entries:
                lines.append(f"  {WEEKDAYS_RU[day_num]}:")
                for e in day_entries:
                    loc = f" ({e.location})" if e.location else ""
                    lines.append(f"    {e.time_str()} — {e.name}{loc}")
        
        if len(lines) == 1:
            return "📭 Расписание пустое. Скажи «добавь пару» чтобы заполнить."
        
        # Дедлайны
        upcoming = [d for d in self.deadlines if not d["done"]]
        if upcoming:
            lines.append("\n  ⏰ Ближайшие дедлайны:")
            for d in sorted(upcoming, key=lambda x: x["date"])[:5]:
                dd = datetime.fromisoformat(d["date"])
                days_left = (dd - datetime.now()).days
                emoji = "🔴" if days_left <= 3 else "🟡" if days_left <= 7 else "🟢"
                lines.append(f"    {emoji} {d['name']} — {dd.strftime('%d.%m')} (через {days_left} дн)")
        
        return "\n".join(lines)
    
    def next_class(self) -> str:
        """Какая следующая пара?"""
        now = datetime.now()
        today = now.weekday()
        current_time = now.hour * 60 + now.minute
        
        # Сначала ищем сегодня
        today_entries = sorted(
            [e for e in self.entries if e.weekday == today],
            key=lambda e: (e.hour, e.minute)
        )
        for e in today_entries:
            entry_time = e.hour * 60 + e.minute
            if entry_time > current_time:
                mins_until = entry_time - current_time
                loc = f" ({e.location})" if e.location else ""
                return f"➡️ Следующая: {e.name} в {e.time_str()}{loc} (через {mins_until} мин)"
        
        # Ищем в ближайшие дни
        for day_offset in range(1, 8):
            check_day = (today + day_offset) % 7
            day_entries = sorted(
                [e for e in self.entries if e.weekday == check_day],
                key=lambda e: (e.hour, e.minute)
            )
            if day_entries:
                e = day_entries[0]
                day_name = WEEKDAYS_RU[check_day]
                loc = f" ({e.location})" if e.location else ""
                return f"➡️ Следующая: {e.name} — {day_name} {e.time_str()}{loc}"
        
        return "📭 Нет пар в расписании."
    
    def remove_class(self, name: str) -> str:
        """Удалить пару из расписания."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.name.lower() != name.lower()]
        removed = before - len(self.entries)
        self._save()
        if removed:
            return f"❌ Удалено: {name} ({removed} записей)"
        return f"Не найдено: {name}"
    
    def _format_day(self, weekday: int, label: str) -> str:
        """Форматирование расписания на день."""
        day_entries = sorted(
            [e for e in self.entries if e.weekday == weekday],
            key=lambda e: (e.hour, e.minute)
        )
        
        if not day_entries:
            return f"📅 {label} ({WEEKDAYS_RU[weekday]}): выходной 🎉"
        
        lines = [f"📅 {label} ({WEEKDAYS_RU[weekday]}):\n"]
        for e in day_entries:
            loc = f" | 📍 {e.location}" if e.location else ""
            lines.append(f"  {e.time_str()}–{e.end_time_str()} | {e.name}{loc}")
        
        return "\n".join(lines)
    
    def _save(self):
        _SCHEDULE_DB.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self.entries],
            "deadlines": self.deadlines,
        }
        with open(_SCHEDULE_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        if _SCHEDULE_DB.exists():
            try:
                with open(_SCHEDULE_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.entries = [ScheduleEntry.from_dict(d) for d in data.get("entries", [])]
                self.deadlines = data.get("deadlines", [])
                logger.info(f"Schedule: {len(self.entries)} пар, {len(self.deadlines)} дедлайнов")
            except Exception:
                pass
