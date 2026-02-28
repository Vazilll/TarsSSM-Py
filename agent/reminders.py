"""
═══════════════════════════════════════════════════════════════
  reminders.py — Напоминания, таймеры, расписание TARS v3
═══════════════════════════════════════════════════════════════

"Напомни через 2 часа позвонить врачу"
"Каждую пятницу в 18:00 — отчёт"
"Что у меня на сегодня?"
"""

import json
import os
import re
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger("Tars.Reminders")

_ROOT = Path(__file__).parent.parent
_REMINDERS_DB = _ROOT / "data" / "reminders.json"


class Reminder:
    """Одно напоминание."""
    def __init__(self, text: str, when: datetime, 
                 recurring: str = None, source: str = "user"):
        self.id = int(time.time() * 1000) % 10**9
        self.text = text
        self.when = when.isoformat()
        self.recurring = recurring  # "daily", "weekly", "monthly", None
        self.source = source
        self.fired = False
        self.created = datetime.now().isoformat()
    
    def is_due(self) -> bool:
        return not self.fired and datetime.now() >= datetime.fromisoformat(self.when)
    
    def fire(self) -> str:
        """Срабатывает напоминание. Возвращает уведомление."""
        self.fired = True
        
        # Для recurring — создаём следующее
        if self.recurring:
            next_when = datetime.fromisoformat(self.when)
            if self.recurring == "daily":
                next_when += timedelta(days=1)
            elif self.recurring == "weekly":
                next_when += timedelta(weeks=1)
            elif self.recurring == "monthly":
                next_when += timedelta(days=30)
            self.when = next_when.isoformat()
            self.fired = False
        
        return f"🔔 Напоминание: {self.text}"
    
    def to_dict(self):
        return {
            "id": self.id, "text": self.text, "when": self.when,
            "recurring": self.recurring, "source": self.source,
            "fired": self.fired, "created": self.created,
        }
    
    @staticmethod
    def from_dict(d):
        r = Reminder.__new__(Reminder)
        r.id = d["id"]; r.text = d["text"]; r.when = d["when"]
        r.recurring = d.get("recurring"); r.source = d.get("source", "user")
        r.fired = d.get("fired", False); r.created = d.get("created", "")
        return r


class ReminderService:
    """
    Сервис напоминаний с фоновым потоком проверки.
    
    Парсит естественный язык:
      "через 30 минут" → now + 30min
      "завтра в 9:00" → tomorrow 9:00
      "каждый день в 8:00" → recurring daily
    """
    
    def __init__(self, callback=None):
        self.reminders: List[Reminder] = []
        self.callback = callback  # Функция уведомления
        self._pending_notifications: List[str] = []
        self._load()
        
        # Фоновый поток для проверки
        self._running = True
        self._thread = threading.Thread(target=self._check_loop, daemon=True)
        self._thread.start()
    
    def add(self, text: str, when_text: str) -> str:
        """
        Добавить напоминание.
        
        text: "Позвонить врачу"
        when_text: "через 2 часа" / "завтра в 9:00" / "каждую пятницу в 18:00"
        """
        when, recurring = self._parse_time(when_text)
        if when is None:
            return f"❌ Не понял время: «{when_text}». Попробуй: «через 30 минут» или «завтра в 9:00»"
        
        reminder = Reminder(text, when, recurring)
        self.reminders.append(reminder)
        self._save()
        
        time_str = when.strftime("%d.%m %H:%M")
        rec_str = f" (повтор: {recurring})" if recurring else ""
        return f"✅ Напоминание установлено: «{text}» → {time_str}{rec_str}"
    
    def add_timer(self, text: str, minutes: int) -> str:
        """Простой таймер на N минут."""
        when = datetime.now() + timedelta(minutes=minutes)
        reminder = Reminder(text, when)
        self.reminders.append(reminder)
        self._save()
        return f"⏱ Таймер на {minutes} мин: «{text}» (в {when.strftime('%H:%M')})"
    
    def list_active(self) -> str:
        """Список активных напоминаний."""
        active = [r for r in self.reminders if not r.fired]
        if not active:
            return "📭 Нет активных напоминаний."
        
        lines = ["📋 Активные напоминания:\n"]
        for i, r in enumerate(active):
            when = datetime.fromisoformat(r.when)
            rec = f" 🔁{r.recurring}" if r.recurring else ""
            lines.append(f"  {i+1}. {r.text} — {when.strftime('%d.%m %H:%M')}{rec}")
        return "\n".join(lines)
    
    def list_today(self) -> str:
        """Что на сегодня?"""
        today = datetime.now().date()
        todays = [r for r in self.reminders if not r.fired 
                  and datetime.fromisoformat(r.when).date() == today]
        
        if not todays:
            return "📭 На сегодня ничего не запланировано."
        
        lines = [f"📅 Сегодня ({today.strftime('%d.%m.%Y')}):\n"]
        for r in sorted(todays, key=lambda x: x.when):
            when = datetime.fromisoformat(r.when)
            lines.append(f"  ⏰ {when.strftime('%H:%M')} — {r.text}")
        return "\n".join(lines)
    
    def cancel(self, index: int) -> str:
        """Отмена напоминания по номеру."""
        active = [r for r in self.reminders if not r.fired]
        if 0 <= index < len(active):
            active[index].fired = True
            self._save()
            return f"❌ Отменено: «{active[index].text}»"
        return "Напоминание не найдено."
    
    def get_pending_notifications(self) -> List[str]:
        """Получить и очистить накопившиеся уведомления."""
        notifications = list(self._pending_notifications)
        self._pending_notifications.clear()
        return notifications
    
    def _parse_time(self, text: str):
        """Парсинг естественного языка для времени."""
        text = text.lower().strip()
        recurring = None
        
        # "через N минут/часов"
        m = re.search(r'через\s+(\d+)\s*(мин|час|секунд|дн)', text)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if 'мин' in unit: return datetime.now() + timedelta(minutes=n), None
            if 'час' in unit: return datetime.now() + timedelta(hours=n), None
            if 'секунд' in unit: return datetime.now() + timedelta(seconds=n), None
            if 'дн' in unit: return datetime.now() + timedelta(days=n), None
        
        # "in N minutes/hours"
        m = re.search(r'in\s+(\d+)\s*(min|hour|sec|day)', text)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            if 'min' in unit: return datetime.now() + timedelta(minutes=n), None
            if 'hour' in unit: return datetime.now() + timedelta(hours=n), None
            if 'day' in unit: return datetime.now() + timedelta(days=n), None
        
        # "завтра в HH:MM"
        m = re.search(r'завтра\s+в?\s*(\d{1,2})[:\.](\d{2})', text)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            tomorrow = datetime.now() + timedelta(days=1)
            return tomorrow.replace(hour=h, minute=mi, second=0), None
        
        # "сегодня в HH:MM"
        m = re.search(r'сегодня\s+в?\s*(\d{1,2})[:\.](\d{2})', text)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            return datetime.now().replace(hour=h, minute=mi, second=0), None
        
        # "в HH:MM" (сегодня или завтра если уже прошло)
        m = re.search(r'в\s+(\d{1,2})[:\.](\d{2})', text)
        if m:
            h, mi = int(m.group(1)), int(m.group(2))
            target = datetime.now().replace(hour=h, minute=mi, second=0)
            if target < datetime.now():
                target += timedelta(days=1)
            return target, None
        
        # Recurring: "каждый день"
        if 'каждый день' in text or 'ежедневно' in text:
            recurring = "daily"
            m = re.search(r'(\d{1,2})[:\.](\d{2})', text)
            if m:
                h, mi = int(m.group(1)), int(m.group(2))
                target = datetime.now().replace(hour=h, minute=mi, second=0)
                if target < datetime.now():
                    target += timedelta(days=1)
                return target, recurring
        
        # "каждую пятницу/понедельник..."
        days_map = {
            'понедельник': 0, 'вторник': 1, 'среду': 2, 'среда': 2,
            'четверг': 3, 'пятницу': 4, 'пятница': 4,
            'субботу': 5, 'суббота': 5, 'воскресенье': 6,
        }
        for day_name, day_num in days_map.items():
            if day_name in text:
                recurring = "weekly"
                now = datetime.now()
                days_ahead = day_num - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target = now + timedelta(days=days_ahead)
                m = re.search(r'(\d{1,2})[:\.](\d{2})', text)
                if m:
                    target = target.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0)
                return target, recurring
        
        return None, None
    
    def _check_loop(self):
        """Фоновый поток проверки напоминаний (каждые 15 секунд)."""
        while self._running:
            try:
                for r in self.reminders:
                    if r.is_due():
                        msg = r.fire()
                        self._pending_notifications.append(msg)
                        logger.info(f"Reminder fired: {msg}")
                        self._save()
                        
                        # Windows toast notification
                        try:
                            from ctypes import windll
                            windll.user32.MessageBeep(0x00000040)
                        except Exception:
                            pass
            except Exception as e:
                logger.debug(f"Reminder check error: {e}")
            time.sleep(15)
    
    def stop(self):
        self._running = False
    
    def _save(self):
        _REMINDERS_DB.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.reminders]
        with open(_REMINDERS_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        if _REMINDERS_DB.exists():
            try:
                with open(_REMINDERS_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.reminders = [Reminder.from_dict(d) for d in data]
                logger.info(f"Reminders: {len(self.reminders)} загружено")
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    svc = ReminderService()
    
    print(svc.add("Позвонить маме", "через 30 минут"))
    print(svc.add("Отчёт", "каждую пятницу в 18:00"))
    print(svc.add("Митинг", "завтра в 10:00"))
    print()
    print(svc.list_active())
    print()
    print(svc.list_today())
    
    svc.stop()
