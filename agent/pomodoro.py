"""
═══════════════════════════════════════════════════════════════
  pomodoro.py — Помодоро-таймер + трекер учёбы TARS v3
═══════════════════════════════════════════════════════════════

"Начни помодоро — учу Python"
"Сколько я сегодня учился?"
"Статистика за неделю"
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("Tars.Pomodoro")

_ROOT = Path(__file__).parent.parent
_STUDY_DB = _ROOT / "data" / "study_log.json"


class StudySession:
    """Одна сессия учёбы."""
    def __init__(self, subject: str, duration_min: int = 25):
        self.subject = subject
        self.duration_min = duration_min
        self.started = datetime.now().isoformat()
        self.ended = None
        self.actual_min = 0
        self.completed = False
    
    def finish(self, completed: bool = True):
        self.ended = datetime.now().isoformat()
        start = datetime.fromisoformat(self.started)
        self.actual_min = (datetime.now() - start).total_seconds() / 60
        self.completed = completed
    
    def to_dict(self):
        return {
            "subject": self.subject, "duration_min": self.duration_min,
            "started": self.started, "ended": self.ended,
            "actual_min": round(self.actual_min, 1), "completed": self.completed,
        }
    
    @staticmethod
    def from_dict(d):
        s = StudySession.__new__(StudySession)
        s.subject = d["subject"]; s.duration_min = d["duration_min"]
        s.started = d["started"]; s.ended = d.get("ended")
        s.actual_min = d.get("actual_min", 0); s.completed = d.get("completed", False)
        return s


class PomodoroTimer:
    """
    Помодоро-таймер с отслеживанием учёбы.
    
    Стандартный цикл: 25 мин работа → 5 мин перерыв
    Каждые 4 цикла: 15 мин длинный перерыв
    
    Логирует все сессии по предметам для статистики.
    """
    
    def __init__(self, work_min: int = 25, break_min: int = 5, long_break_min: int = 15):
        self.work_min = work_min
        self.break_min = break_min
        self.long_break_min = long_break_min
        
        self.current_session: Optional[StudySession] = None
        self.sessions: List[StudySession] = []
        self.cycle_count = 0
        
        self._running = False
        self._on_break = False
        self._thread = None
        self._pending_notification = None
        
        self._load()
    
    def start(self, subject: str = "общее", duration_min: int = None) -> str:
        """Начать помодоро-сессию."""
        if self._running:
            return f"⚠️ Сессия уже идёт: {self.current_session.subject}. Скажи «стоп помодоро» чтобы закончить."
        
        dur = duration_min or self.work_min
        self.current_session = StudySession(subject, dur)
        self._running = True
        self._on_break = False
        
        self._thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._thread.start()
        
        return (
            f"🍅 Помодоро запущен!\n"
            f"📚 Предмет: {subject}\n"
            f"⏱ Длительность: {dur} мин\n"
            f"Сосредоточься — я скажу когда перерыв."
        )
    
    def stop(self) -> str:
        """Остановить текущую сессию."""
        if not self._running:
            return "⚠️ Нет активной сессии."
        
        self._running = False
        if self.current_session:
            self.current_session.finish(completed=False)
            self.sessions.append(self.current_session)
            mins = self.current_session.actual_min
            subj = self.current_session.subject
            self.current_session = None
            self._save()
            return f"⏹ Сессия остановлена: {subj} ({mins:.0f} мин). Молодец!"
        return "Сессия завершена."
    
    def get_status(self) -> Optional[str]:
        """Текущий статус таймера."""
        if not self._running or not self.current_session:
            return None
        
        start = datetime.fromisoformat(self.current_session.started)
        elapsed = (datetime.now() - start).total_seconds() / 60
        remaining = self.current_session.duration_min - elapsed
        
        if self._on_break:
            return f"☕ Перерыв: осталось {max(0, remaining):.0f} мин"
        else:
            return f"🍅 Работа [{self.current_session.subject}]: осталось {max(0, remaining):.0f} мин"
    
    def get_notification(self) -> Optional[str]:
        """Получить pending уведомление от таймера."""
        n = self._pending_notification
        self._pending_notification = None
        return n
    
    def stats_today(self) -> str:
        """Статистика за сегодня."""
        today = date.today().isoformat()
        today_sessions = [s for s in self.sessions 
                         if s.started[:10] == today and s.actual_min > 1]
        
        if not today_sessions:
            return "📊 Сегодня ты ещё не учился. Скажи «помодоро [предмет]» чтобы начать!"
        
        total_min = sum(s.actual_min for s in today_sessions)
        hours = int(total_min // 60)
        mins = int(total_min % 60)
        completed = sum(1 for s in today_sessions if s.completed)
        
        # По предметам
        by_subject = defaultdict(float)
        for s in today_sessions:
            by_subject[s.subject] += s.actual_min
        
        lines = [f"📊 Сегодня: {hours}ч {mins}мин учёбы ({completed} помодоро завершено)\n"]
        for subj, mins_total in sorted(by_subject.items(), key=lambda x: -x[1]):
            bar_len = int(min(mins_total / 30, 10))
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"  📚 {subj}: [{bar}] {mins_total:.0f} мин")
        
        return "\n".join(lines)
    
    def stats_week(self) -> str:
        """Статистика за неделю."""
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        week_sessions = [s for s in self.sessions 
                        if s.started >= week_ago and s.actual_min > 1]
        
        if not week_sessions:
            return "📊 За последнюю неделю нет сессий."
        
        total_min = sum(s.actual_min for s in week_sessions)
        hours = int(total_min // 60)
        completed = sum(1 for s in week_sessions if s.completed)
        
        # По дням
        by_day = defaultdict(float)
        for s in week_sessions:
            day = s.started[:10]
            by_day[day] += s.actual_min
        
        # По предметам
        by_subject = defaultdict(float)
        for s in week_sessions:
            by_subject[s.subject] += s.actual_min
        
        lines = [
            f"📊 Неделя: {hours}ч {int(total_min % 60)}мин учёбы | "
            f"{completed} помодоро | {len(by_day)} дней\n"
        ]
        
        lines.append("  По дням:")
        for day in sorted(by_day.keys()):
            mins = by_day[day]
            bar_len = int(min(mins / 60, 10))
            bar = "█" * bar_len + "░" * (10 - bar_len)
            day_name = datetime.fromisoformat(day).strftime("%a %d.%m")
            lines.append(f"    {day_name}: [{bar}] {mins:.0f} мин")
        
        lines.append("\n  По предметам:")
        for subj, mins_total in sorted(by_subject.items(), key=lambda x: -x[1]):
            pct = mins_total / total_min * 100
            lines.append(f"    📚 {subj}: {mins_total:.0f} мин ({pct:.0f}%)")
        
        return "\n".join(lines)
    
    def _timer_loop(self):
        """Фоновый таймер."""
        while self._running:
            if not self.current_session:
                break
            
            start = datetime.fromisoformat(self.current_session.started)
            elapsed = (datetime.now() - start).total_seconds() / 60
            
            if not self._on_break and elapsed >= self.current_session.duration_min:
                # Работа завершена
                self.current_session.finish(completed=True)
                self.sessions.append(self.current_session)
                self.cycle_count += 1
                self._save()
                
                # Определяем тип перерыва
                if self.cycle_count % 4 == 0:
                    break_dur = self.long_break_min
                    break_type = "длинный перерыв"
                else:
                    break_dur = self.break_min
                    break_type = "перерыв"
                
                self._pending_notification = (
                    f"🍅 Помодоро #{self.cycle_count} завершён!\n"
                    f"📚 {self.current_session.subject}: {self.current_session.actual_min:.0f} мин ✅\n"
                    f"☕ Время на {break_type}: {break_dur} мин\n"
                    f"Скажи «продолжи» для следующего цикла."
                )
                
                # Переходим в перерыв
                self._on_break = True
                self.current_session = StudySession("перерыв", break_dur)
            
            elif self._on_break and elapsed >= self.current_session.duration_min:
                # Перерыв завершён
                self._pending_notification = (
                    f"⏰ Перерыв окончен! Готов к следующему помодоро?\n"
                    f"Скажи «помодоро [предмет]» чтобы продолжить."
                )
                self._on_break = False
                self._running = False
            
            time.sleep(10)
    
    def _save(self):
        _STUDY_DB.parent.mkdir(parents=True, exist_ok=True)
        data = [s.to_dict() for s in self.sessions[-5000:]]
        with open(_STUDY_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        if _STUDY_DB.exists():
            try:
                with open(_STUDY_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.sessions = [StudySession.from_dict(d) for d in data]
                logger.info(f"Pomodoro: {len(self.sessions)} sessions loaded")
            except Exception:
                pass
