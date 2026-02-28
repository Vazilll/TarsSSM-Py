"""
═══════════════════════════════════════════════════════════════
  notifier.py — Агрегатор уведомлений TARS v3
═══════════════════════════════════════════════════════════════

Собирает все проактивные уведомления из подсистем:
  - Напоминания (reminders)
  - Системные алерты (system_monitor)
  - Паттерны рутин (routine_detector)
  - Карточки для повторения (learning_helper)
  - Статус записи встреч (meeting_scribe)

И выдаёт их пользователю при взаимодействии.
"""

import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger("Tars.Notifier")


class TarsNotifier:
    """
    Агрегатор уведомлений — ТАРС пишет первым.
    
    Интегрируется с GIE: при каждом execute_goal() 
    собирает pending notifications из всех подсистем.
    """
    
    def __init__(self, reminders=None, monitor=None, 
                 routine_detector=None, learning_helper=None,
                 meeting_scribe=None):
        self.reminders = reminders
        self.monitor = monitor
        self.routine_detector = routine_detector
        self.learning_helper = learning_helper
        self.meeting_scribe = meeting_scribe
        self._greeted_today = False
    
    def collect_notifications(self) -> List[str]:
        """Собрать все pending уведомления из подсистем."""
        notifications = []
        
        # 1. Напоминания (самый высокий приоритет)
        if self.reminders:
            try:
                for msg in self.reminders.get_pending_notifications():
                    notifications.append(msg)
            except Exception as e:
                logger.debug(f"Reminder notification error: {e}")
        
        # 2. Системные алерты
        if self.monitor:
            try:
                for alert in self.monitor.get_alerts():
                    notifications.append(alert)
            except Exception as e:
                logger.debug(f"Monitor alert error: {e}")
        
        # 3. Рутинные паттерны
        if self.routine_detector:
            try:
                msg = self.routine_detector.get_proactive_message()
                if msg:
                    notifications.append(msg)
            except Exception as e:
                logger.debug(f"Routine notification error: {e}")
        
        # 4. Карточки для повторения (раз в 5 обращений)
        if self.learning_helper:
            try:
                due = self.learning_helper.get_due_cards()
                if due and len(due) >= 3:
                    notifications.append(
                        f"📝 У тебя {len(due)} карточек для повторения! "
                        f"Скажи «повторение» чтобы начать."
                    )
            except Exception as e:
                logger.debug(f"Learning notification error: {e}")
        
        # 5. Статус записи встречи
        if self.meeting_scribe:
            try:
                status = self.meeting_scribe.get_live_status()
                if status:
                    notifications.append(status)
            except Exception as e:
                logger.debug(f"Meeting notification error: {e}")
        
        return notifications
    
    def get_morning_greeting(self) -> Optional[str]:
        """Утреннее приветствие (раз в день)."""
        now = datetime.now()
        
        if self._greeted_today:
            return None
        
        hour = now.hour
        if hour < 5 or hour >= 12:
            return None  # Только утром
        
        self._greeted_today = True
        
        parts = [f"🌅 Доброе утро! Сейчас {now.strftime('%H:%M')}."]
        
        # Напоминания на сегодня
        if self.reminders:
            today_str = self.reminders.list_today()
            if "ничего не запланировано" not in today_str:
                parts.append(today_str)
        
        # Карточки ожидают
        if self.learning_helper:
            due = self.learning_helper.get_due_cards()
            if due:
                parts.append(f"📝 {len(due)} карточек ждут повторения")
        
        return "\n".join(parts)
    
    def format_notifications(self, notifications: List[str]) -> str:
        """Форматирует уведомления в один блок."""
        if not notifications:
            return ""
        
        if len(notifications) == 1:
            return notifications[0]
        
        lines = ["📬 Уведомления:"]
        for n in notifications:
            lines.append(f"  • {n}")
        return "\n".join(lines)


if __name__ == "__main__":
    notifier = TarsNotifier()
    notifs = notifier.collect_notifications()
    print(f"Notifications: {len(notifs)}")
    for n in notifs:
        print(f"  {n}")
