"""
═══════════════════════════════════════════════════════════════
  expense_tracker.py — Трекер расходов TARS v3
═══════════════════════════════════════════════════════════════

"Потратил 500р на обед"
"Сколько я потратил за неделю?"
"Бюджет: 30000р на месяц"
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("Tars.ExpenseTracker")

_ROOT = Path(__file__).parent.parent
_EXPENSE_DB = _ROOT / "data" / "expenses.json"

CATEGORIES = {
    "еда": ["обед", "ужин", "завтрак", "кофе", "перекус", "продукты", "магазин", "food", "lunch"],
    "транспорт": ["метро", "автобус", "такси", "бензин", "проезд", "uber", "transport"],
    "учёба": ["книга", "курс", "подписка", "учебник", "канцелярия", "education"],
    "развлечения": ["кино", "игра", "концерт", "бар", "ресторан", "entertainment"],
    "связь": ["телефон", "интернет", "мобильный", "sim"],
    "здоровье": ["аптека", "врач", "спортзал", "лекарство", "gym"],
    "одежда": ["одежда", "обувь", "clothes"],
    "жильё": ["аренда", "квартира", "коммуналка", "rent"],
}


class Expense:
    """Одна трата."""
    def __init__(self, amount: float, description: str, category: str = "другое"):
        self.amount = amount
        self.description = description
        self.category = category
        self.date = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            "amount": self.amount, "description": self.description,
            "category": self.category, "date": self.date,
        }
    
    @staticmethod
    def from_dict(d):
        e = Expense(d["amount"], d["description"], d.get("category", "другое"))
        e.date = d.get("date", "")
        return e


class ExpenseTracker:
    """Трекер расходов с бюджетом и статистикой."""
    
    def __init__(self):
        self.expenses: List[Expense] = []
        self.budget_monthly: float = 0
        self._load()
    
    def add(self, amount: float, description: str) -> str:
        """Добавить трату."""
        category = self._detect_category(description)
        expense = Expense(amount, description, category)
        self.expenses.append(expense)
        self._save()
        
        # Проверка бюджета
        warning = ""
        if self.budget_monthly > 0:
            month_spent = self._month_total()
            pct = month_spent / self.budget_monthly * 100
            if pct > 90:
                warning = f"\n⚠️ Бюджет почти исчерпан: {month_spent:.0f}/{self.budget_monthly:.0f}р ({pct:.0f}%)"
            elif pct > 75:
                warning = f"\n💡 Бюджет: {month_spent:.0f}/{self.budget_monthly:.0f}р ({pct:.0f}%)"
        
        return f"💰 Записано: {amount:.0f}р — {description} [{category}]{warning}"
    
    def set_budget(self, amount: float) -> str:
        """Установить месячный бюджет."""
        self.budget_monthly = amount
        self._save()
        spent = self._month_total()
        remaining = amount - spent
        return (
            f"💰 Бюджет: {amount:.0f}р/мес\n"
            f"Потрачено: {spent:.0f}р | Осталось: {remaining:.0f}р"
        )
    
    def stats_today(self) -> str:
        """Траты за сегодня."""
        today = datetime.now().date().isoformat()
        todays = [e for e in self.expenses if e.date[:10] == today]
        
        if not todays:
            return "💰 Сегодня трат нет."
        
        total = sum(e.amount for e in todays)
        lines = [f"💰 Сегодня: {total:.0f}р\n"]
        for e in todays:
            lines.append(f"  • {e.amount:.0f}р — {e.description} [{e.category}]")
        return "\n".join(lines)
    
    def stats_week(self) -> str:
        """Статистика за неделю."""
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        week = [e for e in self.expenses if e.date >= week_ago]
        
        if not week:
            return "💰 За неделю трат нет."
        
        total = sum(e.amount for e in week)
        by_cat = defaultdict(float)
        for e in week:
            by_cat[e.category] += e.amount
        
        lines = [f"💰 Неделя: {total:.0f}р\n"]
        for cat, amt in sorted(by_cat.items(), key=lambda x: -x[1]):
            pct = amt / total * 100
            bar_len = int(pct / 10)
            bar = "█" * bar_len + "░" * (10 - bar_len)
            lines.append(f"  [{bar}] {cat}: {amt:.0f}р ({pct:.0f}%)")
        
        if self.budget_monthly > 0:
            month_spent = self._month_total()
            remaining = self.budget_monthly - month_spent
            lines.append(f"\n  Бюджет: {month_spent:.0f}/{self.budget_monthly:.0f}р "
                        f"(осталось {remaining:.0f}р)")
        
        return "\n".join(lines)
    
    def stats_month(self) -> str:
        """Статистика за месяц."""
        month_start = datetime.now().replace(day=1).isoformat()
        month = [e for e in self.expenses if e.date >= month_start]
        
        if not month:
            return "💰 За месяц трат нет."
        
        total = sum(e.amount for e in month)
        by_cat = defaultdict(float)
        by_day = defaultdict(float)
        for e in month:
            by_cat[e.category] += e.amount
            by_day[e.date[:10]] += e.amount
        
        avg_day = total / max(1, len(by_day))
        
        lines = [
            f"💰 Месяц: {total:.0f}р (сред. {avg_day:.0f}р/день)\n",
            "  По категориям:"
        ]
        for cat, amt in sorted(by_cat.items(), key=lambda x: -x[1]):
            pct = amt / total * 100
            lines.append(f"    {cat}: {amt:.0f}р ({pct:.0f}%)")
        
        if self.budget_monthly > 0:
            remaining = self.budget_monthly - total
            days_left = 30 - datetime.now().day
            daily_budget = remaining / max(1, days_left)
            lines.append(f"\n  📊 Бюджет: {total:.0f}/{self.budget_monthly:.0f}р")
            lines.append(f"  📊 На каждый оставшийся день: {daily_budget:.0f}р")
        
        return "\n".join(lines)
    
    def _month_total(self) -> float:
        month_start = datetime.now().replace(day=1).isoformat()
        return sum(e.amount for e in self.expenses if e.date >= month_start)
    
    def _detect_category(self, description: str) -> str:
        """Авто-определение категории."""
        desc_lower = description.lower()
        for category, keywords in CATEGORIES.items():
            for kw in keywords:
                if kw in desc_lower:
                    return category
        return "другое"
    
    def _save(self):
        _EXPENSE_DB.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "expenses": [e.to_dict() for e in self.expenses[-10000:]],
            "budget_monthly": self.budget_monthly,
        }
        with open(_EXPENSE_DB, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _load(self):
        if _EXPENSE_DB.exists():
            try:
                with open(_EXPENSE_DB, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.expenses = [Expense.from_dict(d) for d in data.get("expenses", [])]
                self.budget_monthly = data.get("budget_monthly", 0)
                logger.info(f"Expenses: {len(self.expenses)} loaded, budget={self.budget_monthly}")
            except Exception:
                pass
