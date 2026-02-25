"""
═══════════════════════════════════════════════════════════════
  Integral Auditor — Чикулаев-Кадымов Convergence Control
═══════════════════════════════════════════════════════════════

Мысль = несобственный интеграл: S = ∫_a^∞ f(t) dt
Если f(t) ~ 1/t^p при p > 1 → мысль сошлась (можно остановиться).
Если p < 1 → рекрутировать новые матрицы (IDME).
Если p > 1 но пользователь хочет глубже → повысить порог.

Meta-Auditor: динамический порог по типу задачи.
ZNN Guarantee: T_max ≤ V^(1-a)/(α(1-a)) + V^(1-b)/(α(b-1))
"""

import math
import torch
import torch.nn as nn
import logging
from typing import Tuple, Optional


class IntegralAuditor(nn.Module):
    """
    Вычисляет коэффициент сходимости p в реальном времени.
    
    Метод: МНК на скользящем окне.
        ln f(t) = ln C - p · ln t + ε
    
    Если p > threshold и R² > 0.85 → мысль стабильна.
    """
    
    def __init__(self, window: int = 8, default_threshold: float = 1.1):
        super().__init__()
        self.window = window
        self.default_threshold = default_threshold
        self.logger = logging.getLogger("Tars.IA")
        
        # История f(t) для текущего запроса
        self.history = []
        
        # Счётчик шагов с p < 0.5 (для IDME trigger)
        self.low_p_streak = 0
        
    def reset(self):
        """Сброс между запросами."""
        self.history.clear()
        self.low_p_streak = 0
    
    def observe(self, h_new: torch.Tensor, h_old: torch.Tensor) -> dict:
        """
        Наблюдаем за изменением состояния.
        
        Returns:
            dict с ключами:
            - f_t: float — интенсивность размышления ||h_new - h_old||
            - p: float — коэффициент затухания (p > 1 = сходимость)
            - r_squared: float — качество аппроксимации
            - converged: bool — сошлось ли (p > threshold и R² > 0.85)
            - need_expansion: bool — нужны ли новые матрицы (p < 0.5 streak)
        """
        with torch.no_grad():
            # f(t) = ||h_new - h_old||₂
            f_t = (h_new - h_old).float().norm().item()
            self.history.append(max(f_t, 1e-10))  # clip для log
        
        result = {
            "f_t": f_t,
            "p": 0.0,
            "r_squared": 0.0,
            "converged": False,
            "need_expansion": False,
            "step": len(self.history),
        }
        
        # Нужно минимум 3 точки для МНК
        if len(self.history) < 3:
            return result
        
        # Берём последние W точек
        window = self.history[-self.window:]
        n = len(window)
        
        # МНК: ln f(t) = ln C - p · ln t
        t_vals = list(range(1, n + 1))
        ln_t = [math.log(t) for t in t_vals]
        ln_f = [math.log(max(f, 1e-10)) for f in window]
        
        # Least squares for p
        mean_ln_t = sum(ln_t) / n
        mean_ln_f = sum(ln_f) / n
        
        numerator = sum((ln_t[i] - mean_ln_t) * (ln_f[i] - mean_ln_f) for i in range(n))
        denominator = sum((ln_t[i] - mean_ln_t) ** 2 for i in range(n))
        
        if abs(denominator) < 1e-10:
            return result
        
        slope = numerator / denominator
        p = -slope  # p = -slope (отрицательный наклон = затухание)
        
        # R² (качество фита)
        ss_res = sum((ln_f[i] - (mean_ln_f + slope * (ln_t[i] - mean_ln_t))) ** 2 
                     for i in range(n))
        ss_tot = sum((ln_f[i] - mean_ln_f) ** 2 for i in range(n))
        r_squared = 1 - ss_res / max(ss_tot, 1e-10)
        
        result["p"] = p
        result["r_squared"] = r_squared
        
        # Сходимость
        if p > self.default_threshold and r_squared > 0.85:
            result["converged"] = True
        
        # IDME expansion trigger
        if p < 0.5:
            self.low_p_streak += 1
        else:
            self.low_p_streak = 0
        
        if self.low_p_streak >= 4:
            result["need_expansion"] = True
        
        return result


class MetaAuditor(nn.Module):
    """
    Meta-Auditor: определяет тип задачи и адаптирует порог p.
    
    Простой чат → p=1.05 (быстрый выход)
    Код/математика → p=1.3 (точный ответ)
    "Подумай хорошо" → p=1.5+ (глубокое размышление)
    Бесконечная глубина → p=∞ (пока пул матриц не кончится)
    """
    
    # Ключевые слова для определения типа задачи
    TASK_KEYWORDS = {
        "math": ["вычисли", "формула", "уравнение", "интеграл", "производная",
                 "матрица", "calculate", "solve", "math", "число", "сколько"],
        "code": ["код", "python", "программа", "скрипт", "функция", "класс",
                 "code", "debug", "ошибка", "import", "def ", "class "],
        "chat": ["привет", "как дела", "спасибо", "пока", "что ты", "кто ты",
                 "расскажи", "hello", "thanks"],
        "action": ["открой", "найди файл", "запусти", "нажми", "кликни",
                    "open", "run", "execute", "click"],
        "deep": ["подумай", "тщательно", "внимательно", "проанализируй",
                 "глубоко", "детально", "максимально", "обоснуй",
                 "think hard", "carefully", "analyze deeply"],
    }
    
    THRESHOLDS = {
        "chat": 1.05,
        "action": 1.0,
        "code": 1.2,
        "math": 1.3,
        "deep": 2.0,   # очень высокий = много матриц
        "infinite": 999.0,  # бесконечная глубина
    }
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("Tars.MetaAuditor")
    
    def classify_task(self, text: str) -> Tuple[str, float]:
        """
        Определяет тип задачи и порог p.
        
        Returns: (task_type: str, p_threshold: float)
        """
        text_lower = text.lower()
        
        # Проверка на запрос глубокого размышления
        deep_count = sum(1 for kw in self.TASK_KEYWORDS["deep"] if kw in text_lower)
        if deep_count >= 2:
            self.logger.info(f"MetaAuditor: Запрос глубокого размышления ({deep_count} маркеров)")
            return "deep", self.THRESHOLDS["deep"]
        if deep_count >= 1:
            # Один маркер — повышенный порог
            pass
        
        # Подсчёт совпадений по категориям
        scores = {}
        for task_type, keywords in self.TASK_KEYWORDS.items():
            scores[task_type] = sum(1 for kw in keywords if kw in text_lower)
        
        # Лучшая категория
        best_type = max(scores, key=scores.get)
        if scores[best_type] == 0:
            best_type = "chat"  # default
        
        threshold = self.THRESHOLDS.get(best_type, 1.1)
        
        # Если есть маркер "подумай" + другой тип, повышаем порог
        if deep_count >= 1:
            threshold = max(threshold, 1.5)
        
        return best_type, threshold
    
    def allow_infinite(self, text: str) -> bool:
        """Проверяет, запросил ли пользователь бесконечную глубину."""
        markers = ["без ограничений", "сколько угодно", "не торопись",
                    "максимально глубоко", "бесконечно", "unlimited"]
        return any(m in text.lower() for m in markers)
