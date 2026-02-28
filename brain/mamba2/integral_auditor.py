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
        self.threshold = default_threshold  # Динамически переопределяется MetaAuditor
        self.logger = logging.getLogger("Tars.IA")
        
        # История f(t) для текущего запроса
        self.history = []
        
        # Счётчик шагов с p < 0.5 (для IDME trigger)
        self.low_p_streak = 0
        
    def reset(self):
        """Сброс между запросами."""
        self.history.clear()
        self.low_p_streak = 0
        self.threshold = self.default_threshold
    
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
        
        # Сходимость (threshold переопределяется MetaAuditor)
        if p > self.threshold and r_squared > 0.85:
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


class TDValueEstimator(nn.Module):
    """
    TD-Learning Value Estimator (Temporal Difference, Schultz et al., 1997).
    
    Нейронаука: дофаминергические нейроны VTA кодируют
    prediction error: δ = r + γ·V(s') - V(s).
    
    Модуль обучается предсказывать качество ответа (reward)
    на основе скрытого состояния мозга. TD error используется
    для адаптации порогов p в MetaAuditor:
      - δ > 0 → ответ лучше ожиданий → снижаем порог (быстрее выходим)
      - δ < 0 → ответ хуже → повышаем порог (думаем глубже)
    
    Математика:
      V(s) = value_net(h)          # predicted quality
      δ = r + γ·V(s') - V(s)       # TD error
      threshold *= (1 - β · δ)     # adaptive threshold
    """
    
    def __init__(self, d_model: int = 768, gamma: float = 0.95):
        super().__init__()
        self.gamma = gamma
        self.logger = logging.getLogger("Tars.TDValue")
        
        # Value network: h_state → predicted quality ∈ (0, 1)
        self.value_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
        # Online optimizer for TD updates
        self.lr = 1e-3
        self._optimizer = None
        
        # EMA of TD errors for logging
        self.td_error_ema = 0.0
        self.td_momentum = 0.9
        
        # Threshold adaptation strength
        self.beta = 0.05  # conservative adaptation
    
    def predict_value(self, h_state: torch.Tensor) -> torch.Tensor:
        """Предсказывает ожидаемое качество ответа."""
        return self.value_net(h_state)  # [B, 1]
    
    def td_update(
        self, 
        h_state: torch.Tensor, 
        reward: float,
        h_next: torch.Tensor,
    ) -> float:
        """
        Один шаг TD(0) обновления.
        
        Args:
            h_state: [1, d_model] — состояние перед действием
            reward: скаляр ∈ [0, 1] — фактическое качество
            h_next: [1, d_model] — состояние после действия
        
        Returns:
            td_error: float — prediction error (δ)
        """
        if self._optimizer is None:
            self._optimizer = torch.optim.Adam(
                self.value_net.parameters(), lr=self.lr
            )
        
        # V(s)
        V_s = self.value_net(h_state)
        
        # V(s') — detached (target network)
        with torch.no_grad():
            V_next = self.value_net(h_next)
        
        # TD target: r + γ · V(s')
        td_target = reward + self.gamma * V_next
        
        # TD error: δ = target - V(s)
        td_error = (td_target - V_s).item()
        
        # Update value network
        loss = (V_s - td_target.detach()).pow(2).mean()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        # EMA tracking
        self.td_error_ema = (
            self.td_momentum * self.td_error_ema 
            + (1 - self.td_momentum) * td_error
        )
        
        self.logger.debug(
            f"TD: δ={td_error:.3f}, V(s)={V_s.item():.3f}, "
            f"r={reward:.2f}, EMA(δ)={self.td_error_ema:.3f}"
        )
        
        return td_error
    
    def adapt_threshold(self, base_threshold: float, td_error: float) -> float:
        """
        Адаптирует порог p на основе TD error.
        
        δ > 0 (лучше ожиданий) → снижаем порог → быстрее выходим
        δ < 0 (хуже ожиданий) → повышаем порог → думаем глубже
        """
        adapted = base_threshold * (1.0 - self.beta * td_error)
        # Clamp to reasonable range
        return max(0.5, min(3.0, adapted))
