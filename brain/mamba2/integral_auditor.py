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
        
        # Нужно минимум 4 точки для надёжного МНК
        if len(self.history) < 4:
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
        
        # Сходимость: p > threshold, R² > 0.88, И f(t) реально убывает
        if p > self.threshold and r_squared > 0.88:
            # Дополнительная проверка: последние 3 значения f(t) убывают
            if len(window) >= 3:
                recent = window[-3:]
                if recent[-1] < recent[-2] < recent[-3]:
                    result["converged"] = True
                elif recent[-1] < recent[0] * 0.5:  # Общее снижение > 50%
                    result["converged"] = True
            else:
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


# ═══════════════════════════════════════════════════════════════
# Temporal Embedding — Нейронные часы ТАРС
# ═══════════════════════════════════════════════════════════════

class TemporalEmbedding(nn.Module):
    """
    Внутренние часы ТАРС — осцилляции на разных частотах.
    
    Аналог циркадных ритмов мозга:
      fast    ~100ms  — рабочая память (θ-ритм, 4-8 Hz)
      medium  ~1s     — контекст диалога (α-ритм, 8-13 Hz)
      slow    ~1h     — сессия (ультрадианный цикл)
      circadian ~24h  — суточный ритм (день/ночь)
    
    Выход суммируется с hidden state через обучаемый гейт.
    Модель учится использовать временную информацию для:
      - Понимания срочности ("через 5 минут" vs "завтра")
      - Учёта контекста сессии ("давно не общались")
      - Адаптации стиля (утро → бодрый, ночь → спокойный)
    
    Уникальность: ни одна SSM в мире не имеет встроенного чувства времени.
    """
    
    def __init__(self, d_model: int, n_frequencies: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_freq = n_frequencies
        
        # Обучаемые частоты для каждого ритма
        # Инициализация: fast=10Hz, medium=1Hz, slow=1/3600Hz, circadian=1/86400Hz
        init_periods = torch.tensor([0.1, 1.0, 3600.0, 86400.0])
        self.log_periods = nn.Parameter(torch.log(init_periods))
        
        # Проекция: n_freq*2 (sin+cos) → d_model
        self.proj = nn.Linear(n_frequencies * 2, d_model, bias=False)
        
        # Обучаемый гейт: model решает, сколько временной информации подмешивать
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        
        # Инициализация: гейт начинает с малых значений (не ломает обученную модель)
        nn.init.constant_(self.gate[0].bias, -3.0)  # sigmoid(-3) ≈ 0.05
        
        # Фаза (обучаемый сдвиг для каждой частоты)
        self.phase = nn.Parameter(torch.zeros(n_frequencies))
        
        # Внутренний счётчик (монотонно растёт)
        self.register_buffer('_start_time', torch.tensor(0.0))
        self._initialized = False
    
    def _get_time_seconds(self) -> float:
        """Получить текущее время в секундах от начала сессии."""
        import time as _time
        if not self._initialized:
            self._start_time.fill_(_time.time())
            self._initialized = True
        return _time.time() - self._start_time.item()
    
    def forward(self, h: torch.Tensor, t_override: float = None) -> torch.Tensor:
        """
        Добавляет временное кодирование к hidden state.
        
        h: [B, L, D] или [B, D] — текущий hidden state
        t_override: если задано, использует это значение вместо реального времени
        
        Returns: h + gate * temporal_signal
        """
        t = t_override if t_override is not None else self._get_time_seconds()
        
        # Вычисляем осцилляции
        periods = torch.exp(self.log_periods)  # [n_freq]
        freqs = (2 * math.pi) / periods  # ω = 2π/T
        
        # sin/cos для каждой частоты
        phase = t * freqs + self.phase  # [n_freq]
        oscillations = torch.cat([
            torch.sin(phase),
            torch.cos(phase),
        ])  # [n_freq * 2]
        
        # Проекция в d_model пространство
        temporal = self.proj(oscillations)  # [d_model]
        
        # Гейтирование: модель решает, сколько времени подмешать
        g = self.gate(h)  # [B, ...1]
        
        return h + g * temporal
    
    def reset_clock(self):
        """Сброс внутренних часов (новая сессия)."""
        self._initialized = False


# ═══════════════════════════════════════════════════════════════
# MetaCortex — Метакогнитивный контур (думать о думании)
# ═══════════════════════════════════════════════════════════════

class MetaCortex(nn.Module):
    """
    Метакогнитивный контур ТАРС — модель себя.
    
    Отслеживает историю ошибок и точности для разных типов задач.
    Предсказывает вероятность ошибки для текущего запроса.
    Динамически адаптирует глубину обработки:
      - Высокая вероятность ошибки → увеличить p-threshold, больше IDME
      - Низкая → быстрый выход, экономия compute
    
    Уникальность: замкнутая петля self-assessment → threshold adaptation.
    В отличие от CriticHead (проверка ответа), MetaCortex проверяет
    ПРОЦЕСС мышления — "Правильно ли я думаю?"
    """
    
    def __init__(self, d_model: int = 2048, history_size: int = 100):
        super().__init__()
        self.d_model = d_model
        self.history_size = history_size
        self.logger = logging.getLogger("Tars.MetaCortex")
        
        # Предсказатель ошибки: h_state → P(error)
        self.error_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # История: [(task_type, predicted_error, actual_error, p_threshold)]
        self._history = []
        
        # EMA статистики по типу задачи
        self._type_stats = {}  # type → {errors: int, total: int, ema_error: float}
        
        # Порог адаптации
        self.adaptation_strength = 0.15  # Насколько сильно MetaCortex влияет на threshold
    
    def predict_error(self, h_state: torch.Tensor) -> torch.Tensor:
        """Предсказать P(error) для текущего состояния мозга."""
        return self.error_predictor(h_state.detach())  # detach — не обратный градиент в мозг
    
    def observe_outcome(self, task_type: str, predicted_error: float, 
                        actual_quality: float, p_threshold: float):
        """
        Записать результат: предсказание vs реальность.
        actual_quality ∈ [0, 1] — реальное качество ответа.
        """
        actual_error = 1.0 - actual_quality  # error = 1 - quality
        
        self._history.append({
            "type": task_type,
            "predicted": predicted_error,
            "actual": actual_error,
            "threshold": p_threshold,
        })
        
        # Trim history
        if len(self._history) > self.history_size:
            self._history = self._history[-self.history_size:]
        
        # Update type stats
        if task_type not in self._type_stats:
            self._type_stats[task_type] = {"errors": 0, "total": 0, "ema_error": 0.5}
        
        stats = self._type_stats[task_type]
        stats["total"] += 1
        if actual_error > 0.5:
            stats["errors"] += 1
        stats["ema_error"] = 0.9 * stats["ema_error"] + 0.1 * actual_error
        
        self.logger.debug(
            f"MetaCortex: {task_type} predicted={predicted_error:.2f}, "
            f"actual={actual_error:.2f}, ema={stats['ema_error']:.2f}"
        )
    
    def adapt_threshold(self, base_threshold: float, task_type: str,
                        error_prediction: float) -> float:
        """
        Адаптировать p-threshold на основе метакогнитивной оценки.
        
        Высокая P(error) → повысить порог (думать глубже)
        Низкая P(error) → понизить (быстро)
        
        Safety: не адаптирует пока нет 5+ наблюдений (warm-up).
        """
        # Warm-up guard: пока мало данных — не рискуем
        if len(self._history) < 5:
            return base_threshold
        
        # Учитываем историческую точность для этого типа
        historical_factor = 1.0
        if task_type in self._type_stats:
            stats = self._type_stats[task_type]
            if stats["total"] >= 3:  # минимум 3 примера для типа
                ema = stats["ema_error"]
                historical_factor = 1.0 + (ema - 0.5) * 0.3  # смягчён: 0.5→0.3
        
        # Адаптация: error_prediction > 0.5 → увеличить порог
        delta = (error_prediction - 0.5) * self.adaptation_strength
        adapted = base_threshold * (1.0 + delta) * historical_factor
        
        # Жёсткий clamp: ±15% от базового порога
        min_thresh = base_threshold * 0.85
        max_thresh = base_threshold * 1.15
        
        return max(min_thresh, min(max_thresh, adapted))
    
    def get_stats(self) -> dict:
        """Статистика метакогниции для отладки."""
        return {
            "history_size": len(self._history),
            "type_stats": {
                t: {"error_rate": s["errors"] / max(s["total"], 1),
                    "ema_error": s["ema_error"],
                    "total": s["total"]}
                for t, s in self._type_stats.items()
            }
        }


# ═══════════════════════════════════════════════════════════════
# DreamEngine — Нейронное сновидение (Sleep Replay + Pruning)
# ═══════════════════════════════════════════════════════════════

class DreamEngine:
    """
    Движок сновидений ТАРС — ночная консолидация памяти.
    
    3 фазы сна (аналог NREM + REM у человека):
    
    Phase 1 — Replay (NREM slow-wave):
        Прогоняем дневной опыт через value network.
        Высокоценные воспоминания укрепляются (повторная запись в LEANN).
    
    Phase 2 — Recombination (REM dreaming):
        Случайное смешивание несвязанных воспоминаний.
        Ищем неожиданные связи (комбинации с высокой новизной).
        Это аналог "инсайтов во сне" у человека.
    
    Phase 3 — Pruning (memory garbage collection):
        Удаляем воспоминания с минимальной ценностью.
        Освобождаем место для нового опыта.
    
    Уникальность: ни один ИИ-агент не имеет фазы сна с креативным
    поиском связей. Обычно sleep = просто сохранение.
    """
    
    def __init__(self, noise_scale: float = 0.1, recombine_k: int = 20):
        self.noise_scale = noise_scale
        self.recombine_k = recombine_k
        self.logger = logging.getLogger("Tars.Dream")
        self._dream_log = []
    
    def dream(
        self,
        experiences: list,
        value_net: Optional[nn.Module] = None,
        leann = None,
    ) -> dict:
        """
        Полный цикл сновидения.
        
        Args:
            experiences: list of dicts с ключами:
                - "query": str
                - "response": str
                - "quality": float (0-1)
                - "embedding": Tensor [D] (optional)
            value_net: TDValueEstimator.value_net (optional)
            leann: LeannIndex (optional, для записи/удаления)
        
        Returns:
            dict с результатами каждой фазы
        """
        self.logger.info(f"Dream: Starting sleep cycle ({len(experiences)} experiences)")
        
        results = {
            "phase1_replayed": 0,
            "phase2_insights": 0,
            "phase3_pruned": 0,
        }
        
        if not experiences:
            return results
        
        # ═══ Phase 1: Replay ═══
        self.logger.info("Dream Phase 1: Replay (NREM slow-wave)")
        high_value = []
        for exp in experiences:
            quality = exp.get("quality", 0.5)
            # Высокоценный опыт: перезаписать с усилением
            if quality > 0.7:
                if leann is not None:
                    try:
                        reinforced = f"[Важный опыт] Q: {exp['query']}\nA: {exp['response']}"
                        leann.add_document(reinforced)
                        results["phase1_replayed"] += 1
                    except Exception:
                        pass
                high_value.append(exp)
        
        self.logger.info(f"  Replayed {results['phase1_replayed']} high-value experiences")
        
        # ═══ Phase 2: Recombination (Creative Dreams) ═══
        self.logger.info("Dream Phase 2: Recombination (REM dreaming)")
        import random
        
        if len(experiences) >= 2:
            for _ in range(self.recombine_k):
                a, b = random.sample(experiences, 2)
                
                # Ищем неожиданные связи между разными запросами
                q_a = a.get("query", "")
                q_b = b.get("query", "")
                
                # Простая heuristic: если запросы из разных тем
                # но имеют общие слова — потенциальный инсайт
                words_a = set(q_a.lower().split())
                words_b = set(q_b.lower().split())
                
                common = words_a & words_b - {"", "как", "что", "и", "в", "на", "с"}
                if common and len(common) <= 3:
                    insight = (
                        f"[Связь найдена во сне]\n"
                        f"Тема A: {q_a[:100]}\n"
                        f"Тема B: {q_b[:100]}\n"
                        f"Общее: {', '.join(common)}"
                    )
                    self._dream_log.append(insight)
                    results["phase2_insights"] += 1
                    
                    if leann is not None:
                        try:
                            leann.add_document(insight)
                        except Exception:
                            pass
        
        self.logger.info(f"  Found {results['phase2_insights']} dream insights")
        
        # ═══ Phase 3: Pruning ═══
        self.logger.info("Dream Phase 3: Pruning (memory GC)")
        low_value = [e for e in experiences if e.get("quality", 0.5) < 0.3]
        results["phase3_pruned"] = len(low_value)
        # Примечание: реальное удаление из LEANN требует ID документа,
        # пока просто логируем
        if low_value:
            self.logger.info(f"  Identified {len(low_value)} low-value memories for pruning")
        
        self.logger.info(
            f"Dream complete: {results['phase1_replayed']} replayed, "
            f"{results['phase2_insights']} insights, "
            f"{results['phase3_pruned']} pruned"
        )
        
        return results
    
    @property
    def dream_log(self) -> list:
        """Лог сновидений (найденные инсайты)."""
        return self._dream_log
