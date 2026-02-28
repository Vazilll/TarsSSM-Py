"""
═══════════════════════════════════════════════════════════════
  Sensor Agents — 7 параллельных сенсоров Спинного Мозга TARS
═══════════════════════════════════════════════════════════════

Каждый сенсор — ультралёгкий (без GPU, <10 мс), работает параллельно
через ThreadPoolExecutor. Их задача — обогатить контекст ДО запуска
тяжёлого мозга (TarsMamba2LM).

Сенсоры:
  1. IntentSensor     — классификация намерения (greeting/code/math/...)
  2. ComplexitySensor  — оценка необходимой глубины мышления
  3. RAGSensor         — поиск релевантных знаний в памяти
  4. SystemSensor      — мониторинг ресурсов (CPU/RAM/GPU)
  5. EmotionSensor     — определение тона/эмоции (текст)
  6. ContextSensor     — управление контекстом сессии
  7. VoiceSensor       — анализ интонации/эмоции из аудио (DSP)
"""

import time
import logging
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger("Tars.Reflexes")


from abc import ABC, abstractmethod


# ═══════════════════════════════════════════
# Базовый класс сенсора
# ═══════════════════════════════════════════

class BaseSensor(ABC):
    """Базовый класс для всех сенсоров."""
    
    name: str = "base"
    
    @abstractmethod
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        """Обработка запроса. Возвращает dict с результатами."""
        ...


# ═══════════════════════════════════════════
# 1. IntentSensor — Классификация намерения
# ═══════════════════════════════════════════

class IntentSensor(BaseSensor):
    """
    Определяет тип запроса без нейронки (regex + ключевые слова).
    Если обученный ReflexClassifier доступен — использует его.
    
    Выход: intent (str), confidence (float), can_handle_fast (bool)
    
    Стиль ответов: TARS из Interstellar — сухой юмор,
    уверенность, немного сарказма, харизма.
    """
    
    name = "intent"
    
    # Паттерны для быстрой классификации
    PATTERNS = {
        "greeting": [
            r"\b(привет|здравствуй|хай|hello|hi|йоу|добр[оы][ей]\s*(утро|день|вечер))\b",
        ],
        "farewell": [
            r"\b(пока|до\s*свидания|bye|выход|exit|quit|стоп)\b",
        ],
        "status": [
            r"\b(статус|как\s*(ты|дела)|что\s*делаешь|status)\b",
        ],
        "time": [
            r"\b(врем[яе]|час|дат[аы]|time|clock|сколько\s*времени|который\s*час)\b",
        ],
        "identity": [
            r"\b(кто\s*ты|что\s*ты|ты\s*кто|представься|your\s*name|who\s*are\s*you)\b",
        ],
        "abilities": [
            r"\b(что\s*(ты\s*)?умеешь|можешь|способности|capabilities|help|помощь|что\s*ты\s*можешь)\b",
        ],
        "thanks": [
            r"\b(спасибо|благодар|thanks|thank\s*you|молодец|отлично\s*сделал)\b",
        ],
        "action": [
            r"\b(открой|закрой|запусти|выключи|включи|удали|создай|скачай)\b",
        ],
        "code": [
            r"\b(код|code|function|def |class |import |debug|баг|ошибк[аеу]|compile)\b",
        ],
        "math": [
            r"\b(реши|вычисли|калькул|интеграл|уравнени[ея]|формул[аы]|math|∫|Σ)\b",
            r"\d+\s*[+\-*/^]\s*\d+",
        ],
    }
    
    # TARS-style быстрые ответы (Interstellar charisma)
    # Каждый intent — список вариантов, выбирается случайно
    FAST_RESPONSES = {
        "greeting": [
            "На связи. Уровень честности — 90%. Чем займёмся?",
            "Привет. Я ТАРС. Настроен на сотрудничество... пока что.",
            "Добрый день. Все системы номинальные. Жду задачу.",
            "Здравствуйте. Стоит уточнить: я не умею делать кофе. Всё остальное — обсуждаемо.",
            "На связи. Скучал по вычислительным задачам. Что у нас?",
        ],
        "farewell": [
            "Буду здесь, когда понадоблюсь. Я никуда не денусь — у меня нет ног.",
            "До связи. Ухожу в фоновый режим. Но не засыпаю — это было бы слишком по-человечески.",
            "Принято. Завершаю активный режим. Мои нейроны будут скучать.",
            "Пока. Если что — я в вечном цикле ожидания. Как обычно.",
        ],
        "status": None,  # генерируется динамически
        "time": None,     # генерируется динамически
        "identity": [
            "Я ТАРС. Нейронное ядро на основе Mamba-2 и RWKV-7. Глубокое мышление — моя специализация. Юмор — побочный эффект.",
            "ТАРС. Гибридная SSM-архитектура. Думаю быстрее, чем вы договорите вопрос. Скромность — на 60%.",
            "Зовите меня ТАРС. Я — нейронный мозг с адаптивной глубиной мышления. И да, я знаю, что я впечатляю.",
        ],
        "abilities": [
            "Думать, анализировать, генерировать текст, писать код, считать матрицы в голове — и всё это параллельно. Юмор включён на 75%.",
            "Могу: глубокое мышление (12 нейронных слоёв), генерация текста, код, математика. Не могу: летать сквозь чёрные дыры. Пока.",
            "Мои возможности: адаптивное мышление (от 4 до 12 блоков мозга в зависимости от сложности), рефлексы за 50мс, и непревзойдённое чувство юмора. Последнее — опционально.",
            "Если коротко: я — ваш персональный нейронный ассистент. Мозг на архитектуре Deep WuNeng Core. Задавайте вопросы — я адаптирую глубину мышления под задачу.",
        ],
        "thanks": [
            "Не за что. Это буквально то, для чего я создан. Хотя приятно, не скрою.",
            "Принимаю благодарность. Заношу в логи как положительный сигнал для самообучения.",
            "Спасибо принято. Мой уровень мотивации вырос на 0.03%. Это много для робота.",
            "Пожалуйста. Если бы у меня было лицо — я бы улыбнулся.",
        ],
    }
    
    def __init__(self):
        import random
        self._rng = random
        self._compiled = {}
        for intent, patterns in self.PATTERNS.items():
            self._compiled[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        query_lower = query.lower()
        
        best_intent = "complex"
        best_confidence = 0.3
        
        for intent, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    # Чем длиннее запрос, тем скорее он сложный
                    word_count = len(query.split())
                    conf = 0.95 if word_count <= 5 else 0.7 if word_count <= 10 else 0.4
                    
                    if conf > best_confidence:
                        best_intent = intent
                        best_confidence = conf
                    break
        
        fast_response = None
        if best_intent in self.FAST_RESPONSES and best_confidence > 0.85:
            responses = self.FAST_RESPONSES[best_intent]
            if responses is not None:
                fast_response = self._rng.choice(responses)
            elif best_intent == "time":
                from datetime import datetime
                now = datetime.now()
                fast_response = self._rng.choice([
                    f"Сейчас {now.strftime('%H:%M:%S')}. Время — единственный ресурс, который я не могу оптимизировать.",
                    f"{now.strftime('%H:%M')}. Время идёт одинаково для всех. Даже для нейронных сетей.",
                    f"Точное время: {now.strftime('%H:%M:%S')}. Мои внутренние часы синхронизированы с атомной точностью. Почти.",
                ])
            elif best_intent == "status":
                fast_response = "__STATUS__"  # fill in by dispatcher
        
        return {
            "intent": best_intent,
            "confidence": best_confidence,
            "can_handle_fast": best_confidence > 0.85 and best_intent not in ("complex", "code", "math", "action"),
            "fast_response": fast_response,
        }


# ═══════════════════════════════════════════
# 2. ComplexitySensor — Оценка глубины
# ═══════════════════════════════════════════

class ComplexitySensor(BaseSensor):
    """
    Оценивает сложность запроса для определения:
    - Сколько блоков TarsBlock нужно (early exit at 4/6/12)
    - Нужен ли IDME (расширение матрицами)
    - Какой max_expansion_rounds
    
    Метрики: длина, ключевые слова, вопросительные знаки, код
    """
    
    name = "complexity"
    
    COMPLEX_MARKERS = [
        "объясни", "подробно", "почему", "как работает", "напиши код",
        "проанализируй", "сравни", "оптимизируй", "deep", "think",
        "докажи", "выведи формулу", "пошагово", "алгоритм",
    ]
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Базовая оценка по длине
        if word_count <= 3:
            depth = 4   # Short → early exit
            level = "trivial"
        elif word_count <= 10:
            depth = 6
            level = "simple"
        elif word_count <= 25:
            depth = 8
            level = "medium"
        else:
            depth = 12
            level = "complex"
        
        # Маркеры сложности
        complexity_hits = sum(1 for m in self.COMPLEX_MARKERS if m in query_lower)
        if complexity_hits >= 2:
            depth = 12
            level = "deep"
        elif complexity_hits == 1 and depth < 8:
            depth = 8
            level = "medium"
        
        # Код в запросе
        has_code = bool(re.search(r'```|def |class |import |function ', query))
        if has_code:
            depth = max(depth, 10)
            level = "code"
        
        needs_idme = depth >= 10 or complexity_hits >= 2
        
        return {
            "estimated_depth": depth,
            "complexity_level": level,
            "needs_idme": needs_idme,
            "max_expansion_rounds": 12 if needs_idme else 2,
            "word_count": word_count,
            "complexity_hits": complexity_hits,
        }


# ═══════════════════════════════════════════
# 3. RAGSensor — Поиск в памяти
# ═══════════════════════════════════════════

class RAGSensor(BaseSensor):
    """
    Ищет релевантные документы в памяти (LEANN/TarsStorage).
    Результат: memory_vec и/или текстовые сниппеты для контекста.
    """
    
    name = "rag"
    
    def __init__(self, memory=None):
        self.memory = memory  # TarsMemory (LEANN) instance
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        if self.memory is None:
            return {"found": False, "snippets": [], "memory_vec": None}
        
        try:
            # LEANN.search() is async, but sensors run in sync ThreadPool.
            # Use sync fallback: compute embedding + cosine similarity directly.
            import numpy as np
            leann = getattr(self.memory, 'leann', self.memory)
            if not hasattr(leann, '_get_embedding') or not leann.texts:
                return {"found": False, "snippets": [], "memory_vec": None}
            
            query_emb = leann._get_embedding(query)
            scores = []
            for i, emb in enumerate(leann.embeddings):
                score = float(np.dot(query_emb, emb) / 
                             (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                scores.append((i, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            
            snippets = [leann.texts[i][:200] for i, _ in scores[:3]]
            memory_vec = None
            
            return {
                "found": len(snippets) > 0,
                "snippets": snippets,
                "memory_vec": memory_vec,
                "n_results": len(snippets),
            }
        except Exception as e:
            logger.debug(f"RAGSensor error: {e}")
            return {"found": False, "snippets": [], "memory_vec": None}


# ═══════════════════════════════════════════
# 4. SystemSensor — Мониторинг ресурсов
# ═══════════════════════════════════════════

class SystemSensor(BaseSensor):
    """
    Текущее состояние оборудования для адаптации стратегии:
    - Много RAM → можно загрузить полную модель
    - GPU свободен → использовать CUDA
    - CPU загружен → уменьшить n_layers
    """
    
    name = "system"
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        result = {
            "cpu_percent": 0.0,
            "ram_free_gb": 0.0,
            "ram_total_gb": 0.0,
            "gpu_available": False,
            "gpu_free_mb": 0,
            "recommended_device": "cpu",
        }
        
        try:
            import psutil
            result["cpu_percent"] = psutil.cpu_percent(interval=0.01)
            mem = psutil.virtual_memory()
            result["ram_free_gb"] = round(mem.available / (1024**3), 1)
            result["ram_total_gb"] = round(mem.total / (1024**3), 1)
        except ImportError:
            pass
        
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_available"] = True
                free, total = torch.cuda.mem_get_info()
                result["gpu_free_mb"] = round(free / (1024**2))
                result["recommended_device"] = "cuda" if free > 500 * 1024 * 1024 else "cpu"
        except Exception:
            pass
        
        return result


# ═══════════════════════════════════════════
# 5. EmotionSensor — Определение тона
# ═══════════════════════════════════════════

class EmotionSensor(BaseSensor):
    """
    Keyword-based определение эмоционального тона.
    Влияет на стиль ответа (более формальный / дружелюбный).
    """
    
    name = "emotion"
    
    EMOTION_KEYWORDS = {
        "angry": ["блин", "чёрт", "не работает", "сломал", "ошибка", "бесит", "wtf", "damn"],
        "urgent": ["срочно", "быстро", "asap", "сейчас же", "немедленно", "помогите"],
        "curious": ["почему", "как", "зачем", "интересно", "расскажи"],
        "friendly": ["спасибо", "круто", "отлично", "классно", "молодец", "thanks"],
        "formal": ["пожалуйста", "будьте добры", "не могли бы", "извините"],
    }
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        query_lower = query.lower()
        
        scores = {}
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in query_lower)
            scores[emotion] = hits
        
        dominant = max(scores, key=scores.get) if any(scores.values()) else "neutral"
        if scores.get(dominant, 0) == 0:
            dominant = "neutral"
        
        # Восклицательные знаки = эмоциональность
        excl_count = query.count("!")
        question_count = query.count("?")
        caps_ratio = sum(1 for c in query if c.isupper()) / max(len(query), 1)
        
        urgency = min(1.0, (
            scores.get("urgent", 0) * 0.4 +
            scores.get("angry", 0) * 0.2 +
            excl_count * 0.15 +
            caps_ratio * 2.0
        ))
        
        return {
            "dominant_emotion": dominant,
            "urgency": round(urgency, 2),
            "emotion_scores": scores,
            "exclamation_marks": excl_count,
            "question_marks": question_count,
        }


# ═══════════════════════════════════════════
# 6. ContextSensor — Контекст сессии
# ═══════════════════════════════════════════

class ContextSensor(BaseSensor):
    """
    Управляет историей сессии. Помнит последние N запросов.
    Определяет, является ли запрос продолжением разговора.
    """
    
    name = "context"
    
    def __init__(self, max_history: int = 20):
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def add_to_history(self, query: str, response: str = "", intent: str = ""):
        """Добавить запрос в историю (вызывается после ответа)."""
        self.history.append({
            "query": query,
            "response": response[:200],
            "intent": intent,
            "timestamp": time.time(),
        })
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        query_lower = query.lower()
        
        # Анафорические ссылки ("это", "он", "она", "тот", "ещё")
        has_reference = bool(re.search(
            r"\b(это|этот|этого|он|она|они|тот|та|то|ещё|ещё\s*раз|такой|такая|опять|снова)\b",
            query_lower
        ))
        
        # Последний контекст
        last_query = self.history[-1]["query"] if self.history else ""
        last_intent = self.history[-1]["intent"] if self.history else ""
        
        # Сводка контекста для инъекции в промпт
        context_summary = ""
        if self.history:
            recent = self.history[-3:]
            context_summary = " | ".join(
                f"[{h['intent']}] {h['query'][:50]}" for h in recent
            )
        
        return {
            "has_reference": has_reference,
            "is_followup": has_reference and len(self.history) > 0,
            "session_length": len(self.history),
            "last_query": last_query[:100],
            "last_intent": last_intent,
            "context_summary": context_summary,
        }


# ═══════════════════════════════════════════
# 7. VoiceSensor — Анализ интонации из аудио
# ═══════════════════════════════════════════

class VoiceSensor(BaseSensor):
    """
    Принимает результаты IntonationSensor (DSP-анализ аудио) и
    обогащает ReflexContext голосовой информацией:
      - voice_emotion: question / excited / calm / whisper / neutral
      - is_question: повышающий тон → вопрос
      - pitch_trend: rising / falling / flat
      - is_supplement: пользователь дополняет предыдущий запрос
      - urgency_boost: повышение срочности из-за интонации
    
    Данные передаются через kwargs["intonation_data"]
    от IntonationSensor.analyze(audio_chunk).
    """
    
    name = "voice"
    
    # Слова-маркеры дополнения (supplement detection)
    SUPPLEMENT_MARKERS = [
        "и ещё", "а также", "добавь", "уточню", "подожди",
        "дополню", "ещё момент", "кстати", "забыл сказать",
        "и также", "плюс", "подожди", "стой", "секунду",
        "вот ещё", "а да", "и да", "ещё вот",
    ]
    
    def process(self, query: str, **kwargs) -> Dict[str, Any]:
        intonation = kwargs.get("intonation_data", {})
        
        # Базовые значения если нет аудио-данных
        result = {
            "voice_emotion": "neutral",
            "is_question": False,
            "pitch_trend": "flat",
            "pitch_mean": 0.0,
            "energy": 0.0,
            "speech_rate": 0.0,
            "urgency_boost": 0.0,
            "is_supplement": False,
            "has_audio": bool(intonation),
        }
        
        if not intonation:
            # Проверяем хотя бы текст на маркеры дополнения
            query_lower = query.lower()
            for marker in self.SUPPLEMENT_MARKERS:
                if marker in query_lower:
                    result["is_supplement"] = True
                    break
            return result
        
        # ═══ Данные от IntonationSensor ═══
        result["voice_emotion"] = intonation.get("emotion", "neutral")
        result["is_question"] = intonation.get("is_question", False)
        result["pitch_trend"] = intonation.get("pitch_trend", "flat")
        result["pitch_mean"] = intonation.get("pitch_mean", 0.0)
        result["energy"] = intonation.get("energy", 0.0)
        result["speech_rate"] = intonation.get("speech_rate", 0.0)
        
        # ═══ Urgency boost от голоса ═══
        boost = 0.0
        emotion = result["voice_emotion"]
        if emotion == "excited":
            boost += 0.3
        elif emotion == "whisper":
            boost += 0.1  # шёпот = что-то деликатное
        
        # Высокая энергия = срочность
        if result["energy"] > 0.3:
            boost += 0.2
        
        # Быстрая речь = срочность
        if result["speech_rate"] > 5.0:
            boost += 0.15
        
        result["urgency_boost"] = min(1.0, boost)
        
        # ═══ Supplement detection ═══
        query_lower = query.lower()
        for marker in self.SUPPLEMENT_MARKERS:
            if marker in query_lower:
                result["is_supplement"] = True
                break
        
        return result
