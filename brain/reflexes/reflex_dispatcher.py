"""
═══════════════════════════════════════════════════════════════
  ReflexDispatcher — Параллельный диспетчер спинного мозга
═══════════════════════════════════════════════════════════════

Запускает 6 сенсоров одновременно через ThreadPoolExecutor.
Собирает результаты в ReflexContext — обогащённый контекст для
основного мозга (TarsMamba2LM).

Принцип работы:
  1. Пользователь вводит запрос
  2. ReflexDispatcher.dispatch(query) → 6 сенсоров параллельно (<100мс)
  3. Если рефлексы решили запрос (greeting/time) → мгновенный ответ
  4. Иначе → ReflexContext передаётся в TarsMamba2LM.think()
     с рекомендованной глубиной, RAG-контекстом и метаданными

Использование:
  dispatcher = ReflexDispatcher()
  ctx = dispatcher.dispatch("Привет, как дела?")
  
  if ctx.can_handle_fast:
      print(ctx.fast_response)
  else:
      logits, stats = brain.think(tokens, ctx=ctx)
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from brain.reflexes.sensors import (
    IntentSensor,
    ComplexitySensor,
    RAGSensor,
    SystemSensor,
    EmotionSensor,
    ContextSensor,
    VoiceSensor,
)

logger = logging.getLogger("Tars.ReflexDispatcher")


@dataclass
class ReflexContext:
    """
    Обогащённый контекст после параллельной обработки всеми сенсорами.
    Передаётся в основной мозг для адаптации стратегии.
    """
    
    # ═══ Timing ═══
    query: str = ""
    dispatch_time_ms: float = 0.0
    sensor_times: Dict[str, float] = field(default_factory=dict)
    
    # ═══ Intent (Sensor 1) ═══
    intent: str = "complex"
    confidence: float = 0.0
    can_handle_fast: bool = False
    fast_response: Optional[str] = None
    
    # ═══ Complexity (Sensor 2) ═══
    estimated_depth: int = 12
    complexity_level: str = "complex"
    needs_idme: bool = True
    max_expansion_rounds: int = 12
    
    # ═══ RAG (Sensor 3) ═══
    rag_found: bool = False
    rag_snippets: List[str] = field(default_factory=list)
    memory_vec: Any = None  # torch.Tensor or None
    
    # ═══ System (Sensor 4) ═══
    cpu_percent: float = 0.0
    ram_free_gb: float = 0.0
    gpu_available: bool = False
    recommended_device: str = "cpu"
    
    # ═══ Emotion (Sensor 5) ═══
    dominant_emotion: str = "neutral"
    urgency: float = 0.0
    
    # ═══ Context (Sensor 6) ═══
    is_followup: bool = False
    session_length: int = 0
    context_summary: str = ""
    
    # ═══ Voice/Intonation (Sensor 7) ═══
    voice_emotion: str = "neutral"
    voice_is_question: bool = False
    voice_pitch_trend: str = "flat"
    voice_energy: float = 0.0
    is_supplement: bool = False
    has_voice_data: bool = False
    
    def summary_line(self) -> str:
        """Краткая строка для CLI."""
        emoji_map = {
            "greeting": "👋", "farewell": "👋", "status": "📊",
            "time": "⏰", "action": "⚡", "code": "💻",
            "math": "🔢", "complex": "🧠", "neutral": "💬",
            "identity": "🤖", "abilities": "💪", "thanks": "😊",
        }
        emoji = emoji_map.get(self.intent, "💬")
        
        parts = [
            f"{emoji} {self.intent}({self.confidence:.0%})",
            f"depth={self.estimated_depth}",
            f"{self.complexity_level}",
        ]
        if self.rag_found:
            parts.append(f"RAG:{len(self.rag_snippets)}docs")
        if self.urgency > 0.3:
            parts.append(f"⚠️urgent={self.urgency:.0%}")
        if self.is_followup:
            parts.append("↩️followup")
        if self.is_supplement:
            parts.append("🎤supplement")
        if self.has_voice_data:
            parts.append(f"🗣{self.voice_emotion}")
        
        return " | ".join(parts)


class ReflexDispatcher:
    """
    Параллельный диспетчер: запускает все сенсоры через ThreadPool
    и собирает результаты в ReflexContext.
    
    Типичное время: <50ms для 6 сенсоров (потому что CPU-bound
    операции крошечные, а ThreadPool маскирует I/O от SystemSensor).
    """
    
    def __init__(self, memory=None, max_workers: int = 6):
        """
        Args:
            memory: TarsMemory (LEANN) instance для RAGSensor.
                    None = RAG отключён.
            max_workers: Количество потоков для параллелизма.
        """
        self.sensors = {
            "intent": IntentSensor(),
            "complexity": ComplexitySensor(),
            "rag": RAGSensor(memory=memory),
            "system": SystemSensor(),
            "emotion": EmotionSensor(),
            "context": ContextSensor(),
            "voice": VoiceSensor(),
        }
        self.max_workers = max_workers
        self.total_dispatches = 0
        self.total_fast_handled = 0
        
        logger.info(
            f"ReflexDispatcher: {len(self.sensors)} сенсоров, "
            f"{max_workers} потоков"
        )
    
    def dispatch(self, query: str, intonation_data: dict = None) -> ReflexContext:
        """
        Параллельный запуск всех сенсоров.
        
        Args:
            query: Текст запроса пользователя
            intonation_data: Данные от IntonationSensor (опционально)
        
        Returns:
            ReflexContext с результатами всех сенсоров
        """
        t0 = time.perf_counter()
        self.total_dispatches += 1
        
        ctx = ReflexContext(query=query)
        results = {}
        
        # kwargs для сенсоров (VoiceSensor получит intonation_data)
        sensor_kwargs = {}
        if intonation_data:
            sensor_kwargs["intonation_data"] = intonation_data
        
        # ═══ Параллельный запуск всех сенсоров ═══
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {}
            for name, sensor in self.sensors.items():
                future = pool.submit(self._run_sensor, sensor, query, **sensor_kwargs)
                futures[future] = name
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result, elapsed = future.result()
                    results[name] = result
                    ctx.sensor_times[name] = elapsed
                except Exception as e:
                    logger.warning(f"Sensor '{name}' failed: {e}")
                    results[name] = {}
                    ctx.sensor_times[name] = 0.0
        
        # ═══ Сборка ReflexContext ═══
        self._fill_context(ctx, results)
        
        ctx.dispatch_time_ms = (time.perf_counter() - t0) * 1000
        
        if ctx.can_handle_fast:
            self.total_fast_handled += 1
        
        logger.debug(
            f"Dispatch: {ctx.dispatch_time_ms:.1f}ms | "
            f"{ctx.summary_line()}"
        )
        
        return ctx
    
    def _run_sensor(self, sensor, query: str, **kwargs):
        """Запуск одного сенсора с замером времени."""
        t0 = time.perf_counter()
        result = sensor.process(query, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed
    
    def _fill_context(self, ctx: ReflexContext, results: Dict[str, Dict]):
        """Заполняет ReflexContext из результатов сенсоров."""
        
        # Intent
        r = results.get("intent", {})
        ctx.intent = r.get("intent", "complex")
        ctx.confidence = r.get("confidence", 0.0)
        ctx.can_handle_fast = r.get("can_handle_fast", False)
        ctx.fast_response = r.get("fast_response")
        
        # Подставляем status если нужно
        if ctx.fast_response == "__STATUS__":
            import random
            sys_r = results.get("system", {})
            cpu = sys_r.get("cpu_percent", 0)
            ram = sys_r.get("ram_free_gb", 0)
            gpu = "✅" if sys_r.get("gpu_available") else "OFF"
            n = self.total_dispatches
            ctx.fast_response = random.choice([
                f"Все системы в норме. CPU: {cpu}%, RAM: {ram:.1f}GB свободно, GPU: {gpu}. "
                f"Обработано запросов: {n}. Настроение: оптимистично-вычислительное.",
                f"Работаю штатно. CPU загружен на {cpu}%, памяти {ram:.1f}GB. "
                f"GPU: {gpu}. Количество мыслей за сессию: {n}. Ни одна не была бесполезной. Почти.",
                f"Статус: полностью операционален. {cpu}% CPU, {ram:.1f}GB RAM, GPU: {gpu}. "
                f"Запросов: {n}. Уровень энтузиазма: стабильно высокий.",
            ])
        
        # Complexity
        r = results.get("complexity", {})
        ctx.estimated_depth = r.get("estimated_depth", 12)
        ctx.complexity_level = r.get("complexity_level", "complex")
        ctx.needs_idme = r.get("needs_idme", True)
        ctx.max_expansion_rounds = r.get("max_expansion_rounds", 12)
        
        # RAG
        r = results.get("rag", {})
        ctx.rag_found = r.get("found", False)
        ctx.rag_snippets = r.get("snippets", [])
        ctx.memory_vec = r.get("memory_vec")
        
        # System
        r = results.get("system", {})
        ctx.cpu_percent = r.get("cpu_percent", 0)
        ctx.ram_free_gb = r.get("ram_free_gb", 0)
        ctx.gpu_available = r.get("gpu_available", False)
        ctx.recommended_device = r.get("recommended_device", "cpu")
        
        # Emotion
        r = results.get("emotion", {})
        ctx.dominant_emotion = r.get("dominant_emotion", "neutral")
        ctx.urgency = r.get("urgency", 0)
        
        # Context
        r = results.get("context", {})
        ctx.is_followup = r.get("is_followup", False)
        ctx.session_length = r.get("session_length", 0)
        ctx.context_summary = r.get("context_summary", "")
        
        # Voice
        r = results.get("voice", {})
        ctx.voice_emotion = r.get("voice_emotion", "neutral")
        ctx.voice_is_question = r.get("is_question", False)
        ctx.voice_pitch_trend = r.get("pitch_trend", "flat")
        ctx.voice_energy = r.get("energy", 0.0)
        ctx.is_supplement = r.get("is_supplement", False)
        ctx.has_voice_data = r.get("has_audio", False)
        
        # Merge voice urgency boost into overall urgency
        voice_boost = r.get("urgency_boost", 0.0)
        if voice_boost > 0:
            ctx.urgency = min(1.0, ctx.urgency + voice_boost)
        
        # Voice emotion overrides text emotion when audio is present
        if ctx.has_voice_data and ctx.voice_emotion != "neutral":
            ctx.dominant_emotion = ctx.voice_emotion
    
    def add_to_history(self, query: str, response: str = "", intent: str = ""):
        """Обновляет историю сессии в ContextSensor."""
        self.sensors["context"].add_to_history(query, response, intent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика диспетчера."""
        return {
            "total_dispatches": self.total_dispatches,
            "total_fast_handled": self.total_fast_handled,
            "fast_ratio": (
                self.total_fast_handled / max(self.total_dispatches, 1)
            ),
            "n_sensors": len(self.sensors),
        }
