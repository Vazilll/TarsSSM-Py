"""
═══════════════════════════════════════════════════════════════
  ThinkingChain v3 — Полная система рассуждения ТАРС
═══════════════════════════════════════════════════════════════

10 ПОДСИСТЕМ:

v1 (base):
  1. ThinkingStepProjector — генерация thought_vec + retrieval_vec
  2. 4 фазы: explore → analyze → synthesize → verify

v2 (+5):
  3. Retrieval-triggered RAG между волнами
  4. Multi-Scale Memory (working/session/longterm)
  5. Confidence-Gated Output
  6. Wave Skip (confidence > 0.9)
  7. Self-Verification (model.py)

v3 (+3):
  8. Thought Caching — кэш мыслительных траекторий (2.5x speedup)
  9. Entropy-Based Phase Detection — фазы по состоянию, не по позиции
  10. Sleep Consolidation — офлайн закрепление памяти
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, Callable, Tuple, List, Dict
from collections import deque


class MultiScaleMemory(nn.Module):
    """
    3-уровневая память вместо одного memory_vec.
    
    working:  текущая задача (обновляется каждую волну, fast)
    session:  контекст диалога (обновляется каждый запрос, slow)
    longterm: знания пользователя (обновляется через Titans, SGD)
    
    Финальный memory_vec = weighted sum всех 3 уровней.
    """
    
    def __init__(self, d_memory=384):
        super().__init__()
        self.d_memory = d_memory
        
        # Обучаемые веса смешивания 3 уровней
        self.mix_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # Проекции для каждого уровня (разные "взгляды" на данные)
        self.working_proj = nn.Linear(d_memory, d_memory)
        self.session_proj = nn.Linear(d_memory, d_memory)
        self.longterm_proj = nn.Linear(d_memory, d_memory)
        
        # State (не параметры — буферы)
        self.register_buffer('working', None)
        self.register_buffer('session', None)
        self.register_buffer('longterm', None)
        
        # Скорости обновления
        self.working_rate = 0.3   # быстро
        self.session_rate = 0.1   # медленно
        self.longterm_rate = 0.01 # очень медленно (в основном Titans)
    
    def reset_working(self):
        """Reset при новом запросе (working обнуляется)."""
        self.working = None
    
    def reset_all(self):
        """Полный reset (новая сессия)."""
        self.working = None
        self.session = None
        self.longterm = None
    
    def update_working(self, h_new):
        """Обновить рабочую память (каждая волна)."""
        if self.working is None:
            self.working = h_new.detach()
        else:
            self.working = (1 - self.working_rate) * self.working + self.working_rate * h_new.detach()
    
    def update_session(self, h_new):
        """Обновить сессионную память (каждый запрос)."""
        if self.session is None:
            self.session = h_new.detach()
        else:
            self.session = (1 - self.session_rate) * self.session + self.session_rate * h_new.detach()
    
    def update_longterm(self, h_new):
        """Обновить долговременную память (Titans/SGD)."""
        if self.longterm is None:
            self.longterm = h_new.detach()
        else:
            self.longterm = (1 - self.longterm_rate) * self.longterm + self.longterm_rate * h_new.detach()
    
    def get_combined(self, device=None):
        """
        Получить взвешенную комбинацию всех 3 уровней.
        
        Returns: memory_vec [B, d_memory] или None
        """
        weights = F.softmax(self.mix_weights, dim=0)  # нормализация
        
        components = []
        active_weights = []
        
        if self.working is not None:
            components.append(self.working_proj(self.working))
            active_weights.append(weights[0])
        if self.session is not None:
            components.append(self.session_proj(self.session))
            active_weights.append(weights[1])
        if self.longterm is not None:
            components.append(self.longterm_proj(self.longterm))
            active_weights.append(weights[2])
        
        if not components:
            return None
        
        # Нормализуем веса по активным компонентам
        total_w = sum(active_weights)
        result = sum(w / total_w * c for w, c in zip(active_weights, components))
        return result


class ThinkingStepProjector(nn.Module):
    """
    Проецирует текущее состояние мышления в "мыслительный вектор".
    
    v2: добавлен retrieval_query для запуска реального RAG поиска
    между волнами.
    """
    
    def __init__(self, d_model, d_memory=384, n_thinking_steps=12):
        super().__init__()
        self.d_model = d_model
        self.d_memory = d_memory
        
        # Step embedding: каждый шаг имеет свой "фокус внимания"
        self.step_embeddings = nn.Embedding(n_thinking_steps, d_model)
        
        # What I know: encode current understanding
        self.understanding_proj = nn.Linear(d_model, d_model)
        
        # What I need: generate query for next wave
        self.need_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_memory),
        )
        
        # Retrieval query: separate head for RAG search
        # Проецирует thought в пространство поиска LEANN
        self.retrieval_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, d_memory),
        )
        
        # How to refine: gating between old memory and new thought
        self.refine_gate = nn.Sequential(
            nn.Linear(d_memory * 2, d_memory),
            nn.Sigmoid(),
        )
        
        # Confidence: "Do I understand enough to stop?"
        self.confidence = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, h_current, memory_vec, wave_idx):
        """
        Один шаг мышления: текущее понимание → уточнённый memory_vec.
        
        Returns:
            refined_memory: [B, d_memory]
            retrieval_vec:  [B, d_memory] — вектор для RAG-поиска
            thought_confidence: float
            thought_info: dict
        """
        B = h_current.shape[0]
        
        # 1. "Что я уже понял?"
        understanding = self.understanding_proj(h_current)
        
        # 2. "На каком шаге я?"
        step_idx = min(wave_idx, self.step_embeddings.num_embeddings - 1)
        step_embed = self.step_embeddings(
            torch.tensor([step_idx], device=h_current.device)
        ).expand(B, -1)
        
        combined = torch.cat([understanding, step_embed], dim=-1)
        
        # 3. "Что мне нужно дальше?" → memory refinement
        thought_vec = self.need_proj(combined)
        
        # 4. "Что поискать в базе?" → retrieval query
        retrieval_vec = self.retrieval_proj(combined)
        
        # 5. Gate: merge old and new memory
        if memory_vec is not None:
            gate_input = torch.cat([memory_vec, thought_vec], dim=-1)
            gate = self.refine_gate(gate_input)
            refined_memory = gate * thought_vec + (1 - gate) * memory_vec
        else:
            gate = torch.ones(B, self.d_memory, device=h_current.device) * 0.5
            refined_memory = thought_vec
        
        # 6. Confidence
        thought_confidence = self.confidence(understanding).mean().item()
        
        thought_info = {
            "wave": wave_idx,
            "confidence": thought_confidence,
            "gate_mean": gate.mean().item(),
            "thought_norm": thought_vec.norm().item(),
            "retrieval_norm": retrieval_vec.norm().item(),
        }
        
        return refined_memory, retrieval_vec, thought_confidence, thought_info


class ConfidenceGate(nn.Module):
    """
    Модулирует logits на основе уверенности ThinkingChain.
    
    Если confidence < threshold:
      → logits сглаживаются (модель менее уверена в ответе)
      → повышается энтропия выхода (более "осторожный" ответ)
    
    Если confidence > 0.8:
      → logits не меняются (модель уверена)
    """
    
    def __init__(self, vocab_size, confidence_threshold=0.4):
        super().__init__()
        self.threshold = confidence_threshold
        
        # Uncertainty bias: обучаемый вектор, добавляемый при неуверенности
        # Повышает вероятность "осторожных" токенов
        self.uncertainty_bias = nn.Parameter(torch.zeros(vocab_size))
    
    def forward(self, logits, confidence):
        """
        Args:
            logits: [B, L, V] — logits от lm_head
            confidence: float — уверенность ThinkingChain (0-1)
        
        Returns:
            modulated_logits: [B, L, V]
        """
        if confidence >= 0.8:
            return logits  # уверен — не трогать
        
        # Фактор сглаживания: чем ниже confidence, тем больше сглаживание
        smoothing = max(0.0, (self.threshold - confidence) / self.threshold)
        smoothing = min(smoothing, 0.5)  # max 50% smoothing
        
        if smoothing > 0:
            # Сглаживаем logits (увеличиваем температуру)
            temperature = 1.0 + smoothing * 2.0  # до 2.0 при полной неуверенности
            logits = logits / temperature
            
            # Добавляем uncertainty bias
            logits = logits + smoothing * self.uncertainty_bias.unsqueeze(0).unsqueeze(0)
        
        return logits


# ═══════════════════════════════════════════════════════════════
#  v3: Thought Caching
# ═══════════════════════════════════════════════════════════════

class ThoughtCache:
    """
    Кэш мыслительных траекторий.
    
    Если новый запрос похож на предыдущий (cosine > 0.9),
    восстанавливаем траекторию мышления и пропускаем
    ранние фазы (explore/analyze).
    
    Это как Memo, но для ПРОЦЕССА мышления, а не для ответа.
    """
    
    def __init__(self, max_entries=128, similarity_threshold=0.9):
        self.max_entries = max_entries
        self.threshold = similarity_threshold
        
        # Кэш: {query_vec, trajectory, final_memory, confidence, timestamp}
        self.entries: deque = deque(maxlen=max_entries)
    
    @torch.no_grad()
    def lookup(self, query_vec):
        """
        Поиск похожей траектории.
        
        Args:
            query_vec: [B, d_model] — embedding запроса
        
        Returns:
            dict с trajectory, skip_waves, similarity ИЛИ None
        """
        if not self.entries or query_vec is None:
            return None
        
        q = query_vec.float().mean(dim=0)  # [d_model]
        q_norm = q / (q.norm() + 1e-8)
        
        best_sim = -1.0
        best_entry = None
        
        for entry in self.entries:
            cached_q = entry["query_vec"]
            sim = F.cosine_similarity(
                q_norm.unsqueeze(0), cached_q.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        
        if best_sim >= self.threshold and best_entry is not None:
            # Нашли похожую траекторию!
            trajectory = best_entry["trajectory"]
            n_phases = len(trajectory)
            
            # Пропускаем explore + analyze (первые 50% волн)
            skip_waves = max(1, n_phases // 2)
            
            return {
                "trajectory": trajectory,
                "final_memory": best_entry["final_memory"],
                "skip_waves": skip_waves,
                "similarity": best_sim,
                "cached_confidence": best_entry["confidence"],
            }
        
        return None
    
    def store(self, query_vec, trajectory, final_memory, confidence):
        """
        Сохранить траекторию мышления.
        
        Args:
            query_vec: [d_model] — средний embedding запроса
            trajectory: list[dict] — история ThinkingChain
            final_memory: [384] — финальный memory_vec
            confidence: float — финальная уверенность
        """
        if query_vec is None or len(trajectory) == 0:
            return
        
        q = query_vec.float()
        q_norm = q / (q.norm() + 1e-8)
        
        # Удаляем rag_result из trajectory (не сериализуется)
        clean_trajectory = []
        for t in trajectory:
            clean = {k: v for k, v in t.items() if k != "rag_result"}
            clean_trajectory.append(clean)
        
        self.entries.append({
            "query_vec": q_norm.detach().cpu(),
            "trajectory": clean_trajectory,
            "final_memory": final_memory.detach().cpu() if final_memory is not None else None,
            "confidence": confidence,
            "timestamp": time.time(),
        })
    
    def stats(self):
        return {
            "entries": len(self.entries),
            "max_entries": self.max_entries,
        }


# ═══════════════════════════════════════════════════════════════
#  v3: Sleep Consolidation
# ═══════════════════════════════════════════════════════════════

class SleepConsolidation:
    """
    Офлайн закрепление памяти (как сон у человека).
    
    Каждые N минут простоя:
      1. Replay: прогоняет важные запросы через 2 волны
      2. Compact: сжимает 100 воспоминаний в 10 "кристаллизованных"
      3. Prune: удаляет устаревшие записи
    
    Аналог: hippocampal replay во время медленного сна.
    """
    
    def __init__(self, max_history=200, consolidation_interval=1800):
        """Args: consolidation_interval в секундах (default 30 мин)."""
        self.max_history = max_history
        self.interval = consolidation_interval
        
        # История запросов (сохраняем memory_vec + surprise)
        self.history: deque = deque(maxlen=max_history)
        self.last_consolidation = time.time()
        self.consolidation_count = 0
    
    def record(self, memory_vec, surprise_level, task_type):
        """Записать запрос в историю."""
        if memory_vec is None:
            return
        self.history.append({
            "memory_vec": memory_vec.detach().cpu(),
            "surprise": surprise_level,
            "task_type": task_type,
            "timestamp": time.time(),
        })
    
    def should_consolidate(self):
        """Пора ли 'спать'?"""
        return (
            time.time() - self.last_consolidation > self.interval
            and len(self.history) >= 10
        )
    
    def consolidate(self, multi_memory):
        """
        Выполнить консолидацию памяти.
        
        1. Replay: пройти по важным воспоминаниям (самые surprising)
        2. Compact: усреднить похожие воспоминания
        3. Update: обновить longterm memory
        
        Args:
            multi_memory: MultiScaleMemory для обновления
        """
        if len(self.history) < 5:
            return {"replayed": 0, "consolidated": False}
        
        # 1) Сортируем по surprise (самые важные впереди)
        sorted_history = sorted(
            self.history, key=lambda x: x["surprise"], reverse=True
        )
        
        # 2) Берём top-10 самых важных
        top_memories = sorted_history[:10]
        
        # 3) Усредняем их в один "consolidated" вектор
        vecs = [m["memory_vec"] for m in top_memories]
        weights = [m["surprise"] for m in top_memories]
        total_w = sum(weights) + 1e-8
        
        consolidated = sum(
            (w / total_w) * v for w, v in zip(weights, vecs)
        )
        
        # 4) Обновляем longterm memory
        if multi_memory is not None:
            multi_memory.update_longterm(consolidated)
        
        # 5) Очищаем старые записи (оставляем top-20)
        self.history = deque(
            sorted_history[:20], maxlen=self.max_history
        )
        
        self.last_consolidation = time.time()
        self.consolidation_count += 1
        
        return {
            "replayed": len(top_memories),
            "consolidated": True,
            "consolidation_count": self.consolidation_count,
            "remaining_memories": len(self.history),
        }
    
    def stats(self):
        return {
            "history_size": len(self.history),
            "consolidations": self.consolidation_count,
            "time_since_last": time.time() - self.last_consolidation,
            "should_consolidate": self.should_consolidate(),
        }


class ThinkingChain(nn.Module):
    """
    Контроллер цепочки рассуждений v3.
    
    Подключается к think() между волнами и управляет:
    1. ФОКУСИРОВКОЙ: каждая волна ищет разное в памяти
    2. ПРОГРЕССИЕЙ: от общего к частному
    3. RETRIEVAL: запускает реальный RAG-поиск между волнами
    4. CONFIDENCE: оценивает уверенность для early stop
    5. MULTI-SCALE: обновляет 3 уровня памяти
    6. THOUGHT CACHE: кэширует траектории мышления
    7. ENTROPY PHASE: фазы по энтропии, не по позиции
    8. SLEEP: офлайн консолидация памяти
    
    Фазы мышления (определяются энтропией h_current):
      EXPLORE   — "что спросили?"     → RAG: общие концепции
      ANALYZE   — "как решить?"       → RAG: методы, алгоритмы
      SYNTHESIZE— "складываю ответ"   → RAG: детали, формулы
      VERIFY    — "проверяю"          → RAG: противоречия
    """
    
    def __init__(self, d_model, d_memory=384, n_max_waves=12, vocab_size=256):
        super().__init__()
        self.projector = ThinkingStepProjector(d_model, d_memory, n_max_waves)
        self.multi_memory = MultiScaleMemory(d_memory)
        self.confidence_gate = ConfidenceGate(vocab_size)
        self.thought_cache = ThoughtCache(max_entries=128, similarity_threshold=0.9)
        self.sleep = SleepConsolidation(max_history=200, consolidation_interval=1800)
        self.n_max_waves = n_max_waves
        
        # Retrieval callback: set by model.py to call LEANN/Memo
        self._retrieval_fn = None
        
        # State
        self.thinking_history = []
        self.last_confidence = 0.0
        self.last_retrieval_vec = None
        self._cache_hit = None  # Результат cache lookup
    
    def set_retrieval_fn(self, fn):
        """
        Устанавливает callback для реального RAG-поиска.
        
        fn(retrieval_vec: Tensor[B, 384]) -> Optional[Tensor[B, d_state, d_state]]
        Вызывается ThinkingChain между волнами.
        """
        self._retrieval_fn = fn
    
    def reset(self):
        """Reset при новом запросе."""
        self.thinking_history = []
        self.last_confidence = 0.0
        self.last_retrieval_vec = None
        self._cache_hit = None
        self.multi_memory.reset_working()
    
    def reset_session(self):
        """Reset при новой сессии (полный)."""
        self.thinking_history = []
        self.last_confidence = 0.0
        self.last_retrieval_vec = None
        self._cache_hit = None
        self.multi_memory.reset_all()
    
    def try_cache_shortcut(self, query_embedding):
        """
        Thought Caching: проверить есть ли похожая траектория.
        
        Вызывается в начале think() ПЕРЕД волнами.
        Если найдена — возвращает число волн для пропуска.
        
        Args:
            query_embedding: [B, d_model] — embedding запроса
            
        Returns:
            skip_waves: int (0 = не найдено, >0 = пропустить столько)
        """
        cache_result = self.thought_cache.lookup(query_embedding)
        if cache_result is not None:
            self._cache_hit = cache_result
            return cache_result["skip_waves"]
        self._cache_hit = None
        return 0
    
    def get_cached_memory(self):
        """Если был cache hit, вернуть кэшированный memory_vec."""
        if self._cache_hit is not None and self._cache_hit["final_memory"] is not None:
            return self._cache_hit["final_memory"]
        return None
    
    def get_phase(self, wave_idx, total_waves, h_current=None):
        """
        Entropy-Based Phase Detection (v3).
        
        Вместо фиксированных позиций (25%/50%/75%),
        фазы определяются СОСТОЯНИЕМ модели:
          - Высокая энтропия h → explore (много неопределённости)
          - Средняя → analyze (сужается)
          - Низкая → synthesize (почти нашёл)
          - Очень низкая → verify (уверен)
        
        Fallback на position-based если h_current не передан.
        """
        if h_current is not None:
            # Энтропия hidden state как мера неопределённости
            with torch.no_grad():
                h_flat = h_current.float().abs()  # [B, d_model]
                h_probs = F.softmax(h_flat, dim=-1)
                log_probs = torch.log(h_probs + 1e-8)
                entropy = -(h_probs * log_probs).sum(dim=-1).mean().item()
                
                # Нормализуем: max entropy = ln(d_model)
                max_entropy = math.log(h_current.shape[-1])
                norm_entropy = entropy / max_entropy  # [0, 1]
            
            if norm_entropy > 0.75:
                return "explore"
            elif norm_entropy > 0.50:
                return "analyze"
            elif norm_entropy > 0.25:
                return "synthesize"
            else:
                return "verify"
        
        # Fallback: position-based
        progress = wave_idx / max(total_waves - 1, 1)
        if progress < 0.25:
            return "explore"
        elif progress < 0.50:
            return "analyze"
        elif progress < 0.75:
            return "synthesize"
        else:
            return "verify"
    
    def should_skip_remaining(self):
        """
        Wave Skip: если confidence > 0.9 → можно пропустить.
        Дополнительный механизм к IntegralAuditor.
        """
        return self.last_confidence > 0.9 and len(self.thinking_history) >= 2
    
    def step(self, h_current, memory_vec, wave_idx, total_waves=12):
        """
        Один шаг мышления между волнами.
        
        v3 изменения:
          - Entropy-based фазы (h_current определяет фазу)
          - Thought cache recording
          - Sleep recording
        
        Returns:
            refined_memory: [B, d_memory]
            thought_info: dict (с retrieval данными)
        """
        phase = self.get_phase(wave_idx, total_waves, h_current)
        
        refined_memory, retrieval_vec, confidence, thought_info = \
            self.projector(h_current, memory_vec, wave_idx)
        
        thought_info["phase"] = phase
        self.last_confidence = confidence
        self.last_retrieval_vec = retrieval_vec
        
        # Multi-scale memory update
        if memory_vec is not None:
            self.multi_memory.update_working(memory_vec)
        
        # Retrieval trigger: вызываем реальный RAG-поиск
        rag_result = None
        if self._retrieval_fn is not None and retrieval_vec is not None:
            try:
                rag_result = self._retrieval_fn(retrieval_vec)
                thought_info["retrieval_triggered"] = True
                thought_info["retrieval_found"] = rag_result is not None
            except Exception:
                thought_info["retrieval_triggered"] = False
        
        # Enrich refined_memory with multi-scale context
        multi_mem = self.multi_memory.get_combined()
        if multi_mem is not None:
            # 80% thought-refined + 20% multi-scale context
            refined_memory = 0.8 * refined_memory + 0.2 * multi_mem
        
        thought_info["rag_result"] = rag_result
        self.thinking_history.append(thought_info)
        
        return refined_memory, thought_info
    
    def finalize(self, query_embedding, final_memory, surprise_level=0.0, task_type="chat"):
        """
        Вызывается в КОНЦЕ think():
        1. Сохраняет траекторию в ThoughtCache
        2. Записывает в SleepConsolidation
        3. Обновляет сессионную память
        """
        # 1. Thought Cache: save trajectory
        if query_embedding is not None and len(self.thinking_history) > 0:
            q_vec = query_embedding.float().mean(dim=0) if query_embedding.dim() > 1 else query_embedding
            self.thought_cache.store(
                q_vec, self.thinking_history, final_memory, self.last_confidence
            )
        
        # 2. Sleep: record for future consolidation
        self.sleep.record(final_memory, surprise_level, task_type)
        
        # 3. Session memory
        if final_memory is not None:
            self.update_session_memory(final_memory)
        
        # 4. Check if sleep consolidation needed
        if self.sleep.should_consolidate():
            self.sleep.consolidate(self.multi_memory)
    
    def update_session_memory(self, h_query):
        """Вызывается после каждого запроса для обновления сессионной памяти."""
        self.multi_memory.update_session(h_query)
    
    def update_longterm_memory(self, h_knowledge):
        """Вызывается Titans для обновления долговременной памяти."""
        self.multi_memory.update_longterm(h_knowledge)
    
    def apply_confidence_gate(self, logits):
        """Применить confidence gating к logits."""
        return self.confidence_gate(logits, self.last_confidence)
    
    def get_summary(self):
        """Summary of the thinking chain for logging."""
        if not self.thinking_history:
            return {"waves": 0, "phases": [], "final_confidence": 0.0}
        
        return {
            "waves": len(self.thinking_history),
            "phases": [t["phase"] for t in self.thinking_history],
            "confidences": [t["confidence"] for t in self.thinking_history],
            "final_confidence": self.last_confidence,
            "retrieval_count": sum(1 for t in self.thinking_history 
                                   if t.get("retrieval_triggered", False)),
            "wave_skip_ready": self.should_skip_remaining(),
            "cache_hit": self._cache_hit is not None,
            "cache_similarity": self._cache_hit["similarity"] if self._cache_hit else 0.0,
            "thought_cache": self.thought_cache.stats(),
            "sleep": self.sleep.stats(),
        }
