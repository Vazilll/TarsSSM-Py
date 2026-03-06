"""
═══════════════════════════════════════════════════════════════
  QueryRouter — Neural Memory Retrieval Router
═══════════════════════════════════════════════════════════════


Определяет:
  1. Нужен ли RAG вообще (greeting/math → skip)
  2. Какой источник памяти использовать (LEANN / Titans / Web)
  3. Уточняет поисковый запрос (query_refiner)

Обучается через backprop вместе с моделью.
Решает проблему: "как система понимает, что ей искать".

Использование:
  router = QueryRouter(d_model=768, mem_dim=384)
  decision = router(query_embedding)
  # decision["needs_rag"] → bool
  # decision["source_weights"] → [leann, titans, web, history]
  # decision["refined_query"] → refined embedding for search
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, List, Set

from brain.mamba2.bitnet import UniversalLinear, RMSNorm


class QueryRouter(nn.Module):
    """
    Neural Query Router — определяет стратегию поиска памяти.
    
    Выходы:
      needs_rag: bool — нужен ли поиск вообще
      source_weights: [4] — веса источников [leann, titans, web, history]
      urgency: float — насколько срочно нужна память (0=можно подождать, 1=блокирующе)
      refined_query: [d_model] — уточнённый вектор для поиска
    
    Архитектура:
      query → RMSNorm → MLP(d_model → d_model//4 → 6) → sigmoid
                              ↓
                    query_refiner(d_model → mem_dim)
    """
    
    N_SOURCES = 4  # leann, titans, web, history
    
    def __init__(self, d_model: int = 768, mem_dim: int = 384, quant_mode: str = "fp16"):
        super().__init__()
        self.d_model = d_model
        self.mem_dim = mem_dim
        
        # ═══ Нормализация входа ═══
        self.norm = RMSNorm(d_model)
        
        # ═══ Классификатор: needs_rag (1) + source_weights (4) + urgency (1) = 6 ═══
        hidden = max(64, d_model // 4)
        self.classifier = nn.Sequential(
            UniversalLinear(d_model, hidden, bias=True, mode=quant_mode),
            nn.SiLU(),
            UniversalLinear(hidden, 6, bias=True, mode=quant_mode),
        )
        
        # ═══ Query Refiner: уточняет embedding для поиска ═══
        # Может отличаться от исходного query: "что было вчера" → дата-специфичный вектор
        self.query_refiner = nn.Sequential(
            UniversalLinear(d_model, hidden, bias=True, mode=quant_mode),
            nn.SiLU(),
            UniversalLinear(hidden, mem_dim, bias=True, mode=quant_mode),
        )
        
        # ═══ Порог needs_rag (обучаемый) ═══
        self.rag_threshold = nn.Parameter(torch.tensor(0.5))
        
        # ═══ Статистика (для логирования и отладки) ═══
        self._last_decision = {}
    
    def forward(self, query_embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            query_embedding: [B, d_model] — embedding запроса
        
        Returns:
            dict с ключами:
              needs_rag: [B] bool tensor
              source_weights: [B, 4] — нормализованные веса источников
              urgency: [B] float 0..1
              refined_query: [B, mem_dim] — уточнённый вектор для поиска
        """
        x = self.norm(query_embedding)  # [B, d_model]
        
        # Классификация
        logits = self.classifier(x)     # [B, 6]
        
        # Разделяем выходы
        needs_rag_logit = logits[:, 0]         # [B]
        source_logits = logits[:, 1:5]         # [B, 4]
        urgency_logit = logits[:, 5]           # [B]
        
        # Активации
        needs_rag_score = torch.sigmoid(needs_rag_logit)  # [B]
        needs_rag = needs_rag_score > self.rag_threshold   # [B] bool
        source_weights = F.softmax(source_logits, dim=-1)  # [B, 4]
        urgency = torch.sigmoid(urgency_logit)             # [B]
        
        # Уточнённый запрос
        refined_query = self.query_refiner(x)  # [B, mem_dim]
        
        self._last_decision = {
            "needs_rag_score": needs_rag_score.detach(),
            "source_names": ["leann", "titans", "web", "history"],
        }
        
        return {
            "needs_rag": needs_rag,
            "needs_rag_score": needs_rag_score,
            "source_weights": source_weights,
            "urgency": urgency,
            "refined_query": refined_query,
        }
    
    def get_primary_source(self, source_weights: torch.Tensor) -> str:
        """Возвращает имя источника с наибольшим весом."""
        names = ["leann", "titans", "web", "history"]
        idx = source_weights.argmax(dim=-1).item()
        return names[idx]


class ProgressiveMemoryManager:
    """
    Progressive Memory Injection — менеджер асинхронной подачи памяти.
    
    Решает проблему: блоки не ждут RAG, но когда память приходит —
    она инжектится в ближайшую волну + сбрасывается аудитор сходимости.
    
    Использование (в think()):
        pmm = ProgressiveMemoryManager()
        pmm.start_retrieval(query_embedding, router_decision)
        
        for wave_idx in range(max_waves):
            # Проверяем готовность RAG (non-blocking)
            pmm.check_and_inject(wave_idx)
            memory_vec = pmm.get_memory_vec()
            rag_state = pmm.get_rag_state()
            
            # ... блоки работают ...
            pmm.update_spine(h_curr)
    """
    
    def __init__(self):
        self.memory_vec = None
        self.rag_state = None
        self._rag_ready = False
        self._rag_future = None
        self._injection_wave = -1  # волна, на которой был инжект
        self._needs_auditor_reset = False
        
        # ═══ RAG Delivery Tracking ═══
        self._sources_requested: List[str] = []   # какие источники запрошены
        self._sources_delivered: Set[str] = set()  # какие реально доставили
        self._request_time: float = 0.0            # время запроса
        self._request_urgency: float = 0.0         # срочность
        self._logger = logging.getLogger("Tars.PMM")
    
    def register_request(self, source_names: List[str], urgency: float = 0.0):
        """Зарегистрировать запрос RAG-источников."""
        self._sources_requested = list(source_names)
        self._request_urgency = urgency
        self._request_time = time.time()
        self._logger.debug(
            f"PMM: RAG request registered: {source_names}, urgency={urgency:.2f}"
        )
    
    def mark_delivered(self, source_name: str):
        """Отметить источник как доставленный."""
        self._sources_delivered.add(source_name)
        self._logger.debug(f"PMM: source '{source_name}' delivered")
    
    def get_delivery_report(self) -> Dict:
        """
        Отчёт: requested vs delivered.
        
        Returns:
            {
                "requested": [...],
                "delivered": [...],
                "missing": [...],
                "all_loaded": bool,
                "latency": float,
                "urgency": float,
            }
        """
        missing = [s for s in self._sources_requested if s not in self._sources_delivered]
        latency = time.time() - self._request_time if self._request_time > 0 else 0.0
        return {
            "requested": list(self._sources_requested),
            "delivered": list(self._sources_delivered),
            "missing": missing,
            "all_loaded": len(missing) == 0 and len(self._sources_requested) > 0,
            "no_rag_needed": len(self._sources_requested) == 0,
            "latency": latency,
            "urgency": self._request_urgency,
        }
    
    def set_initial_memory(self, memory_vec=None, rag_state=None):
        """Установить начальную память (из reflex_ctx или предыдущего запроса)."""
        self.memory_vec = memory_vec
        self.rag_state = rag_state
        if memory_vec is not None:
            self._rag_ready = True
    
    def register_future(self, future):
        """Зарегистрировать Future от asyncio RAG retrieval."""
        self._rag_future = future
        self._rag_ready = False
    
    def check_ready(self) -> bool:
        """Non-blocking проверка: RAG данные готовы?"""
        if self._rag_ready:
            return True
        
        if self._rag_future is not None:
            # asyncio.Future или concurrent.futures.Future
            if hasattr(self._rag_future, 'done') and self._rag_future.done():
                try:
                    result = self._rag_future.result()
                    if isinstance(result, dict):
                        self.memory_vec = result.get("memory_vec", self.memory_vec)
                        self.rag_state = result.get("rag_state", self.rag_state)
                        # Auto-mark delivered sources from result
                        for src in result.get("sources", []):
                            self.mark_delivered(src)
                    elif isinstance(result, tuple) and len(result) == 2:
                        self.memory_vec, self.rag_state = result
                    else:
                        self.memory_vec = result
                    self._rag_ready = True
                    self._needs_auditor_reset = True
                    # If memory arrived, mark primary source as delivered
                    if self.memory_vec is not None and self._sources_requested:
                        self.mark_delivered(self._sources_requested[0])
                except Exception:
                    self._rag_ready = False
                    self._rag_future = None
        
        return self._rag_ready
    
    def check_and_inject(self, wave_idx: int) -> bool:
        """
        Проверить и инжектировать RAG если готов.
        
        Returns: True если инжект произошёл на этой волне (нужен auditor reset)
        """
        if self._needs_auditor_reset and self.check_ready():
            self._injection_wave = wave_idx
            self._needs_auditor_reset = False
            return True
        return False
    
    def needs_auditor_reset(self) -> bool:
        """Вернуть True если нужно сбросить Integral Auditor."""
        return self._needs_auditor_reset
    
    def get_memory_vec(self):
        return self.memory_vec
    
    def get_rag_state(self):
        return self.rag_state
    
    def update_spine(self, h_curr, to_memory_space_fn, alpha=0.3):
        """Обновить memory_vec из текущего hidden state (spine update)."""
        if to_memory_space_fn is None:
            return
        try:
            h_for_mem = to_memory_space_fn(h_curr)
            if self.memory_vec is None:
                self.memory_vec = h_for_mem.detach()
            else:
                self.memory_vec = (1 - alpha) * self.memory_vec + alpha * h_for_mem.detach()
        except Exception:
            pass
    
    @property
    def injection_wave(self) -> int:
        """Волна, на которой произошёл RAG injection (-1 если не было)."""
        return self._injection_wave
