# memory/__init__.py
"""
TARS v3 Memory System.

Модули:
  - LEANN:       Семантическое хранилище (384d MiniLM эмбеддинги, графовый индекс)
  - Memo:        LRU кэш для быстрого повторного поиска (O(1) vs O(N))
  - Titans:      Surprise-Based Learning (нейронная LTM, 384d)
  - Store:       TarsMemoryHub — единый интерфейс (LEANN + Memo + Titans)
  - Matryoshka:  2-stage retrieval (64d coarse → 384d fine)
  - Gaussian:    N(μ,σ²) uncertainty embeddings
  - DocToLoRA:   Document → LoRA adapter for MoLE experts
  - Context:     Sliding window + auto-summary

Поток данных:
  Запрос → Memo → LEANN/Matryoshka → Результат
  Новый факт → LEANN → Titans (surprise?) → JSON log
  Документ → DocToLoRA → MoLE expert
"""
from memory.leann import LeannIndex, TarsMemory
from memory.memo import MemoCache
from memory.titans import TitansMemory
from memory.store import TarsMemoryHub, TarsStorage
from memory.matryoshka import MatryoshkaRetriever
from memory.gaussian_embed import GaussianEmbedding, GaussianVector
from memory.doc_to_lora import DocToLoRA, DocLoRAInfo
from memory.context_manager import ContextManager

__all__ = [
    "LeannIndex", "TarsMemory",
    "MemoCache",
    "TitansMemory",
    "TarsMemoryHub", "TarsStorage",
    "MatryoshkaRetriever",
    "GaussianEmbedding", "GaussianVector",
    "DocToLoRA", "DocLoRAInfo",
    "ContextManager",
]
