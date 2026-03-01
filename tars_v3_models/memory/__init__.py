# memory/__init__.py
"""
TARS v3 Memory System.

Модули:
  - LEANN:  Семантическое хранилище (384d MiniLM эмбеддинги, графовый индекс)
  - Memo:   LRU кэш для быстрого повторного поиска (O(1) vs O(N))
  - Titans: Surprise-Based Learning (нейронная LTM, 384d)
  - Store:  TarsMemoryHub — единый интерфейс (LEANN + Memo + Titans)

Поток данных:
  Запрос → Memo → LEANN → Результат
  Новый факт → LEANN → Titans (surprise?) → JSON log
"""
from memory.leann import LeannIndex, TarsMemory
from memory.memo import MemoCache
from memory.titans import TitansMemory
from memory.store import TarsMemoryHub, TarsStorage

__all__ = [
    "LeannIndex", "TarsMemory",
    "MemoCache",
    "TitansMemory",
    "TarsMemoryHub", "TarsStorage",
]
