"""
memo.py — Быстрый LRU-кэш для семантического поиска TARS.

Работает между TarsBlock (мозг) и LEANN (главная память).
Избавляет от O(N) перебора всех эмбеддингов при повторных запросах.

Архитектура:
  Запрос → hash(embedding[:16]) → LRU Cache
    ├─ HIT  → мгновенный ответ (0ms)
    └─ MISS → LEANN.search() → кэш → ответ (50-200ms)
    
Размер: 256 записей × top_k=5 результатов.
"""
import numpy as np
import logging
import time
from collections import OrderedDict
from typing import List, Optional, Tuple


logger = logging.getLogger("Tars.Memo")


class MemoCache:
    """
    LRU Semantic Cache для LEANN.
    
    Принцип: если два запроса имеют похожий эмбеддинг → вернуть кэшированный результат.
    Кэш-ключ: квантизированный hash первых 16 компонент embedding вектора.
    
    Атрибуты:
      capacity: макс. записей в кэше (default=256)
      similarity_threshold: порог похожести для кэш-хита (default=0.92)
    """
    
    def __init__(self, capacity: int = 256, similarity_threshold: float = 0.92):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        
        # LRU: OrderedDict(key → (embedding, results, timestamp))
        self._cache: OrderedDict = OrderedDict()
        
        # Статистика
        self.hits = 0
        self.misses = 0
        self.total_saved_ms = 0.0
    
    def _make_key(self, embedding: np.ndarray) -> str:
        """
        Квантизированный ключ из первых 16 компонент.
        Группирует похожие вектора в одну корзину.
        """
        # Берём 16 ведущих компонент, квантизуем до 4 бит (0-15)
        head = embedding[:16]
        quantized = np.clip((head + 1) * 8, 0, 15).astype(np.int8)
        return quantized.tobytes().hex()
    
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Косинусное сходство между двумя векторами."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def get(self, query_embedding: np.ndarray) -> Optional[List[str]]:
        """
        Поиск в кэше.
        
        Args:
            query_embedding: [384] — вектор запроса
        Returns:
            list[str] — кэшированные результаты, или None (промах)
        """
        key = self._make_key(query_embedding)
        
        if key in self._cache:
            cached_emb, cached_results, cached_time = self._cache[key]
            
            # Проверяем точное сходство (квантизация может дать false positives)
            sim = self._cosine_sim(query_embedding, cached_emb)
            
            if sim >= self.similarity_threshold:
                # HIT! Двигаем в конец (LRU)
                self._cache.move_to_end(key)
                self.hits += 1
                return cached_results
        
        self.misses += 1
        return None
    
    def put(self, query_embedding: np.ndarray, results: List[str], search_time_ms: float = 0.0):
        """
        Сохранить результат поиска в кэш.
        
        Args:
            query_embedding: [384] — вектор запроса
            results: list[str] — результаты из LEANN.search()
            search_time_ms: время поиска (для статистики)
        """
        key = self._make_key(query_embedding)
        
        # Если ключ уже есть — обновляем
        if key in self._cache:
            self._cache.move_to_end(key)
        
        self._cache[key] = (query_embedding.copy(), results, time.time())
        self.total_saved_ms += search_time_ms
        
        # LRU eviction
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
    
    def invalidate(self):
        """Очистить весь кэш (при добавлении новых документов в LEANN)."""
        self._cache.clear()
        logger.debug("Memo: Кэш очищен")
    
    def get_stats(self) -> dict:
        """Статистика кэша."""
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "capacity": self.capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / max(1, total),
            "total_saved_ms": self.total_saved_ms,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    memo = MemoCache(capacity=128)
    
    # Тест: 3 запроса, 2 одинаковых
    emb1 = np.random.randn(384).astype(np.float32)
    emb2 = emb1 + np.random.randn(384).astype(np.float32) * 0.01  # Почти такой же
    emb3 = np.random.randn(384).astype(np.float32)  # Совершенно другой
    
    # Промах
    assert memo.get(emb1) is None
    print("1. Cache miss (first query)")
    
    # Сохраняем
    memo.put(emb1, ["Результат 1", "Результат 2"], search_time_ms=50.0)
    
    # Хит (похожий вектор)
    result = memo.get(emb2)
    print(f"2. Cache {'hit' if result else 'miss'} (similar query): {result}")
    
    # Промах (другой вектор)
    result = memo.get(emb3)
    print(f"3. Cache {'hit' if result else 'miss'} (different query)")
    
    print(f"\nStats: {memo.get_stats()}")
