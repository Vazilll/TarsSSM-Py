"""
store.py — Единый интерфейс памяти TARS v3.

Объединяет все подсистемы памяти:
  - LEANN: семантический поиск (384d MiniLM эмбеддинги)
  - Memo: LRU кэш (мгновенный повторный поиск)
  - Titans: surprise-based LTM (запоминание нового)
  - Storage: JSON-лог фактов с временными метками

Архитектура потока данных:
  запрос → Memo.get()
    ├─ HIT  → результат (0ms)
    └─ MISS → LEANN.search() → Memo.put() → результат
  
  новый факт → LEANN.add_document() → Titans.update()
    └─ surprise? → Fast Weight Update (3 шага)
"""
import logging
import os
import json
import time
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger("Tars.Memory")


class TarsMemoryHub:
    """
    Центральный хаб памяти ТАРС.
    
    Связывает LEANN (хранилище) + Memo (кэш) + Titans (обучение).
    Единственная точка входа для всех операций с памятью.
    """
    
    def __init__(self, storage_path="data/tars_memories.json"):
        self.storage_path = storage_path
        self._fact_log = []
        
        # Ленивая инициализация компонентов
        self._leann = None
        self._memo = None
        self._titans = None
        
        self._load_facts()
    
    @property
    def leann(self):
        if self._leann is None:
            from memory.leann import LeannIndex
            self._leann = LeannIndex()
        return self._leann
    
    @property
    def memo(self):
        if self._memo is None:
            from memory.memo import MemoCache
            self._memo = MemoCache(capacity=256)
        return self._memo
    
    @property
    def titans(self):
        if self._titans is None:
            from memory.titans import TitansMemory
            self._titans = TitansMemory(dim=384, brain_dim=768)
        return self._titans
    
    def _load_facts(self):
        """Загрузка лога фактов."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self._fact_log = json.load(f)
                logger.info(f"Memory Hub: {len(self._fact_log)} фактов загружено")
            except Exception:
                self._fact_log = []
    
    def _save_facts(self):
        """Сохранение лога фактов."""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._fact_log, f, ensure_ascii=False, indent=2)
    
    async def remember(self, text: str, user_id: str = "tars_user") -> dict:
        """
        Запомнить новый факт.
        
        1. Добавляет в LEANN (семантический индекс)
        2. Инвалидирует Memo кэш
        3. Отправляет эмбеддинг в Titans для surprise-check
        4. Логирует в JSON с временной меткой
        
        Returns:
            dict: surprised (bool), fact_id (int)
        """
        # 1. LEANN
        self.leann.add_document(text)
        
        # 2. Memo — инвалидируем (новый документ может изменить результаты)
        self.memo.invalidate()
        
        # 3. Titans — проверяем, новый ли это факт
        import numpy as np
        emb = self.leann._get_embedding(text)
        import torch
        emb_tensor = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
        titans_result = await self.titans.update(emb_tensor)
        
        # 4. JSON лог
        fact_id = len(self._fact_log)
        self._fact_log.append({
            "id": fact_id,
            "text": text,
            "user": user_id,
            "time": datetime.now().isoformat(),
            "surprised": titans_result.get("surprised", False),
        })
        
        # Ограничение размера (последние 1000)
        if len(self._fact_log) > 1000:
            self._fact_log = self._fact_log[-1000:]
        self._save_facts()
        
        logger.debug(f"Memory: '{text[:50]}...' (surprise={titans_result.get('surprised', False)})")
        
        return {"fact_id": fact_id, **titans_result}
    
    async def recall(self, query: str, top_k: int = 5) -> List[str]:
        """
        Вспомнить релевантные факты.
        
        1. Проверяем Memo кэш (мгновенно)
        2. Если промах → LEANN.search() → кэшируем
        
        Returns:
            list[str] — релевантные факты
        """
        import numpy as np
        
        # 1. Memo check
        query_emb = self.leann._get_embedding(query)
        cached = self.memo.get(query_emb)
        if cached is not None:
            return cached
        
        # 2. LEANN search
        t0 = time.time()
        results = await self.leann.search(query, top_k=top_k)
        search_ms = (time.time() - t0) * 1000
        
        # 3. Кэшируем
        self.memo.put(query_emb, results, search_time_ms=search_ms)
        
        return results
    
    def get_stats(self) -> dict:
        """Статистика всех подсистем памяти."""
        stats = {
            "facts": len(self._fact_log),
            "leann_docs": len(self.leann.texts) if self._leann else 0,
        }
        if self._memo:
            stats["memo"] = self.memo.get_stats()
        if self._titans:
            stats["titans"] = self.titans.get_stats()
        return stats


# ═══ Backward compatibility ═══
class TarsStorage:
    """Legacy wrapper → TarsMemoryHub."""
    def __init__(self, storage_path="data/tars_memories.json"):
        self._hub = TarsMemoryHub(storage_path)
    
    async def remember(self, text, user_id="tars_user"):
        return await self._hub.remember(text, user_id)
    
    async def retrieve_memories(self, query, user_id="tars_user", top_k=5):
        return await self._hub.recall(query, top_k)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hub = TarsMemoryHub()
    
    import asyncio
    async def test():
        await hub.remember("Пользователь любит кофе без сахара.")
        await hub.remember("Проект ТАРС в C:\\Users\\Public\\Tarsfull")
        
        result = await hub.recall("Что любит пользователь?")
        print(f"Recall: {result}")
        
        # Повторный запрос — из кэша
        result2 = await hub.recall("Что любит пользователь?")
        print(f"Cached: {result2}")
        
        print(f"\nStats: {hub.get_stats()}")
    
    asyncio.run(test())
