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
        Запомнить новый факт (Total Memory — НИЧЕГО не удаляем).
        
        1. Добавляет в LEANN (семантический индекс)
        2. Инвалидирует Memo кэш
        3. Отправляет эмбеддинг в Titans для surprise-check
        4. Логирует в JSON + SQLite с временной меткой
        
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
        
        # 4. JSON лог (Total Memory — без жёсткого лимита)
        fact_id = len(self._fact_log)
        self._fact_log.append({
            "id": fact_id,
            "text": text,
            "user": user_id,
            "time": datetime.now().isoformat(),
            "surprised": titans_result.get("surprised", False),
        })
        
        # SQLite архив для бесконечного хранения
        self._archive_to_sqlite(self._fact_log[-1])
        
        # JSON сохраняем последние 50000 (горячий кэш)
        if len(self._fact_log) > 50000:
            self._fact_log = self._fact_log[-50000:]
        self._save_facts()
        
        logger.debug(f"Memory: '{text[:50]}...' (surprise={titans_result.get('surprised', False)})")
        
        return {"fact_id": fact_id, **titans_result}
    
    async def recall(self, query: str, top_k: int = 5) -> List[str]:
        """
        Вспомнить релевантные факты (Total Memory).
        
        1. Проверяем Memo кэш (мгновенно)
        2. Если промах → LEANN.search() → кэшируем
        3. Дополняем из SQLite архива (full-text search)
        
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
        
        # 3. SQLite fallback для старых фактов
        sqlite_results = self._recall_from_sqlite(query, top_k=3)
        for sr in sqlite_results:
            if sr not in results:
                results.append(sr)
        
        # 4. Кэшируем
        self.memo.put(query_emb, results, search_time_ms=search_ms)
        
        return results
    
    def _archive_to_sqlite(self, fact: dict):
        """
        SQLite архив для бесконечного хранения (Total Memory).
        Каждый факт сохраняется навсегда — ничего не удаляется.
        """
        import sqlite3
        db_path = self.storage_path.replace('.json', '.db')
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    user_id TEXT DEFAULT 'tars_user',
                    timestamp TEXT NOT NULL,
                    surprised INTEGER DEFAULT 0
                )
            """)
            conn.execute(
                "INSERT INTO facts (text, user_id, timestamp, surprised) VALUES (?, ?, ?, ?)",
                (fact["text"], fact.get("user", "tars_user"), 
                 fact.get("time", ""), int(fact.get("surprised", False)))
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"SQLite archive error: {e}")
    
    def _recall_from_sqlite(self, query: str, top_k: int = 3) -> List[str]:
        """
        Full-text поиск по SQLite архиву (для старых фактов за пределами JSON).
        """
        import sqlite3
        db_path = self.storage_path.replace('.json', '.db')
        if not os.path.exists(db_path):
            return []
        try:
            conn = sqlite3.connect(db_path)
            # Простой LIKE-поиск по ключевым словам
            keywords = query.split()[:3]  # первые 3 слова
            conditions = " OR ".join([f"text LIKE ?" for _ in keywords])
            params = [f"%{kw}%" for kw in keywords]
            cursor = conn.execute(
                f"SELECT text FROM facts WHERE {conditions} ORDER BY id DESC LIMIT ?",
                params + [top_k]
            )
            results = [row[0] for row in cursor.fetchall()]
            conn.close()
            return results
        except Exception as e:
            logger.warning(f"SQLite recall error: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Статистика всех подсистем памяти."""
        stats = {
            "facts_hot": len(self._fact_log),
            "leann_docs": len(self.leann.texts) if self._leann else 0,
        }
        # SQLite total count
        import sqlite3
        db_path = self.storage_path.replace('.json', '.db')
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
                conn.close()
                stats["facts_total_sqlite"] = count
            except Exception:
                pass
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
