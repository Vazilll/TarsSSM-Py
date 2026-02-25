"""
store.py — TARS v3 Локальное хранилище памяти.

Полностью локальное. Никаких внешних API.
Хранит факты и персоналию в JSON на диске.
"""
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger("Tars.Storage")


class TarsStorage:
    """
    Локальное хранилище фактов TARS.
    Файловая система: data/tars_memories.json
    """

    def __init__(self, storage_path="data/tars_memories.json"):
        self.storage_path = storage_path
        self.memories = []
        self._load()

    def _load(self):
        """Загрузка из JSON."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.memories = json.load(f)
                logger.info(f"Storage: {len(self.memories)} записей загружено")
            except Exception:
                self.memories = []

    def _save(self):
        """Сохранение в JSON."""
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    async def remember(self, text: str, user_id: str = "tars_user"):
        """Запомнить факт."""
        entry = {
            "text": text,
            "user": user_id,
            "time": datetime.now().isoformat()
        }
        self.memories.append(entry)
        # Ограничение размера (последние 1000 записей)
        if len(self.memories) > 1000:
            self.memories = self.memories[-1000:]
        self._save()
        logger.debug(f"Storage: '{text[:50]}...'")
        return True

    async def retrieve_memories(self, query: str, user_id: str = "tars_user", top_k: int = 5):
        """Поиск по ключевым словам."""
        query_words = set(query.lower().split())
        scored = []
        for entry in self.memories:
            text = entry.get("text", "")
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                scored.append((overlap, text))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    storage = TarsStorage()

    import asyncio
    async def test():
        await storage.remember("Пользователь любит кофе без сахара.")
        await storage.remember("Проект ТАРС в C:\\Users\\Public\\Tarsfull")
        mems = await storage.retrieve_memories("Что любит пользователь?")
        print(f"Результат: {mems}")

    asyncio.run(test())
