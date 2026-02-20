from mem0 import Memory
import logging
import asyncio

class TarsStorage:
    """
    Интеграция с Mem0 для управления персоналией и опытом пользователя.
    Дополняет LEANN, предоставляя API для сохранения фактов.
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger("Tars.Storage")
        # Инициализация Mem0 в полностью локальном режиме
        local_config = {
            "vector_store": {"provider": "chroma", "config": {"path": "data/mem0_db"}},
            "embedder": {"provider": "local", "config": {"model": "all-MiniLM-L6-v2"}},
            "llm": {"provider": "local", "config": {"model": "mamba2"}}
        }
        try:
            self.memory = Memory.from_config(local_config)
            self.logger.info("Storage: Mem0 успешно проинициализирован в локальном режиме.")
        except Exception as e:
            self.logger.error(f"Storage: Ошибка Mem0 - {e}")
            self.memory = None

    async def remember(self, text: str, user_id: str = "tars_user"):
        """ Сохранение факта в долгосрочную память Mem0. """
        if self.memory:
            self.memory.add(text, user_id=user_id)
            self.logger.info(f"Storage: Запомнено - '{text}'")
        return True

    async def retrieve_memories(self, query: str, user_id: str = "tars_user"):
        """ Поиск связанных воспоминаний. """
        if self.memory:
            results = self.memory.search(query, user_id=user_id)
            return [res['text'] for res in results]
        return []

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    storage = TarsStorage()
    async def test():
        await storage.remember("Пользователь любит кофе без сахара.")
        mems = await storage.retrieve_memories("Что любит пользователь?")
        print(f"Воспоминания: {mems}")
    asyncio.run(test())
