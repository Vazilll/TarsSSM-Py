import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import os
import json

class LeannIndex:
    """
    LEANN (Lightweight Efficient ANN) - Сверхлегкий векторный индекс.
    Ключевая особенность: Выборочный пересчет эмбеддингов (Selective Recompute).
    Вместо того чтобы хранить миллиарды векторов, мы храним графовую структуру и восстанавливаем
    нужные вектора только в момент поиска, что экономит до 97% места.
    """
    def __init__(self, model_path='models/embeddings', index_path="memory/leann.index"):
        # Используем SentenceTransformer с улучшенным механизмом отката
        try:
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "tokenizer.json")):
                self.model = SentenceTransformer(model_path)
                print(f"LEANN: Инициализирован из локальной папки {model_path}")
            else:
                raise ImportError("Local model folder incomplete")
        except Exception as e:
            print(f"LEANN: Локальная загрузка не удалась ({e}), использую стандартный 'all-MiniLM-L6-v2'")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        self.index_path = index_path
        self.texts = []
        self.graph = {}
        if os.path.exists(index_path):
            self.load()

    def add_document(self, text: str):
        """
        Добавляет документ в индекс.
        """
        doc_id = len(self.texts)
        self.texts.append(text)
        
        # Получаем эмбеддинг для вычисления связей в графе (HGF прунинг)
        emb = self.model.encode([text])[0]
        
        # В этом прототипе просто инициализируем список соседей
        self.graph[doc_id] = [] 
        
        print(f"LEANN: Документ {doc_id} проиндексирован.")

    async def search(self, query: str, top_k: int = 5):
        """
        Поиск похожих документов.
        В LEANN мы не храним все вектора в RAM, а вычисляем их для кандидатов из графа.
        """
        query_emb = self.model.encode([query])[0]
        
        # Упрощенный поиск по всей базе (в полноценной версии используется обход графа)
        results = []
        for i, text in enumerate(self.texts):
            # Selective Recompute: пересчитываем эмбеддинг 'на лету'
            emb = self.model.encode([text])[0]
            # Косинусное сходство
            score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            results.append((i, score))
            
        # Сортировка по релевантности
        results.sort(key=lambda x: x[1], reverse=True)
        
        return [self.texts[i] for i, score in results[:top_k]]

    def save(self):
        """ Сохранение текстов и структуры графа на диск. """
        with open(self.index_path, "w", encoding='utf-8') as f:
            json.dump({"texts": self.texts, "graph": self.graph}, f)

    def load(self):
        """ Загрузка индекса с диска. """
        with open(self.index_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            self.texts = data["texts"]
            self.graph = data["graph"]

class TarsMemory:
    """ Интерфейс управления памятью Tars. """
    def __init__(self):
        self.leann = LeannIndex()
        # Позже здесь будет интеграция с Mem0 для долгосрочного хранения

    async def get_context(self, query: str) -> str:
        """ Получение контекста (наиболее релевантных фактов) для запроса. """
        results = await self.leann.search(query, top_k=3)
        return "\n".join(results)

if __name__ == "__main__":
    memory = LeannIndex()
    memory.add_document("Любимый цвет пользователя - бирюзовый.")
    memory.add_document("У Тарса сегодня запланирована встреча в 18:00.")
    
    import asyncio
    async def test():
        res = await memory.search("Какой цвет?")
        print(f"Результат поиска LEANN: {res}")
    
    asyncio.run(test())
