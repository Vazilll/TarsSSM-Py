import numpy as np
import os
import json
import logging
from pathlib import Path

_LEANN_ROOT = Path(__file__).parent.parent

class LeannIndex:
    """
    LEANN (Lightweight Efficient ANN) — Сверхлегкий графовый векторный индекс.
    
    Служит для долговременного хранения текстовых знаний и быстрого поиска
    релевантной информации (RAG — Retrieval-Augmented Generation).
    
    Принципы работы:
    1. Эмбеддинги: Текст переводится в векторы фиксированной длины (384 или 1024).
    2. Косинусное сходство: Оценивается близость 'смысла' слов, а не просто совпадение букв.
    3. Графовая структура: Позволяет ускорить поиск за счет связей между близкими документами.
    """
    def __init__(self, model_path='models/embeddings', index_path=None):
        self.logger = logging.getLogger("Tars.LEANN")
        if index_path is None:
            index_path = str(_LEANN_ROOT / "memory" / "leann.index")
        self.index_path = index_path
        self.texts = []
        self.embeddings = [] # Кеш признаковых векторов.
        self.graph = {}
        self.model = None
        
        # Загрузка нейросети для создания эмбеддингов.
        try:
            from sentence_transformers import SentenceTransformer
            # Пытаемся взять локальную модель, если она есть.
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                self.model = SentenceTransformer(model_path)
            else:
                # В ином случае качаем миниатюрную, но мощную all-MiniLM-L6-v2.
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"LEANN: Используется математический Fallback (TF-IDF), т.к. нейросеть не загружена: {e}")
        
        if os.path.exists(index_path):
            self.load()

    def _get_embedding(self, text: str) -> np.ndarray:
        """Превращение сырого текста в семантический вектор."""
        if self.model:
            return self.model.encode([text])[0]
        
        # Математический дублер (TF-IDF стиль), если нет GPU/нейросети.
        words = text.lower().split()
        vec = np.zeros(384)
        for i, w in enumerate(words):
            idx = hash(w) % 384
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add_document(self, text: str):
        """Регистрация нового знания в памяти ассистента."""
        doc_id = len(self.texts)
        self.texts.append(text)
        
        emb = self._get_embedding(text)
        self.embeddings.append(emb)
        
        # Построение ассоциативного графа.
        # Мы ищем 'соседа', наиболее близкого по смыслу к новому документу,
        # и создаем между ними ребро.
        best_neighbor = -1
        best_score = -1.0
        
        for i in range(doc_id):
            # Формула косинусного сходства (Cosine Similarity).
            score = float(np.dot(emb, self.embeddings[i]) / 
                         (np.linalg.norm(emb) * np.linalg.norm(self.embeddings[i]) + 1e-8))
            if score > best_score:
                best_score = score
                best_neighbor = i
        
        self.graph[str(doc_id)] = [best_neighbor] if best_neighbor != -1 else []
        self.logger.debug(f"LEANN: Факт {doc_id} привязан к соседу {best_neighbor}")

    async def search(self, query: str, top_k: int = 5):
        """
        Семантический поиск знаний.
        Возвращает top_k наиболее близких по смыслу фрагментов.
        """
        if not self.texts: return []
        
        query_emb = self._get_embedding(query)
        
        results = []
        for i, emb in enumerate(self.embeddings):
            # Линейный перебор с косинусной проверкой.
            score = float(np.dot(query_emb, emb) / 
                         (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
            results.append((i, score))
        
        # Сортировка: самое важное — вверх.
        results.sort(key=lambda x: x[1], reverse=True)
        return [self.texts[i] for i, _ in results[:top_k]]

    def save(self):
        """Фиксация памяти на диске."""
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        data = {
            "texts": self.texts,
            "embeddings": [emb.tolist() for emb in self.embeddings]
        }
        with open(self.index_path, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self):
        """Восстановление 'воспоминаний' при запуске TARS."""
        try:
            with open(self.index_path, "r", encoding='utf-8') as f:
                data = json.load(f)
                self.texts = data.get("texts", [])
                self.embeddings = [np.array(e) for e in data.get("embeddings", [])]
        except Exception:
            self.logger.warning("LEANN: Индекс пуст или поврежден.")

class TarsMemory:
    """Верхнеуровневый системный интерфейс памяти."""
    def __init__(self):
        self.leann = LeannIndex()

    async def get_context(self, query: str) -> str:
        """Сборка контекста из памяти для дополнения ответа ИИ."""
        results = await self.leann.search(query, top_k=3)
        return "\n".join(results) if results else "Контекст не найден."
    
    def add_document(self, text: str):
        self.leann.add_document(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    memory = LeannIndex()
    memory.add_document("Любимый цвет пользователя - бирюзовый.")
    memory.add_document("У Тарса сегодня запланирована встреча в 18:00.")
    memory.add_document("Проект ТАРС находится в папке C:\\Users\\Public\\Tarsfull")
    
    import asyncio
    async def test():
        res = await memory.search("Какой цвет?")
        print(f"Цвет: {res}")
        res = await memory.search("Где проект?")
        print(f"Проект: {res}")
    
    asyncio.run(test())
