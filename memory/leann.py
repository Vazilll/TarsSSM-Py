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
    
    Хранение: numpy binary (float16) + JSON тексты.
    RAM: ~750 MB на 1M документов (вместо 8.9 GB в JSON float64).
    """
    def __init__(self, model_path='models/embeddings', index_path=None):
        self.logger = logging.getLogger("Tars.LEANN")
        if index_path is None:
            # На Colab → сохраняем прямо на Drive (symlink для npz не работает)
            drive_idx = _LEANN_ROOT / ".." / "drive" / "MyDrive" / "TarsMemory" / "leann.index"
            if drive_idx.parent.exists():
                index_path = str(drive_idx.resolve())
            else:
                index_path = str(_LEANN_ROOT / "memory" / "leann.index")
        self.index_path = index_path
        self.texts = []
        self.embeddings = None  # numpy array (int8) или None
        self.emb_scales = None  # numpy array (float32) per-vector scales
        self.model = None
        
        # IVF (Inverted File Index) для O(√N) поиска
        self.ivf_centroids = None   # [K, 384] float32
        self.ivf_assignments = None # [N] int — кластер каждого документа
        self.ivf_n_clusters = 256
        self.ivf_n_probe = 8        # сколько кластеров проверять при поиске
        
        # Загрузка нейросети для создания эмбеддингов.
        try:
            from sentence_transformers import SentenceTransformer
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                self.model = SentenceTransformer(model_path)
            else:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"LEANN: Fallback TF-IDF: {e}")
        
        self.load()

    def _get_embedding(self, text: str):
        """Превращение сырого текста в семантический вектор. Returns (int8_vec, scale)."""
        if self.model:
            emb = self.model.encode([text])[0].astype(np.float32)
        else:
            # TF-IDF fallback
            words = text.lower().split()
            emb = np.zeros(384, dtype=np.float32)
            for w in words:
                idx = hash(w) % 384
                emb[idx] += 1.0
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
        
        # Квантизация в int8
        scale = np.abs(emb).max() / 127.0
        if scale < 1e-8:
            scale = 1e-8
        emb_q = np.clip(np.round(emb / scale), -128, 127).astype(np.int8)
        return emb_q, np.float32(scale)

    def _build_ivf(self):
        """Построить IVF индекс (k-means кластеризация)."""
        if self.embeddings is None or len(self.embeddings) < self.ivf_n_clusters * 2:
            self.ivf_centroids = None
            self.ivf_assignments = None
            return
        
        # Деквантизация для кластеризации
        embs_f32 = self.embeddings.astype(np.float32) * self.emb_scales[:, np.newaxis]
        norms = np.linalg.norm(embs_f32, axis=1, keepdims=True) + 1e-8
        embs_norm = (embs_f32 / norms).astype(np.float32)
        
        n = len(embs_norm)
        k = min(self.ivf_n_clusters, n // 4)
        
        # Простой k-means (10 итераций достаточно)
        rng = np.random.RandomState(42)
        idx = rng.choice(n, k, replace=False)
        centroids = embs_norm[idx].copy()
        
        for _ in range(10):
            dists = embs_norm @ centroids.T  # [N, K]
            assignments = dists.argmax(axis=1)
            for c in range(k):
                mask = assignments == c
                if mask.sum() > 0:
                    centroids[c] = embs_norm[mask].mean(axis=0)
                    norm_c = np.linalg.norm(centroids[c])
                    if norm_c > 0:
                        centroids[c] /= norm_c
        
        self.ivf_centroids = centroids.astype(np.float32)
        self.ivf_assignments = assignments.astype(np.int32)
        self.logger.info(f"LEANN: IVF индекс: {k} кластеров, {n} документов")

    def add_document(self, text: str):
        """Регистрация нового знания в памяти ассистента."""
        self.texts.append(text)
        emb_q, scale = self._get_embedding(text)
        
        if self.embeddings is not None and len(self.embeddings) > 0:
            self.embeddings = np.vstack([self.embeddings, emb_q.reshape(1, -1)])
            self.emb_scales = np.append(self.emb_scales, scale)
        else:
            self.embeddings = emb_q.reshape(1, -1)
            self.emb_scales = np.array([scale], dtype=np.float32)
        
        # IVF инвалидируется при добавлении
        self.ivf_centroids = None

    async def search(self, query: str, top_k: int = 5):
        """
        Семантический поиск знаний.
        IVF: O(√N) для больших индексов, brute-force для малых.
        """
        if not self.texts or self.embeddings is None:
            return []
        
        q_emb_q, q_scale = self._get_embedding(query)
        q_emb = q_emb_q.astype(np.float32) * q_scale
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        # Если IVF доступен и индекс большой — используем кластерный поиск
        if (self.ivf_centroids is not None and 
            self.ivf_assignments is not None and 
            len(self.embeddings) > 1000):
            
            # Найти ближайшие кластеры
            centroid_scores = self.ivf_centroids @ q_norm
            top_clusters = np.argsort(centroid_scores)[::-1][:self.ivf_n_probe]
            
            # Собрать кандидатов из этих кластеров
            candidate_mask = np.isin(self.ivf_assignments, top_clusters)
            candidate_indices = np.where(candidate_mask)[0]
            
            if len(candidate_indices) == 0:
                candidate_indices = np.arange(len(self.embeddings))
            
            # Поиск только среди кандидатов
            cand_embs = self.embeddings[candidate_indices].astype(np.float32)
            cand_scales = self.emb_scales[candidate_indices]
            cand_f32 = cand_embs * cand_scales[:, np.newaxis]
            cand_norms = np.linalg.norm(cand_f32, axis=1, keepdims=True) + 1e-8
            cand_norm = cand_f32 / cand_norms
            
            scores = cand_norm @ q_norm
            top_local = np.argsort(scores)[::-1][:top_k]
            top_indices = candidate_indices[top_local]
        else:
            # Brute-force для малых индексов
            embs_f32 = self.embeddings.astype(np.float32) * self.emb_scales[:, np.newaxis]
            norms = np.linalg.norm(embs_f32, axis=1, keepdims=True) + 1e-8
            embs_norm = embs_f32 / norms
            scores = embs_norm @ q_norm
            top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [self.texts[i] for i in top_indices]

    def save(self):
        """Фиксация памяти на диске (int8 + IVF)."""
        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        
        npz_path = self.index_path.replace(".index", ".npz")
        texts_path = self.index_path.replace(".index", ".texts.json")
        
        # Построить IVF перед сохранением
        if self.ivf_centroids is None and self.embeddings is not None:
            self._build_ivf()
        
        if self.embeddings is not None:
            save_dict = {
                'embeddings': self.embeddings,
                'scales': self.emb_scales,
            }
            if self.ivf_centroids is not None:
                save_dict['ivf_centroids'] = self.ivf_centroids
                save_dict['ivf_assignments'] = self.ivf_assignments
            np.savez_compressed(npz_path, **save_dict)
        
        with open(texts_path, 'w', encoding='utf-8') as f:
            json.dump(self.texts, f, ensure_ascii=False)

    def load(self):
        """Восстановление 'воспоминаний' при запуске TARS."""
        npz_path = self.index_path.replace(".index", ".npz")
        texts_path = self.index_path.replace(".index", ".texts.json")
        
        # Новый формат (npz + texts.json)
        if os.path.exists(npz_path) and os.path.exists(texts_path):
            try:
                data = np.load(npz_path)
                self.embeddings = data["embeddings"]
                # Поддержка int8 (scales) и float16 (старый формат)
                if "scales" in data:
                    self.emb_scales = data["scales"]
                else:
                    # Миграция float16 → int8
                    embs_f32 = self.embeddings.astype(np.float32)
                    scales = np.abs(embs_f32).max(axis=1) / 127.0
                    scales = np.maximum(scales, 1e-8)
                    self.embeddings = np.clip(np.round(embs_f32 / scales[:, np.newaxis]), -128, 127).astype(np.int8)
                    self.emb_scales = scales.astype(np.float32)
                    # Сохранить в новом формате
                    self.save()
                    self.logger.info(f"LEANN: Мигрировано float16 → int8")
                with open(texts_path, 'r', encoding='utf-8') as f:
                    self.texts = json.load(f)
                ram_mb = self.embeddings.nbytes / (1024 * 1024)
                self.logger.info(f"LEANN: Загружено {len(self.texts)} документов ({ram_mb:.0f} MB int8)")
                
                # Загрузить или построить IVF
                if 'ivf_centroids' in data and 'ivf_assignments' in data:
                    self.ivf_centroids = data['ivf_centroids']
                    self.ivf_assignments = data['ivf_assignments']
                    self.logger.info(f"LEANN: IVF загружен ({len(self.ivf_centroids)} кластеров)")
                else:
                    self._build_ivf()
                return
            except Exception as e:
                self.logger.warning(f"LEANN: Ошибка npz: {e}")
        
        # Старый формат (JSON)
        if os.path.exists(self.index_path):
            try:
                file_size = os.path.getsize(self.index_path)
                if file_size < 100:
                    return
                with open(self.index_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                self.texts = data.get("texts", [])
                raw = data.get("embeddings", [])
                if raw:
                    embs_f32 = np.array(raw, dtype=np.float32)
                    scales = np.abs(embs_f32).max(axis=1) / 127.0
                    scales = np.maximum(scales, 1e-8)
                    self.embeddings = np.clip(np.round(embs_f32 / scales[:, np.newaxis]), -128, 127).astype(np.int8)
                    self.emb_scales = scales.astype(np.float32)
                    self.save()
                    os.remove(self.index_path)
                    self.logger.info(f"LEANN: Мигрировано {len(self.texts)} документов JSON → int8")
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
