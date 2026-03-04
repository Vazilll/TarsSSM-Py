import numpy as np
import os
import json
import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from memory.search_utils import (
    SearchResult, BM25Index,
    mmr_rerank, apply_temporal_decay, hybrid_merge,
)

try:
    from tools import expand_query
except ImportError:
    def expand_query(q, **_): return q

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
        self.timestamps = []    # Unix timestamp каждого документа
        self.embeddings = None  # numpy array (int8) или None
        self.emb_scales = None  # numpy array (float32) per-vector scales
        self.model = None
        
        # IVF (Inverted File Index) для O(√N) поиска
        self.ivf_centroids = None   # [K, 384] float32
        self.ivf_assignments = None # [N] int — кластер каждого документа
        self.ivf_n_clusters = 256
        self.ivf_n_probe = 8        # сколько кластеров проверять при поиске
        
        # BM25 индекс (строится лениво при первом поиске)
        self._bm25: Optional[BM25Index] = None
        self._bm25_dirty = True
        
        # ═══ CPU-OPT: LRU cache for query embeddings ═══
        # SentenceTransformer.encode() is ~50-100ms per call on CPU
        # Caching avoids re-encoding identical/repeated queries
        self._emb_cache: OrderedDict = OrderedDict()  # key → (int8_vec, scale)
        self._emb_cache_max = 256
        
        # Настройки поиска (OpenClaw-inspired)
        self.mmr_lambda = 0.7       # MMR: 0=diversity, 1=relevance
        self.decay_half_life = 30.0 # Temporal decay: дни
        self.vector_weight = 0.7    # Hybrid: вес vector search
        self.bm25_weight = 0.3      # Hybrid: вес BM25
        
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
        # ═══ CPU-OPT: check LRU cache first ═══
        cache_key = hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()
        if cache_key in self._emb_cache:
            self._emb_cache.move_to_end(cache_key)
            return self._emb_cache[cache_key]
        
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
        result = (emb_q, np.float32(scale))
        
        # ═══ Store in LRU cache ═══
        self._emb_cache[cache_key] = result
        if len(self._emb_cache) > self._emb_cache_max:
            self._emb_cache.popitem(last=False)  # evict oldest
        
        return result

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

    def add_document(self, text: str, timestamp: float = 0.0):
        """Регистрация нового знания в памяти ассистента."""
        self.texts.append(text)
        self.timestamps.append(timestamp or time.time())
        emb_q, scale = self._get_embedding(text)
        
        if self.embeddings is not None and len(self.embeddings) > 0:
            self.embeddings = np.vstack([self.embeddings, emb_q.reshape(1, -1)])
            self.emb_scales = np.append(self.emb_scales, scale)
        else:
            self.embeddings = emb_q.reshape(1, -1)
            self.emb_scales = np.array([scale], dtype=np.float32)
        
        # IVF и BM25 инвалидируются при добавлении
        self.ivf_centroids = None
        self._bm25_dirty = True

    def _ensure_bm25(self):
        """Построить BM25 индекс если нужно (лениво)."""
        if self._bm25_dirty and self.texts:
            self._bm25 = BM25Index()
            self._bm25.build(self.texts)
            self._bm25_dirty = False
    
    def _vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Чистый vector search (IVF или brute-force)."""
        if not self.texts or self.embeddings is None:
            return []
        
        q_emb_q, q_scale = self._get_embedding(query)
        q_emb = q_emb_q.astype(np.float32) * q_scale
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-8)
        
        # Если IVF доступен и индекс большой — кластерный поиск
        if (self.ivf_centroids is not None and 
            self.ivf_assignments is not None and 
            len(self.embeddings) > 1000):
            
            centroid_scores = self.ivf_centroids @ q_norm
            top_clusters = np.argsort(centroid_scores)[::-1][:self.ivf_n_probe]
            candidate_mask = np.isin(self.ivf_assignments, top_clusters)
            candidate_indices = np.where(candidate_mask)[0]
            
            if len(candidate_indices) == 0:
                candidate_indices = np.arange(len(self.embeddings))
            
            cand_embs = self.embeddings[candidate_indices].astype(np.float32)
            cand_scales = self.emb_scales[candidate_indices]
            cand_f32 = cand_embs * cand_scales[:, np.newaxis]
            cand_norms = np.linalg.norm(cand_f32, axis=1, keepdims=True) + 1e-8
            cand_norm = cand_f32 / cand_norms
            
            scores = cand_norm @ q_norm
            top_local = np.argsort(scores)[::-1][:top_k]
            top_indices = candidate_indices[top_local]
            top_scores = scores[top_local]
        else:
            # Brute-force для малых индексов
            embs_f32 = self.embeddings.astype(np.float32) * self.emb_scales[:, np.newaxis]
            norms = np.linalg.norm(embs_f32, axis=1, keepdims=True) + 1e-8
            embs_norm = embs_f32 / norms
            all_scores = embs_norm @ q_norm
            top_indices = np.argsort(all_scores)[::-1][:top_k]
            top_scores = all_scores[top_indices]
        
        results = []
        for i, idx in enumerate(top_indices):
            ts = self.timestamps[idx] if idx < len(self.timestamps) else 0.0
            results.append(SearchResult(
                index=int(idx),
                text=self.texts[idx],
                score=float(top_scores[i]),
                vector_score=float(top_scores[i]),
                timestamp=ts,
            ))
        return results
    
    async def search(self, query: str, top_k: int = 5, 
                     use_hybrid: bool = True,
                     use_mmr: bool = True,
                     use_decay: bool = True) -> list:
        """
        Гибридный поиск знаний.
        
        Алгоритмы (портированы из OpenClaw):
          1. Vector Search (IVF/brute-force cosine) — семантическая близость
          2. BM25 Keyword Search — точные совпадения терминов  
          3. Hybrid Merge — объединение vector + BM25 с весами
          4. Temporal Decay — старые воспоминания угасают
          5. MMR Re-ranking — разнообразие результатов
        
        Returns:
            List[str] — тексты документов (backward-compatible)
        """
        if not self.texts or self.embeddings is None:
            return []
        
        # Расширяем top_k для пост-обработки (MMR фильтрует дубли)
        fetch_k = min(top_k * 3, len(self.texts))
        
        # ── Шаг 1: Vector search ──
        vector_results = self._vector_search(query, fetch_k)
        
        # ── Шаг 2: Hybrid merge с BM25 ──
        if use_hybrid and self.bm25_weight > 0:
            self._ensure_bm25()
            if self._bm25:
                expanded = expand_query(query)  # Расширение синонимами
                bm25_hits = self._bm25.search(expanded, top_k=fetch_k)
                results = hybrid_merge(
                    vector_results, bm25_hits,
                    self.texts, self.timestamps,
                    self.vector_weight, self.bm25_weight,
                )
            else:
                results = vector_results
        else:
            results = vector_results
        
        # ── Шаг 3: Temporal Decay ──
        if use_decay and self.decay_half_life > 0:
            results = apply_temporal_decay(results, self.decay_half_life)
            results.sort(key=lambda r: r.score, reverse=True)
        
        # ── Шаг 4: MMR re-ranking для разнообразия ──
        if use_mmr and len(results) > 1:
            results = mmr_rerank(results[:fetch_k], self.mmr_lambda)
        
        # Вернуть top_k текстов (backward compatible)
        return [r.text for r in results[:top_k]]
    
    async def search_rich(self, query: str, top_k: int = 5,
                          use_hybrid: bool = True,
                          use_mmr: bool = True,
                          use_decay: bool = True) -> List[SearchResult]:
        """
        Поиск с полной информацией о результатах.
        Возвращает SearchResult с score, bm25_score, vector_score, timestamp.
        """
        if not self.texts or self.embeddings is None:
            return []
        
        fetch_k = min(top_k * 3, len(self.texts))
        vector_results = self._vector_search(query, fetch_k)
        
        if use_hybrid and self.bm25_weight > 0:
            self._ensure_bm25()
            if self._bm25:
                expanded = expand_query(query)
                bm25_hits = self._bm25.search(expanded, top_k=fetch_k)
                results = hybrid_merge(
                    vector_results, bm25_hits,
                    self.texts, self.timestamps,
                    self.vector_weight, self.bm25_weight,
                )
            else:
                results = vector_results
        else:
            results = vector_results
        
        if use_decay and self.decay_half_life > 0:
            results = apply_temporal_decay(results, self.decay_half_life)
            results.sort(key=lambda r: r.score, reverse=True)
        
        if use_mmr and len(results) > 1:
            results = mmr_rerank(results[:fetch_k], self.mmr_lambda)
        
        return results[:top_k]

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
            # Сохраняем timestamps
            if self.timestamps:
                save_dict['timestamps'] = np.array(self.timestamps, dtype=np.float64)
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
                
                # Загрузить timestamps
                if 'timestamps' in data:
                    self.timestamps = data['timestamps'].tolist()
                else:
                    # Legacy: нет timestamps → ставим 0
                    self.timestamps = [0.0] * len(self.texts)
                
                # Загрузить или построить IVF
                if 'ivf_centroids' in data and 'ivf_assignments' in data:
                    self.ivf_centroids = data['ivf_centroids']
                    self.ivf_assignments = data['ivf_assignments']
                    self.logger.info(f"LEANN: IVF загружен ({len(self.ivf_centroids)} кластеров)")
                else:
                    self._build_ivf()
                
                # BM25 будет построен лениво при первом search
                self._bm25_dirty = True
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
