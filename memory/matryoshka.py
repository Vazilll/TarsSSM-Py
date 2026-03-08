"""
matryoshka.py — 2-Stage Matryoshka Retrieval (J12).

Двухэтапный поиск по "матрёшке" размерностей:
  Stage 1: Быстрый грубый поиск по первым 64 dims из 384d (6× меньше вычислений)
  Stage 2: Точный re-rank полными 384d только для top candidates

Принцип Matryoshka Representations (Kusupati et al., NeurIPS 2022):
  Embedding обучен так, что prefix из первых M dims сохраняет семантику.
  Мы используем M=64 → M=384 двухэтапную воронку.

Связь с пайплайном:
  MatryoshkaRetriever оборачивает LeannIndex — использует его embeddings.
  Для больших индексов (>5K документов) даёт ускорение ~3-5× при тех же результатах.

RAM: 0 дополнительных — переиспользует embeddings из LeannIndex.
"""

import numpy as np
import time
import logging
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from memory.leann import LeannIndex

from memory.search_utils import SearchResult

logger = logging.getLogger("Tars.Matryoshka")

# ═══════════════════════════════════════
# Размерности "матрёшки" (от грубого к точному)
# ═══════════════════════════════════════
# all-MiniLM-L6-v2 embeddings = 384d
# Matryoshka truncation: 64 → 128 → 384
COARSE_DIM = 64    # Stage 1: быстрый scan
FINE_DIM = 384     # Stage 2: точный rerank
COARSE_EXPANSION = 8  # берём top_k × 8 на Stage 1 для re-rank


class MatryoshkaRetriever:
    """
    2-Stage Matryoshka Retrieval — ускоренный поиск для больших индексов.

    Алгоритм:
      1. Truncate embeddings до первых 64d → cosine на сжатых векторах
      2. Отобрать top_k × 8 кандидатов
      3. Re-rank по полным 384d embeddings → top_k результатов

    Threshold:
      Если индекс < min_docs_for_matryoshka → прямой brute-force (нет смысла).

    Attributes:
        leann: LeannIndex instance (embeddings source)
        coarse_dim: размерность для грубого поиска (default=64)
        expansion: множитель для количества кандидатов на Stage 1
        min_docs: минимум документов для включения Matryoshka
    """

    def __init__(self, leann: "LeannIndex",
                 coarse_dim: int = COARSE_DIM,
                 expansion: int = COARSE_EXPANSION,
                 min_docs: int = 500):
        self.leann = leann
        self.coarse_dim = coarse_dim
        self.expansion = expansion
        self.min_docs = min_docs

    def _should_use_matryoshka(self) -> bool:
        """Матрёшка полезна только для больших индексов."""
        if self.leann.embeddings is None:
            return False
        return len(self.leann.embeddings) >= self.min_docs

    def _dequantize_embeddings(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Деквантизация INT8 → float32 из LeannIndex.

        Args:
            indices: если указаны, деквантизировать только эти строки
        Returns:
            [N, 384] float32 нормализованные embeddings
        """
        embs = self.leann.embeddings
        scales = self.leann.emb_scales

        if indices is not None:
            embs = embs[indices]
            scales = scales[indices]

        embs_f32 = embs.astype(np.float32) * scales[:, np.newaxis]
        norms = np.linalg.norm(embs_f32, axis=1, keepdims=True) + 1e-8
        return embs_f32 / norms

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Получить float32 query embedding из LeannIndex."""
        q_int8, q_scale = self.leann._get_embedding(query)
        q_f32 = q_int8.astype(np.float32) * q_scale
        q_norm = np.linalg.norm(q_f32) + 1e-8
        return q_f32 / q_norm

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        2-stage Matryoshka поиск.

        Args:
            query: текстовый запрос
            top_k: количество результатов

        Returns:
            List[SearchResult] — отсортированные по score
        """
        if self.leann.embeddings is None or len(self.leann.texts) == 0:
            return []

        t0 = time.time()

        # Если индекс маленький — обычный полный поиск
        if not self._should_use_matryoshka():
            results = self.leann._vector_search(query, top_k)
            return results

        q_emb = self._get_query_embedding(query)
        n_docs = len(self.leann.embeddings)

        # ═══ Stage 1: Coarse search (64d) ═══
        # Truncate query и все embeddings до coarse_dim
        q_coarse = q_emb[:self.coarse_dim]
        q_coarse = q_coarse / (np.linalg.norm(q_coarse) + 1e-8)

        # Деквантизация только первых coarse_dim столбцов
        embs_int8 = self.leann.embeddings[:, :self.coarse_dim]
        scales = self.leann.emb_scales
        embs_coarse = embs_int8.astype(np.float32) * scales[:, np.newaxis]
        norms_c = np.linalg.norm(embs_coarse, axis=1, keepdims=True) + 1e-8
        embs_coarse = embs_coarse / norms_c

        # Cosine similarity на 64d
        coarse_scores = embs_coarse @ q_coarse
        n_candidates = min(top_k * self.expansion, n_docs)
        candidate_indices = np.argpartition(coarse_scores, -n_candidates)[-n_candidates:]
        candidate_indices = candidate_indices[np.argsort(coarse_scores[candidate_indices])[::-1]]

        # ═══ Stage 2: Fine re-rank (384d) ═══
        fine_embs = self._dequantize_embeddings(candidate_indices)
        fine_scores = fine_embs @ q_emb  # полные 384d

        # Сортировка и отбор top_k
        top_local = np.argsort(fine_scores)[::-1][:top_k]

        elapsed_ms = (time.time() - t0) * 1000

        results = []
        for i in top_local:
            idx = int(candidate_indices[i])

            # Пропуск удалённых документов (tombstone)
            if idx < len(self.leann.deleted) and self.leann.deleted[idx]:
                continue

            ts = self.leann.timestamps[idx] if idx < len(self.leann.timestamps) else 0.0
            results.append(SearchResult(
                index=idx,
                text=self.leann.texts[idx],
                score=float(fine_scores[i]),
                vector_score=float(fine_scores[i]),
                timestamp=ts,
            ))

        logger.debug(
            f"Matryoshka: {n_docs} docs → {n_candidates} candidates "
            f"→ {len(results)} results ({elapsed_ms:.1f}ms)"
        )
        return results

    def get_stats(self) -> dict:
        """Статистика Matryoshka retriever."""
        n_docs = len(self.leann.embeddings) if self.leann.embeddings is not None else 0
        return {
            "total_docs": n_docs,
            "coarse_dim": self.coarse_dim,
            "fine_dim": FINE_DIM,
            "matryoshka_active": self._should_use_matryoshka(),
            "expansion": self.expansion,
            "min_docs_threshold": self.min_docs,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import asyncio
    from memory.leann import LeannIndex

    idx = LeannIndex()
    for i in range(20):
        idx.add_document(f"Тестовый документ номер {i} с уникальным содержимым")

    retriever = MatryoshkaRetriever(idx, min_docs=10)

    async def test():
        results = await retriever.search("тестовый документ", top_k=3)
        for r in results:
            print(f"  [{r.index}] score={r.score:.4f}: {r.text[:60]}")
        print(f"\nStats: {retriever.get_stats()}")

    asyncio.run(test())
