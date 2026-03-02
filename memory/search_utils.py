"""
═══════════════════════════════════════════════════════════════
  LEANN Search Utilities — Advanced Search Algorithms
═══════════════════════════════════════════════════════════════

Портировано из OpenClaw (TypeScript → Python):
  1. MMR  (Maximal Marginal Relevance) — разнообразие результатов
  2. Temporal Decay — старые воспоминания угасают
  3. BM25 Keyword Search — текстовый поиск по ключевым словам
  4. Hybrid Merge — объединение Vector + BM25 с весами

Ссылки:
  - Carbonell & Goldstein, "The Use of MMR" (1998)
  - OpenClaw: src/memory/mmr.ts, temporal-decay.ts, hybrid.ts
"""

import math
import re
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════
# Типы данных
# ═══════════════════════════════════════════

@dataclass
class SearchResult:
    """Результат поиска с метаданными."""
    index: int          # Индекс в texts[]
    text: str           # Текст документа
    score: float        # Финальный score
    vector_score: float = 0.0   # Cosine similarity от vector search
    bm25_score: float = 0.0     # BM25 score от keyword search
    timestamp: float = 0.0      # Unix timestamp когда добавлен


# ═══════════════════════════════════════════
# 1. MMR — Maximal Marginal Relevance
# ═══════════════════════════════════════════

def _tokenize(text: str) -> set:
    """Токенизация текста для Jaccard similarity."""
    tokens = re.findall(r'[a-zа-яё0-9_]+', text.lower())
    return set(tokens)


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A∩B| / |A∪B|."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def mmr_rerank(
    results: List[SearchResult],
    lambda_param: float = 0.7,
) -> List[SearchResult]:
    """
    MMR (Maximal Marginal Relevance) re-ranking.
    
    Балансирует релевантность и разнообразие:
      MMR = λ × relevance - (1-λ) × max_similarity_to_selected
    
    λ=1.0 → только релевантность (без diversification)
    λ=0.5 → баланс
    λ=0.0 → максимальное разнообразие
    
    Args:
        results: Отсортированные по score результаты
        lambda_param: Параметр баланса (default 0.7)
    
    Returns:
        Переранжированные результаты
    """
    if len(results) <= 1:
        return list(results)
    
    lambda_param = max(0.0, min(1.0, lambda_param))
    
    if lambda_param == 1.0:
        return sorted(results, key=lambda r: r.score, reverse=True)
    
    # Кэш токенов
    token_cache = {r.index: _tokenize(r.text) for r in results}
    
    # Нормализация scores к [0, 1]
    max_score = max(r.score for r in results)
    min_score = min(r.score for r in results)
    score_range = max_score - min_score
    
    def normalize(score: float) -> float:
        if score_range == 0:
            return 1.0
        return (score - min_score) / score_range
    
    selected = []
    remaining = set(range(len(results)))
    
    while remaining:
        best_idx = None
        best_mmr = -float('inf')
        
        for idx in remaining:
            candidate = results[idx]
            norm_relevance = normalize(candidate.score)
            
            # Максимальная похожесть на уже выбранные
            max_sim = 0.0
            if selected:
                cand_tokens = token_cache[candidate.index]
                for sel in selected:
                    sel_tokens = token_cache[sel.index]
                    sim = _jaccard_similarity(cand_tokens, sel_tokens)
                    max_sim = max(max_sim, sim)
            
            mmr_score = lambda_param * norm_relevance - (1 - lambda_param) * max_sim
            
            if mmr_score > best_mmr or (mmr_score == best_mmr and 
                                         candidate.score > (results[best_idx].score if best_idx is not None else -float('inf'))):
                best_mmr = mmr_score
                best_idx = idx
        
        if best_idx is not None:
            selected.append(results[best_idx])
            remaining.discard(best_idx)
        else:
            break
    
    return selected


# ═══════════════════════════════════════════
# 2. Temporal Decay — Угасание старых воспоминаний
# ═══════════════════════════════════════════

DAY_SECONDS = 86400.0

def temporal_decay_multiplier(age_days: float, half_life_days: float = 30.0) -> float:
    """
    Экспоненциальный decay: score × e^(-λ × age_days)
    
    half_life = 30 дней → через 30 дней score × 0.5
                        → через 60 дней score × 0.25
                        → через 90 дней score × 0.125
    
    Args:
        age_days: Возраст документа в днях
        half_life_days: Период полураспада в днях
    
    Returns:
        Множитель [0, 1]
    """
    if half_life_days <= 0 or not math.isfinite(age_days):
        return 1.0
    
    lambda_decay = math.log(2) / half_life_days
    age = max(0.0, age_days)
    return math.exp(-lambda_decay * age)


def apply_temporal_decay(
    results: List[SearchResult],
    half_life_days: float = 30.0,
    now: Optional[float] = None,
) -> List[SearchResult]:
    """
    Применить temporal decay к результатам поиска.
    
    Документы без timestamp (=0) не меняются.
    """
    if half_life_days <= 0:
        return results
    
    now = now or time.time()
    
    decayed = []
    for r in results:
        if r.timestamp > 0:
            age_days = (now - r.timestamp) / DAY_SECONDS
            multiplier = temporal_decay_multiplier(age_days, half_life_days)
            new_r = SearchResult(
                index=r.index, text=r.text,
                score=r.score * multiplier,
                vector_score=r.vector_score,
                bm25_score=r.bm25_score,
                timestamp=r.timestamp,
            )
            decayed.append(new_r)
        else:
            decayed.append(r)
    
    return decayed


# ═══════════════════════════════════════════
# 3. BM25 — Keyword Search
# ═══════════════════════════════════════════

class BM25Index:
    """
    BM25 текстовый поиск по ключевым словам.
    
    Дополняет vector search: ищет точные совпадения слов,
    а не семантическую близость.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_dl = 0.0
        self.doc_lens = []           # длина каждого документа
        self.term_freqs = []         # [{term: count}] для каждого документа
        self.doc_freq = {}           # {term: кол-во документов с этим термом}
        self._tokenize_cache = {}
    
    def _tokenize_doc(self, text: str) -> list:
        """Токенизация документа для BM25."""
        return re.findall(r'[a-zа-яё0-9_]+', text.lower())
    
    def build(self, texts: List[str]):
        """Построить BM25 индекс из списка текстов."""
        self.doc_count = len(texts)
        self.doc_lens = []
        self.term_freqs = []
        self.doc_freq = {}
        
        total_len = 0
        for text in texts:
            tokens = self._tokenize_doc(text)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            
            # Term frequency для документа
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            
            # Document frequency
            for t in set(tokens):
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1
        
        self.avg_dl = total_len / max(self.doc_count, 1)
    
    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Поиск по BM25.
        
        Returns:
            Список (index, bm25_score) отсортированных по score
        """
        if self.doc_count == 0:
            return []
        
        query_tokens = self._tokenize_doc(query)
        if not query_tokens:
            return []
        
        scores = []
        for doc_idx in range(self.doc_count):
            score = 0.0
            dl = self.doc_lens[doc_idx]
            tf_dict = self.term_freqs[doc_idx]
            
            for term in query_tokens:
                if term not in tf_dict:
                    continue
                
                tf = tf_dict[term]
                df = self.doc_freq.get(term, 0)
                
                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
                
                # BM25 TF component
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                
                score += idf * tf_norm
            
            if score > 0:
                scores.append((doc_idx, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ═══════════════════════════════════════════
# 4. Hybrid Merge — Vector + BM25
# ═══════════════════════════════════════════

def hybrid_merge(
    vector_results: List[SearchResult],
    bm25_results: List[Tuple[int, float]],
    texts: List[str],
    timestamps: Optional[List[float]] = None,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3,
) -> List[SearchResult]:
    """
    Объединение результатов vector search и BM25.
    
    Формула: score = vector_weight × vector_score + bm25_weight × bm25_score
    
    Args:
        vector_results: Результаты из vector search
        bm25_results: Результаты из BM25 [(index, score)]
        texts: Все тексты документов
        timestamps: Timestamps документов (optional)
        vector_weight: Вес vector search (default 0.7)
        bm25_weight: Вес BM25 (default 0.3)
    
    Returns:
        Merged и отсортированные результаты
    """
    merged = {}  # index → SearchResult
    
    # Нормализация BM25 scores к [0, 1]
    bm25_max = max((s for _, s in bm25_results), default=1.0) or 1.0
    
    # Добавить vector results
    for r in vector_results:
        merged[r.index] = SearchResult(
            index=r.index,
            text=r.text,
            score=0.0,
            vector_score=r.score,
            bm25_score=0.0,
            timestamp=r.timestamp,
        )
    
    # Добавить BM25 results
    for idx, bm25_score in bm25_results:
        normalized_bm25 = bm25_score / bm25_max
        if idx in merged:
            merged[idx].bm25_score = normalized_bm25
        else:
            ts = timestamps[idx] if timestamps and idx < len(timestamps) else 0.0
            merged[idx] = SearchResult(
                index=idx,
                text=texts[idx] if idx < len(texts) else "",
                score=0.0,
                vector_score=0.0,
                bm25_score=normalized_bm25,
                timestamp=ts,
            )
    
    # Вычислить финальный score
    for r in merged.values():
        r.score = vector_weight * r.vector_score + bm25_weight * r.bm25_score
    
    results = sorted(merged.values(), key=lambda r: r.score, reverse=True)
    return results
