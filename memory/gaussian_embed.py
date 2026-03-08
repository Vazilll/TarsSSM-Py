"""
gaussian_embed.py — Gaussian Uncertainty Embeddings (J13).

Каждый документ хранится не как точка μ ∈ ℝ^d, а как распределение N(μ, σ²):
  - μ = средний вектор (семантика)
  - σ = неопределённость (ширина распределения)

Зачем:
  Точечные embeddings теряют информацию об "уверенности".
  "Банк" → чёткий термин, σ маленькая → точное совпадение.
  "Финансовые вещи" → размытый запрос, σ большая → широкий поиск.

Similarity:
  Вместо cosine(μ₁, μ₂) используем:
    sim = -W₂(N₁, N₂) — отрицательное расстояние Вассерштейна-2
  W₂² = ||μ₁ - μ₂||² + ||σ₁ - σ₂||² (для диагональных ковариаций)

CPU-friendly: всё = numpy, никакого PyTorch на hot path.
RAM: +4 bytes/dim/doc (float32 σ вектор).
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger("Tars.GaussianEmbed")

# ═══════════════════════════════════════
# Конфигурация
# ═══════════════════════════════════════
DEFAULT_DIM = 384              # Совпадает с all-MiniLM-L6-v2
DEFAULT_SIGMA_INIT = 0.1       # Начальная неопределённость
MIN_SIGMA = 1e-4               # Нижняя граница σ (избегаем вырождения)
MAX_SIGMA = 2.0                # Верхняя граница σ (избегаем бесконечной размытости)
SIGMA_DECAY_ON_ACCESS = 0.95   # При recall σ уменьшается (уверенность растёт)
SIGMA_BOOST_ON_UPDATE = 1.1    # При обновлении содержимого σ растёт (нестабильность)


@dataclass
class GaussianVector:
    """Gaussian embedding: N(μ, diag(σ²))."""
    mu: np.ndarray       # [d] — mean vector (float32)
    sigma: np.ndarray    # [d] — std dev per dimension (float32)

    @property
    def dim(self) -> int:
        return len(self.mu)

    def to_dict(self) -> dict:
        return {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> "GaussianVector":
        return GaussianVector(
            mu=np.array(data["mu"], dtype=np.float32),
            sigma=np.array(data["sigma"], dtype=np.float32),
        )


class GaussianEmbedding:
    """
    Gaussian Uncertainty Embedding engine.

    Кодирует текст в N(μ, σ²) и считает similarity через Wasserstein-2.

    Attributes:
        dim: размерность embeddings (default=384)
        sigma_init: начальная σ для новых документов
        embedder: callable(text) → np.ndarray[float32], e.g. SentenceTransformer
    """

    def __init__(self, dim: int = DEFAULT_DIM,
                 sigma_init: float = DEFAULT_SIGMA_INIT,
                 embedder=None):
        self.dim = dim
        self.sigma_init = sigma_init
        self._embedder = embedder  # Lazy: если None → будет TF-IDF fallback

    @property
    def embedder(self):
        """Lazy-load SentenceTransformer или TF-IDF fallback."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                self._embedder = lambda text: model.encode([text])[0].astype(np.float32)
            except ImportError:
                def tfidf_fallback(text):
                    words = text.lower().split()
                    emb = np.zeros(self.dim, dtype=np.float32)
                    for w in words:
                        idx = hash(w) % self.dim
                        emb[idx] += 1.0
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb /= norm
                    return emb
                self._embedder = tfidf_fallback
                logger.warning("GaussianEmbed: SentenceTransformer unavailable, using TF-IDF fallback")
        return self._embedder

    def encode(self, text: str, sigma: Optional[np.ndarray] = None) -> GaussianVector:
        """
        Encode text → Gaussian N(μ, σ²).

        Args:
            text: входной текст
            sigma: если указан — использовать этот σ вместо автоматического

        Returns:
            GaussianVector(mu, sigma)
        """
        mu = self.embedder(text)

        if sigma is not None:
            sigma_vec = np.clip(sigma, MIN_SIGMA, MAX_SIGMA).astype(np.float32)
        else:
            # Автоматическая σ: короткие/точные тексты → низкая σ, длинные/размытые → высокая
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))

            # Heuristic: more unique words → more specific → lower σ
            specificity = unique_words / max(word_count, 1)
            auto_sigma = self.sigma_init * (1.5 - specificity)
            auto_sigma = np.clip(auto_sigma, MIN_SIGMA, MAX_SIGMA)

            sigma_vec = np.full(self.dim, auto_sigma, dtype=np.float32)

        return GaussianVector(mu=mu, sigma=sigma_vec)

    def encode_batch(self, texts: List[str]) -> List[GaussianVector]:
        """Batch encode: кодирует список текстов."""
        return [self.encode(text) for text in texts]

    @staticmethod
    def wasserstein2_distance(a: GaussianVector, b: GaussianVector) -> float:
        """
        Wasserstein-2 distance между двумя Gaussians (diagonal cov).

        W₂²(N₁, N₂) = ||μ₁ - μ₂||² + ||σ₁ - σ₂||²

        Для диагональных ковариаций — точная closed-form.
        """
        mu_diff_sq = np.sum((a.mu - b.mu) ** 2)
        sigma_diff_sq = np.sum((a.sigma - b.sigma) ** 2)
        return float(np.sqrt(mu_diff_sq + sigma_diff_sq))

    @staticmethod
    def similarity(a: GaussianVector, b: GaussianVector,
                   max_dist: float = 20.0) -> float:
        """
        Similarity = 1 / (1 + W₂).

        Нормализована в [0, 1]:
          - Идентичные → 1.0
          - Совершенно разные → ~0.0
        """
        w2 = GaussianEmbedding.wasserstein2_distance(a, b)
        return 1.0 / (1.0 + w2)

    @staticmethod
    def expected_likelihood_kernel(a: GaussianVector, b: GaussianVector) -> float:
        """
        Expected Likelihood Kernel (ELK):
          ELK(a, b) = ∫ N(x | μ_a, Σ_a) · N(x | μ_b, Σ_b) dx

        Для диагональных Gaussians:
          ELK = ∏ᵢ (2π(σ_aᵢ² + σ_bᵢ²))^(-½) · exp(-Δμᵢ²/(2(σ_aᵢ² + σ_bᵢ²)))

        Логарифмическая версия для численной стабильности.
        """
        sigma_sq_sum = a.sigma ** 2 + b.sigma ** 2 + 1e-8
        mu_diff_sq = (a.mu - b.mu) ** 2

        log_elk = -0.5 * np.sum(np.log(2 * np.pi * sigma_sq_sum)) \
                  - 0.5 * np.sum(mu_diff_sq / sigma_sq_sum)

        # Clamp to avoid overflow
        log_elk = np.clip(log_elk, -500, 0)
        return float(np.exp(log_elk))

    @staticmethod
    def update_sigma_on_recall(gv: GaussianVector) -> GaussianVector:
        """
        При вспоминании (recall) σ уменьшается — память становится чётче.
        Аналог реконсолидации: вспомненные воспоминания укрепляются.
        """
        new_sigma = np.clip(gv.sigma * SIGMA_DECAY_ON_ACCESS, MIN_SIGMA, MAX_SIGMA)
        return GaussianVector(mu=gv.mu, sigma=new_sigma)

    @staticmethod
    def update_sigma_on_conflict(gv: GaussianVector) -> GaussianVector:
        """
        При конфликте/обновлении σ растёт — память становится размытее.
        Аналог интерференции: новая информация делает существующую менее уверенной.
        """
        new_sigma = np.clip(gv.sigma * SIGMA_BOOST_ON_UPDATE, MIN_SIGMA, MAX_SIGMA)
        return GaussianVector(mu=gv.mu, sigma=new_sigma)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ge = GaussianEmbedding()

    # Тест: точный запрос vs размытый
    precise = ge.encode("Москва — столица России")
    vague = ge.encode("что-то про города и страны где-то в мире")
    same = ge.encode("Москва — столица Российской Федерации")

    print(f"Precise σ mean: {precise.sigma.mean():.4f}")
    print(f"Vague σ mean:   {vague.sigma.mean():.4f}")

    sim_same = GaussianEmbedding.similarity(precise, same)
    sim_diff = GaussianEmbedding.similarity(precise, vague)
    print(f"\nSimilarity (precise ↔ same):  {sim_same:.4f}")
    print(f"Similarity (precise ↔ vague): {sim_diff:.4f}")

    # Тест: реконсолидация
    recalled = GaussianEmbedding.update_sigma_on_recall(precise)
    print(f"\nAfter recall σ:  {recalled.sigma.mean():.4f} (was {precise.sigma.mean():.4f})")
