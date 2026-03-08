/*
 * Embedding Lookup — AVX2 Implementation
 *
 * out[i] = table[token_id * d_model + i] * sqrt(d_model)
 *
 * Simple but important: the √d_model scaling is standard for transformer
 * embeddings (prevents vanishing gradients in deep models).
 *
 * Agent 1 — Week 1
 */

#include "embedding_lookup.h"
#include <cmath>
#include <cstring>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

void tars_embedding_lookup(
    const float* table,
    int token_id,
    int d_model,
    float* out
) {
    const float* row = table + (size_t)token_id * d_model;
    float scale = sqrtf((float)d_model);

#if USE_AVX2
    __m256 vscale = _mm256_set1_ps(scale);
    int i = 0;
    for (; i + 7 < d_model; i += 8) {
        __m256 v = _mm256_loadu_ps(row + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, vscale));
    }
    for (; i < d_model; ++i) {
        out[i] = row[i] * scale;
    }
#else
    for (int i = 0; i < d_model; ++i) {
        out[i] = row[i] * scale;
    }
#endif
}
