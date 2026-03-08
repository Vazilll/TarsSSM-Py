/*
 * RoPE Fused — AVX2 Implementation
 *
 * Rotary Position Embeddings with θ = 500,000 (Qwen3/LLaMA3 style).
 *
 * Two phases:
 *   1. Precompute: build frequency table [max_seq, dim/2]
 *   2. Apply:      rotate Q, K vectors using cos/sin pairs
 *
 * Python reference: training/custom_kernels.py → _rope_python()
 *
 * Agent 1 — Week 2
 */

#include "rope_fused.h"
#include <cmath>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// ═══════════════════════════════════════
// Precompute frequency table
// ═══════════════════════════════════════

void tars_rope_precompute(
    float* freqs,
    double base,
    int dim,
    int max_seq
) {
    int half_d = dim / 2;

    for (int pos = 0; pos < max_seq; ++pos) {
        for (int i = 0; i < half_d; ++i) {
            // θ_i = pos / (base^(2i/dim))
            double theta = (double)pos / pow(base, (double)(2 * i) / (double)dim);
            freqs[pos * half_d + i] = (float)theta;
        }
    }
}

// ═══════════════════════════════════════
// Apply RoPE rotation (in-place)
// ═══════════════════════════════════════

static inline void rope_rotate_vec(
    float* vec,         // [dim], modified in-place
    const float* freq,  // [dim/2], angles for this position
    int dim
) {
    int half_d = dim / 2;

#if USE_AVX2
    int i = 0;
    for (; i + 3 < half_d; i += 4) {
        // Load 4 pairs: (x1, x2) for dimensions (2i, 2i+1)
        // We need to interleave: load x[2i] and x[2i+half_d]
        // Actually with RoPE, the standard layout is:
        //   first half = x[0..half_d-1]
        //   second half = x[half_d..dim-1]

        __m128 vx1 = _mm_loadu_ps(vec + i);           // x1 = [x[i], x[i+1], x[i+2], x[i+3]]
        __m128 vx2 = _mm_loadu_ps(vec + half_d + i);  // x2 = [x[half+i], ...]
        __m128 vf = _mm_loadu_ps(freq + i);            // angles

        // cos/sin (scalar, since sincos isn't in SSE)
        float c0 = cosf(freq[i]),   s0 = sinf(freq[i]);
        float c1 = cosf(freq[i+1]), s1 = sinf(freq[i+1]);
        float c2 = cosf(freq[i+2]), s2 = sinf(freq[i+2]);
        float c3 = cosf(freq[i+3]), s3 = sinf(freq[i+3]);

        __m128 vc = _mm_set_ps(c3, c2, c1, c0);
        __m128 vs = _mm_set_ps(s3, s2, s1, s0);

        // Rotation: [x1', x2'] = [x1*cos - x2*sin, x1*sin + x2*cos]
        __m128 new_x1 = _mm_sub_ps(_mm_mul_ps(vx1, vc), _mm_mul_ps(vx2, vs));
        __m128 new_x2 = _mm_add_ps(_mm_mul_ps(vx1, vs), _mm_mul_ps(vx2, vc));

        _mm_storeu_ps(vec + i, new_x1);
        _mm_storeu_ps(vec + half_d + i, new_x2);
    }
    // Scalar tail
    for (; i < half_d; ++i) {
        float x1 = vec[i];
        float x2 = vec[half_d + i];
        float c = cosf(freq[i]);
        float s = sinf(freq[i]);
        vec[i]          = x1 * c - x2 * s;
        vec[half_d + i] = x1 * s + x2 * c;
    }
#else
    for (int i = 0; i < half_d; ++i) {
        float x1 = vec[i];
        float x2 = vec[half_d + i];
        float c = cosf(freq[i]);
        float s = sinf(freq[i]);
        vec[i]          = x1 * c - x2 * s;
        vec[half_d + i] = x1 * s + x2 * c;
    }
#endif
}

void tars_rope_apply(
    float* Q,
    float* K,
    const float* freqs,
    int seq_len,
    int dim,
    int offset
) {
    int half_d = dim / 2;

    for (int t = 0; t < seq_len; ++t) {
        int pos = t + offset;
        const float* freq_t = freqs + pos * half_d;

        rope_rotate_vec(Q + t * dim, freq_t, dim);
        rope_rotate_vec(K + t * dim, freq_t, dim);
    }
}
