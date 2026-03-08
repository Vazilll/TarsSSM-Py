/*
 * Online Softmax — AVX2 Implementation
 *
 * Algorithm (numerically stable, 2-pass):
 *   Pass 1: Find max_val = max(x)
 *   Pass 2: Compute exp(x[i] - max_val), accumulate sum
 *   Normalize: out[i] /= sum
 *
 * AVX2 path: vectorized max, exp approximation, and sum.
 *
 * Agent 1 — Week 1
 */

#include "softmax_fused.h"
#include <cmath>
#include <cfloat>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif


// ═══════════════════════════════════════
// Softmax
// ═══════════════════════════════════════

void tars_softmax(const float* x, int n, float* out) {
    if (n <= 0) return;
    if (n == 1) { out[0] = 1.0f; return; }

    // ── Pass 1: find max ──
    float max_val = x[0];
#if USE_AVX2
    {
        __m256 vmax = _mm256_set1_ps(-FLT_MAX);
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 v = _mm256_loadu_ps(x + i);
            vmax = _mm256_max_ps(vmax, v);
        }
        // Horizontal max
        __m128 hi = _mm256_extractf128_ps(vmax, 1);
        __m128 lo = _mm256_castps256_ps128(vmax);
        __m128 m = _mm_max_ps(lo, hi);
        m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
        max_val = _mm_cvtss_f32(m);
        // Scalar tail
        for (; i < n; ++i) {
            if (x[i] > max_val) max_val = x[i];
        }
    }
#else
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
#endif

    // ── Pass 2: exp(x - max) and sum ──
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = expf(x[i] - max_val);
        sum += out[i];
    }

    // ── Pass 3: normalize ──
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
#if USE_AVX2
        {
            __m256 vinv = _mm256_set1_ps(inv_sum);
            int i = 0;
            for (; i + 7 < n; i += 8) {
                __m256 v = _mm256_loadu_ps(out + i);
                _mm256_storeu_ps(out + i, _mm256_mul_ps(v, vinv));
            }
            for (; i < n; ++i) {
                out[i] *= inv_sum;
            }
        }
#else
        for (int i = 0; i < n; ++i) {
            out[i] *= inv_sum;
        }
#endif
    }
}


// ═══════════════════════════════════════
// Argmax
// ═══════════════════════════════════════

int tars_argmax(const float* x, int n) {
    if (n <= 0) return -1;
    int best = 0;
    float best_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > best_val) {
            best_val = x[i];
            best = i;
        }
    }
    return best;
}


// ═══════════════════════════════════════
// Softmax max (confidence for early exit)
// ═══════════════════════════════════════

float tars_softmax_max(const float* x, int n) {
    if (n <= 0) return 0.0f;
    if (n == 1) return 1.0f;

    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    float max_exp = 0.0f;
    for (int i = 0; i < n; ++i) {
        float e = expf(x[i] - max_val);
        sum += e;
        if (e > max_exp) max_exp = e;
    }

    // max(softmax) = max_exp / sum
    return (sum > 0.0f) ? max_exp / sum : 0.0f;
}
