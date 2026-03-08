/*
 * RMSNorm Fused Kernel — AVX2 Implementation
 *
 * Formula: out[i] = gamma[i] * x[i] / sqrt(mean(x²) + eps)
 *
 * Two passes:
 *   Pass 1: Compute sum_sq = Σ x[i]²   (AVX2 vectorized)
 *   Pass 2: scale = 1/sqrt(sum_sq/dim + eps), out[i] = gamma[i] * x[i] * scale
 *
 * Python reference: brain/mamba2/core/bitnet.py → RMSNorm
 *
 * Agent 1 — Week 1
 */

#include "rmsnorm_fused.h"
#include <cmath>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

void tars_rmsnorm(
    const float* x,
    const float* gamma,
    float eps,
    int dim,
    float* out
) {
    // ═══ Pass 1: sum of squares ═══
    float sum_sq = 0.0f;

#if USE_AVX2
    {
        __m256 vsum = _mm256_setzero_ps();
        int i = 0;
        // Main AVX2 loop: 8 floats at a time
        for (; i + 7 < dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            vsum = _mm256_fmadd_ps(vx, vx, vsum);  // vsum += x*x
        }
        // Horizontal sum of 8 lanes
        // vsum = [a0 a1 a2 a3 | a4 a5 a6 a7]
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        __m128 s = _mm_add_ps(lo, hi);           // [a0+a4, a1+a5, a2+a6, a3+a7]
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));  // [a0+a4+a2+a6, a1+a5+a3+a7, ...]
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1)); // total in lane 0
        sum_sq = _mm_cvtss_f32(s);
        // Scalar tail
        for (; i < dim; ++i) {
            sum_sq += x[i] * x[i];
        }
    }
#else
    for (int i = 0; i < dim; ++i) {
        sum_sq += x[i] * x[i];
    }
#endif

    // ═══ Pass 2: scale and apply gamma ═══
    float rms_inv = 1.0f / sqrtf(sum_sq / (float)dim + eps);

#if USE_AVX2
    {
        __m256 vscale = _mm256_set1_ps(rms_inv);
        int i = 0;
        for (; i + 7 < dim; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vg = _mm256_loadu_ps(gamma + i);
            __m256 vr = _mm256_mul_ps(_mm256_mul_ps(vx, vscale), vg);
            _mm256_storeu_ps(out + i, vr);
        }
        // Scalar tail
        for (; i < dim; ++i) {
            out[i] = gamma[i] * x[i] * rms_inv;
        }
    }
#else
    for (int i = 0; i < dim; ++i) {
        out[i] = gamma[i] * x[i] * rms_inv;
    }
#endif
}
