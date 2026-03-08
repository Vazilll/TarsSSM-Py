/*
 * SwiGLU Fused — AVX2 Implementation
 *
 * y = SiLU(gate) ⊙ value
 * SiLU(x) = x / (1 + exp(-x))
 *
 * With optional Double Sparsity: zero where |y| < threshold.
 *
 * AVX2 path: fast SiLU approximation using polynomial.
 *
 * Python reference: training/custom_kernels.py → _swiglu_python()
 *
 * Agent 1 — Week 2
 */

#include "swiglu_fused.h"
#include <cmath>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// Fast scalar SiLU
static inline float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

#if USE_AVX2
// Fast AVX2 exp approximation (Schraudolph-style, ~1e-3 relative error)
static inline __m256 _mm256_fast_exp_ps(__m256 x) {
    // Clamp to prevent overflow/underflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    // exp(x) ≈ 2^(x * log2e)
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 C1 = _mm256_set1_ps(0.693359375f);
    const __m256 C2 = _mm256_set1_ps(-2.12194440e-4f);
    const __m256 one = _mm256_set1_ps(1.0f);

    __m256 fx = _mm256_mul_ps(x, log2e);
    __m256 fxf = _mm256_floor_ps(fx);
    __m256 fxr = _mm256_sub_ps(fx, fxf);

    // Reduced x
    __m256 xr = _mm256_sub_ps(x, _mm256_mul_ps(fxf, C1));
    xr = _mm256_sub_ps(xr, _mm256_mul_ps(fxf, C2));

    // Polynomial: 1 + x + x²/2 + x³/6
    __m256 p = _mm256_set1_ps(1.0f / 6.0f);
    p = _mm256_fmadd_ps(p, xr, _mm256_set1_ps(0.5f));
    p = _mm256_fmadd_ps(p, xr, one);
    p = _mm256_fmadd_ps(p, xr, one);

    // Scale by 2^integer_part
    __m256i fi = _mm256_cvtps_epi32(fxf);
    fi = _mm256_add_epi32(fi, _mm256_set1_epi32(127));
    fi = _mm256_slli_epi32(fi, 23);
    __m256 pow2 = _mm256_castsi256_ps(fi);

    return _mm256_mul_ps(p, pow2);
}

// AVX2 SiLU: x / (1 + exp(-x))
static inline __m256 _mm256_silu_ps(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 exp_neg = _mm256_fast_exp_ps(neg_x);
    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg);
    return _mm256_div_ps(x, denom);
}
#endif

int tars_swiglu_fused(
    const float* gate,
    const float* value,
    int dim,
    float sparsity_threshold,
    float* out
) {
    int nnz = 0;

#if USE_AVX2
    int i = 0;
    if (sparsity_threshold <= 0.0f) {
        // No sparsity: just SiLU(gate) * value
        for (; i + 7 < dim; i += 8) {
            __m256 vg = _mm256_loadu_ps(gate + i);
            __m256 vv = _mm256_loadu_ps(value + i);
            __m256 silu_g = _mm256_silu_ps(vg);
            __m256 result = _mm256_mul_ps(silu_g, vv);
            _mm256_storeu_ps(out + i, result);
        }
        nnz = dim;
        for (; i < dim; ++i) {
            out[i] = silu_f(gate[i]) * value[i];
        }
    } else {
        // With sparsity mask
        __m256 vthresh = _mm256_set1_ps(sparsity_threshold);
        __m256 vneg_thresh = _mm256_set1_ps(-sparsity_threshold);
        __m256 vzero = _mm256_setzero_ps();

        for (; i + 7 < dim; i += 8) {
            __m256 vg = _mm256_loadu_ps(gate + i);
            __m256 vv = _mm256_loadu_ps(value + i);
            __m256 silu_g = _mm256_silu_ps(vg);
            __m256 result = _mm256_mul_ps(silu_g, vv);

            // Sparsity: zero where |result| < threshold
            __m256 mask_pos = _mm256_cmp_ps(result, vthresh, _CMP_GE_OQ);
            __m256 mask_neg = _mm256_cmp_ps(result, vneg_thresh, _CMP_LE_OQ);
            __m256 mask = _mm256_or_ps(mask_pos, mask_neg);
            result = _mm256_blendv_ps(vzero, result, mask);

            _mm256_storeu_ps(out + i, result);

            // Count non-zeros (approximate via movemask)
            nnz += __builtin_popcount(_mm256_movemask_ps(mask));
        }
        for (; i < dim; ++i) {
            float y = silu_f(gate[i]) * value[i];
            if (y > sparsity_threshold || y < -sparsity_threshold) {
                out[i] = y;
                nnz++;
            } else {
                out[i] = 0.0f;
            }
        }
    }
#else
    for (int i = 0; i < dim; ++i) {
        float y = silu_f(gate[i]) * value[i];
        if (sparsity_threshold > 0.0f &&
            y < sparsity_threshold && y > -sparsity_threshold) {
            out[i] = 0.0f;
        } else {
            out[i] = y;
            nnz++;
        }
    }
#endif

    return nnz;
}
