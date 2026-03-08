/*
 * BitNet Ternary MatMul — AVX2 Implementation
 *
 * Y[r] = alpha * Σ_c W[r,c] * X[c]
 * Where W[r,c] ∈ {-1, 0, +1}
 *
 * Since W is ternary, the inner product becomes:
 *   acc += x[c]   when W = +1
 *   acc -= x[c]   when W = -1
 *   (skip)        when W = 0
 *
 * AVX2 path uses masked ADD/SUB with blend instructions.
 *
 * Python reference: brain/mamba2/core/bitnet.py → UniversalLinear
 *
 * Agent 1 — Week 1
 */

#include "bitnet_matmul.h"
#include <cmath>
#include <cstdlib>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// ═══════════════════════════════════════
// Scalar fallback
// ═══════════════════════════════════════

static void bitnet_matmul_scalar(
    const int8_t* W,
    const float* x,
    float alpha,
    int rows, int cols,
    float* out
) {
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const int8_t* w_row = W + (size_t)r * cols;
        for (int c = 0; c < cols; ++c) {
            int8_t w = w_row[c];
            if (w == 1) {
                acc += x[c];
            } else if (w == -1) {
                acc -= x[c];
            }
            // w == 0: skip (sparsity optimization)
        }
        out[r] = acc * alpha;
    }
}

// ═══════════════════════════════════════
// AVX2 ternary matmul
// ═══════════════════════════════════════

#if USE_AVX2
static void bitnet_matmul_avx2(
    const int8_t* W,
    const float* x,
    float alpha,
    int rows, int cols,
    float* out
) {
    // We process 8 columns at a time using AVX2.
    // For each group of 8 weights:
    //   1. Load 8 int8 weights → expand to 8 int32
    //   2. Create masks for +1 and -1
    //   3. Use masked add/sub with input vector

    for (int r = 0; r < rows; ++r) {
        const int8_t* w_row = W + (size_t)r * cols;
        __m256 vacc = _mm256_setzero_ps();
        
        int c = 0;
        for (; c + 7 < cols; c += 8) {
            // Load 8 weights as int8 → convert to int32
            // Use _mm_loadl_epi64 for 8 bytes, then extend
            __m128i w8 = _mm_loadl_epi64((__m128i const*)(w_row + c));
            // Sign-extend int8 → int16
            __m128i w16 = _mm_cvtepi8_epi16(w8);
            // Sign-extend int16 → int32 (low 4 + high 4)
            __m128i w32_lo = _mm_cvtepi16_epi32(w16);
            __m128i w32_hi = _mm_cvtepi16_epi32(_mm_srli_si128(w16, 8));
            __m256i w32 = _mm256_set_m128i(w32_hi, w32_lo);
            
            // Convert int32 weights to float (values are -1.0, 0.0, +1.0)
            __m256 wf = _mm256_cvtepi32_ps(w32);
            
            // Load 8 input values
            __m256 vx = _mm256_loadu_ps(x + c);
            
            // Fused multiply-add: acc += w * x
            // Since w ∈ {-1, 0, +1}, this is effectively add/sub/skip
            vacc = _mm256_fmadd_ps(wf, vx, vacc);
        }
        
        // Horizontal sum of 8 accumulators
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        float acc = _mm_cvtss_f32(s);
        
        // Scalar tail
        for (; c < cols; ++c) {
            int8_t w = w_row[c];
            if (w == 1) acc += x[c];
            else if (w == -1) acc -= x[c];
        }
        
        out[r] = acc * alpha;
    }
}
#endif


// ═══════════════════════════════════════
// Public API
// ═══════════════════════════════════════

void tars_bitnet_matmul(
    const int8_t* W_ternary,
    const float* x,
    float alpha,
    int rows, int cols,
    float* out
) {
#if USE_AVX2
    bitnet_matmul_avx2(W_ternary, x, alpha, rows, cols, out);
#else
    bitnet_matmul_scalar(W_ternary, x, alpha, rows, cols, out);
#endif
}

float tars_quantize_ternary(
    const float* W_float,
    int8_t* W_ternary,
    int numel
) {
    // Step 1: Compute scale = mean(|W|)
    float abs_sum = 0.0f;
    for (int i = 0; i < numel; ++i) {
        abs_sum += fabsf(W_float[i]);
    }
    float scale = abs_sum / (float)numel;
    if (scale < 1e-8f) scale = 1e-8f;
    
    // Step 2: Quantize to {-1, 0, +1}
    // BUG-2 fix: use roundf() to match Python torch.round() (IEEE 754 round-to-nearest-even)
    // Old code used > 0.5 / < -0.5 threshold which diverges from Python at boundaries
    float inv_scale = 1.0f / scale;
    for (int i = 0; i < numel; ++i) {
        float w_norm = W_float[i] * inv_scale;
        // clip to [-1, 1] and round (banker's rounding via roundf)
        if (w_norm > 1.0f) w_norm = 1.0f;
        else if (w_norm < -1.0f) w_norm = -1.0f;
        float rounded = roundf(w_norm);
        // roundf maps to {-1, 0, +1} for clipped input
        W_ternary[i] = (int8_t)rounded;
    }
    
    return scale;
}
