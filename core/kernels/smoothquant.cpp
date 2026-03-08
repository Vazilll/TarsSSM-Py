/*
 * SmoothQuant INT8 — AVX2 Implementation
 *
 * Y = (X · S⁻¹) · (S · W)
 *
 * INT8 matmul with float32 accumulation.
 * On CPUs with VNNI: could use _mm256_dpbusd_epi32 for 4× throughput.
 * This version uses standard AVX2 int16→int32 widening multiply.
 *
 * Python reference: brain/mamba2/core/bitnet.py
 *
 * Agent 1 — Week 3
 */

#include "smoothquant.h"
#include <cmath>
#include <cstdlib>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// ═══════════════════════════════════════
// Compute SmoothQuant scales
// ═══════════════════════════════════════

void tars_smoothquant_compute_scales(
    const float* act_scales,
    const float* wt_scales,
    float alpha,
    int dim,
    float* S_out
) {
    float one_minus_alpha = 1.0f - alpha;

    for (int j = 0; j < dim; ++j) {
        float a = act_scales[j];
        float w = wt_scales[j];

        // Avoid division by zero
        if (a < 1e-8f) a = 1e-8f;
        if (w < 1e-8f) w = 1e-8f;

        // S[j] = a^α / w^(1-α)
        S_out[j] = powf(a, alpha) / powf(w, one_minus_alpha);
    }
}

// ═══════════════════════════════════════
// INT8 Matmul with float accumulation
// ═══════════════════════════════════════

void tars_smoothquant_matmul_int8(
    const int8_t* X_int8,
    const int8_t* W_int8,
    float x_scale,
    float w_scale,
    int M, int K, int N,
    float* Y
) {
    float dequant_scale = x_scale * w_scale;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            const int8_t* x_row = X_int8 + m * K;
            const int8_t* w_col_start = W_int8 + n;  // column n, stride N

#if USE_AVX2
            // Process 16 elements at a time using int16 widening
            __m256i vacc = _mm256_setzero_si256();
            int k = 0;
            for (; k + 15 < K; k += 16) {
                // Load 16 int8 values from X
                __m128i vx = _mm_loadu_si128((__m128i const*)(x_row + k));

                // Load 16 int8 values from W (strided — need gather or scalar)
                // For row-major W[K,N], column n elements are at offsets n, n+N, n+2N...
                // This is non-contiguous — use scalar for W column access
                // TODO: optimize with transposed W for VNNI
                int8_t w_buf[16];
                for (int i = 0; i < 16 && k + i < K; ++i) {
                    w_buf[i] = W_int8[(k + i) * N + n];
                }
                __m128i vw = _mm_loadu_si128((__m128i const*)w_buf);

                // Widen int8→int16, multiply, accumulate to int32
                __m256i vx16 = _mm256_cvtepi8_epi16(vx);
                __m256i vw16 = _mm256_cvtepi8_epi16(vw);
                __m256i prod = _mm256_mullo_epi16(vx16, vw16);

                // Widen int16→int32 and accumulate
                __m128i prod_lo = _mm256_castsi256_si128(prod);
                __m128i prod_hi = _mm256_extracti128_si256(prod, 1);
                __m256i prod32_lo = _mm256_cvtepi16_epi32(prod_lo);
                __m256i prod32_hi = _mm256_cvtepi16_epi32(prod_hi);
                vacc = _mm256_add_epi32(vacc, prod32_lo);
                vacc = _mm256_add_epi32(vacc, prod32_hi);
            }

            // Horizontal sum of int32 accumulator
            __m128i hi = _mm256_extracti128_si256(vacc, 1);
            __m128i lo = _mm256_castsi256_si128(vacc);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
            sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
            acc = _mm_cvtsi128_si32(sum);

            // Scalar tail
            for (; k < K; ++k) {
                acc += (int32_t)x_row[k] * (int32_t)W_int8[k * N + n];
            }
#else
            for (int k = 0; k < K; ++k) {
                acc += (int32_t)x_row[k] * (int32_t)W_int8[k * N + n];
            }
#endif
            Y[m * N + n] = (float)acc * dequant_scale;
        }
    }
}

// ═══════════════════════════════════════
// Symmetric INT8 quantization
// ═══════════════════════════════════════

float tars_quantize_int8(
    const float* x_float,
    int numel,
    int8_t* x_int8
) {
    // Find max absolute value
    float max_abs = 0.0f;
    for (int i = 0; i < numel; ++i) {
        float a = fabsf(x_float[i]);
        if (a > max_abs) max_abs = a;
    }

    if (max_abs < 1e-8f) {
        // All zeros
        for (int i = 0; i < numel; ++i) x_int8[i] = 0;
        return 1e-8f;
    }

    float scale = max_abs / 127.0f;
    float inv_scale = 127.0f / max_abs;

    for (int i = 0; i < numel; ++i) {
        float v = x_float[i] * inv_scale;
        // Clamp to [-127, 127] and round
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        x_int8[i] = (int8_t)(v > 0 ? v + 0.5f : v - 0.5f);
    }

    return scale;
}
