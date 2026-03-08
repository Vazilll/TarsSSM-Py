/*
 * Diff Attention — AVX2 Implementation
 *
 * Differential Transformer (Ye et al., 2024):
 *   attn₁ = softmax(Q₁K₁ᵀ / √d)
 *   attn₂ = softmax(Q₂K₂ᵀ / √d)
 *   attn  = attn₁ − λ · attn₂
 *   out   = attn @ V
 *
 * Allocates two [seq×seq] scratch buffers for the attention maps.
 *
 * Python reference: training/custom_kernels.py → _diff_attention_python()
 *
 * Agent 1 — Week 3
 */

#include "attention_diff.h"
#include <cmath>
#include <cfloat>
#include <cstring>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// ═══════════════════════════════════════
// Helpers
// ═══════════════════════════════════════

// Compute QKᵀ: out[i][j] = Σ_k Q[i][k] * K[j][k] * scale
static void matmul_QKt(
    const float* Q,     // [M, D]
    const float* K,     // [N, D]
    float scale,
    int M, int N, int D,
    float* out          // [M, N]
) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            const float* qi = Q + i * D;
            const float* kj = K + j * D;
#if USE_AVX2
            __m256 vacc = _mm256_setzero_ps();
            int k = 0;
            for (; k + 7 < D; k += 8) {
                __m256 vq = _mm256_loadu_ps(qi + k);
                __m256 vk = _mm256_loadu_ps(kj + k);
                vacc = _mm256_fmadd_ps(vq, vk, vacc);
            }
            __m128 hi = _mm256_extractf128_ps(vacc, 1);
            __m128 lo = _mm256_castps256_ps128(vacc);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            dot = _mm_cvtss_f32(s);
            for (; k < D; ++k) dot += qi[k] * kj[k];
#else
            for (int k = 0; k < D; ++k) dot += qi[k] * kj[k];
#endif
            out[i * N + j] = dot * scale;
        }
    }
}

// Online softmax: row-wise, numerically stable
static void softmax_rows(float* mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        float* row = mat + i * cols;

        // Find max
        float max_val = -FLT_MAX;
        for (int j = 0; j < cols; ++j) {
            if (row[j] > max_val) max_val = row[j];
        }

        // exp and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }

        // Normalize
        float inv_sum = 1.0f / (sum + 1e-8f);
        for (int j = 0; j < cols; ++j) {
            row[j] *= inv_sum;
        }
    }
}

// Apply causal mask (upper triangle → -inf)
static void apply_causal_mask(float* mat, int seq_len) {
    for (int i = 0; i < seq_len; ++i) {
        for (int j = i + 1; j < seq_len; ++j) {
            mat[i * seq_len + j] = -FLT_MAX;
        }
    }
}

// ═══════════════════════════════════════
// Public API
// ═══════════════════════════════════════

void tars_diff_attention(
    const float* Q1,
    const float* Q2,
    const float* K1,
    const float* K2,
    const float* V,
    float lambda_,
    int seq_len,
    int dim,
    float* out,
    float* scratch
) {
    int ss = seq_len * seq_len;
    float scale = 1.0f / sqrtf((float)dim);

    // scratch layout: [attn1 | attn2], each [seq_len × seq_len]
    float* attn1 = scratch;
    float* attn2 = scratch + ss;

    // 1. Compute attention maps
    matmul_QKt(Q1, K1, scale, seq_len, seq_len, dim, attn1);
    matmul_QKt(Q2, K2, scale, seq_len, seq_len, dim, attn2);

    // 2. Apply causal mask
    apply_causal_mask(attn1, seq_len);
    apply_causal_mask(attn2, seq_len);

    // 3. Softmax
    softmax_rows(attn1, seq_len, seq_len);
    softmax_rows(attn2, seq_len, seq_len);

    // 4. Differential: attn = attn1 - λ·attn2
    for (int i = 0; i < ss; ++i) {
        attn1[i] -= lambda_ * attn2[i];
    }

    // 5. Output: out = attn @ V
    // out[i][d] = Σ_j attn[i][j] * V[j][d]
    memset(out, 0, (size_t)seq_len * dim * sizeof(float));

    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < seq_len; ++j) {
            float a = attn1[i * seq_len + j];
            if (a == 0.0f) continue;  // sparsity skip

            const float* vj = V + j * dim;
            float* oi = out + i * dim;

#if USE_AVX2
            __m256 va = _mm256_set1_ps(a);
            int d = 0;
            for (; d + 7 < dim; d += 8) {
                __m256 vv = _mm256_loadu_ps(vj + d);
                __m256 vo = _mm256_loadu_ps(oi + d);
                vo = _mm256_fmadd_ps(va, vv, vo);
                _mm256_storeu_ps(oi + d, vo);
            }
            for (; d < dim; ++d) {
                oi[d] += a * vj[d];
            }
#else
            for (int d = 0; d < dim; ++d) {
                oi[d] += a * vj[d];
            }
#endif
        }
    }
}
