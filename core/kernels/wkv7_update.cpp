/*
 * WKV-7 State Update — AVX2 Implementation
 *
 * Generalized Delta Rule (RWKV-7):
 *   S' = S · (diag(w) + a^T · b) + v^T · k
 *   y  = r ⊙ (S' · k)
 *
 * Key computation: the transition matrix T = diag(w) + outer(a, b)
 * is NOT formed explicitly.  Instead we compute S·diag(w) and
 * S·outer(a,b) separately (the latter via two matvecs).
 *
 * Complexity: O(dim²) per step — dominated by the S·k matvec.
 *
 * Python reference: brain/mamba2/core/ssd.py → _wkv_step()
 *
 * Agent 1 — Week 2
 */

#include "wkv7_update.h"
#include <cstring>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// ═══════════════════════════════════════
// Helper: element-wise row scaling
// ═══════════════════════════════════════

static inline void row_scale_add_outer(
    const float* S,          // [dim × dim]
    const float* w,          // [dim] diagonal decay
    const float* alpha,      // [dim] learning rate gate (bonus)
    const float* k,          // [dim] key
    const float* v,          // [dim] value
    int dim,
    float* S_out             // [dim × dim]
) {
    // BUG-1 fix: aligned with Python _wkv_step (delta rule)
    //
    // Step 1: S_decay[i][j] = S[i][j] * w[j]        (diagonal decay)
    // Step 2: Sk[i] = Σ_j S[i][j] * k[j]            (prediction = S · k)
    // Step 3: delta[i] = v[i] - Sk[i]                (prediction error)
    // Step 4: S_out[i][j] = S_decay[i][j] + alpha[i] * k[i] * delta[j]
    //                                        ^^^ outer(alpha*k, delta)
    //
    // NOTE: The outer product dimensions follow Python exactly:
    //   k_t.unsqueeze(-1) * delta.unsqueeze(-2)  →  outer[i][j] = k[i] * delta[j]
    //   then multiplied by alpha (b_t) per-row

    // First pass: compute Sk[i] = S[i] · k  (matvec for prediction)
    // and the decayed state
    float Sk[1024];  // stack-allocate (dim should be ≤ 1024)

    for (int i = 0; i < dim; ++i) {
        const float* row = S + i * dim;
        float* out_row = S_out + i * dim;
        float sk_i = 0.0f;

#if USE_AVX2
        {
            __m256 vsk = _mm256_setzero_ps();
            int j = 0;
            for (; j + 7 < dim; j += 8) {
                __m256 vs_ij = _mm256_loadu_ps(row + j);
                __m256 vw_j = _mm256_loadu_ps(w + j);
                __m256 vk_j = _mm256_loadu_ps(k + j);

                // S_decay[i][j] = S[i][j] * w[j]
                __m256 scaled = _mm256_mul_ps(vs_ij, vw_j);
                _mm256_storeu_ps(out_row + j, scaled);

                // Sk[i] += S[i][j] * k[j]
                vsk = _mm256_fmadd_ps(vs_ij, vk_j, vsk);
            }
            // horizontal sum
            __m128 hi = _mm256_extractf128_ps(vsk, 1);
            __m128 lo = _mm256_castps256_ps128(vsk);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            sk_i = _mm_cvtss_f32(s);
            for (; j < dim; ++j) {
                out_row[j] = row[j] * w[j];
                sk_i += row[j] * k[j];
            }
        }
#else
        for (int j = 0; j < dim; ++j) {
            out_row[j] = row[j] * w[j];
            sk_i += row[j] * k[j];
        }
#endif
        Sk[i] = sk_i;
    }

    // Second pass: delta[i] = v[i] - Sk[i], then S_out += alpha[i]*k[i]*delta[j]
    // Python: state += b_t.unsqueeze(-1) * (k_t.unsqueeze(-1) * delta.unsqueeze(-2))
    //       = for each i,j: += alpha[i] * k[i] * delta[j]
    for (int i = 0; i < dim; ++i) {
        float* out_row = S_out + i * dim;
        float delta_i = v[i] - Sk[i];  // prediction error for dim i
        float ak_i = alpha[i] * k[i];  // alpha[i] * k[i]

#if USE_AVX2
        {
            // We need outer(alpha*k, delta), so out_row[j] += ak_i * delta_j
            // But delta is per-element of the OTHER dimension...
            // Wait: Python does outer(k, delta) where k is [S] and delta is [S]
            // Result[i][j] = k[i] * delta[j]
            // Then alpha[i] * Result[i][j] = alpha[i] * k[i] * delta[j]

            // So we precompute alpha[i]*k[i] and iterate over j (delta[j])
            // But delta[j] = v[j] - Sk[j], so we need all Sk first (done above)
            __m256 vak_i = _mm256_set1_ps(ak_i);
            int j = 0;
            for (; j + 7 < dim; j += 8) {
                __m256 vout = _mm256_loadu_ps(out_row + j);
                // delta[j] = v[j] - Sk[j]
                __m256 vv_j = _mm256_loadu_ps(v + j);
                __m256 vSk_j = _mm256_loadu_ps(Sk + j);
                __m256 vdelta_j = _mm256_sub_ps(vv_j, vSk_j);
                // out += ak_i * delta[j]
                vout = _mm256_fmadd_ps(vak_i, vdelta_j, vout);
                _mm256_storeu_ps(out_row + j, vout);
            }
            for (; j < dim; ++j) {
                float delta_j = v[j] - Sk[j];
                out_row[j] += ak_i * delta_j;
            }
        }
#else
        for (int j = 0; j < dim; ++j) {
            float delta_j = v[j] - Sk[j];
            out_row[j] += ak_i * delta_j;
        }
#endif
    }
}

// ═══════════════════════════════════════
// Readout: y = r ⊙ (S · k)
// ═══════════════════════════════════════

static inline void readout(
    const float* S,   // [dim × dim]
    const float* k,   // [dim]
    const float* r,   // [dim]
    int dim,
    float* y          // [dim]
) {
    for (int i = 0; i < dim; ++i) {
        const float* row = S + i * dim;
        float dot = 0.0f;

#if USE_AVX2
        {
            __m256 vacc = _mm256_setzero_ps();
            int j = 0;
            for (; j + 7 < dim; j += 8) {
                __m256 vs = _mm256_loadu_ps(row + j);
                __m256 vk = _mm256_loadu_ps(k + j);
                vacc = _mm256_fmadd_ps(vs, vk, vacc);
            }
            __m128 hi = _mm256_extractf128_ps(vacc, 1);
            __m128 lo = _mm256_castps256_ps128(vacc);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            dot = _mm_cvtss_f32(s);
            for (; j < dim; ++j) {
                dot += row[j] * k[j];
            }
        }
#else
        for (int j = 0; j < dim; ++j) {
            dot += row[j] * k[j];
        }
#endif
        y[i] = r[i] * dot;
    }
}

// ═══════════════════════════════════════
// Public API
// ═══════════════════════════════════════

void tars_wkv7_step(
    const float* S,
    const float* w,
    const float* a,
    const float* b,
    const float* v,
    const float* k,
    const float* r,
    int dim,
    float* S_out,
    float* y_out
) {
    // BUG-1 fix: delta rule — S' = S·diag(w) + α·outer(k, v - S·k)
    // 'b' parameter kept in signature for ABI compatibility but
    // we pass 'a' as alpha (learning rate) and 'k' directly.
    // The old formula S' = S·(diag(w) + outer(a,b)) + outer(v,k) is gone.
    row_scale_add_outer(S, w, a, k, v, dim, S_out);

    // 2. Readout: y = r ⊙ (S' · k)
    readout(S_out, k, r, dim, y_out);
}


void tars_wkv7_scan(
    float* S,
    const float* r,
    const float* k,
    const float* v,
    const float* w,
    const float* bonus,
    int seq_len,
    int dim,
    float* S_out,
    float* Y
) {
    // Delta rule scan: bonus = alpha (learning rate gate)
    // S' = S·diag(w) + α·outer(k, v - S·k)

    for (int t = 0; t < seq_len; ++t) {
        const float* r_t = r + t * dim;
        const float* k_t = k + t * dim;
        const float* v_t = v + t * dim;
        const float* w_t = w + t * dim;
        const float* a_t = bonus + t * dim;  // bonus = alpha (learning rate)
        float* y_t = Y + t * dim;

        tars_wkv7_step(S, w_t, a_t, k_t, v_t, k_t, r_t, dim, S_out, y_t);

        // Copy S_out → S for next step
        memcpy(S, S_out, (size_t)dim * dim * sizeof(float));
    }
}

