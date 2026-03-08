/*
 * SSD Scan — AVX2 Implementation
 *
 * s_{t+1} = γ · s_t + B · x_t      (state update)
 * y_t     = C · s_t                 (readout)
 *
 * Two entry points:
 *   tars_ssd_scan_step  — single token (autoregressive inference)
 *   tars_ssd_scan_seq   — full sequence (prefill / training)
 *
 * AVX2 path: FMA for γ·s + B·x in 8-wide vectors.
 *
 * Python reference: brain/mamba2/core/ssd.py → ssd_step()
 *
 * Agent 1 — Week 2
 */

#include "ssd_scan.h"
#include <cmath>

#if defined(TARS_HAS_AVX2) || defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX2 1
#else
#define USE_AVX2 0
#endif

// ═══════════════════════════════════════
// Single-step: s' = γ·s + B·x
// ═══════════════════════════════════════

void tars_ssd_scan_step(
    const float* state,
    const float* gamma,
    const float* B,
    const float* x,
    int dim,
    float* out_state
) {
#if USE_AVX2
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 vs = _mm256_loadu_ps(state + i);
        __m256 vg = _mm256_loadu_ps(gamma + i);
        __m256 vb = _mm256_loadu_ps(B + i);
        __m256 vx = _mm256_loadu_ps(x + i);

        // s' = γ·s + B·x  (FMA: fused multiply-add)
        __m256 result = _mm256_fmadd_ps(vg, vs, _mm256_mul_ps(vb, vx));
        _mm256_storeu_ps(out_state + i, result);
    }
    // Scalar tail
    for (; i < dim; ++i) {
        out_state[i] = gamma[i] * state[i] + B[i] * x[i];
    }
#else
    for (int i = 0; i < dim; ++i) {
        out_state[i] = gamma[i] * state[i] + B[i] * x[i];
    }
#endif
}

// ═══════════════════════════════════════
// Sequence scan: recurrent over T steps
// ═══════════════════════════════════════

void tars_ssd_scan_seq(
    float* states,
    const float* gamma,
    const float* B,
    const float* X,
    const float* C,
    int seq_len,
    int dim,
    float* Y
) {
    // states layout: [seq_len+1, dim]
    //   states[0*dim .. dim-1]   = initial state (input)
    //   states[t*dim .. (t+1)*dim-1] = state after step t

    for (int t = 0; t < seq_len; ++t) {
        const float* s_prev = states + t * dim;
        float* s_next = states + (t + 1) * dim;
        const float* g_t = gamma + t * dim;
        const float* b_t = B + t * dim;
        const float* x_t = X + t * dim;

        // State update: s_{t+1} = γ_t · s_t + B_t · x_t
        tars_ssd_scan_step(s_prev, g_t, b_t, x_t, dim, s_next);

        // Readout: y_t = Σ(C_t[i] · s_{t+1}[i])
        const float* c_t = C + t * dim;
        float dot = 0.0f;

#if USE_AVX2
        {
            __m256 vacc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 7 < dim; i += 8) {
                __m256 vc = _mm256_loadu_ps(c_t + i);
                __m256 vs = _mm256_loadu_ps(s_next + i);
                vacc = _mm256_fmadd_ps(vc, vs, vacc);
            }
            // Horizontal sum
            __m128 hi = _mm256_extractf128_ps(vacc, 1);
            __m128 lo = _mm256_castps256_ps128(vacc);
            __m128 s = _mm_add_ps(lo, hi);
            s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            dot = _mm_cvtss_f32(s);
            for (; i < dim; ++i) {
                dot += c_t[i] * s_next[i];
            }
        }
#else
        for (int i = 0; i < dim; ++i) {
            dot += c_t[i] * s_next[i];
        }
#endif
        Y[t] = dot;
    }
}
