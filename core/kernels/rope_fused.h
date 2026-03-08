/*
 * RoPE Fused — Rotary Position Embeddings (θ=500K)
 *
 * QK-Norm → RoPE rotation (correct order per K6).
 *
 * For each position p and dimension pair (2i, 2i+1):
 *   θ_i = p / (base^(2i/d))
 *   q_rot[2i]   = q[2i] * cos(θ_i) - q[2i+1] * sin(θ_i)
 *   q_rot[2i+1] = q[2i] * sin(θ_i) + q[2i+1] * cos(θ_i)
 *
 * Python reference: training/custom_kernels.py → _rope_python()
 *
 * Agent 1 — Week 2
 */

#pragma once

#ifdef _WIN32
    #ifdef TARS_BUILDING_DLL
        #define TARS_API __declspec(dllexport)
    #else
        #define TARS_API __declspec(dllimport)
    #endif
#else
    #define TARS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// Precompute RoPE frequency table.
///
/// @param freqs     Output [max_seq, half_dim], float32
/// @param base      θ base frequency (500000.0 for TARS)
/// @param dim       Head dimension (must be even)
/// @param max_seq   Maximum sequence length
TARS_API void tars_rope_precompute(
    float* freqs,
    double base,
    int dim,
    int max_seq
);

/// Apply RoPE rotation to Q and K vectors (in-place).
///
/// @param Q         Query [seq_len, dim], float32 (modified in-place)
/// @param K         Key [seq_len, dim], float32 (modified in-place)
/// @param freqs     Precomputed frequencies [max_seq, dim/2]
/// @param seq_len   Current sequence length
/// @param dim       Head dimension (even)
/// @param offset    Position offset for cached generation
TARS_API void tars_rope_apply(
    float* Q,
    float* K,
    const float* freqs,
    int seq_len,
    int dim,
    int offset
);

#ifdef __cplusplus
}
#endif
