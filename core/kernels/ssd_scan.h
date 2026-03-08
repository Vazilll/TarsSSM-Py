/*
 * SSD Scan — State-Space Dual Discrete Scan
 *
 * s_{t+1} = γ · s_t + B · x_t
 * y_t     = C · s_t
 *
 * AVX2: vectorized multiply-accumulate, 8 floats at a time.
 *
 * Python reference: brain/mamba2/core/ssd.py → ssd_step(), ssd_scan()
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

/// Single-step SSD scan: s_new = gamma * s + B * x
///
/// @param state     Current state [dim], float32
/// @param gamma     Decay factor [dim], float32
/// @param B         Input projection [dim], float32
/// @param x         Input vector [dim], float32
/// @param dim       State dimension
/// @param out_state New state [dim], float32 (can alias state for in-place)
TARS_API void tars_ssd_scan_step(
    const float* state,
    const float* gamma,
    const float* B,
    const float* x,
    int dim,
    float* out_state
);

/// Multi-step SSD scan over a sequence: s_t = gamma * s_{t-1} + B_t * x_t
///
/// @param states    State buffer [seq_len+1, dim] — states[0] = initial
/// @param gamma     Decay [seq_len, dim], float32
/// @param B         Input projection [seq_len, dim], float32
/// @param X         Input sequence [seq_len, dim], float32
/// @param C         Output projection [seq_len, dim], float32
/// @param seq_len   Sequence length
/// @param dim       State dimension
/// @param Y         Output sequence [seq_len, dim], float32
TARS_API void tars_ssd_scan_seq(
    float* states,
    const float* gamma,
    const float* B,
    const float* X,
    const float* C,
    int seq_len,
    int dim,
    float* Y
);

#ifdef __cplusplus
}
#endif
