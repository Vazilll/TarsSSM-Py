/*
 * WKV-7 State Update — RWKV-7 Generalized Delta Rule
 *
 * S' = S · (diag(w) + a^T · b) + v^T · k
 *
 * Non-diagonal transition matrix: the a^T·b rank-1 outer product
 * allows cross-channel state mixing (vs diagonal-only in RWKV-6).
 *
 * Python reference: brain/mamba2/core/ssd.py → _wkv_step()
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

/// WKV-7 single-step state update:
///   S' = S · (diag(w) + a^T · b) + v^T · k
///   y  = r ⊙ (S' · k)
///
/// @param S          State matrix [dim × dim], float32 (row-major)
/// @param w          Decay vector [dim], float32 (diagonal part)
/// @param a          Alpha vector [dim], float32 (transition row)
/// @param b          Beta vector [dim], float32 (transition col)
/// @param v          Value vector [dim], float32
/// @param k          Key vector [dim], float32
/// @param r          Receptance vector [dim], float32
/// @param dim        State dimension
/// @param S_out      New state [dim × dim], float32 (can alias S)
/// @param y_out      Output vector [dim], float32
TARS_API void tars_wkv7_step(
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
);

/// WKV-7 sequential scan over T steps.
///
/// @param S          Initial state matrix [dim × dim]
/// @param r          Receptance [seq_len, dim]
/// @param k          Keys [seq_len, dim]
/// @param v          Values [seq_len, dim]
/// @param w          Decay [seq_len, dim]
/// @param bonus      Learning rate gate [seq_len, dim]
/// @param seq_len    Sequence length
/// @param dim        State dimension
/// @param S_out      Final state [dim × dim]
/// @param Y          Output [seq_len, dim]
TARS_API void tars_wkv7_scan(
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
);

#ifdef __cplusplus
}
#endif
