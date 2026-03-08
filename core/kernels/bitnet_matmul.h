/*
 * BitNet Ternary MatMul — TARS Inference Core
 *
 * Y = alpha * (W_ternary @ X)
 * Where W ∈ {-1, 0, +1}^{rows × cols}
 *
 * Key insight: with ternary weights, matmul becomes pure ADD/SUB.
 * Zero FPU usage on the hot path — only integer arithmetic.
 *
 * Python reference: brain/mamba2/core/bitnet.py → UniversalLinear.forward() 158bit mode
 *   w_q = round(clip(W / scale, -1, 1))  →  {-1, 0, +1}
 *   y = w_q @ x * scale
 *
 * Agent 1 — Week 1
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

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Ternary matrix-vector multiply: out = alpha * (W @ x)
/// W is stored as int8 with values in {-1, 0, +1}.
/// The multiply becomes pure ADD/SUB — no FPU needed.
///
/// @param W_ternary  Weight matrix [rows × cols], int8 values ∈ {-1, 0, +1}
/// @param x          Input vector [cols], float32
/// @param alpha      Scale factor (e.g., mean(|W_full|))
/// @param rows       Number of output features
/// @param cols       Number of input features
/// @param out        Output vector [rows], float32
TARS_API void tars_bitnet_matmul(
    const int8_t* W_ternary,
    const float* x,
    float alpha,
    int rows,
    int cols,
    float* out
);

/// Quantize float weights to ternary {-1, 0, +1}.
/// Returns the scale factor (mean(|W|)).
///
/// @param W_float    Input weights [rows × cols], float32
/// @param W_ternary  Output ternary weights [rows × cols], int8
/// @param numel      Total number of elements (rows * cols)
/// @return           Scale factor α = mean(|W|)
TARS_API float tars_quantize_ternary(
    const float* W_float,
    int8_t* W_ternary,
    int numel
);

#ifdef __cplusplus
}
#endif
