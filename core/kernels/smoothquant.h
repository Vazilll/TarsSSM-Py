/*
 * SmoothQuant INT8 — Activation-Aware Quantization
 *
 * Y = (X · S⁻¹) · (S · W)
 *
 * SmoothQuant shifts the quantization difficulty from activations
 * to weights using a per-channel scaling factor S.
 *
 * S[j] = max(|X[:,j]|)^α / max(|W[j,:]|)^(1-α)
 *
 * After smoothing, both X and W can be safely quantized to INT8.
 *
 * Python reference: brain/mamba2/core/bitnet.py (SmoothQuant path)
 *
 * Agent 1 — Week 3
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

/// Compute SmoothQuant scaling factors from activation and weight statistics.
///
/// S[j] = max(|X[:,j]|)^α / max(|W[j,:]|)^(1-α)
///
/// @param act_scales   Per-channel max(|activation|) [dim], float32
/// @param wt_scales    Per-channel max(|weight|) [dim], float32
/// @param alpha        Balance factor (0.5 = balanced)
/// @param dim          Number of channels
/// @param S_out        Output scaling factors [dim], float32
TARS_API void tars_smoothquant_compute_scales(
    const float* act_scales,
    const float* wt_scales,
    float alpha,
    int dim,
    float* S_out
);

/// Apply SmoothQuant: Y = (X · diag(S⁻¹)) @ (diag(S) · W)
/// All in INT8 with float accumulation.
///
/// @param X_int8       Quantized activations [M, K], int8
/// @param W_int8       Quantized weights [K, N], int8 (pre-smoothed)
/// @param x_scale      Activation quantization scale (float)
/// @param w_scale      Weight quantization scale (float)
/// @param M            Batch size (rows of X)
/// @param K            Inner dimension
/// @param N            Output dimension (cols of W)
/// @param Y            Output [M, N], float32
TARS_API void tars_smoothquant_matmul_int8(
    const int8_t* X_int8,
    const int8_t* W_int8,
    float x_scale,
    float w_scale,
    int M, int K, int N,
    float* Y
);

/// Quantize a float vector to INT8 with symmetric quantization.
///
/// @param x_float      Input [numel], float32
/// @param numel        Number of elements
/// @param x_int8       Output [numel], int8
/// @return             Scale factor (max(|x|) / 127)
TARS_API float tars_quantize_int8(
    const float* x_float,
    int numel,
    int8_t* x_int8
);

#ifdef __cplusplus
}
#endif
