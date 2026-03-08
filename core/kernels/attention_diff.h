/*
 * Diff Attention — Differential Transformer
 *
 * attn = softmax(Q₁K₁ᵀ/√d) − λ · softmax(Q₂K₂ᵀ/√d)
 * output = attn @ V
 *
 * Cancels attention noise by subtracting a secondary attention map.
 *
 * Python reference: training/custom_kernels.py → _diff_attention_python()
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

#ifdef __cplusplus
extern "C" {
#endif

/// Differential Attention: attn₁ − λ·attn₂ → matmul with V
///
/// All inputs are single-head, 2D matrices [seq_len × dim].
///
/// @param Q1, Q2    Query matrices [seq_len, dim]
/// @param K1, K2    Key matrices [seq_len, dim]
/// @param V         Value matrix [seq_len, dim]
/// @param lambda_   Subtraction weight (typically 0.5, learnable)
/// @param seq_len   Sequence length
/// @param dim       Head dimension
/// @param out       Output [seq_len, dim]
/// @param scratch   Scratch buffer, size >= 2 * seq_len * seq_len floats
TARS_API void tars_diff_attention(
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
);

#ifdef __cplusplus
}
#endif
