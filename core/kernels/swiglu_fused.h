/*
 * SwiGLU Fused — SiLU(W₁x) ⊙ W₂x + Double Sparsity
 *
 * y = SiLU(W₁·x) ⊙ W₂·x
 * With optional sparsity: zero where |y| < ε
 *
 * SiLU(x) = x · σ(x) = x / (1 + exp(-x))
 *
 * Python reference: training/custom_kernels.py → _swiglu_python()
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

/// Fused SwiGLU: y = SiLU(gate) ⊙ value  (element-wise)
/// where gate = W₁·x and value = W₂·x are pre-computed.
///
/// @param gate     SiLU input (W₁·x) [dim], float32
/// @param value    Value input (W₂·x) [dim], float32
/// @param dim      Vector dimension
/// @param sparsity_threshold  Zero outputs where |y| < threshold (0 = disabled)
/// @param out      Output [dim], float32
/// @return         Number of non-zero elements (for sparsity stats)
TARS_API int tars_swiglu_fused(
    const float* gate,
    const float* value,
    int dim,
    float sparsity_threshold,
    float* out
);

#ifdef __cplusplus
}
#endif
