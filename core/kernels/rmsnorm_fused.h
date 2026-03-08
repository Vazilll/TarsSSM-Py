/*
 * RMSNorm Fused Kernel — TARS Inference Core
 *
 * Formula: out[i] = gamma[i] * x[i] / sqrt(mean(x²) + eps)
 * 
 * Python reference: brain/mamba2/core/bitnet.py → RMSNorm.forward()
 *   x_fp32 = x.float()
 *   rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
 *   return (x_fp32 * rms).type_as(x) * self.weight
 *
 * AVX2 path: vectorized sum-of-squares, fused multiply-scale loop.
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

#ifdef __cplusplus
extern "C" {
#endif

/// RMSNorm: out[i] = gamma[i] * x[i] / sqrt(mean(x²) + eps)
/// @param x      Input vector [dim]
/// @param gamma  Scale weights [dim]
/// @param eps    Epsilon for numerical stability (1e-8 typical)
/// @param dim    Vector dimension
/// @param out    Output vector [dim] (can alias x for in-place)
TARS_API void tars_rmsnorm(
    const float* x,
    const float* gamma,
    float eps,
    int dim,
    float* out
);

#ifdef __cplusplus
}
#endif
