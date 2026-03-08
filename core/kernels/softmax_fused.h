/*
 * Online Softmax — TARS Inference Core
 *
 * Numerically stable softmax with max-subtract trick.
 * Single-pass online algorithm where possible.
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

/// Numerically stable softmax: out[i] = exp(x[i] - max(x)) / Σexp(x[j] - max(x))
/// @param x    Input logits [n]
/// @param n    Vector length
/// @param out  Output probabilities [n] (can alias x for in-place)
TARS_API void tars_softmax(const float* x, int n, float* out);

/// Argmax: returns index of maximum element
/// @param x    Input vector [n]
/// @param n    Vector length
/// @return     Index of max element
TARS_API int tars_argmax(const float* x, int n);

/// Max value of softmax output (confidence score for early exit)
/// @param x    Input logits [n]
/// @param n    Vector length
/// @return     max(softmax(x))
TARS_API float tars_softmax_max(const float* x, int n);

#ifdef __cplusplus
}
#endif
