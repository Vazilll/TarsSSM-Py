/*
 * Embedding Lookup — TARS Inference Core
 *
 * Formula: out[i] = table[token_id * d_model + i] * sqrt(d_model)
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

/// Embedding lookup with √d_model scaling.
/// @param table     Embedding table [vocab_size × d_model], row-major
/// @param token_id  Token index to look up
/// @param d_model   Embedding dimension
/// @param out       Output vector [d_model]
TARS_API void tars_embedding_lookup(
    const float* table,
    int token_id,
    int d_model,
    float* out
);

#ifdef __cplusplus
}
#endif
