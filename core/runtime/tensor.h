/*
 * TarsTensor — Lightweight tensor descriptor for TARS inference core.
 * Header-only. No allocations — just metadata pointing to existing memory.
 *
 * Agent 1 — Week 1
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cassert>

// ═══════════════════════════════════════
// Dtype enum
// ═══════════════════════════════════════
enum TarsDtype : uint8_t {
    TARS_FLOAT32 = 0,
    TARS_INT8    = 1,
    TARS_TERNARY = 2,  // {-1, 0, +1} stored as int8
    TARS_UINT8   = 3,
};

static inline size_t tars_dtype_size(TarsDtype dt) {
    switch (dt) {
        case TARS_FLOAT32: return 4;
        case TARS_INT8:    return 1;
        case TARS_TERNARY: return 1;
        case TARS_UINT8:   return 1;
        default:           return 0;
    }
}

// ═══════════════════════════════════════
// TarsTensor — non-owning view
// ═══════════════════════════════════════
static constexpr int TARS_MAX_DIMS = 4;

struct TarsTensor {
    void*      data;
    int        dims[TARS_MAX_DIMS];    // shape
    int        stride[TARS_MAX_DIMS];  // stride in elements
    int        ndim;
    TarsDtype  dtype;

    // ── Helpers ──

    /// Total number of elements
    inline int numel() const {
        int n = 1;
        for (int i = 0; i < ndim; ++i) n *= dims[i];
        return n;
    }

    /// Total size in bytes
    inline size_t nbytes() const {
        return (size_t)numel() * tars_dtype_size(dtype);
    }

    /// Flat element access (float)
    inline float& at_f32(int idx) {
        assert(dtype == TARS_FLOAT32);
        return static_cast<float*>(data)[idx];
    }
    inline const float& at_f32(int idx) const {
        assert(dtype == TARS_FLOAT32);
        return static_cast<const float*>(data)[idx];
    }

    /// Flat element access (int8)
    inline int8_t& at_i8(int idx) {
        assert(dtype == TARS_INT8 || dtype == TARS_TERNARY);
        return static_cast<int8_t*>(data)[idx];
    }
    inline const int8_t& at_i8(int idx) const {
        assert(dtype == TARS_INT8 || dtype == TARS_TERNARY);
        return static_cast<const int8_t*>(data)[idx];
    }

    /// Float pointer shorthand
    inline float* f32() { return static_cast<float*>(data); }
    inline const float* f32() const { return static_cast<const float*>(data); }

    /// Int8 pointer shorthand
    inline int8_t* i8() { return static_cast<int8_t*>(data); }
    inline const int8_t* i8() const { return static_cast<const int8_t*>(data); }
};

// ═══════════════════════════════════════
// Factory helpers
// ═══════════════════════════════════════

/// Create a 1D float tensor view over existing memory
static inline TarsTensor tars_tensor_1d(float* data, int dim0) {
    TarsTensor t{};
    t.data = data;
    t.dims[0] = dim0;
    t.stride[0] = 1;
    t.ndim = 1;
    t.dtype = TARS_FLOAT32;
    return t;
}

/// Create a 2D float tensor view (row-major)
static inline TarsTensor tars_tensor_2d(float* data, int dim0, int dim1) {
    TarsTensor t{};
    t.data = data;
    t.dims[0] = dim0;
    t.dims[1] = dim1;
    t.stride[0] = dim1;
    t.stride[1] = 1;
    t.ndim = 2;
    t.dtype = TARS_FLOAT32;
    return t;
}

/// Create a 1D int8/ternary tensor view
static inline TarsTensor tars_tensor_1d_i8(int8_t* data, int dim0, TarsDtype dt = TARS_INT8) {
    TarsTensor t{};
    t.data = data;
    t.dims[0] = dim0;
    t.stride[0] = 1;
    t.ndim = 1;
    t.dtype = dt;
    return t;
}

/// Create a 2D int8/ternary tensor view (row-major)
static inline TarsTensor tars_tensor_2d_i8(int8_t* data, int dim0, int dim1, TarsDtype dt = TARS_INT8) {
    TarsTensor t{};
    t.data = data;
    t.dims[0] = dim0;
    t.dims[1] = dim1;
    t.stride[0] = dim1;
    t.stride[1] = 1;
    t.ndim = 2;
    t.dtype = dt;
    return t;
}
