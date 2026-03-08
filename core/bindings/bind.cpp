/*
 * pybind11 Bindings — TARS Core Python Interface
 *
 * Exposes ALL C++ kernels as Python functions via numpy arrays.
 * Module name: tars_core
 *
 * Usage:
 *   import tars_core
 *   out = tars_core.rmsnorm(x, gamma, eps)
 *   out = tars_core.bitnet_matmul(W_ternary, x, alpha)
 *   out = tars_core.ssd_scan_step(state, gamma, B, x)
 *   out = tars_core.wkv7_step(S, w, a, b, v, k, r)
 *   out = tars_core.swiglu_fused(gate, value, sparsity)
 *   Q, K = tars_core.rope_apply(Q, K, freqs, offset)
 *
 * Agent 1 — Week 1-4
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cstring>

#include "kernels/rmsnorm_fused.h"
#include "kernels/embedding_lookup.h"
#include "kernels/bitnet_matmul.h"
#include "kernels/softmax_fused.h"
#include "kernels/ssd_scan.h"
#include "kernels/wkv7_update.h"
#include "kernels/swiglu_fused.h"
#include "kernels/rope_fused.h"
#include "kernels/attention_diff.h"
#include "kernels/smoothquant.h"
#include "runtime/arena.h"

namespace py = pybind11;


// ═══════════════════════════════════════
// Week 1 Kernel wrappers
// ═══════════════════════════════════════

/// RMSNorm: out = gamma * x / sqrt(mean(x²) + eps)
static py::array_t<float> py_rmsnorm(
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    py::array_t<float, py::array::c_style | py::array::forcecast> gamma,
    float eps
) {
    auto x_buf = x.request();
    auto g_buf = gamma.request();
    int dim = (int)x_buf.size;
    
    if (g_buf.size != dim) {
        throw std::runtime_error("gamma must have same size as x");
    }
    
    auto result = py::array_t<float>(dim);
    auto r_buf = result.request();
    
    tars_rmsnorm(
        static_cast<const float*>(x_buf.ptr),
        static_cast<const float*>(g_buf.ptr),
        eps, dim,
        static_cast<float*>(r_buf.ptr)
    );
    
    return result;
}

/// Embedding lookup: out = table[token_id] * sqrt(d_model)
static py::array_t<float> py_embedding_lookup(
    py::array_t<float, py::array::c_style | py::array::forcecast> table,
    int token_id
) {
    auto buf = table.request();
    if (buf.ndim != 2) {
        throw std::runtime_error("table must be 2D [vocab_size × d_model]");
    }
    int d_model = (int)buf.shape[1];
    
    auto result = py::array_t<float>(d_model);
    auto r_buf = result.request();
    
    tars_embedding_lookup(
        static_cast<const float*>(buf.ptr),
        token_id, d_model,
        static_cast<float*>(r_buf.ptr)
    );
    
    return result;
}

/// BitNet ternary matmul: out = alpha * W @ x
static py::array_t<float> py_bitnet_matmul(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> W,
    py::array_t<float, py::array::c_style | py::array::forcecast> x,
    float alpha
) {
    auto w_buf = W.request();
    auto x_buf = x.request();
    
    if (w_buf.ndim != 2) {
        throw std::runtime_error("W must be 2D [rows × cols]");
    }
    int rows = (int)w_buf.shape[0];
    int cols = (int)w_buf.shape[1];
    
    if (x_buf.size != cols) {
        throw std::runtime_error("x dimension must match W columns");
    }
    
    auto result = py::array_t<float>(rows);
    auto r_buf = result.request();
    
    tars_bitnet_matmul(
        static_cast<const int8_t*>(w_buf.ptr),
        static_cast<const float*>(x_buf.ptr),
        alpha, rows, cols,
        static_cast<float*>(r_buf.ptr)
    );
    
    return result;
}

/// Softmax: out = exp(x - max(x)) / sum(exp(x - max(x)))
static py::array_t<float> py_softmax(
    py::array_t<float, py::array::c_style | py::array::forcecast> x
) {
    auto x_buf = x.request();
    int n = (int)x_buf.size;
    
    auto result = py::array_t<float>(n);
    auto r_buf = result.request();
    
    tars_softmax(
        static_cast<const float*>(x_buf.ptr), n,
        static_cast<float*>(r_buf.ptr)
    );
    
    return result;
}

/// Argmax
static int py_argmax(
    py::array_t<float, py::array::c_style | py::array::forcecast> x
) {
    auto x_buf = x.request();
    return tars_argmax(static_cast<const float*>(x_buf.ptr), (int)x_buf.size);
}

/// Quantize float weights to ternary
static std::pair<py::array_t<int8_t>, float> py_quantize_ternary(
    py::array_t<float, py::array::c_style | py::array::forcecast> W
) {
    auto w_buf = W.request();
    int numel = (int)w_buf.size;
    
    auto result = py::array_t<int8_t>(numel);
    auto r_buf = result.request();
    
    float scale = tars_quantize_ternary(
        static_cast<const float*>(w_buf.ptr),
        static_cast<int8_t*>(r_buf.ptr),
        numel
    );
    
    if (w_buf.ndim == 2) {
        result.resize({w_buf.shape[0], w_buf.shape[1]});
    }
    
    return {result, scale};
}


// ═══════════════════════════════════════
// Week 2-3 Kernel wrappers
// ═══════════════════════════════════════

/// SSD scan step: s' = gamma * s + B * x
static py::array_t<float> py_ssd_scan_step(
    py::array_t<float, py::array::c_style | py::array::forcecast> state,
    py::array_t<float, py::array::c_style | py::array::forcecast> gamma,
    py::array_t<float, py::array::c_style | py::array::forcecast> B,
    py::array_t<float, py::array::c_style | py::array::forcecast> x
) {
    auto s_buf = state.request();
    int dim = (int)s_buf.size;

    auto result = py::array_t<float>(dim);
    auto r_buf = result.request();

    tars_ssd_scan_step(
        static_cast<const float*>(s_buf.ptr),
        static_cast<const float*>(gamma.request().ptr),
        static_cast<const float*>(B.request().ptr),
        static_cast<const float*>(x.request().ptr),
        dim,
        static_cast<float*>(r_buf.ptr)
    );

    return result;
}

/// WKV-7 step: S' = S·(diag(w) + aᵀb) + vᵀk, y = r⊙(S'·k)
static std::pair<py::array_t<float>, py::array_t<float>> py_wkv7_step(
    py::array_t<float, py::array::c_style | py::array::forcecast> S,
    py::array_t<float, py::array::c_style | py::array::forcecast> w,
    py::array_t<float, py::array::c_style | py::array::forcecast> a,
    py::array_t<float, py::array::c_style | py::array::forcecast> b,
    py::array_t<float, py::array::c_style | py::array::forcecast> v,
    py::array_t<float, py::array::c_style | py::array::forcecast> k,
    py::array_t<float, py::array::c_style | py::array::forcecast> r
) {
    auto w_buf = w.request();
    int dim = (int)w_buf.size;

    auto S_out = py::array_t<float>(dim * dim);
    auto y_out = py::array_t<float>(dim);

    tars_wkv7_step(
        static_cast<const float*>(S.request().ptr),
        static_cast<const float*>(w_buf.ptr),
        static_cast<const float*>(a.request().ptr),
        static_cast<const float*>(b.request().ptr),
        static_cast<const float*>(v.request().ptr),
        static_cast<const float*>(k.request().ptr),
        static_cast<const float*>(r.request().ptr),
        dim,
        static_cast<float*>(S_out.request().ptr),
        static_cast<float*>(y_out.request().ptr)
    );

    S_out.resize({(long)dim, (long)dim});
    return {S_out, y_out};
}

/// SwiGLU fused: y = SiLU(gate) ⊙ value
static py::array_t<float> py_swiglu_fused(
    py::array_t<float, py::array::c_style | py::array::forcecast> gate,
    py::array_t<float, py::array::c_style | py::array::forcecast> value,
    float sparsity_threshold
) {
    auto g_buf = gate.request();
    int dim = (int)g_buf.size;

    auto result = py::array_t<float>(dim);
    auto r_buf = result.request();

    tars_swiglu_fused(
        static_cast<const float*>(g_buf.ptr),
        static_cast<const float*>(value.request().ptr),
        dim, sparsity_threshold,
        static_cast<float*>(r_buf.ptr)
    );

    return result;
}

/// RoPE precompute: build frequency table
static py::array_t<float> py_rope_precompute(
    double base, int dim, int max_seq
) {
    int half_d = dim / 2;
    auto result = py::array_t<float>(max_seq * half_d);
    auto r_buf = result.request();

    tars_rope_precompute(
        static_cast<float*>(r_buf.ptr),
        base, dim, max_seq
    );

    result.resize({(long)max_seq, (long)half_d});
    return result;
}

/// RoPE apply: rotate Q, K in-place
static std::pair<py::array_t<float>, py::array_t<float>> py_rope_apply(
    py::array_t<float, py::array::c_style | py::array::forcecast> Q,
    py::array_t<float, py::array::c_style | py::array::forcecast> K,
    py::array_t<float, py::array::c_style | py::array::forcecast> freqs,
    int offset
) {
    auto q_buf = Q.request();
    int seq_len = (q_buf.ndim >= 2) ? (int)q_buf.shape[0] : 1;
    int dim = (q_buf.ndim >= 2) ? (int)q_buf.shape[1] : (int)q_buf.size;

    // Copy to outputs (non-destructive)
    auto Q_out = py::array_t<float>(q_buf.size);
    auto K_out = py::array_t<float>(K.request().size);
    memcpy(Q_out.request().ptr, q_buf.ptr, q_buf.size * sizeof(float));
    memcpy(K_out.request().ptr, K.request().ptr, K.request().size * sizeof(float));

    tars_rope_apply(
        static_cast<float*>(Q_out.request().ptr),
        static_cast<float*>(K_out.request().ptr),
        static_cast<const float*>(freqs.request().ptr),
        seq_len, dim, offset
    );

    return {Q_out, K_out};
}

/// SmoothQuant INT8 quantize
static std::pair<py::array_t<int8_t>, float> py_quantize_int8(
    py::array_t<float, py::array::c_style | py::array::forcecast> x
) {
    auto x_buf = x.request();
    int numel = (int)x_buf.size;

    auto result = py::array_t<int8_t>(numel);
    auto r_buf = result.request();

    float scale = tars_quantize_int8(
        static_cast<const float*>(x_buf.ptr),
        numel,
        static_cast<int8_t*>(r_buf.ptr)
    );

    return {result, scale};
}


// ═══════════════════════════════════════
// Python module definition
// ═══════════════════════════════════════

PYBIND11_MODULE(tars_core, m) {
    m.doc() = "TARS C++ Inference Core — 12 AVX2 kernels";
    
    // Week 1 Kernels
    m.def("rmsnorm", &py_rmsnorm,
          "RMSNorm: out = gamma * x / sqrt(mean(x²) + eps)",
          py::arg("x"), py::arg("gamma"), py::arg("eps") = 1e-8f);
    
    m.def("embedding_lookup", &py_embedding_lookup,
          "Embedding lookup with sqrt(d_model) scaling",
          py::arg("table"), py::arg("token_id"));
    
    m.def("bitnet_matmul", &py_bitnet_matmul,
          "Ternary matmul: out = alpha * W @ x (W in {-1,0,+1})",
          py::arg("W_ternary"), py::arg("x"), py::arg("alpha") = 1.0f);
    
    m.def("softmax", &py_softmax,
          "Numerically stable softmax",
          py::arg("x"));
    
    m.def("argmax", &py_argmax,
          "Argmax: index of maximum element",
          py::arg("x"));
    
    m.def("quantize_ternary", &py_quantize_ternary,
          "Quantize float weights to ternary {-1, 0, +1}. Returns (W_ternary, scale).",
          py::arg("W"));

    // Week 2-3 Kernels
    m.def("ssd_scan", &py_ssd_scan_step,
          "SSD scan step: s' = γ·s + B·x (AVX2 FMA)",
          py::arg("state"), py::arg("gamma"), py::arg("B"), py::arg("x"));

    m.def("wkv7_update", &py_wkv7_step,
          "WKV-7 step: S'=S·(diag(w)+aᵀb)+vᵀk, y=r⊙(S'·k)",
          py::arg("S"), py::arg("w"), py::arg("a"), py::arg("b"),
          py::arg("v"), py::arg("k"), py::arg("r"));

    m.def("swiglu_fused", &py_swiglu_fused,
          "SwiGLU: y = SiLU(gate)⊙value + Double Sparsity",
          py::arg("gate"), py::arg("value"), py::arg("sparsity_threshold") = 0.0f);

    m.def("rope_precompute", &py_rope_precompute,
          "Precompute RoPE frequency table",
          py::arg("base") = 500000.0, py::arg("dim") = 64, py::arg("max_seq") = 32768);

    m.def("rope", &py_rope_apply,
          "Apply RoPE rotation to Q, K",
          py::arg("Q"), py::arg("K"), py::arg("freqs"), py::arg("offset") = 0);

    m.def("quantize_int8", &py_quantize_int8,
          "SmoothQuant symmetric INT8 quantization. Returns (x_int8, scale).",
          py::arg("x"));

    // Arena
    py::class_<TarsArena>(m, "Arena")
        .def(py::init<size_t>(), py::arg("capacity") = TARS_ARENA_DEFAULT_CAPACITY)
        .def("alloc_bytes", [](TarsArena& a, size_t n) -> uintptr_t {
            return (uintptr_t)a.alloc(n);
        })
        .def("reset", &TarsArena::reset)
        .def("used", &TarsArena::used)
        .def("capacity", &TarsArena::capacity)
        .def("remaining", &TarsArena::remaining);
    
    // Version info
    m.attr("__version__") = "0.2.0";
    m.attr("n_kernels") = 12;
}
