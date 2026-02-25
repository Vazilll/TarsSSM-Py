/*
 * OmegaCore Pure — TARS v3 High-Performance C++ Kernel
 * Zero external dependencies (no PyTorch, no PyBind11, no Python.h)
 * Compile: zig c++ -shared -O3 -mavx2 -mfma omega_core_pure.cpp -o
 * omega_core.dll
 *
 * Functions:
 *   bit_linear()     — 1.58-bit ternary MatMul with AVX2 LUT
 *   ssm_step()       — Ω-SSM state update h' = h + (Ah + Bx)·dt
 *   cayley_update()  — SO(n) Cayley Transform for gradient stability
 *   advanced_sample() — Top-P / Top-K / Min-P token sampler
 *   integral_audit() — p-convergence check (Чикулаев-Кадымов)
 *   hankel_rank()    — Hankel SVD rank-collapse detector
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// AVX2 (optional, graceful degrade)
#if defined(__AVX2__)
#include <immintrin.h>
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

// --------------------------------------------------------------------------
// Export macros
// --------------------------------------------------------------------------
#ifdef _WIN32
#define OMEGA_API extern "C" __declspec(dllexport)
#else
#define OMEGA_API extern "C" __attribute__((visibility("default")))
#endif

// --------------------------------------------------------------------------
// Global LUT for 1.58-bit ternary weight unpacking
// --------------------------------------------------------------------------
static uint32_t UNPACK_LUT[256];
static bool lut_initialized = false;

OMEGA_API void omega_init() {
  if (lut_initialized)
    return;
  for (int i = 0; i < 256; ++i) {
    int8_t w[4];
    w[0] = (int8_t)((i & 0x03) - 1);        // bits 0-1
    w[1] = (int8_t)(((i >> 2) & 0x03) - 1); // bits 2-3
    w[2] = (int8_t)(((i >> 4) & 0x03) - 1); // bits 4-5
    w[3] = (int8_t)(((i >> 6) & 0x03) - 1); // bits 6-7
    uint32_t val = 0;
    uint8_t *p = (uint8_t *)&val;
    p[0] = (uint8_t)w[0];
    p[1] = (uint8_t)w[1];
    p[2] = (uint8_t)w[2];
    p[3] = (uint8_t)w[3];
    UNPACK_LUT[i] = val;
  }
  lut_initialized = true;
}

// --------------------------------------------------------------------------
// SiLU activation (used internally)
// --------------------------------------------------------------------------
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// --------------------------------------------------------------------------
// 1. BitLinear AVX2 Forward:  Y = W_ternary * X
//    weights_packed: uint8 array, 4 ternary weights per byte
//    input: float[cols], output: float[rows]
// --------------------------------------------------------------------------
OMEGA_API void bit_linear(const float *input, int cols,
                          const uint8_t *weights_packed, int rows,
                          float scale_x, float scale_w, float *output) {
  omega_init();
  int packed_stride = (cols + 3) / 4;

  for (int r = 0; r < rows; ++r) {
    float acc = 0.0f;
    const uint8_t *row_w = weights_packed + r * packed_stride;

    for (int c = 0; c < cols; ++c) {
      int byte_idx = c / 4;
      int bit_pos = (c % 4) * 2;
      int w_val = ((row_w[byte_idx] >> bit_pos) & 0x03) - 1; // {-1, 0, 1}

      // Skip zero weights (sparsity optimization)
      if (w_val != 0) {
        acc += (float)w_val * input[c];
      }
    }
    output[r] = acc * scale_x * scale_w;
  }
}

// --------------------------------------------------------------------------
// 2. SSM State Update (Ω-SSM, Euler discretization)
//    h_next[i] = h_prev[i] + (sum_j(A[i,j]*h_prev[j]) + B[i]*x[i]) * dt
// --------------------------------------------------------------------------
OMEGA_API void ssm_step(const float *x,      // input  [dim]
                        const float *h_prev, // state  [dim]
                        const float *A,      // matrix [dim x dim]
                        const float *B,      // vector [dim]
                        float dt, int dim,
                        float *h_next // output [dim]
) {
  for (int i = 0; i < dim; ++i) {
    float drift = 0.0f;
    const float *A_row = A + i * dim;
    for (int j = 0; j < dim; ++j) {
      drift += A_row[j] * h_prev[j];
    }
    float input_drive = B[i] * x[i];
    h_next[i] = h_prev[i] + (drift + input_drive) * dt;
  }
}

// --------------------------------------------------------------------------
// 3. Cayley Transform: SO(n) update for gradient-stable recurrence
//    R = (I + S/2)^{-1} (I - S/2)  where S is skew-symmetric
//    Then h_next = R @ h_prev
//    For efficiency, uses Neumann series approximation:
//    R ≈ I + S + S²/2  (valid when ||S|| < 1)
// --------------------------------------------------------------------------
OMEGA_API void
cayley_update(const float *skew_S, // skew-symmetric matrix [dim x dim]
              const float *h_prev, // state [dim]
              int dim,
              float *h_next // output [dim]
) {
  // h_next = h_prev + S @ h_prev + 0.5 * S @ (S @ h_prev)
  // Step 1: Sh = S @ h_prev
  std::vector<float> Sh(dim, 0.0f);
  for (int i = 0; i < dim; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < dim; ++j) {
      acc += skew_S[i * dim + j] * h_prev[j];
    }
    Sh[i] = acc;
  }
  // Step 2: SSh = S @ Sh
  std::vector<float> SSh(dim, 0.0f);
  for (int i = 0; i < dim; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < dim; ++j) {
      acc += skew_S[i * dim + j] * Sh[j];
    }
    SSh[i] = acc;
  }
  // Step 3: h_next = h_prev + Sh + 0.5*SSh
  for (int i = 0; i < dim; ++i) {
    h_next[i] = h_prev[i] + Sh[i] + 0.5f * SSh[i];
  }
}

// --------------------------------------------------------------------------
// 4. Advanced Token Sampler (Top-P, Top-K, Min-P)
//    Returns token ID
// --------------------------------------------------------------------------
struct TokenProb {
  int id;
  float val;
};

OMEGA_API int advanced_sample(const float *logits, int vocab_size, float temp,
                              float min_p, float top_p, int top_k) {
  // Greedy
  if (temp < 0.05f) {
    int best = 0;
    float max_l = -1e30f;
    for (int i = 0; i < vocab_size; ++i) {
      if (logits[i] > max_l) {
        max_l = logits[i];
        best = i;
      }
    }
    return best;
  }

  // Softmax with temperature
  std::vector<TokenProb> probs(vocab_size);
  float max_logit = -1e30f;
  for (int i = 0; i < vocab_size; ++i)
    if (logits[i] > max_logit)
      max_logit = logits[i];

  double sum_exp = 0.0;
  for (int i = 0; i < vocab_size; ++i) {
    probs[i].id = i;
    probs[i].val = std::exp((logits[i] - max_logit) / temp);
    sum_exp += probs[i].val;
  }
  for (int i = 0; i < vocab_size; ++i)
    probs[i].val /= (float)sum_exp;

  // Sort descending
  int sort_len = vocab_size;
  if (top_k > 0 && top_k < vocab_size) {
    sort_len = std::min(vocab_size, top_k + 5);
    std::partial_sort(
        probs.begin(), probs.begin() + sort_len, probs.end(),
        [](const TokenProb &a, const TokenProb &b) { return a.val > b.val; });
  } else {
    std::sort(
        probs.begin(), probs.end(),
        [](const TokenProb &a, const TokenProb &b) { return a.val > b.val; });
  }

  // Min-P filter
  int n_probs = vocab_size;
  if (min_p > 0.0f) {
    float thr = probs[0].val * min_p;
    for (int i = 1; i < n_probs; ++i) {
      if (probs[i].val < thr) {
        n_probs = i;
        break;
      }
    }
  }
  // Top-P (nucleus)
  if (top_p < 1.0f) {
    double cs = 0.0;
    for (int i = 0; i < n_probs; ++i) {
      cs += probs[i].val;
      if (cs >= top_p) {
        n_probs = i + 1;
        break;
      }
    }
  }
  // Top-K
  if (top_k > 0 && top_k < n_probs)
    n_probs = top_k;

  // Weighted random choice
  static std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  double new_sum = 0.0;
  for (int i = 0; i < n_probs; ++i)
    new_sum += probs[i].val;
  float r = dist(gen) * (float)new_sum;
  double cdf = 0.0;
  for (int i = 0; i < n_probs; ++i) {
    cdf += probs[i].val;
    if (r < cdf)
      return probs[i].id;
  }
  return probs[n_probs - 1].id;
}

// --------------------------------------------------------------------------
// 5. Integral Auditor: p-convergence check (Чикулаев-Кадымов)
//    Given history of f(t) = ||h(t) - h(t-1)||, fits ln(f) = lnC - p*ln(t)
//    Returns: p value.  If p > 1.2 → converged.  If p < 0.5 → searching.
// --------------------------------------------------------------------------
OMEGA_API float
integral_audit(const float *f_history, // array of ||delta_h|| values
               int n,                  // number of values
               int window              // sliding window size (e.g. 8)
) {
  if (n < 3)
    return 0.0f;

  int start = (n > window) ? n - window : 0;
  int count = n - start;
  if (count < 3)
    return 0.0f;

  // Least squares: ln(f) = a - p * ln(t)
  // Y = ln(f), X = ln(t), fit Y = a + b*X  where b = -p
  double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
  int valid = 0;
  for (int i = start; i < n; ++i) {
    float f_val = f_history[i];
    if (f_val <= 1e-12f)
      f_val = 1e-12f; // clamp
    double x = log((double)(i - start + 1));
    double y = log((double)f_val);
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_xx += x * x;
    valid++;
  }
  if (valid < 3)
    return 0.0f;

  double denom = valid * sum_xx - sum_x * sum_x;
  if (fabs(denom) < 1e-15)
    return 0.0f;

  double b = (valid * sum_xy - sum_x * sum_y) / denom;
  float p = (float)(-b); // p = -slope
  return p;
}

// --------------------------------------------------------------------------
// 6. Hankel Rank Collapse Detector
//    Builds Hankel matrix from last W states, computes ratio σ₂/σ₁
//    If ratio < threshold → system is looping (rank collapse)
//    Returns: ratio (0..1).  Low = looping.
//    Uses power iteration for top-2 singular values (no LAPACK needed).
// --------------------------------------------------------------------------
OMEGA_API float
hankel_rank(const float *state_history, // flattened [n_steps x dim]
            int n_steps, int dim,
            int hankel_rows // how many rows for Hankel (e.g. 5)
) {
  if (n_steps < hankel_rows + 1)
    return 1.0f;

  int hankel_cols = n_steps - hankel_rows;
  if (hankel_cols < 2)
    return 1.0f;

  // Build Hankel matrix H[i,j] = norm(state[i+j])
  int H_rows = hankel_rows;
  int H_cols = hankel_cols;
  std::vector<float> H(H_rows * H_cols, 0.0f);

  for (int i = 0; i < H_rows; ++i) {
    for (int j = 0; j < H_cols; ++j) {
      int idx = i + j;
      float norm = 0.0f;
      const float *s = state_history + idx * dim;
      for (int d = 0; d < dim; ++d)
        norm += s[d] * s[d];
      H[i * H_cols + j] = sqrtf(norm);
    }
  }

  // Power iteration to get σ₁ (top singular value of H)
  // H^T H v = σ² v
  std::vector<float> v(H_cols, 1.0f / sqrtf((float)H_cols));
  std::vector<float> Hv(H_rows), HTHv(H_cols);

  float sigma1 = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    // Hv = H @ v
    for (int i = 0; i < H_rows; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < H_cols; ++j)
        acc += H[i * H_cols + j] * v[j];
      Hv[i] = acc;
    }
    // HTHv = H^T @ Hv
    for (int j = 0; j < H_cols; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < H_rows; ++i)
        acc += H[i * H_cols + j] * Hv[i];
      HTHv[j] = acc;
    }
    // Normalize
    float norm = 0.0f;
    for (int j = 0; j < H_cols; ++j)
      norm += HTHv[j] * HTHv[j];
    norm = sqrtf(norm);
    if (norm < 1e-12f)
      return 0.0f;
    sigma1 = sqrtf(norm);
    for (int j = 0; j < H_cols; ++j)
      v[j] = HTHv[j] / norm;
  }

  // Deflate: H2 = H - σ₁ * u * v^T, then power iterate for σ₂
  // u = Hv / σ₁
  std::vector<float> u(H_rows);
  for (int i = 0; i < H_rows; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < H_cols; ++j)
      acc += H[i * H_cols + j] * v[j];
    u[i] = acc / (sigma1 + 1e-12f);
  }
  // H2 = H - σ₁ * u * v^T
  std::vector<float> H2(H_rows * H_cols);
  for (int i = 0; i < H_rows; ++i)
    for (int j = 0; j < H_cols; ++j)
      H2[i * H_cols + j] = H[i * H_cols + j] - sigma1 * u[i] * v[j];

  // Power iteration on H2 for σ₂
  std::vector<float> v2(H_cols, 1.0f / sqrtf((float)H_cols));
  float sigma2 = 0.0f;
  for (int iter = 0; iter < 20; ++iter) {
    for (int i = 0; i < H_rows; ++i) {
      float acc = 0.0f;
      for (int j = 0; j < H_cols; ++j)
        acc += H2[i * H_cols + j] * v2[j];
      Hv[i] = acc;
    }
    for (int j = 0; j < H_cols; ++j) {
      float acc = 0.0f;
      for (int i = 0; i < H_rows; ++i)
        acc += H2[i * H_cols + j] * Hv[i];
      HTHv[j] = acc;
    }
    float norm = 0.0f;
    for (int j = 0; j < H_cols; ++j)
      norm += HTHv[j] * HTHv[j];
    norm = sqrtf(norm);
    if (norm < 1e-12f) {
      sigma2 = 0.0f;
      break;
    }
    sigma2 = sqrtf(norm);
    for (int j = 0; j < H_cols; ++j)
      v2[j] = HTHv[j] / norm;
  }

  if (sigma1 < 1e-12f)
    return 0.0f;
  return sigma2 / sigma1; // low ratio = rank collapse = looping
}

// --------------------------------------------------------------------------
// 7. Utility: Get version
// --------------------------------------------------------------------------
OMEGA_API int omega_version() {
  return 300; // v3.0.0
}
