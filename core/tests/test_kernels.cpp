/*
 * TARS Core — Unit Tests (Week 1)
 *
 * Standalone tests (no Google Test), simple assert + printf.
 * Tests all Week 1 kernels against known reference values.
 *
 * Build: cmake --build build --config Release --target test_kernels
 * Run:   build/Release/test_kernels.exe
 *
 * Agent 1 — Week 1
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

#include "kernels/rmsnorm_fused.h"
#include "kernels/embedding_lookup.h"
#include "kernels/bitnet_matmul.h"
#include "kernels/softmax_fused.h"
#include "runtime/arena.h"
#include "runtime/tensor.h"

// ═══════════════════════════════════════
// Test helpers
// ═══════════════════════════════════════

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT_NEAR(got, expected, tol, msg) do { \
    float _g = (got), _e = (expected), _t = (tol); \
    if (fabsf(_g - _e) > _t) { \
        printf("  FAIL: %s: got %.8f, expected %.8f (tol %.1e)\n", msg, _g, _e, _t); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(got, expected, msg) do { \
    if ((got) != (expected)) { \
        printf("  FAIL: %s: got %d, expected %d\n", msg, (int)(got), (int)(expected)); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_PASS() do { tests_passed++; } while(0)


// ═══════════════════════════════════════
// Test: Arena allocator
// ═══════════════════════════════════════

static void test_arena() {
    printf("[test_arena] ");
    
    TarsArena arena(1024 * 1024);  // 1 MB
    ASSERT_EQ(arena.used(), 0, "initial used");
    ASSERT_EQ(arena.capacity(), 1024 * 1024, "capacity");
    
    // Allocate some memory
    float* buf = arena.alloc_array<float>(256);
    ASSERT_EQ(buf != nullptr, 1, "alloc returned non-null");
    
    // Write to it — should not crash
    for (int i = 0; i < 256; ++i) buf[i] = (float)i;
    ASSERT_NEAR(buf[100], 100.0f, 1e-6f, "write test");
    
    size_t used_after = arena.used();
    ASSERT_EQ(used_after > 0, 1, "used increased");
    
    // Allocate more
    float* buf2 = arena.alloc_array<float>(512);
    ASSERT_EQ(buf2 != nullptr, 1, "second alloc non-null");
    ASSERT_EQ(arena.used() > used_after, 1, "used increased again");
    
    // Reset — reclaim all
    arena.reset();
    ASSERT_EQ(arena.used(), 0, "after reset used=0");
    
    // Re-allocate from beginning
    float* buf3 = arena.alloc_array<float>(128);
    ASSERT_EQ(buf3 != nullptr, 1, "post-reset alloc non-null");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: Tensor struct
// ═══════════════════════════════════════

static void test_tensor() {
    printf("[test_tensor] ");
    
    float data[12] = {1,2,3,4,5,6,7,8,9,10,11,12};
    
    // 1D tensor
    TarsTensor t1 = tars_tensor_1d(data, 12);
    ASSERT_EQ(t1.numel(), 12, "1D numel");
    ASSERT_EQ(t1.ndim, 1, "1D ndim");
    ASSERT_NEAR(t1.at_f32(5), 6.0f, 1e-6f, "1D access");
    
    // 2D tensor (3×4 view)
    TarsTensor t2 = tars_tensor_2d(data, 3, 4);
    ASSERT_EQ(t2.numel(), 12, "2D numel");
    ASSERT_EQ(t2.ndim, 2, "2D ndim");
    ASSERT_EQ(t2.stride[0], 4, "2D stride[0]");
    ASSERT_EQ(t2.stride[1], 1, "2D stride[1]");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: RMSNorm
// ═══════════════════════════════════════

static void test_rmsnorm() {
    printf("[test_rmsnorm] ");
    
    // x = [1, 2, 3, 4], gamma = [1, 1, 1, 1], eps = 1e-8
    // mean(x²) = (1+4+9+16)/4 = 7.5
    // rms_inv = 1/sqrt(7.5 + 1e-8) ≈ 0.365148
    // out = [0.365148, 0.730297, 1.095445, 1.460594]
    
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];
    
    tars_rmsnorm(x, gamma, 1e-8f, 4, out);
    
    float rms_inv = 1.0f / sqrtf(7.5f + 1e-8f);
    ASSERT_NEAR(out[0], 1.0f * rms_inv, 1e-5f, "rmsnorm[0]");
    ASSERT_NEAR(out[1], 2.0f * rms_inv, 1e-5f, "rmsnorm[1]");
    ASSERT_NEAR(out[2], 3.0f * rms_inv, 1e-5f, "rmsnorm[2]");
    ASSERT_NEAR(out[3], 4.0f * rms_inv, 1e-5f, "rmsnorm[3]");
    
    // Test with non-unit gamma
    float gamma2[] = {2.0f, 0.5f, 1.0f, 3.0f};
    tars_rmsnorm(x, gamma2, 1e-8f, 4, out);
    ASSERT_NEAR(out[0], 2.0f * 1.0f * rms_inv, 1e-5f, "rmsnorm gamma[0]");
    ASSERT_NEAR(out[1], 0.5f * 2.0f * rms_inv, 1e-5f, "rmsnorm gamma[1]");
    
    // Large vector (d_model=1280) to test AVX2 path
    const int DIM = 1280;
    std::vector<float> xv(DIM), gv(DIM, 1.0f), ov(DIM);
    float sq_sum = 0.0f;
    for (int i = 0; i < DIM; ++i) {
        xv[i] = 0.01f * (float)(i - DIM/2);
        sq_sum += xv[i] * xv[i];
    }
    tars_rmsnorm(xv.data(), gv.data(), 1e-8f, DIM, ov.data());
    float expected_rms_inv = 1.0f / sqrtf(sq_sum / DIM + 1e-8f);
    ASSERT_NEAR(ov[0], xv[0] * expected_rms_inv, 1e-4f, "rmsnorm large[0]");
    ASSERT_NEAR(ov[640], xv[640] * expected_rms_inv, 1e-4f, "rmsnorm large[640]");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: Embedding lookup
// ═══════════════════════════════════════

static void test_embedding_lookup() {
    printf("[test_embedding_lookup] ");
    
    // table 4×3 (4 tokens, d_model=3)
    float table[] = {
        0.1f, 0.2f, 0.3f,   // token 0
        0.4f, 0.5f, 0.6f,   // token 1
        0.7f, 0.8f, 0.9f,   // token 2
        1.0f, 1.1f, 1.2f,   // token 3
    };
    float out[3];
    
    float scale = sqrtf(3.0f);
    
    tars_embedding_lookup(table, 0, 3, out);
    ASSERT_NEAR(out[0], 0.1f * scale, 1e-5f, "embed[0][0]");
    ASSERT_NEAR(out[1], 0.2f * scale, 1e-5f, "embed[0][1]");
    
    tars_embedding_lookup(table, 2, 3, out);
    ASSERT_NEAR(out[0], 0.7f * scale, 1e-5f, "embed[2][0]");
    ASSERT_NEAR(out[2], 0.9f * scale, 1e-5f, "embed[2][2]");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: BitNet ternary matmul
// ═══════════════════════════════════════

static void test_bitnet_matmul() {
    printf("[test_bitnet_matmul] ");
    
    // W = [[1, -1, 0],
    //      [0,  1, 1],
    //      [-1, 0, 1]]
    // x = [2, 3, 4]
    // alpha = 0.5
    // naive: W@x = [2-3+0, 0+3+4, -2+0+4] = [-1, 7, 2]
    // out = alpha * W@x = [-0.5, 3.5, 1.0]
    
    int8_t W[] = {1, -1, 0,   0, 1, 1,   -1, 0, 1};
    float x[] = {2.0f, 3.0f, 4.0f};
    float out[3];
    
    tars_bitnet_matmul(W, x, 0.5f, 3, 3, out);
    ASSERT_NEAR(out[0], -0.5f, 1e-5f, "bitnet[0]");
    ASSERT_NEAR(out[1],  3.5f, 1e-5f, "bitnet[1]");
    ASSERT_NEAR(out[2],  1.0f, 1e-5f, "bitnet[2]");
    
    // Large matrix to test AVX2 path (32×32)
    const int N = 32;
    std::vector<int8_t> Wl(N * N);
    std::vector<float> xl(N, 1.0f), ol(N);
    // W = all 1s → each row sums to N → out[r] = alpha * N
    for (auto& w : Wl) w = 1;
    tars_bitnet_matmul(Wl.data(), xl.data(), 1.0f, N, N, ol.data());
    ASSERT_NEAR(ol[0], (float)N, 1e-4f, "bitnet large[0]");
    ASSERT_NEAR(ol[N-1], (float)N, 1e-4f, "bitnet large[N-1]");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: Ternary quantization
// ═══════════════════════════════════════

static void test_quantize_ternary() {
    printf("[test_quantize_ternary] ");
    
    float W[] = {1.0f, -0.5f, 0.1f, -1.2f, 0.0f, 0.8f};
    int8_t Wq[6];
    
    float scale = tars_quantize_ternary(W, Wq, 6);
    
    // scale = mean(|W|) = (1 + 0.5 + 0.1 + 1.2 + 0 + 0.8) / 6 = 3.6/6 = 0.6
    ASSERT_NEAR(scale, 0.6f, 1e-5f, "scale");
    
    // W/scale = [1.67, -0.83, 0.17, -2.0, 0.0, 1.33]
    // clipped round: [1, -1, 0, -1, 0, 1]
    ASSERT_EQ(Wq[0],  1, "Wq[0]");
    ASSERT_EQ(Wq[1], -1, "Wq[1]");
    ASSERT_EQ(Wq[2],  0, "Wq[2]");
    ASSERT_EQ(Wq[3], -1, "Wq[3]");
    ASSERT_EQ(Wq[4],  0, "Wq[4]");
    ASSERT_EQ(Wq[5],  1, "Wq[5]");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: Softmax
// ═══════════════════════════════════════

static void test_softmax() {
    printf("[test_softmax] ");
    
    // Simple case: [1, 2, 3]
    float x[] = {1.0f, 2.0f, 3.0f};
    float out[3];
    tars_softmax(x, 3, out);
    
    // Verify: sum = 1
    float sum = out[0] + out[1] + out[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax sum=1");
    
    // out[2] > out[1] > out[0]
    ASSERT_EQ(out[2] > out[1], 1, "softmax ordering 2>1");
    ASSERT_EQ(out[1] > out[0], 1, "softmax ordering 1>0");
    
    // Known values: softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    ASSERT_NEAR(out[0], 0.0900f, 1e-3f, "softmax[0]");
    ASSERT_NEAR(out[1], 0.2447f, 1e-3f, "softmax[1]");
    ASSERT_NEAR(out[2], 0.6652f, 1e-3f, "softmax[2]");
    
    // Numerical stability: large values
    float x_large[] = {1000.0f, 1000.1f, 1000.2f};
    tars_softmax(x_large, 3, out);
    sum = out[0] + out[1] + out[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f, "softmax large sum=1");
    
    // Single element
    float x1[] = {42.0f};
    float out1[1];
    tars_softmax(x1, 1, out1);
    ASSERT_NEAR(out1[0], 1.0f, 1e-6f, "softmax single");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: Argmax
// ═══════════════════════════════════════

static void test_argmax() {
    printf("[test_argmax] ");
    
    float x[] = {0.1f, 0.5f, 0.3f, 0.9f, 0.2f};
    int idx = tars_argmax(x, 5);
    ASSERT_EQ(idx, 3, "argmax");
    
    float x2[] = {5.0f, 1.0f, 2.0f};
    ASSERT_EQ(tars_argmax(x2, 3), 0, "argmax first");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Test: Softmax max (confidence)
// ═══════════════════════════════════════

static void test_softmax_max() {
    printf("[test_softmax_max] ");
    
    // Uniform logits → low confidence
    float x_flat[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float conf_flat = tars_softmax_max(x_flat, 4);
    ASSERT_NEAR(conf_flat, 0.25f, 1e-3f, "flat confidence ≈ 0.25");
    
    // One dominant → high confidence
    float x_peak[] = {0.0f, 0.0f, 0.0f, 100.0f};
    float conf_peak = tars_softmax_max(x_peak, 4);
    ASSERT_EQ(conf_peak > 0.99f, 1, "peak confidence > 0.99");
    
    printf("PASSED\n");
    TEST_PASS();
}


// ═══════════════════════════════════════
// Benchmark (optional, for information)
// ═══════════════════════════════════════

static void bench_bitnet_matmul() {
    printf("[bench] BitNet matmul 1280×1280 × 1280: ");
    
    const int DIM = 1280;
    std::vector<int8_t> W(DIM * DIM);
    std::vector<float> x(DIM), out(DIM);
    
    // Fill with alternating {-1, 0, 1}
    for (int i = 0; i < DIM * DIM; ++i) W[i] = (int8_t)((i % 3) - 1);
    for (int i = 0; i < DIM; ++i) x[i] = 0.01f * (float)i;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int ITERS = 100;
    for (int it = 0; it < ITERS; ++it) {
        tars_bitnet_matmul(W.data(), x.data(), 1.0f, DIM, DIM, out.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
    
    printf("%.3f ms/iter\n", ms);
}

static void bench_rmsnorm() {
    printf("[bench] RMSNorm dim=1280: ");
    
    const int DIM = 1280;
    std::vector<float> x(DIM), gamma(DIM, 1.0f), out(DIM);
    for (int i = 0; i < DIM; ++i) x[i] = 0.01f * (float)i;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    const int ITERS = 10000;
    for (int it = 0; it < ITERS; ++it) {
        tars_rmsnorm(x.data(), gamma.data(), 1e-8f, DIM, out.data());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
    
    printf("%.3f us/iter\n", us);
}


// ═══════════════════════════════════════
// Main
// ═══════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   TARS Core — Unit Tests (Week 1)       ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");
    
    test_arena();
    test_tensor();
    test_rmsnorm();
    test_embedding_lookup();
    test_bitnet_matmul();
    test_quantize_ternary();
    test_softmax();
    test_argmax();
    test_softmax_max();
    
    printf("\n── Benchmarks ──\n");
    bench_bitnet_matmul();
    bench_rmsnorm();
    
    printf("\n═══════════════════════════════════════════\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════════\n");
    
    return tests_failed > 0 ? 1 : 0;
}
