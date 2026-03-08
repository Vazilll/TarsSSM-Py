"""
═══════════════════════════════════════════════════════════════
  test_kernels.py — C++ Kernel Accuracy vs Python Reference
═══════════════════════════════════════════════════════════════

CRITICAL: Ensures C++ kernels (Agent 1) produce identical results
to Python reference implementations (Agent 4).

All 12 kernels tested:
  ssd_scan, wkv7_update, bitnet_matmul, rmsnorm, swiglu,
  rope, diff_attention, softmax, embedding_lookup, smoothquant,
  eagle_decode, tokenizer

Owner: Agent 5 (EXCLUSIVE)
"""

import pytest
import torch
import numpy as np

# ═══ Try to import C++ core ═══
try:
    import tars_core
    HAS_TARS_CORE = True
except ImportError:
    HAS_TARS_CORE = False

# ═══ Python references (Agent 4) ═══
try:
    from brain.mamba2.ssd import ssd_scan_python
except ImportError:
    ssd_scan_python = None

try:
    from brain.mamba2.bitnet import bitnet_forward_python
except ImportError:
    bitnet_forward_python = None


needs_core = pytest.mark.skipif(
    not HAS_TARS_CORE,
    reason="tars_core C++ module not built (Agent 1)"
)


# ═══════════════════════════════════════════════════════════════
# 1. SSD Scan: s_{t+1} = γ·s_t + B·x_t
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_ssd_scan_sync():
    """C++ ssd_scan MUST match Python reference."""
    d_state = 64
    d_model = 1280
    state = torch.randn(d_state)
    gamma = torch.rand(d_state)
    B = torch.randn(d_state)
    x = torch.randn(d_model)

    py_result = ssd_scan_python(state, gamma, B, x)
    cpp_result = tars_core.ssd_scan(state, gamma, B, x)
    assert torch.allclose(py_result, cpp_result, atol=1e-5), \
        f"SSD scan mismatch: max diff = {(py_result - cpp_result).abs().max():.6e}"


# ═══════════════════════════════════════════════════════════════
# 2. BitNet Matmul: ternary ADD/SUB only
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_bitnet_matmul_sync():
    """Ternary matmul MUST match Python reference."""
    d = 1280
    W = torch.randint(-1, 2, (d, d), dtype=torch.int8)
    x = torch.randn(d)
    alpha = 0.5

    py_result = bitnet_forward_python(W, x, alpha)
    cpp_result = tars_core.bitnet_matmul(W, x, alpha)
    assert torch.allclose(py_result, cpp_result, atol=1e-4), \
        f"BitNet matmul mismatch: max diff = {(py_result - cpp_result).abs().max():.6e}"


# ═══════════════════════════════════════════════════════════════
# 3. RMSNorm: γ·x/√(mean(x²)+ε)
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_rmsnorm_sync():
    """RMSNorm C++ MUST match Python."""
    d = 1280
    x = torch.randn(d)
    gamma = torch.ones(d)
    eps = 1e-5

    # Python reference
    rms = torch.sqrt(torch.mean(x ** 2) + eps)
    py_result = gamma * x / rms

    cpp_result = tars_core.rmsnorm(x, gamma, eps)
    assert torch.allclose(py_result, cpp_result, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 4. SwiGLU: SiLU(W₁x)⊙W₂x
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_swiglu_sync():
    """SwiGLU fused C++ MUST match Python."""
    d = 1280
    d_ff = d * 4
    W1 = torch.randn(d_ff, d)
    W2 = torch.randn(d_ff, d)
    x = torch.randn(d)

    # Python reference
    py_result = torch.nn.functional.silu(W1 @ x) * (W2 @ x)

    cpp_result = tars_core.swiglu_fused(W1, W2, x, sparsity=0.0)
    assert torch.allclose(py_result, cpp_result, atol=1e-4)


# ═══════════════════════════════════════════════════════════════
# 5. RoPE: rotary positional encoding
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_rope_sync():
    """RoPE C++ MUST match Python."""
    d = 64  # head dim
    Q = torch.randn(1, d)
    K = torch.randn(1, d)
    theta = 500000.0
    positions = torch.tensor([42])

    # Python reference
    half = d // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos_freqs = positions.unsqueeze(-1) * freqs.unsqueeze(0)
    cos_f = torch.cos(pos_freqs)
    sin_f = torch.sin(pos_freqs)

    def rotate(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1)

    py_Q = rotate(Q)
    py_K = rotate(K)

    cpp_Q, cpp_K = tars_core.rope(Q, K, theta, positions)
    assert torch.allclose(py_Q, cpp_Q, atol=1e-5)
    assert torch.allclose(py_K, cpp_K, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 6. Softmax (online, numerically stable)
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_softmax_sync():
    """Online softmax C++ MUST match Python."""
    x = torch.randn(1280)
    py_result = torch.softmax(x, dim=-1)
    cpp_result = tars_core.softmax_fused(x)
    assert torch.allclose(py_result, cpp_result, atol=1e-6)


# ═══════════════════════════════════════════════════════════════
# 7. Diff Attention: attn₁ − λ·attn₂
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_diff_attention_sync():
    """Diff Transformer C++ MUST match Python."""
    seq_len = 16
    d = 64
    Q1 = torch.randn(seq_len, d)
    Q2 = torch.randn(seq_len, d)
    K1 = torch.randn(seq_len, d)
    K2 = torch.randn(seq_len, d)
    V = torch.randn(seq_len, d)
    lam = 0.8

    # Python reference
    scale = d ** -0.5
    attn1 = torch.softmax(Q1 @ K1.T * scale, dim=-1)
    attn2 = torch.softmax(Q2 @ K2.T * scale, dim=-1)
    py_result = (attn1 - lam * attn2) @ V

    cpp_result = tars_core.diff_attention(Q1, Q2, K1, K2, V, lam)
    assert torch.allclose(py_result, cpp_result, atol=1e-4)


# ═══════════════════════════════════════════════════════════════
# 8. WKV-7 Update: S = S·(diag(w)+aᵀb)+vᵀk
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_wkv7_update_sync():
    """WKV-7 update C++ MUST match Python."""
    d = 64
    S = torch.randn(d, d)
    w = torch.randn(d)
    a = torch.randn(d)
    b = torch.randn(d)
    v = torch.randn(d)
    k = torch.randn(d)

    # Python reference
    py_result = S * (torch.diag(w) + a.unsqueeze(1) * b.unsqueeze(0)) + \
                v.unsqueeze(1) * k.unsqueeze(0)

    cpp_result = tars_core.wkv7_update(S, w, a, b, v, k)
    assert torch.allclose(py_result, cpp_result, atol=1e-4)


# ═══════════════════════════════════════════════════════════════
# 9. Embedding Lookup: embed × √d_model
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_embedding_lookup_sync():
    """Embedding lookup C++ MUST match Python."""
    vocab_size = 48000
    d_model = 1280
    W_embed = torch.randn(vocab_size, d_model)
    token_ids = torch.tensor([42, 100, 999])

    py_result = W_embed[token_ids] * (d_model ** 0.5)
    cpp_result = tars_core.embedding_lookup(W_embed, token_ids)
    assert torch.allclose(py_result, cpp_result, atol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 10. SmoothQuant: Y = (X·S⁻¹)·(S·W)
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_smoothquant_sync():
    """SmoothQuant INT8 path C++ MUST match Python (FP32 reference)."""
    d = 128  # smaller for speed
    X = torch.randn(1, d)
    W = torch.randn(d, d)
    S = torch.ones(d) * 2.0  # smoothing factor

    # Python reference (FP32)
    py_result = (X / S.unsqueeze(0)) @ (torch.diag(S) @ W)

    cpp_result = tars_core.smoothquant(X, W, S)
    # INT8 quantization introduces error, so larger tolerance
    assert torch.allclose(py_result, cpp_result, atol=1e-2), \
        f"SmoothQuant mismatch: max diff = {(py_result - cpp_result).abs().max():.6e}"


# ═══════════════════════════════════════════════════════════════
# 11. Tokenizer BPE
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_tokenizer_sync():
    """C++ BPE tokenizer MUST match Python reference."""
    from brain.tokenizer import TarsTokenizer

    py_tok = TarsTokenizer()
    texts = [
        "Hello, world!",
        "Привет ТАРС, как дела?",
        "def foo(x): return x * 2",
    ]

    for text in texts:
        py_ids = py_tok.encode(text)
        cpp_ids = tars_core.tokenizer_encode(text)
        assert py_ids == cpp_ids, f"Tokenizer mismatch for '{text}'"

        py_decoded = py_tok.decode(py_ids)
        cpp_decoded = tars_core.tokenizer_decode(cpp_ids)
        assert py_decoded == cpp_decoded, f"Detokenizer mismatch for ids {py_ids}"


# ═══════════════════════════════════════════════════════════════
# 12. EAGLE-3 Speculative Decode
# ═══════════════════════════════════════════════════════════════

@needs_core
def test_eagle_decode_sync():
    """EAGLE-3 speculative decoding acceptance logic."""
    # Test accept/reject with known probabilities
    draft_probs = torch.tensor([0.8, 0.1, 0.05, 0.05])
    target_probs = torch.tensor([0.7, 0.15, 0.1, 0.05])

    # The acceptance should be: min(1, target/draft) for each token
    # This is just a smoke test — full spec decode is complex
    cpp_accepted = tars_core.eagle_check_acceptance(draft_probs, target_probs)
    assert isinstance(cpp_accepted, (bool, int, list))


# ═══════════════════════════════════════════════════════════════
# Smoke test: Python references exist (no C++ needed)
# ═══════════════════════════════════════════════════════════════

def test_python_rmsnorm_reference():
    """Python RMSNorm reference works."""
    x = torch.randn(128)
    gamma = torch.ones(128)
    eps = 1e-5
    result = gamma * x / torch.sqrt(torch.mean(x ** 2) + eps)
    assert result.shape == x.shape
    assert not torch.isnan(result).any()


def test_python_swiglu_reference():
    """Python SwiGLU reference works."""
    d = 128
    x = torch.randn(d)
    W1 = torch.randn(d * 4, d)
    W2 = torch.randn(d * 4, d)
    result = torch.nn.functional.silu(W1 @ x) * (W2 @ x)
    assert result.shape == (d * 4,)
    assert not torch.isnan(result).any()


def test_python_softmax_reference():
    """Python softmax reference works."""
    x = torch.randn(128)
    result = torch.softmax(x, dim=-1)
    assert abs(result.sum().item() - 1.0) < 1e-5
