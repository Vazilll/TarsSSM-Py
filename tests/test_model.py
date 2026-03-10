"""
═══════════════════════════════════════════════════════════════
  test_model.py — Python Model Structure & Forward Pass Tests
═══════════════════════════════════════════════════════════════

Tests for brain/ module: model structure, block, forward pass.
Consolidated from tests/brain/*.

Owner: Agent 5 (EXCLUSIVE)
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════
# Model structure
# ═══════════════════════════════════════════════════════════════

def test_model_import():
    """Model class can be imported."""
    from brain.mamba2.model import TarsMamba2LM
    assert TarsMamba2LM is not None


def test_model_creation(small_config):
    """Model can be instantiated with small config."""
    from brain.mamba2.model import TarsMamba2LM
    model = TarsMamba2LM(
        d_model=small_config["d_model"],
        n_layers=small_config["n_layers"],
        vocab_size=small_config["vocab_size"],
        mingru_dim=small_config.get("d_state", 16),
    )
    assert model is not None
    total = sum(p.numel() for p in model.parameters())
    assert total > 0


def test_model_forward(small_config, device):
    """Model forward pass produces correct output shape."""
    from brain.mamba2.model import TarsMamba2LM
    model = TarsMamba2LM(
        d_model=small_config["d_model"],
        n_layers=small_config["n_layers"],
        vocab_size=small_config["vocab_size"],
        mingru_dim=small_config.get("d_state", 16),
    ).to(device)
    model.eval()

    batch = 1
    seq_len = 8
    x = torch.randint(0, small_config["vocab_size"], (batch, seq_len), device=device)

    with torch.no_grad():
        output = model(x)

    # Output should be logits [B, S, V] or similar
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    assert logits.dim() >= 2
    assert not torch.isnan(logits).any(), "NaN in model output"


# ═══════════════════════════════════════════════════════════════
# TarsHelixLite (CURRENT model — used for training)
# ═══════════════════════════════════════════════════════════════

def test_helix_lite_import():
    """TarsHelixLite can be imported."""
    from brain.mamba2.core.model_lite import TarsHelixLite
    assert TarsHelixLite is not None


def test_helix_lite_creation():
    """TarsHelixLite can be instantiated with small config."""
    from config import TarsConfig
    from brain.mamba2.core.model_lite import TarsHelixLite
    cfg = TarsConfig(d_model=128, n_layers=2, vocab_size=256, d_state=16, headdim=32)
    model = TarsHelixLite(cfg)
    assert model is not None
    total = sum(p.numel() for p in model.parameters())
    assert total > 0


def test_helix_lite_forward():
    """TarsHelixLite forward pass with loss computation."""
    from config import TarsConfig
    from brain.mamba2.core.model_lite import TarsHelixLite
    cfg = TarsConfig(d_model=128, n_layers=2, vocab_size=256, d_state=16, headdim=32)
    model = TarsHelixLite(cfg)
    model.eval()

    B, L = 2, 32
    x = torch.randint(0, 256, (B, L))
    labels = torch.randint(0, 256, (B, L))

    with torch.no_grad():
        result = model(x, labels=labels)

    assert 'logits' in result
    assert 'loss' in result
    assert result['logits'].shape == (B, L, 256)
    assert not torch.isnan(result['logits']).any(), "NaN in TarsHelixLite output"
    assert not torch.isnan(result['loss']), "NaN loss in TarsHelixLite"


def test_helix_lite_generate():
    """TarsHelixLite generation produces tokens."""
    from config import TarsConfig
    from brain.mamba2.core.model_lite import TarsHelixLite
    cfg = TarsConfig(d_model=128, n_layers=2, vocab_size=256, d_state=16, headdim=32)
    model = TarsHelixLite(cfg)

    prompt = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=4)

    assert generated.shape[0] == 1
    assert generated.shape[1] > prompt.shape[1], "No tokens generated"


# ═══════════════════════════════════════════════════════════════
# DoubtEngine
# ═══════════════════════════════════════════════════════════════

def test_doubt_engine_creation():
    """DoubtEngine can be created."""
    from brain.doubt_engine import DoubtEngine
    de = DoubtEngine(d_model=64, d_doubt=32)
    total = sum(p.numel() for p in de.parameters())
    assert total > 0


def test_doubt_engine_forward():
    """DoubtEngine forward pass works."""
    from brain.doubt_engine import DoubtEngine
    de = DoubtEngine(d_model=64, d_doubt=32)
    de.eval()

    q = torch.randn(1, 64)
    r = torch.randn(1, 64)

    with torch.no_grad():
        scores = de(q, r)

    assert "coherence" in scores
    assert "safety" in scores
    assert "repetition" in scores
    assert 0 <= scores["coherence"].item() <= 1
    assert 0 <= scores["safety"].item() <= 1
    assert 0 <= scores["repetition"].item() <= 1


def test_doubt_repetition_detection():
    """Repetition detection works."""
    from brain.doubt_engine import DoubtEngine

    # No repetition
    score1 = DoubtEngine.compute_repetition("The quick brown fox jumps over the lazy dog")
    assert score1 < 0.3

    # High repetition
    score2 = DoubtEngine.compute_repetition("hello world " * 50)
    assert score2 > 0.7


# ═══════════════════════════════════════════════════════════════
# Tokenizer
# ═══════════════════════════════════════════════════════════════

def test_tokenizer():
    """TarsTokenizer encode/decode roundtrip."""
    from brain.tokenizer import TarsTokenizer
    tok = TarsTokenizer()

    texts = ["Привет", "Hello world", "123 + 456 = 579"]
    for text in texts:
        ids = tok.encode(text)
        assert len(ids) > 0
        decoded = tok.decode(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0


# ═══════════════════════════════════════════════════════════════
# SafetyGate
# ═══════════════════════════════════════════════════════════════

def test_safety_gate_blocks_dangerous():
    """SafetyGate blocks dangerous commands."""
    from brain.doubt_engine import SafetyGate

    # Should block
    v = SafetyGate.check("shell", {"command": "rm -rf /"})
    assert v.is_blocked

    v = SafetyGate.check("shell", {"command": "format C:"})
    assert v.is_blocked

    v = SafetyGate.check("shell", {"command": "shutdown"})
    assert v.is_blocked

    # Should pass
    v = SafetyGate.check("shell", {"command": "pip list"})
    assert v.is_passed
