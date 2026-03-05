"""Shared fixtures for TARS tests."""
import sys
import os
import pytest
import torch

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def device():
    """Default device for tests (CPU)."""
    return torch.device("cpu")


@pytest.fixture
def small_d_model():
    """Small d_model for fast tests."""
    return 64


@pytest.fixture
def small_config():
    """Minimal model config for smoke tests."""
    return {
        "d_model": 64,
        "d_state": 16,
        "headdim": 16,
        "omega_dim": 8,
        "n_experts": 4,
        "n_layers": 2,
        "vocab_size": 256,
    }
