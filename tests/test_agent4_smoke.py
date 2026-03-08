"""
Agent 4 — Smoke tests for all new modules.
Run: python -m pytest tests/test_agent4_smoke.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config():
    from config import TarsConfig
    c = TarsConfig()
    assert c.d_model == 2048
    assert c.n_layers == 24
    assert c.vocab_size == 32000
    assert c.d_state == 128
    assert c.quant_mode == "fp16"
    assert c.use_cpp_core == False
    s = c.summary()
    assert "2048" in s
    print(f"[OK] config.py: {s}")


def test_config_serialization():
    import tempfile, json
    from config import TarsConfig
    c = TarsConfig(d_model=512, n_layers=4)
    path = os.path.join(tempfile.gettempdir(), "test_tars_config.json")
    c.to_file(path)
    c2 = TarsConfig.from_file(path)
    assert c2.d_model == 512
    assert c2.n_layers == 4
    os.remove(path)
    print("[OK] config serialization")


def test_safe_file():
    import tempfile
    from utils.safe_file import safe_write, safe_write_json, safe_read_json, file_checksum
    path = os.path.join(tempfile.gettempdir(), "test_safe.bin")
    safe_write(path, b"hello tars")
    assert os.path.exists(path)
    cs = file_checksum(path)
    assert len(cs) == 64  # SHA256 hex

    jpath = os.path.join(tempfile.gettempdir(), "test_safe.json")
    safe_write_json(jpath, {"key": "value"})
    data = safe_read_json(jpath)
    assert data["key"] == "value"

    os.remove(path)
    os.remove(jpath)
    print("[OK] safe_file.py")


def test_tensor_pool():
    import torch
    from utils.tensor_pool import TensorPool
    pool = TensorPool(max_tensors=8)
    t = pool.get((4, 16), dtype=torch.float32)
    assert t.shape == (4, 16)
    pool.release(t)
    t2 = pool.get((4, 16), dtype=torch.float32)
    assert t2.shape == (4, 16)
    stats = pool.stats()
    assert stats["hits"] >= 1
    print(f"[OK] tensor_pool.py: {pool}")


def test_tensor_pool_context():
    import torch
    from utils.tensor_pool import TensorPool
    pool = TensorPool()
    with pool.borrow((2, 8), dtype=torch.float32) as t:
        assert t.shape == (2, 8)
    assert pool.stats()["cached_tensors"] >= 1
    print("[OK] tensor_pool context manager")


def test_input_sanitizer():
    from sensory.input_sanitizer import sanitize, is_safe_text
    r = sanitize("Hello\x00World\x01!")
    assert "Hello" in r.text
    assert "\x00" not in r.text
    assert "\x01" not in r.text
    assert len(r.warnings) > 0
    assert is_safe_text("clean text")
    assert not is_safe_text("bad\x00text")
    print(f"[OK] input_sanitizer.py: warnings={r.warnings}")


def test_nan_guard():
    import torch
    import torch.nn as nn
    from utils.nan_guard import NanGuard
    model = nn.Linear(16, 16)
    guard = NanGuard(model, replace_nan=True, raise_on_nan=False)
    guard.enable()
    x = torch.randn(2, 16)
    out = model(x)
    assert not torch.isnan(out).any()
    guard.disable()
    stats = guard.get_stats()
    assert stats["total_nan_events"] == 0
    print(f"[OK] nan_guard.py: {stats}")


def test_disk_guardian():
    from utils.disk_guardian import DiskGuardian
    g = DiskGuardian(warn_gb=1.0, block_gb=0.5)
    free = g.free_space_gb()
    assert free > 0
    assert g.can_write(1024)
    print(f"[OK] disk_guardian.py: {free:.1f} GB free")


def test_power_manager():
    from utils.power_manager import PowerManager
    pm = PowerManager()
    info = pm.cpu_info()
    assert "cores_physical" in info
    assert info["cores_physical"] >= 1
    mem = pm.memory_status()
    assert "total_gb" in mem
    print(f"[OK] power_manager.py: {info['cores_physical']}P cores")


def test_checksummed_lora():
    import tempfile, torch, torch.nn as nn
    from utils.checksummed_lora import save_lora, load_lora, LoRACorruptedError
    model = nn.Linear(16, 16)
    path = os.path.join(tempfile.gettempdir(), "test_lora.pt")
    cs = save_lora(model, path, rank=4)
    assert len(cs) == 64
    state, meta = load_lora(path, verify=True)
    assert meta["checksum"] == cs
    assert meta["rank"] == 4
    os.remove(path)
    print(f"[OK] checksummed_lora.py: checksum={cs[:12]}...")


def test_wuneng_fusion():
    import torch
    from brain.omega_core.wuneng_fusion import WuNengFusion, FlashSigmoidFusion
    d = 64
    fusion = WuNengFusion(d, quant_mode="fp16")
    y_ssd = torch.randn(2, 4, d)
    y_wkv = torch.randn(2, 4, d)
    out = fusion(y_ssd, y_wkv)
    assert out.shape == (2, 4, d)
    stats = fusion.get_gate_stats(y_ssd, y_wkv)
    assert 0 <= stats["gate_mean"] <= 1
    # Test FlashSigmoid variant
    flash = FlashSigmoidFusion(d, quant_mode="fp16")
    out2 = flash(y_ssd, y_wkv)
    assert out2.shape == (2, 4, d)
    print(f"[OK] wuneng_fusion.py: gate_mean={stats['gate_mean']:.3f}")


def test_spine_router():
    import torch
    from brain.min_gru.spine_router import SpineRouter
    router = SpineRouter(d_model=64, n_blocks=8, spine_dim=32)
    h = torch.randn(2, 64)
    probs = router(h)
    assert probs.shape == (2, 8)
    assert (probs >= 0).all() and (probs <= 1).all()
    mask = router.get_skip_mask(h, threshold=0.5)
    assert mask.shape == (2, 8)
    eff = router.compute_efficiency(mask)
    assert 0 <= eff <= 1
    print(f"[OK] spine_router.py: skip_probs mean={probs.mean():.3f}, efficiency={eff:.1%}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"Agent 4 Smoke Tests: {passed} passed, {failed} failed")
