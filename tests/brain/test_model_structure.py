"""TarsMamba2LM structural tests — weights_only, speculative verification."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestModelStructure:
    """TarsMamba2LM structural tests."""

    def test_weights_only_true(self):
        import inspect
        from brain.mamba2.model import TarsMamba2LM
        assert "weights_only=True" in inspect.getsource(TarsMamba2LM.load_pretrained)

    def test_no_shadow_os_import(self):
        from pathlib import Path
        ROOT = Path(__file__).resolve().parent.parent.parent
        source = (ROOT / "brain" / "mamba2" / "model.py").read_text(encoding='utf-8')
        assert "import os as _os" not in source

    def test_speculative_has_verification(self):
        import inspect
        from brain.mamba2.model import TarsMamba2LM
        source = inspect.getsource(TarsMamba2LM.generate_speculative)
        assert "verify" in source.lower() or "snapshot" in source.lower()

    def test_has_inference_engine_property(self):
        from brain.mamba2.model import TarsMamba2LM
        assert hasattr(TarsMamba2LM, 'inference_engine')
