# Backward-compatible shim: brain.mamba2.model → brain.mamba2.core.model
from brain.mamba2.core.model import *  # noqa: F401,F403
try:
    from brain.mamba2.core.model import TarsMamba2LM  # noqa: F401
except ImportError:
    pass
