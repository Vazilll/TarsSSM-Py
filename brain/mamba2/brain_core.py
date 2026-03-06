# Backward-compatible shim: brain.mamba2.brain_core -> brain.mamba2.core.brain_core
from brain.mamba2.core.brain_core import *  # noqa: F401,F403
