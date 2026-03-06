# Backward-compatible shim: brain.mamba2.generate_mamba -> brain.mamba2.inference.generate_mamba
from brain.mamba2.inference.generate_mamba import *  # noqa: F401,F403
