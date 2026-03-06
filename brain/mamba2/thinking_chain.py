# Backward-compatible shim: brain.mamba2.thinking_chain -> brain.mamba2.inference.thinking_chain
from brain.mamba2.inference.thinking_chain import *  # noqa: F401,F403
