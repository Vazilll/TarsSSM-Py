# Backward-compatible shim: brain.mamba2.inference_engine -> brain.mamba2.inference.inference_engine
from brain.mamba2.inference.inference_engine import *  # noqa: F401,F403
