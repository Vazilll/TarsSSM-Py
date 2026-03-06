# Backward-compatible shim: brain.mamba2.tars_block -> brain.mamba2.core.tars_block
from brain.mamba2.core.tars_block import *  # noqa: F401,F403
