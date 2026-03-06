# Backward-compatible shim: brain.mamba2.matrix_pool -> brain.mamba2.routing.matrix_pool
from brain.mamba2.routing.matrix_pool import *  # noqa: F401,F403
