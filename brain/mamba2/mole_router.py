# Backward-compatible shim: brain.mamba2.mole_router -> brain.mamba2.routing.mole_router
from brain.mamba2.routing.mole_router import *  # noqa: F401,F403
