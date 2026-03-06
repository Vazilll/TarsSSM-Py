# Backward-compatible shim: brain.mamba2.query_router -> brain.mamba2.routing.query_router
from brain.mamba2.routing.query_router import *  # noqa: F401,F403
