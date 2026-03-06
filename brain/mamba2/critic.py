# Backward-compatible shim: brain.mamba2.critic -> brain.mamba2.verification.critic
from brain.mamba2.verification.critic import *  # noqa: F401,F403
