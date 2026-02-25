"""
ssm_vad.py — Voice Activity Detection using SSM resonance.
Uses OmegaCore (C++ or Python fallback) for efficient audio analysis.
"""
import numpy as np
import logging

logger = logging.getLogger("SSM_VAD")


class SsmVAD:
    """SSM-based Voice Activity Detection using resonant bit-matrices."""

    def __init__(self, dim=256, threshold=0.8):
        self.dim = dim
        self.threshold = threshold
        self._state = np.zeros(dim, dtype=np.float32)

        # Try to load OmegaCore
        try:
            from brain.omega_core import get_omega_core
            self.core = get_omega_core()
        except ImportError:
            self.core = None

    def detect(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if audio chunk contains speech.
        Returns True if voice detected.
        """
        if len(audio_chunk) == 0:
            return False

        # Simple energy-based detection as baseline
        energy = np.sqrt(np.mean(audio_chunk.astype(np.float32) ** 2))
        if energy < 0.01:
            return False

        # SSM resonance check (if core available)
        if self.core and hasattr(self.core, 'integral_audit'):
            # Project audio into SSM space
            chunk_proj = np.zeros(self.dim, dtype=np.float32)
            step = max(1, len(audio_chunk) // self.dim)
            for i in range(min(self.dim, len(audio_chunk))):
                chunk_proj[i] = audio_chunk[i * step] if i * step < len(audio_chunk) else 0.0

            # Use integral audit to check for speech pattern convergence
            f_history = np.abs(np.diff(chunk_proj[:32])).tolist()
            if len(f_history) >= 3:
                p_audio = self.core.integral_audit(f_history, window=8)
                return p_audio > self.threshold

        # Fallback: simple energy threshold
        return energy > 0.05

    def reset(self):
        self._state = np.zeros(self.dim, dtype=np.float32)
