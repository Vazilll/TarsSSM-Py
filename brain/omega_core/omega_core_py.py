"""
OmegaCore Python Wrapper — ctypes bridge to the pure C++ kernel.
Zero dependencies beyond stdlib + numpy/torch for tensor conversion.

Usage:
    from brain.omega_core.omega_core_py import OmegaCore
    core = OmegaCore()
    if core.available:
        output = core.bit_linear(input_tensor, packed_weights, scale_x, scale_w)
        h_next = core.ssm_step(x, h, A, B, dt)
        token  = core.sample(logits, temp=0.8)
        p_val  = core.integral_audit(f_history)
        ratio  = core.hankel_rank(state_history_2d)
"""
import ctypes
import os
import sys
import logging
import numpy as np

logger = logging.getLogger("OmegaCore")


class OmegaCore:
    """Lightweight ctypes wrapper for omega_core.dll / .so"""

    _instance = None
    _lib = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """Try to load the compiled DLL/SO."""
        core_dir = os.path.dirname(os.path.abspath(__file__))
        if sys.platform == "win32":
            candidates = [
                os.path.join(core_dir, "omega_core.dll"),
                os.path.join(core_dir, "omega_core_pure.dll"),
            ]
        else:
            candidates = [
                os.path.join(core_dir, "omega_core.so"),
                os.path.join(core_dir, "libomega_core.so"),
            ]

        for path in candidates:
            if os.path.exists(path):
                try:
                    self._lib = ctypes.CDLL(path)
                    self._setup_signatures()
                    self._lib.omega_init()
                    logger.info(f"OmegaCore: Loaded {os.path.basename(path)}")
                    return
                except Exception as e:
                    logger.warning(f"OmegaCore: Failed to load {path}: {e}")

        logger.warning("OmegaCore: No compiled binary found. Using pure Python fallback.")
        self._lib = None

    def _setup_signatures(self):
        """Define C function signatures for type safety."""
        L = self._lib
        c_float_p = ctypes.POINTER(ctypes.c_float)
        c_uint8_p = ctypes.POINTER(ctypes.c_uint8)

        L.omega_init.restype = None
        L.omega_init.argtypes = []

        L.omega_version.restype = ctypes.c_int
        L.omega_version.argtypes = []

        L.bit_linear.restype = None
        L.bit_linear.argtypes = [
            c_float_p, ctypes.c_int,       # input, cols
            c_uint8_p, ctypes.c_int,       # weights_packed, rows
            ctypes.c_float, ctypes.c_float, # scale_x, scale_w
            c_float_p                       # output
        ]

        L.ssm_step.restype = None
        L.ssm_step.argtypes = [
            c_float_p, c_float_p,   # x, h_prev
            c_float_p, c_float_p,   # A, B
            ctypes.c_float,          # dt
            ctypes.c_int,            # dim
            c_float_p                # h_next
        ]

        L.cayley_update.restype = None
        L.cayley_update.argtypes = [
            c_float_p, c_float_p,   # skew_S, h_prev
            ctypes.c_int,            # dim
            c_float_p                # h_next
        ]

        L.advanced_sample.restype = ctypes.c_int
        L.advanced_sample.argtypes = [
            c_float_p, ctypes.c_int,   # logits, vocab_size
            ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int  # temp, min_p, top_p, top_k
        ]

        L.integral_audit.restype = ctypes.c_float
        L.integral_audit.argtypes = [c_float_p, ctypes.c_int, ctypes.c_int]

        L.hankel_rank.restype = ctypes.c_float
        L.hankel_rank.argtypes = [c_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

    @property
    def available(self) -> bool:
        return self._lib is not None

    @property
    def version(self) -> int:
        return self._lib.omega_version() if self._lib else 0

    # ---- High-level API (accepts numpy arrays or torch tensors) ----

    @staticmethod
    def _to_np(t) -> np.ndarray:
        """Convert torch.Tensor or list to contiguous float32 numpy array."""
        if hasattr(t, 'detach'):  # torch.Tensor
            return t.detach().cpu().numpy().astype(np.float32).ravel()
        return np.ascontiguousarray(t, dtype=np.float32).ravel()

    @staticmethod
    def _ptr(arr):
        return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def bit_linear(self, input_vec, weights_packed, scale_x=1.0, scale_w=1.0):
        """BitLinear: Y = W_ternary @ X. Returns numpy array [rows]."""
        inp = self._to_np(input_vec)
        cols = len(inp)
        w = np.ascontiguousarray(weights_packed, dtype=np.uint8)
        rows = len(w) // ((cols + 3) // 4)
        out = np.zeros(rows, dtype=np.float32)
        self._lib.bit_linear(
            self._ptr(inp), cols,
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), rows,
            ctypes.c_float(scale_x), ctypes.c_float(scale_w),
            self._ptr(out)
        )
        return out

    def ssm_step(self, x, h_prev, A, B, dt=1.0):
        """SSM state update. Returns h_next as numpy [dim]."""
        x_np = self._to_np(x)
        h_np = self._to_np(h_prev)
        dim = len(h_np)
        A_np = self._to_np(A)
        B_np = self._to_np(B)
        h_next = np.zeros(dim, dtype=np.float32)
        self._lib.ssm_step(
            self._ptr(x_np), self._ptr(h_np),
            self._ptr(A_np), self._ptr(B_np),
            ctypes.c_float(dt), dim,
            self._ptr(h_next)
        )
        return h_next

    def cayley_update(self, skew_S, h_prev):
        """SO(n) Cayley transform state update. Returns h_next [dim]."""
        h_np = self._to_np(h_prev)
        dim = len(h_np)
        S_np = self._to_np(skew_S)
        h_next = np.zeros(dim, dtype=np.float32)
        self._lib.cayley_update(self._ptr(S_np), self._ptr(h_np), dim, self._ptr(h_next))
        return h_next

    def sample(self, logits, temp=0.8, min_p=0.05, top_p=0.95, top_k=50):
        """Sample token from logits. Returns token ID (int)."""
        l_np = self._to_np(logits)
        return self._lib.advanced_sample(
            self._ptr(l_np), len(l_np),
            ctypes.c_float(temp), ctypes.c_float(min_p),
            ctypes.c_float(top_p), top_k
        )

    def integral_audit(self, f_history, window=8):
        """Compute p-convergence coefficient. Returns float."""
        f_np = self._to_np(f_history)
        return self._lib.integral_audit(self._ptr(f_np), len(f_np), window)

    def hankel_rank(self, state_history_2d, hankel_rows=5):
        """Compute Hankel σ₂/σ₁ ratio. Low = looping. Returns float."""
        s = np.ascontiguousarray(state_history_2d, dtype=np.float32)
        n_steps, dim = s.shape
        return self._lib.hankel_rank(
            self._ptr(s.ravel()), n_steps, dim, hankel_rows
        )


# ---- Pure Python Fallback (if DLL not compiled) ----

class OmegaCoreFallback:
    """Pure Python/NumPy implementation for systems without compiled kernel."""
    available = False
    version = 0

    def ssm_step(self, x, h_prev, A, B, dt=1.0):
        x = np.asarray(x, dtype=np.float32).ravel()
        h = np.asarray(h_prev, dtype=np.float32).ravel()
        A = np.asarray(A, dtype=np.float32).reshape(len(h), len(h))
        B = np.asarray(B, dtype=np.float32).ravel()
        return h + (A @ h + B * x) * dt

    def sample(self, logits, temp=0.8, **_):
        logits = np.asarray(logits, dtype=np.float32).ravel()
        if temp < 0.05:
            return int(np.argmax(logits))
        logits = logits / temp
        logits -= logits.max()
        probs = np.exp(logits) / np.exp(logits).sum()
        return int(np.random.choice(len(probs), p=probs))

    def integral_audit(self, f_history, window=8):
        f = np.asarray(f_history, dtype=np.float32)
        if len(f) < 3: return 0.0
        f = f[-window:]
        f = np.clip(f, 1e-12, None)
        x = np.log(np.arange(1, len(f) + 1, dtype=np.float32))
        y = np.log(f)
        n = len(x)
        b = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x**2).sum() - x.sum()**2 + 1e-15)
        return float(-b)

    def hankel_rank(self, state_history_2d, hankel_rows=5):
        return 1.0  # No collapse detection in fallback

    def cayley_update(self, skew_S, h_prev):
        S = np.asarray(skew_S, dtype=np.float32).reshape(len(h_prev), -1)
        h = np.asarray(h_prev, dtype=np.float32)
        Sh = S @ h
        return h + Sh + 0.5 * (S @ Sh)


def get_omega_core():
    """Factory: returns OmegaCore (C++) if available, else OmegaCoreFallback (Python)."""
    core = OmegaCore()
    if core.available:
        return core
    logger.info("OmegaCore: Falling back to pure Python implementation.")
    return OmegaCoreFallback()
