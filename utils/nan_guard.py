"""
═══════════════════════════════════════════════════════════════
  nan_guard.py — NaN/Inf Detection & Recovery (Agent 4)
═══════════════════════════════════════════════════════════════

Forward hooks that catch NaN/Inf in module outputs.
Critical for 24/7 operation — prevents silent corruption.

Usage:
    from utils.nan_guard import NanGuard

    guard = NanGuard(model)
    guard.enable()       # register hooks
    output = model(x)   # NaN checked after each module
    guard.disable()      # remove hooks
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, List, Dict

logger = logging.getLogger("Tars.NanGuard")


class NanGuard:
    """
    Register forward hooks on all modules to detect NaN/Inf.

    Options:
        replace_nan: if True, replace NaN with zeros (keep running)
        raise_on_nan: if True, raise RuntimeError on first NaN
        log_only: just log, don't fix or raise
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        replace_nan: bool = True,
        raise_on_nan: bool = False,
        log_only: bool = False,
        check_grad: bool = False,
    ):
        self.model = model
        self.replace_nan = replace_nan
        self.raise_on_nan = raise_on_nan
        self.log_only = log_only
        self.check_grad = check_grad
        self._hooks: List = []
        self._nan_counts: Dict[str, int] = {}
        self._enabled = False

    def enable(self) -> "NanGuard":
        """Register hooks on all modules."""
        if self._enabled:
            return self

        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(
                self._make_hook(name)
            )
            self._hooks.append(hook)

            if self.check_grad:
                try:
                    bh = module.register_full_backward_hook(
                        self._make_grad_hook(name)
                    )
                    self._hooks.append(bh)
                except Exception:
                    pass  # Some modules don't support backward hooks

        self._enabled = True
        logger.info(f"NanGuard enabled on {len(self._hooks)} modules")
        return self

    def disable(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._enabled = False
        logger.info("NanGuard disabled")

    def _make_hook(self, name: str):
        """Create a forward hook for a named module."""
        def hook(module, input, output):
            self._check_output(name, output)
        return hook

    def _make_grad_hook(self, name: str):
        """Create a backward hook for gradient checking."""
        def hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_output):
                if g is not None and isinstance(g, torch.Tensor):
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        logger.error(
                            f"NanGuard: NaN/Inf in GRADIENT of '{name}' "
                            f"output[{i}], shape={g.shape}"
                        )
        return hook

    def _check_output(self, name: str, output) -> None:
        """Check a module's output for NaN/Inf."""
        tensors = self._extract_tensors(output)
        for i, t in enumerate(tensors):
            has_nan = torch.isnan(t).any()
            has_inf = torch.isinf(t).any()

            if has_nan or has_inf:
                issue = []
                if has_nan:
                    issue.append(f"NaN({torch.isnan(t).sum().item()})")
                if has_inf:
                    issue.append(f"Inf({torch.isinf(t).sum().item()})")

                key = f"{name}[{i}]"
                self._nan_counts[key] = self._nan_counts.get(key, 0) + 1

                msg = (
                    f"NanGuard: {'+'.join(issue)} in '{name}' "
                    f"output[{i}], shape={t.shape}, "
                    f"count={self._nan_counts[key]}"
                )
                logger.error(msg)

                if self.raise_on_nan:
                    raise RuntimeError(msg)

                if self.replace_nan and not self.log_only:
                    # Replace NaN with 0, Inf with large finite
                    t.nan_to_num_(nan=0.0, posinf=1e4, neginf=-1e4)
                    logger.warning(f"NanGuard: replaced NaN/Inf in '{name}'")

    @staticmethod
    def _extract_tensors(output) -> List[torch.Tensor]:
        """Extract all tensors from output (handles tuples, dicts, etc.)."""
        tensors = []
        if isinstance(output, torch.Tensor):
            tensors.append(output)
        elif isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    tensors.append(item)
        elif isinstance(output, dict):
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    tensors.append(v)
        return tensors

    def get_stats(self) -> dict:
        """Return NaN occurrence statistics."""
        return {
            "enabled": self._enabled,
            "total_nan_events": sum(self._nan_counts.values()),
            "affected_modules": len(self._nan_counts),
            "details": dict(self._nan_counts),
        }

    def reset_stats(self) -> None:
        """Reset NaN counters."""
        self._nan_counts.clear()
