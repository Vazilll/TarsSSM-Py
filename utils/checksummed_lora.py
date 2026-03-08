"""
═══════════════════════════════════════════════════════════════
  checksummed_lora.py — LoRA with Integrity Checks (Agent 4)
═══════════════════════════════════════════════════════════════

Save/load LoRA adapters with SHA256 checksums in metadata.
Prevents loading corrupted adapters that could crash inference.

Usage:
    from utils.checksummed_lora import save_lora, load_lora

    save_lora(model, "adapters/expert_code.pt")
    state, meta = load_lora("adapters/expert_code.pt")
    # meta = {"checksum": "abc...", "params": 16384, "rank": 8}
"""

import hashlib
import io
import logging
import os
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger("Tars.ChecksummedLoRA")

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def _compute_state_checksum(state_dict: dict) -> str:
    """Compute SHA256 checksum of state dict tensors."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(key.encode("utf-8"))
        tensor = state_dict[key]
        if hasattr(tensor, "numpy"):
            h.update(tensor.cpu().numpy().tobytes())
        elif hasattr(tensor, "tobytes"):
            h.update(tensor.tobytes())
        else:
            h.update(str(tensor).encode())
    return h.hexdigest()


def save_lora(
    model_or_state: Any,
    path: str,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    description: str = "",
    extra_meta: Optional[dict] = None,
) -> str:
    """
    Save LoRA adapter with checksum metadata.

    Args:
        model_or_state: nn.Module (extracts LoRA params) or state_dict
        path: save path
        rank: LoRA rank (for metadata)
        alpha: LoRA alpha (for metadata)
        description: human-readable description
        extra_meta: additional metadata dict

    Returns:
        checksum string
    """
    if torch is None:
        raise RuntimeError("PyTorch not available")

    # Extract state dict
    if isinstance(model_or_state, dict):
        state = model_or_state
    elif hasattr(model_or_state, "state_dict"):
        # Filter LoRA parameters only
        full_state = model_or_state.state_dict()
        state = {
            k: v for k, v in full_state.items()
            if "lora" in k.lower() or "adapter" in k.lower()
               or ".A." in k or ".B." in k
        }
        if not state:
            # Fallback: save everything
            state = full_state
            logger.warning("No LoRA parameters found, saving full state dict")
    else:
        raise TypeError(f"Expected dict or nn.Module, got {type(model_or_state)}")

    # Compute checksum
    checksum = _compute_state_checksum(state)

    # Count parameters
    total_params = sum(
        v.numel() for v in state.values()
        if hasattr(v, "numel")
    )

    # Build metadata
    meta = {
        "checksum": checksum,
        "params": total_params,
        "rank": rank,
        "alpha": alpha,
        "description": description,
        "format_version": 1,
    }
    if extra_meta:
        meta.update(extra_meta)

    # Save
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_dict = {
        "lora_state_dict": state,
        "lora_metadata": meta,
    }
    torch.save(save_dict, path)
    logger.info(
        f"LoRA saved: {path} ({total_params} params, "
        f"rank={rank}, checksum={checksum[:12]}...)"
    )
    return checksum


def load_lora(
    path: str,
    *,
    verify: bool = True,
    device: str = "cpu",
) -> Tuple[dict, dict]:
    """
    Load LoRA adapter with integrity verification.

    Args:
        path: path to saved LoRA
        verify: if True, verify checksum (recommended)
        device: target device

    Returns:
        (state_dict, metadata) tuple

    Raises:
        LoRACorruptedError: if checksum doesn't match
        FileNotFoundError: if file doesn't exist
    """
    if torch is None:
        raise RuntimeError("PyTorch not available")

    if not os.path.exists(path):
        raise FileNotFoundError(f"LoRA file not found: {path}")

    # Load with safety
    try:
        save_dict = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        logger.warning("weights_only=True failed, trying legacy load")
        save_dict = torch.load(path, map_location=device, weights_only=False)

    # Handle both new format (with metadata) and legacy (bare state dict)
    if isinstance(save_dict, dict) and "lora_state_dict" in save_dict:
        state = save_dict["lora_state_dict"]
        meta = save_dict.get("lora_metadata", {})
    else:
        # Legacy format: just a state dict
        state = save_dict if isinstance(save_dict, dict) else save_dict.get("model_state_dict", {})
        meta = {"format_version": 0, "checksum": "unknown"}

    # Verify checksum
    if verify and meta.get("checksum") and meta["checksum"] != "unknown":
        actual = _compute_state_checksum(state)
        expected = meta["checksum"]
        if actual != expected:
            msg = (
                f"LoRA CORRUPTED: checksum mismatch!\n"
                f"  Expected: {expected}\n"
                f"  Got:      {actual}\n"
                f"  File:     {path}"
            )
            logger.error(msg)
            raise LoRACorruptedError(msg)
        logger.debug(f"LoRA checksum verified: {actual[:12]}...")

    logger.info(
        f"LoRA loaded: {path} "
        f"({meta.get('params', '?')} params, "
        f"rank={meta.get('rank', '?')})"
    )
    return state, meta


class LoRACorruptedError(ValueError):
    """Raised when a LoRA adapter fails integrity check."""
    pass
