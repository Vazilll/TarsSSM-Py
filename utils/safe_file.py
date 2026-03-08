"""
═══════════════════════════════════════════════════════════════
  safe_file.py — Atomic File Operations (Agent 4)
═══════════════════════════════════════════════════════════════

Atomic writes: write to temp → fsync → rename.
Prevents corruption on crash/power loss.

Usage:
    from utils.safe_file import safe_write, safe_read_json, safe_write_json

    safe_write("model.safetensors", data_bytes)
    safe_write_json("config.json", {"d_model": 2048})
    cfg = safe_read_json("config.json")
"""

import json
import os
import tempfile
import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger("Tars.SafeFile")


def safe_write(path: str, data: bytes, *, make_dirs: bool = True) -> str:
    """
    Atomic binary write: temp file → fsync → rename.

    Args:
        path: target file path
        data: bytes to write
        make_dirs: create parent directories if needed

    Returns:
        absolute path of written file

    Raises:
        OSError: if write fails (original file untouched)
    """
    path = os.path.abspath(path)
    dir_name = os.path.dirname(path)

    if make_dirs:
        os.makedirs(dir_name, exist_ok=True)

    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, prefix=".safe_", suffix=".tmp")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        fd = -1  # mark as closed

        # Atomic rename (on same filesystem)
        # On Windows, need to remove target first
        if os.name == "nt" and os.path.exists(path):
            backup = path + ".bak"
            try:
                if os.path.exists(backup):
                    os.remove(backup)
                os.rename(path, backup)
                os.rename(tmp_path, path)
                os.remove(backup)
            except Exception:
                # Restore backup if rename failed
                if os.path.exists(backup) and not os.path.exists(path):
                    os.rename(backup, path)
                raise
        else:
            os.replace(tmp_path, path)

        logger.debug(f"safe_write: {path} ({len(data)} bytes)")
        return path

    except Exception:
        if fd >= 0:
            os.close(fd)
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def safe_write_text(path: str, text: str, encoding: str = "utf-8",
                    *, make_dirs: bool = True) -> str:
    """Atomic text write."""
    return safe_write(path, text.encode(encoding), make_dirs=make_dirs)


def safe_write_json(path: str, obj: Any, *, indent: int = 2,
                    make_dirs: bool = True) -> str:
    """Atomic JSON write with pretty-print."""
    text = json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    return safe_write_text(path, text + "\n", make_dirs=make_dirs)


def safe_read_json(path: str, default: Any = None) -> Any:
    """
    Read JSON file with error handling.

    Returns:
        parsed JSON object, or `default` if file doesn't exist or is corrupted
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"safe_read_json: Failed to read {path}: {e}")
        return default


def file_checksum(path: str, algorithm: str = "sha256") -> str:
    """Compute file checksum (SHA256 by default)."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_backup(path: str, max_backups: int = 3) -> Optional[str]:
    """
    Create numbered backup of a file. Keeps at most max_backups.

    Returns:
        backup path, or None if source doesn't exist
    """
    if not os.path.exists(path):
        return None

    # Rotate backups
    for i in range(max_backups - 1, 0, -1):
        old = f"{path}.bak.{i}"
        new = f"{path}.bak.{i + 1}"
        if os.path.exists(old):
            if i + 1 > max_backups:
                os.remove(old)
            else:
                if os.path.exists(new):
                    os.remove(new)
                os.rename(old, new)

    backup_path = f"{path}.bak.1"
    if os.path.exists(backup_path):
        os.remove(backup_path)

    import shutil
    shutil.copy2(path, backup_path)
    logger.debug(f"safe_backup: {path} → {backup_path}")
    return backup_path
