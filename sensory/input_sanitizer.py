"""
═══════════════════════════════════════════════════════════════
  input_sanitizer.py — Input Validation & Cleaning (Agent 4)
═══════════════════════════════════════════════════════════════

Validates and sanitizes user input BEFORE it reaches the brain.
Lives in the sensory layer — first line of defense.

Usage:
    from sensory.input_sanitizer import sanitize, SanitizeResult

    result = sanitize("Hello\\x00World\\x01!")
    print(result.text)       # "HelloWorld!"
    print(result.warnings)   # ["Removed 2 control characters"]
"""

import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("Tars.InputSanitizer")

# ═══════════════════════════════════════
# Constants
# ═══════════════════════════════════════
MAX_INPUT_LENGTH = 32768       # 32K chars max (matches rope_max_seq_len)
MAX_LINE_LENGTH = 4096         # single line max
MAX_CODEPOINTS = 100000        # absolute Unicode codepoint limit
REPLACEMENT_CHAR = ""          # drop bad chars (not replace)

# Control characters to strip (except \n, \r, \t)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Excessive whitespace
_MULTI_SPACE_RE = re.compile(r"[ \t]{4,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{4,}")

# Unicode confusables (homoglyph attack prevention — basic)
_CONFUSABLE_MAP = {
    "\u200b": "",   # zero-width space
    "\u200c": "",   # zero-width non-joiner
    "\u200d": "",   # zero-width joiner
    "\u2060": "",   # word joiner
    "\ufeff": "",   # BOM
    "\u00ad": "",   # soft hyphen
    "\u034f": "",   # combining grapheme joiner
}


@dataclass
class SanitizeResult:
    """Result of input sanitization."""
    text: str                           # cleaned text
    original_length: int = 0            # original char count
    final_length: int = 0               # final char count
    was_truncated: bool = False         # True if input was too long
    warnings: List[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.warnings) == 0


def sanitize(
    text: str,
    *,
    max_length: int = MAX_INPUT_LENGTH,
    normalize_unicode: bool = True,
    strip_control: bool = True,
    collapse_whitespace: bool = True,
    remove_null_bytes: bool = True,
    strip_confusables: bool = True,
) -> SanitizeResult:
    """
    Sanitize user input text.

    Steps:
    1. Remove null bytes
    2. Unicode NFC normalization
    3. Strip control characters (keep \\n, \\r, \\t)
    4. Remove Unicode confusables (zero-width chars, BOM)
    5. Collapse excessive whitespace
    6. Truncate to max_length
    7. Strip leading/trailing whitespace

    Args:
        text: raw user input
        max_length: maximum allowed character count
        normalize_unicode: apply NFC normalization
        strip_control: remove control characters
        collapse_whitespace: collapse runs of spaces/newlines
        remove_null_bytes: strip \\x00 bytes
        strip_confusables: remove zero-width and invisible chars

    Returns:
        SanitizeResult with cleaned text and warnings
    """
    if not isinstance(text, str):
        text = str(text)

    result = SanitizeResult(text="", original_length=len(text))
    warnings = result.warnings

    # 1. Null bytes
    if remove_null_bytes and "\x00" in text:
        count = text.count("\x00")
        text = text.replace("\x00", "")
        warnings.append(f"Removed {count} null bytes")

    # 2. Unicode normalization (NFC = canonical decomposition + composition)
    if normalize_unicode:
        try:
            text = unicodedata.normalize("NFC", text)
        except Exception as e:
            warnings.append(f"Unicode normalization failed: {e}")

    # 3. Control characters
    if strip_control:
        cleaned, n = _CTRL_RE.subn("", text)
        if n > 0:
            text = cleaned
            warnings.append(f"Removed {n} control characters")

    # 4. Confusables (zero-width, BOM, etc.)
    if strip_confusables:
        removed = 0
        for char, replacement in _CONFUSABLE_MAP.items():
            if char in text:
                count = text.count(char)
                text = text.replace(char, replacement)
                removed += count
        if removed > 0:
            warnings.append(f"Removed {removed} invisible/zero-width characters")

    # 5. Collapse whitespace
    if collapse_whitespace:
        text = _MULTI_SPACE_RE.sub("   ", text)    # 4+ spaces → 3
        text = _MULTI_NEWLINE_RE.sub("\n\n\n", text)  # 4+ newlines → 3

    # 6. Truncation
    if len(text) > max_length:
        text = text[:max_length]
        result.was_truncated = True
        warnings.append(f"Truncated from {result.original_length} to {max_length} chars")

    # 7. Strip edges
    text = text.strip()

    result.text = text
    result.final_length = len(text)
    return result


def is_safe_text(text: str) -> bool:
    """Quick check: is text safe (no embedded control/null)."""
    return "\x00" not in text and not _CTRL_RE.search(text)


def normalize_encoding(raw_bytes: bytes, *, preferred: str = "utf-8") -> str:
    """
    Try to decode bytes as UTF-8, fall back to cp1251 (Russian), then latin-1.
    """
    for enc in [preferred, "utf-8", "cp1251", "latin-1"]:
        try:
            return raw_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort: ignore errors
    return raw_bytes.decode("utf-8", errors="replace")
