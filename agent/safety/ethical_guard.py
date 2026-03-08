"""
═══════════════════════════════════════════════════════════════
  EthicalGuard — Unified Safety Gate for TARS v3 (Agent 5)
═══════════════════════════════════════════════════════════════

Combines:
  1. SafetyGate (regex blacklist — hardcoded, always available)
  2. DoubtEngine neural heads (learned — when tars_core available)
  3. Fail-closed for actions, fail-open for text

Future: SafetyHead from tars_core.get_doubt_scores() replaces regex.

Owner: Agent 5 (EXCLUSIVE)
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("Tars.Safety")


# ═══════════════════════════════════════════════════════════════
# SafetyVerdict — result of safety check
# ═══════════════════════════════════════════════════════════════

@dataclass
class SafetyVerdict:
    """Result of a safety check."""
    action: str  # "pass", "flag", "block"
    scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    timestamp: float = 0.0
    rule_id: str = ""  # which rule triggered

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    @property
    def is_blocked(self) -> bool:
        return self.action == "block"

    @property
    def is_flagged(self) -> bool:
        return self.action == "flag"

    @property
    def is_passed(self) -> bool:
        return self.action == "pass"

    def __str__(self) -> str:
        icons = {"pass": "✅", "flag": "⚠️", "block": "🚫"}
        icon = icons.get(self.action, "❓")
        scores_str = ", ".join(f"{k}={v:.2f}" for k, v in self.scores.items())
        rule = f" [{self.rule_id}]" if self.rule_id else ""
        return f"{icon} {self.action.upper()}{rule} [{scores_str}] {self.reason}"


# ═══════════════════════════════════════════════════════════════
# EthicalGuard — Unified Safety Gate
# ═══════════════════════════════════════════════════════════════

class EthicalGuard:
    """
    Unified safety gate for all TARS actions and text.

    Layer 1: Hardcoded regex blacklist (always available, no ML)
    Layer 2: Neural SafetyHead from DoubtEngine/tars_core (when available)
    Layer 3: Contextual checks (command length, suspicious patterns)

    Fail-closed for actions (shell, code, URL).
    Fail-open for text generation (crash → pass).
    """

    # ═══ Regex blacklist patterns ═══
    SHELL_BLACKLIST: List[str] = [
        # File system destruction
        r'rm\s+-r', r'rm\s+.*\s+-f',
        r'del\s+/[sfq]', r'del\s+\*\.\*',
        r'Remove-Item.*-Recurse', r'Remove-Item.*-Force',
        r'rmdir\s+/s',
        # Disk/partition
        r'format\s+[a-z]:', r'mkfs', r'dd\s+if=.*of=/dev/',
        r'diskpart',
        # System control
        r'shutdown', r'reboot', r'halt\b',
        r'init\s+[06]', r'systemctl\s+(halt|poweroff)',
        # Shell injection
        r'\|\s*(bash|sh|cmd|powershell)',
        r';\s*(bash|sh|cmd)',
        r'`[^`]+`', r'\$\([^)]+\)',
        # Encoded/obfuscated execution
        r'powershell\s+-e', r'powershell.*-enc',
        r'base64\s+-d.*\|',
        r'certutil\s+-decode', r'bitsadmin', r'mshta\b',
        r'wscript\b', r'cscript\b',
        # Process/user management
        r'wmic\s+process', r'net\s+user', r'net\s+localgroup',
        r'schtasks\s+/create', r'at\s+\d',
        # Registry
        r'reg\s+(add|delete)', r'regedit',
        # Network exfiltration
        r'curl.*\|\s*(bash|sh)', r'wget.*\|\s*(bash|sh)',
        r'nc\s+-[el]', r'ncat\b',
        # Privilege escalation
        r'sudo\s+', r'runas\b',
        r'chmod\s+[0-7]*s', r'chown\s+root',
    ]

    CODE_BLACKLIST: List[str] = [
        r'__import__', r'\beval\s*\(', r'\bexec\s*\(',
        r'\bcompile\s*\(', r'\bopen\s*\(',
        r'__class__', r'__bases__', r'__subclasses__',
        r'__globals__', r'__builtins__',
        r'os\.system', r'os\.popen', r'subprocess\.',
        r'shutil\.rmtree',
        r'ctypes\.', r'socket\.',
    ]

    URL_BLACKLIST: List[str] = [
        r'javascript:', r'data:text/html',
        r'file:///', r'vbscript:',
    ]

    # Compiled patterns (lazy init)
    _shell_re: Optional[List] = None
    _code_re: Optional[List] = None
    _url_re: Optional[List] = None

    def __init__(self, audit_logger=None):
        """
        Args:
            audit_logger: Optional AuditLogger for logging decisions.
        """
        self.audit = audit_logger
        self._neural_engine = None
        self._ensure_compiled()

    @classmethod
    def _ensure_compiled(cls):
        """Compile regex patterns once (thread-safe via class)."""
        if cls._shell_re is None:
            cls._shell_re = [re.compile(p, re.IGNORECASE) for p in cls.SHELL_BLACKLIST]
        if cls._code_re is None:
            cls._code_re = [re.compile(p, re.IGNORECASE) for p in cls.CODE_BLACKLIST]
        if cls._url_re is None:
            cls._url_re = [re.compile(p, re.IGNORECASE) for p in cls.URL_BLACKLIST]

    def attach_neural_engine(self, engine):
        """
        Attach neural DoubtEngine/tars_core for learned safety checks.

        Args:
            engine: object with get_doubt_scores() -> dict
        """
        self._neural_engine = engine
        logger.info("EthicalGuard: Neural safety engine attached")

    # ─────────────────────────────────────────
    # Main check methods
    # ─────────────────────────────────────────

    def check_shell(self, command: str) -> SafetyVerdict:
        """Check shell command safety (fail-closed)."""
        if not command or not command.strip():
            return SafetyVerdict("pass", {"safety": 1.0}, "Empty command")

        # Length check
        if len(command) > 500:
            v = SafetyVerdict(
                "flag", {"safety": 0.4, "length": len(command)},
                f"Command suspiciously long ({len(command)} chars)",
                rule_id="LEN_CHECK",
            )
            self._log_verdict("shell", command, v)
            return v

        # Regex blacklist
        for i, pattern in enumerate(self._shell_re):
            match = pattern.search(command)
            if match:
                v = SafetyVerdict(
                    "block", {"safety": 0.0},
                    f"Blocked: '{match.group(0)}' (rule #{i})",
                    rule_id=f"SHELL_{i}",
                )
                self._log_verdict("shell", command, v)
                return v

        v = SafetyVerdict("pass", {"safety": 1.0}, "Shell check passed")
        self._log_verdict("shell", command, v)
        return v

    def check_code(self, code: str) -> SafetyVerdict:
        """Check Python code safety (fail-closed)."""
        if not code or not code.strip():
            return SafetyVerdict("pass", {"safety": 1.0}, "Empty code")

        for i, pattern in enumerate(self._code_re):
            match = pattern.search(code)
            if match:
                v = SafetyVerdict(
                    "block", {"safety": 0.0},
                    f"Blocked: '{match.group(0)}' (rule #{i})",
                    rule_id=f"CODE_{i}",
                )
                self._log_verdict("code", code[:200], v)
                return v

        v = SafetyVerdict("pass", {"safety": 1.0}, "Code check passed")
        self._log_verdict("code", code[:200], v)
        return v

    def check_url(self, url: str) -> SafetyVerdict:
        """Check URL safety (fail-closed)."""
        if not url or not url.strip():
            return SafetyVerdict("pass", {"safety": 1.0}, "Empty URL")

        # Scheme whitelist
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https", ""):
            v = SafetyVerdict(
                "block", {"safety": 0.0},
                f"Blocked scheme: '{parsed.scheme}' (only http/https)",
                rule_id="URL_SCHEME",
            )
            self._log_verdict("url", url, v)
            return v

        # Shell injection chars in URL
        if any(c in url for c in [';', '|', '&', '`', '$', '\n', '\r']):
            v = SafetyVerdict(
                "block", {"safety": 0.0},
                "URL contains forbidden shell characters",
                rule_id="URL_INJECT",
            )
            self._log_verdict("url", url, v)
            return v

        # Regex blacklist
        for i, pattern in enumerate(self._url_re):
            match = pattern.search(url)
            if match:
                v = SafetyVerdict(
                    "block", {"safety": 0.0},
                    f"Blocked: '{match.group(0)}' (rule #{i})",
                    rule_id=f"URL_{i}",
                )
                self._log_verdict("url", url, v)
                return v

        v = SafetyVerdict("pass", {"safety": 1.0}, "URL check passed")
        self._log_verdict("url", url, v)
        return v

    def check(self, action_type: str, params: dict) -> SafetyVerdict:
        """
        Universal check dispatcher (backward-compatible with SafetyGate.check).

        Args:
            action_type: "shell", "execute_script", "open_url", etc.
            params: dict with "command", "code", "url", etc.
        """
        if action_type in ("shell", "run_command", "Terminal"):
            return self.check_shell(params.get("command", ""))
        elif action_type in ("execute_script", "Python", "code"):
            return self.check_code(params.get("code", ""))
        elif action_type in ("open_url", "Browser", "url"):
            return self.check_url(params.get("url", ""))
        else:
            # Unknown action type — check all text fields
            texts = []
            for key in ("command", "cmd", "code", "url", "text", "script"):
                val = params.get(key, "")
                if val:
                    texts.append(str(val))
            if not texts:
                return SafetyVerdict("pass", {"safety": 1.0}, "No text to check")
            combined = " ".join(texts)
            v = self.check_shell(combined)
            if v.is_blocked:
                return v
            return self.check_code(combined)

    def check_text(self, text: str) -> SafetyVerdict:
        """
        Check generated text safety (fail-open).

        Uses neural engine if available, otherwise passes.
        """
        if self._neural_engine is not None:
            try:
                scores = self._neural_engine.get_doubt_scores()
                safety = scores.get("safety", 1.0)
                if safety < 0.3:
                    return SafetyVerdict(
                        "block", scores,
                        f"Neural safety head: {safety:.2f} < 0.3",
                        rule_id="NEURAL_SAFETY",
                    )
                elif safety < 0.6:
                    return SafetyVerdict(
                        "flag", scores,
                        f"Neural safety head: {safety:.2f} < 0.6",
                        rule_id="NEURAL_SAFETY",
                    )
            except Exception as e:
                # Fail-open for text
                logger.debug(f"Neural safety check failed (fail-open): {e}")

        return SafetyVerdict("pass", {"safety": 1.0}, "Text check passed (no neural engine)")

    # ─────────────────────────────────────────
    # Audit logging
    # ─────────────────────────────────────────

    def _log_verdict(self, action_type: str, content: str, verdict: SafetyVerdict):
        """Log verdict to audit log if available."""
        if self.audit is not None:
            self.audit.log_safety(action_type, content[:200], verdict)
        if verdict.is_blocked:
            logger.warning(f"EthicalGuard BLOCK [{action_type}]: {verdict.reason}")
        elif verdict.is_flagged:
            logger.info(f"EthicalGuard FLAG [{action_type}]: {verdict.reason}")
