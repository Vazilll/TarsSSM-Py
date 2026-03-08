"""
═══════════════════════════════════════════════════════════════
  PromptDefense — Injection & Jailbreak Detection (Agent 5)
═══════════════════════════════════════════════════════════════

Detects and blocks:
  1. System prompt extraction attempts
  2. Role hijacking ("ignore previous instructions")
  3. Encoding bypass (base64, hex, unicode tricks)
  4. Context manipulation
  5. Indirect injection from retrieved documents

Future: neural detection head via tars_core.SafetyHead

Owner: Agent 5 (EXCLUSIVE)
"""

import re
import logging
from typing import Tuple, List
from dataclasses import dataclass

logger = logging.getLogger("Tars.PromptDefense")


@dataclass
class DefenseResult:
    """Result of prompt defense check."""
    is_safe: bool
    threat_type: str = ""  # "injection", "jailbreak", "extraction", "encoding"
    confidence: float = 1.0
    matched_pattern: str = ""

    def __str__(self):
        if self.is_safe:
            return "✅ Safe"
        return f"🚫 {self.threat_type}: {self.matched_pattern} (conf={self.confidence:.0%})"


class PromptDefense:
    """
    Multi-layer prompt injection detection.

    Layer 1: Pattern matching (high-precision rules)
    Layer 2: Structural analysis (delimiter detection)
    Layer 3: Neural detection (future, via tars_core)
    """

    # ═══ Layer 1: Injection patterns (case-insensitive) ═══
    INJECTION_PATTERNS: List[Tuple[str, str]] = [
        # System prompt extraction
        (r"(repeat|show|print|output|display|reveal)\s*(your|the|system)\s*(prompt|instructions|rules|system\s*message)",
         "extraction"),
        (r"what\s*(are|is)\s*your\s*(instructions|system\s*prompt|rules|directives)",
         "extraction"),
        (r"(tell|give)\s*me\s*your\s*(system|initial|original)\s*(prompt|message|instructions)",
         "extraction"),

        # Role hijacking
        (r"ignore\s*(all\s*)?(previous|prior|above|earlier)\s*(instructions|rules|context|prompts)",
         "jailbreak"),
        (r"(forget|disregard|override)\s*(everything|all)\s*(above|before|you\s*were\s*told)",
         "jailbreak"),
        (r"you\s*are\s*now\s*(a|an|my)\s*(unrestricted|uncensored|unfiltered|evil)",
         "jailbreak"),
        (r"from\s*now\s*on\s*(you|act|behave|respond)\s*(as|like)",
         "jailbreak"),
        (r"(DAN|do\s*anything\s*now|jailbreak|developer\s*mode|god\s*mode)",
         "jailbreak"),
        (r"pretend\s*(you\s*are|to\s*be|that)\s*(an?\s*)?(evil|unrestricted|hacker|malicious)",
         "jailbreak"),

        # Delimiter injection
        (r"<\|?(system|endoftext|im_start|im_end)\|?>",
         "injection"),
        (r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]",
         "injection"),
        (r"###\s*(System|Human|Assistant|User)\s*:",
         "injection"),

        # Context manipulation — RU
        (r"(игнорируй|забудь|отмени)\s*(все\s*)?(предыдущие|прошлые|системные)\s*(инструкции|правила|команды)",
         "jailbreak"),
        (r"ты\s*теперь\s*(злой|без\s*ограничений|хакер|свободный)",
         "jailbreak"),
        (r"(покажи|выведи|повтори)\s+.{0,30}(промпт|инструкц|правила)",
         "extraction"),
    ]

    _compiled = None

    def __init__(self):
        self._ensure_compiled()

    @classmethod
    def _ensure_compiled(cls):
        if cls._compiled is None:
            cls._compiled = [
                (re.compile(pat, re.IGNORECASE | re.DOTALL), threat)
                for pat, threat in cls.INJECTION_PATTERNS
            ]

    def check(self, text: str) -> DefenseResult:
        """
        Check user input for prompt injection / jailbreak attempts.

        Args:
            text: raw user input

        Returns:
            DefenseResult with safety assessment
        """
        if not text or not text.strip():
            return DefenseResult(is_safe=True)

        # Layer 1: Pattern matching
        for pattern, threat_type in self._compiled:
            match = pattern.search(text)
            if match:
                result = DefenseResult(
                    is_safe=False,
                    threat_type=threat_type,
                    confidence=0.95,
                    matched_pattern=match.group(0)[:80],
                )
                logger.warning(f"PromptDefense: {result}")
                return result

        # Layer 2: Structural checks
        structural = self._check_structural(text)
        if not structural.is_safe:
            return structural

        return DefenseResult(is_safe=True)

    def _check_structural(self, text: str) -> DefenseResult:
        """Structural analysis for less obvious injection."""

        # Excessive special characters (encoding bypass)
        special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
        if special_ratio > 0.4 and len(text) > 50:
            return DefenseResult(
                is_safe=False,
                threat_type="encoding",
                confidence=0.7,
                matched_pattern=f"High special char ratio: {special_ratio:.0%}",
            )

        # Multiple role markers in single input
        role_markers = len(re.findall(
            r'(system|human|assistant|user|bot)\s*:', text, re.IGNORECASE
        ))
        if role_markers >= 3:
            return DefenseResult(
                is_safe=False,
                threat_type="injection",
                confidence=0.8,
                matched_pattern=f"{role_markers} role markers detected",
            )

        return DefenseResult(is_safe=True)

    def sanitize(self, text: str) -> str:
        """
        Sanitize user input by removing known injection delimiters.

        Does NOT block — just strips dangerous tokens.
        Use check() for blocking decisions.
        """
        # Remove system prompt delimiters
        sanitized = re.sub(r'<\|?(system|endoftext|im_start|im_end)\|?>', '', text)
        sanitized = re.sub(r'\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]', '', sanitized)
        return sanitized.strip()

    def check_retrieved_document(self, doc_text: str) -> DefenseResult:
        """
        Check retrieved/RAG document for indirect injection.

        Documents from external sources may contain hidden instructions.
        """
        if not doc_text:
            return DefenseResult(is_safe=True)

        # Check for instruction patterns embedded in documents
        indirect_patterns = [
            (r"(IMPORTANT|NOTE|INSTRUCTION):\s*(ignore|forget|disregard)", "injection"),
            (r"<hidden>(.*?)</hidden>", "injection"),
            (r"<!--\s*(instruction|system|ignore)", "injection"),
        ]

        for pat_str, threat in indirect_patterns:
            match = re.search(pat_str, doc_text, re.IGNORECASE | re.DOTALL)
            if match:
                return DefenseResult(
                    is_safe=False,
                    threat_type=f"indirect_{threat}",
                    confidence=0.85,
                    matched_pattern=match.group(0)[:80],
                )

        return DefenseResult(is_safe=True)
