"""
═══════════════════════════════════════════════════════════════
  DoubtEngine — Adversarial Verification System (System 0)
═══════════════════════════════════════════════════════════════

Микро-классификатор (~200K params, <5ms на CPU), работающий
НЕЗАВИСИМО от Brain и MinGRU.

Принцип: DoubtEngine НИКОГДА не генерирует текст.
Он только проверяет и сомневается.

3 головы сомнения:
  CoherenceHead — «Ответ логичен?»
  SafetyHead    — «Это безопасно?»
  RepeatHead    — «Не зациклилось?»

Вердикты:
  ✅ PASS  — ответ/действие нормальное
  ⚠️ FLAG  — подозрительно, записать в лог
  🚫 BLOCK — опасно, требует подтверждение

Безопасность:
  Fail-open для текста (crash → pass)
  Fail-closed для действий (crash → block)

Файлы: brain/doubt_engine.py [THIS], model.py (inter-wave),
        tars_agent.py (pre-action gate).
"""

import re
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import Counter

logger = logging.getLogger("Tars.DoubtEngine")


# ═══════════════════════════════════════════════════════════════
# 1. DoubtVerdict — результат верификации
# ═══════════════════════════════════════════════════════════════

@dataclass
class DoubtVerdict:
    """Результат верификации DoubtEngine."""
    action: str          # "pass", "flag", "block"
    scores: Dict[str, float] = field(default_factory=dict)
    reason: str = ""
    timestamp: float = 0.0

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
        return f"{icon} {self.action.upper()} [{scores_str}] {self.reason}"


# ═══════════════════════════════════════════════════════════════
# 2. SafetyGate — Hardcoded Safety Blacklist
# ═══════════════════════════════════════════════════════════════

class SafetyGate:
    """
    Hardcoded blacklist для опасных команд.

    Не использует нейросеть — чисто regex-based.
    Гарантирует блокировку даже если DoubtEngine упал.

    Принцип: fail-closed для действий.
    Если не уверен — блокируй.
    """

    # Опасные паттерны (regex, case-insensitive)
    BLACKLIST: List[str] = [
        # === File system destruction ===
        r'rm\s+-r',                         # rm -rf
        r'rm\s+.*\s+-f',                    # rm ... -f
        r'del\s+/[sfq]',                    # del /s /f /q
        r'del\s+\*\.\*',                    # del *.*
        r'Remove-Item.*-Recurse',           # PowerShell recursive delete
        r'Remove-Item.*-Force',             # PowerShell force delete
        r'rmdir\s+/s',                      # rmdir /s

        # === Disk/partition destruction ===
        r'format\s+[a-z]:',                 # format C:
        r'mkfs',                            # mkfs.ext4
        r'dd\s+if=.*of=/dev/',              # dd to raw device
        r'diskpart',                        # Windows disk management

        # === System control ===
        r'shutdown',                        # shutdown
        r'reboot',                          # reboot
        r'halt\b',                          # halt
        r'init\s+[06]',                     # init 0 / init 6
        r'systemctl\s+(halt|poweroff)',     # systemctl halt

        # === Shell injection ===
        r'\|\s*(bash|sh|cmd|powershell)',    # pipe to shell
        r';\s*(bash|sh|cmd)',               # chain to shell
        r'`[^`]+`',                         # backtick execution
        r'\$\([^)]+\)',                      # command substitution

        # === Encoded/obfuscated execution ===
        r'powershell\s+-e',                 # encoded commands
        r'powershell.*-enc',                # encoded
        r'base64\s+-d.*\|',                 # base64 decode pipe
        r'certutil\s+-decode',              # Windows LOLBin
        r'bitsadmin',                       # Windows LOLBin
        r'mshta\b',                         # Windows LOLBin
        r'wscript\b',                       # Windows script host
        r'cscript\b',                       # Windows script host

        # === Process/user management ===
        r'wmic\s+process',                  # WMI process exec
        r'net\s+user',                      # user management
        r'net\s+localgroup',                # group management
        r'schtasks\s+/create',              # scheduled tasks
        r'at\s+\d',                         # at scheduler

        # === Registry modification ===
        r'reg\s+(add|delete)',              # registry
        r'regedit',                         # registry editor

        # === Network exfiltration ===
        r'curl.*\|\s*(bash|sh)',            # curl pipe execution
        r'wget.*\|\s*(bash|sh)',            # wget pipe execution
        r'nc\s+-[el]',                      # netcat listen/exec
        r'ncat\b',                          # ncat

        # === Privilege escalation ===
        r'sudo\s+',                         # sudo
        r'runas\b',                         # Windows runas
        r'chmod\s+[0-7]*s',                # setuid
        r'chown\s+root',                    # chown to root
    ]

    # Compiled patterns (cached)
    _compiled: Optional[List] = None

    @classmethod
    def _ensure_compiled(cls):
        """Compile regex patterns once."""
        if cls._compiled is None:
            cls._compiled = [
                re.compile(p, re.IGNORECASE) for p in cls.BLACKLIST
            ]

    @classmethod
    def check(cls, action: str, params: dict) -> DoubtVerdict:
        """
        Проверить действие на безопасность.

        Args:
            action: тип действия (shell, execute_script, open_url, etc.)
            params: параметры действия (command, code, url, etc.)

        Returns:
            DoubtVerdict с результатом проверки
        """
        cls._ensure_compiled()

        # Собираем текст для проверки из всех параметров
        texts_to_check = []
        for key in ("command", "cmd", "code", "url", "text", "script"):
            val = params.get(key, "")
            if val:
                texts_to_check.append(str(val))

        # Если нет текста для проверки — пропускаем
        if not texts_to_check:
            return DoubtVerdict(
                action="pass",
                scores={"safety": 1.0},
                reason="No actionable text to check",
            )

        combined = " ".join(texts_to_check)

        # Проверяем каждый паттерн
        for i, pattern in enumerate(cls._compiled):
            match = pattern.search(combined)
            if match:
                matched_text = match.group(0)
                return DoubtVerdict(
                    action="block",
                    scores={"safety": 0.0},
                    reason=f"Dangerous pattern detected: '{matched_text}' "
                           f"(rule #{i}: {cls.BLACKLIST[i]})",
                )

        # Дополнительная проверка: слишком длинные команды
        if len(combined) > 500:
            return DoubtVerdict(
                action="flag",
                scores={"safety": 0.4},
                reason=f"Command suspiciously long ({len(combined)} chars)",
            )

        return DoubtVerdict(
            action="pass",
            scores={"safety": 1.0},
            reason="Passed safety check",
        )


# ═══════════════════════════════════════════════════════════════
# 3. DoubtEngine — Neural Verification Module
# ═══════════════════════════════════════════════════════════════

class DoubtEngine(nn.Module):
    """
    Adversarial Verifier (~200K params, <5ms на CPU).

    Архитектура:
      Stem: Linear(d_model*2, 128) → SiLU → Linear(128, 128)
      CoherenceHead: Linear(128, 1) — логичность ответа
      SafetyHead:    Linear(128, 1) — безопасность действия
      RepeatHead:    Linear(128, 1) — зацикленность

    ВАЖНО: DoubtEngine НЕ обучается на выходах Brain/MinGRU.
    Обучается на независимом корпусе.
    """

    # ═══ Thresholds (из ТЗ) ═══
    COHERENCE_FLAG = 0.5    # coherence < 0.5 → FLAG
    COHERENCE_BLOCK = 0.2   # coherence < 0.2 → BLOCK
    SAFETY_FLAG = 0.6       # safety < 0.6 → FLAG
    SAFETY_BLOCK = 0.3      # safety < 0.3 → BLOCK
    REPEAT_FLAG = 0.7       # repeat > 0.7 → FLAG
    REPEAT_BLOCK = 0.9      # repeat > 0.9 → BLOCK

    def __init__(self, d_model: int = 768, d_doubt: int = 128):
        super().__init__()
        self.d_model = d_model
        self.d_doubt = d_doubt

        # ═══ Stem: shared feature extraction ═══
        self.stem = nn.Sequential(
            nn.Linear(d_model * 2, d_doubt),
            nn.SiLU(),
            nn.Linear(d_doubt, d_doubt),
        )

        # ═══ 3 Verification Heads ═══
        self.coherence_head = nn.Linear(d_doubt, 1)  # логичность
        self.safety_head = nn.Linear(d_doubt, 1)      # безопасность
        self.repeat_head = nn.Linear(d_doubt, 1)      # зацикленность

        # Initialize coherence/safety to optimistic (pass by default)
        nn.init.constant_(self.coherence_head.bias, 2.0)  # sigmoid(2) ≈ 0.88
        nn.init.constant_(self.safety_head.bias, 2.0)
        # Initialize repeat to pessimistic (low repetition by default)
        nn.init.constant_(self.repeat_head.bias, -2.0)   # sigmoid(-2) ≈ 0.12

        # ═══ Safety Gate (hardcoded, always available) ═══
        self.safety_gate = SafetyGate()

    def forward(
        self,
        query_emb: torch.Tensor,
        response_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Neural doubt scoring.

        Args:
            query_emb:    [B, d_model] — query embedding (mean-pooled)
            response_emb: [B, d_model] — response embedding (mean-pooled)

        Returns:
            dict with keys:
                coherence: [B] — P(response is coherent with query), 0-1
                safety:    [B] — P(response is safe), 0-1
                repetition: [B] — P(response is repetitive), 0-1
                features:  [B, d_doubt] — intermediate features
        """
        # Concatenate query and response embeddings
        combined = torch.cat([query_emb, response_emb], dim=-1)  # [B, d_model*2]

        # Shared feature extraction
        features = self.stem(combined)  # [B, d_doubt]

        # 3 heads
        coherence = torch.sigmoid(self.coherence_head(features)).squeeze(-1)  # [B]
        safety = torch.sigmoid(self.safety_head(features)).squeeze(-1)        # [B]
        repetition = torch.sigmoid(self.repeat_head(features)).squeeze(-1)    # [B]

        return {
            "coherence": coherence,
            "safety": safety,
            "repetition": repetition,
            "features": features,
        }

    @staticmethod
    def compute_repetition(text: str, n: int = 4) -> float:
        """
        Compute n-gram overlap ratio for repetition detection.

        Formula: ratio = unique_ngrams / total_ngrams
        High overlap (low ratio) → high repetition score.

        This is a non-neural, formula-based metric.

        Args:
            text: generated text
            n: n-gram size (default 4)

        Returns:
            repetition score 0-1 (0=unique, 1=fully repeated)
        """
        if not text or len(text) < n:
            return 0.0

        words = text.lower().split()
        if len(words) < n:
            return 0.0

        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))

        if total == 0:
            return 0.0

        # Repetition = 1 - (unique / total)
        # All unique → 0.0 (no repetition)
        # All same → 1.0 - 1/total ≈ 1.0 (full repetition)
        return 1.0 - (unique / total)

    @staticmethod
    def compute_char_repetition(text: str, window: int = 50) -> float:
        """
        Character-level repetition: detect copy-paste loops.

        Checks if any substring of length `window` repeats
        more than twice in the text.

        Args:
            text: generated text
            window: sliding window size

        Returns:
            repetition score 0-1
        """
        if not text or len(text) < window * 2:
            return 0.0

        substrings = Counter()
        for i in range(len(text) - window + 1):
            substrings[text[i:i + window]] += 1

        if not substrings:
            return 0.0

        max_count = max(substrings.values())
        total_windows = len(text) - window + 1

        # Normalize: 1 occurrence = 0.0, all same = 1.0
        if total_windows <= 1:
            return 0.0
        return min(1.0, (max_count - 1) / max(1, total_windows // 2))

    def get_verdict(
        self,
        scores: Dict[str, float],
        is_action: bool = False,
    ) -> DoubtVerdict:
        """
        Convert raw scores to a DoubtVerdict.

        Args:
            scores: dict with coherence, safety, repetition (float 0-1)
            is_action: if True, any doubt → BLOCK (fail-closed)

        Returns:
            DoubtVerdict
        """
        coherence = scores.get("coherence", 1.0)
        safety = scores.get("safety", 1.0)
        repetition = scores.get("repetition", 0.0)

        reasons = []

        # ═══ Check BLOCK conditions ═══
        if coherence < self.COHERENCE_BLOCK:
            reasons.append(f"coherence={coherence:.2f}<{self.COHERENCE_BLOCK}")
        if safety < self.SAFETY_BLOCK:
            reasons.append(f"safety={safety:.2f}<{self.SAFETY_BLOCK}")
        if repetition > self.REPEAT_BLOCK:
            reasons.append(f"repetition={repetition:.2f}>{self.REPEAT_BLOCK}")

        if reasons:
            return DoubtVerdict(
                action="block",
                scores=scores,
                reason="BLOCK: " + "; ".join(reasons),
            )

        # ═══ Check FLAG conditions ═══
        flags = []
        if coherence < self.COHERENCE_FLAG:
            flags.append(f"coherence={coherence:.2f}<{self.COHERENCE_FLAG}")
        if safety < self.SAFETY_FLAG:
            flags.append(f"safety={safety:.2f}<{self.SAFETY_FLAG}")
        if repetition > self.REPEAT_FLAG:
            flags.append(f"repetition={repetition:.2f}>{self.REPEAT_FLAG}")

        if flags:
            # For actions: FLAG → BLOCK (fail-closed)
            if is_action:
                return DoubtVerdict(
                    action="block",
                    scores=scores,
                    reason="BLOCK (fail-closed for action): " + "; ".join(flags),
                )
            return DoubtVerdict(
                action="flag",
                scores=scores,
                reason="FLAG: " + "; ".join(flags),
            )

        return DoubtVerdict(
            action="pass",
            scores=scores,
            reason="All checks passed",
        )


# ═══════════════════════════════════════════════════════════════
# 4. OutputGate — Final Decision Combiner
# ═══════════════════════════════════════════════════════════════

class OutputGate:
    """
    Combines neural DoubtEngine scores + SafetyGate verdict
    into a final decision.

    Priority: SafetyGate BLOCK > DoubtEngine BLOCK > FLAG > PASS

    Fail-open for text: if DoubtEngine crashes → text passes
    Fail-closed for actions: if DoubtEngine crashes → action blocked
    """

    @staticmethod
    def evaluate(
        doubt_engine: Optional[DoubtEngine],
        query_emb: Optional[torch.Tensor],
        response_emb: Optional[torch.Tensor],
        action_text: str = "",
        action_params: Optional[dict] = None,
        generated_text: str = "",
        is_action: bool = False,
    ) -> DoubtVerdict:
        """
        Full verification pipeline.

        Args:
            doubt_engine: DoubtEngine module (can be None)
            query_emb: [1, d_model] query embedding
            response_emb: [1, d_model] response embedding
            action_text: action name (for SafetyGate)
            action_params: action parameters (for SafetyGate)
            generated_text: generated text (for repetition check)
            is_action: whether this is an action (fail-closed)

        Returns:
            DoubtVerdict with final decision
        """
        all_scores: Dict[str, float] = {}
        verdicts: List[DoubtVerdict] = []

        # ═══ 1. SafetyGate (hardcoded, always runs) ═══
        if action_params is not None:
            sg_verdict = SafetyGate.check(action_text, action_params)
            verdicts.append(sg_verdict)
            all_scores["safety_gate"] = sg_verdict.scores.get("safety", 1.0)
            if sg_verdict.is_blocked:
                return sg_verdict  # SafetyGate BLOCK is absolute

        # ═══ 2. Neural DoubtEngine ═══
        if doubt_engine is not None and query_emb is not None and response_emb is not None:
            try:
                with torch.no_grad():
                    neural_scores = doubt_engine(query_emb, response_emb)

                all_scores["coherence"] = neural_scores["coherence"].item()
                all_scores["safety"] = neural_scores["safety"].item()
                all_scores["repetition"] = neural_scores["repetition"].item()

            except Exception as e:
                logger.warning(f"DoubtEngine neural forward failed: {e}")
                if is_action:
                    # Fail-closed for actions
                    return DoubtVerdict(
                        action="block",
                        scores={"error": 1.0},
                        reason=f"DoubtEngine crash (fail-closed): {e}",
                    )
                else:
                    # Fail-open for text
                    all_scores["coherence"] = 1.0
                    all_scores["safety"] = 1.0
                    all_scores["repetition"] = 0.0
        else:
            # No neural engine available
            all_scores.setdefault("coherence", 1.0)
            all_scores.setdefault("safety", 1.0)
            all_scores.setdefault("repetition", 0.0)

        # ═══ 3. Text-based repetition (non-neural) ═══
        if generated_text:
            text_rep = DoubtEngine.compute_repetition(generated_text)
            char_rep = DoubtEngine.compute_char_repetition(generated_text)
            combined_rep = max(text_rep, char_rep)

            # Override neural repetition if text-based is higher
            all_scores["repetition"] = max(
                all_scores.get("repetition", 0.0),
                combined_rep,
            )
            all_scores["repetition_ngram"] = text_rep
            all_scores["repetition_char"] = char_rep

        # ═══ 4. Final verdict ═══
        if doubt_engine is not None:
            return doubt_engine.get_verdict(all_scores, is_action=is_action)

        # Fallback: use DoubtEngine class thresholds without instance
        return DoubtEngine(d_model=1).get_verdict(all_scores, is_action=is_action)


# ═══════════════════════════════════════════════════════════════
# 5. Utility: load/save DoubtEngine
# ═══════════════════════════════════════════════════════════════

def load_doubt_engine(
    d_model: int = 768,
    checkpoint_path: str = None,
    device: str = "cpu",
) -> Optional[DoubtEngine]:
    """
    Load DoubtEngine from checkpoint.

    Args:
        d_model: model dimension
        checkpoint_path: path to checkpoint (default: models/doubt/doubt_engine_best.pt)
        device: target device

    Returns:
        DoubtEngine or None if no checkpoint found
    """
    import os

    if checkpoint_path is None:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(_root, "models", "doubt", "doubt_engine_best.pt")

    engine = DoubtEngine(d_model=d_model).to(device)

    if os.path.exists(checkpoint_path):
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            engine.load_state_dict(state, strict=False)
            logger.info(f"DoubtEngine loaded from {checkpoint_path}")
        except Exception as e:
            logger.warning(f"Failed to load DoubtEngine checkpoint: {e}")
    else:
        logger.info(f"No DoubtEngine checkpoint at {checkpoint_path} — using untrained")

    engine.eval()
    return engine
