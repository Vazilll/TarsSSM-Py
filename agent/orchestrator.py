"""
═══════════════════════════════════════════════════════════════
  TarsOrchestrator — Central Agent Coordinator (Agent 5)
═══════════════════════════════════════════════════════════════

Unified orchestrator consolidating logic from:
  - agent/core/gie.py (GieAgent)
  - agent/core/tars_agent.py (TarsAgent)

Pipeline:
  1. PromptDefense → check user input
  2. Mode Router → reflex/think/deep (neural, tars_core)
  3. Tool dispatch (via ToolRegistry + EthicalGuard)
  4. TarsEngine (C++ core) or Python fallback for inference
  5. DoubtEngine verification
  6. AuditLogger logging

Owner: Agent 5 (EXCLUSIVE)
"""

import os
import sys
import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger("Tars.Orchestrator")

# Project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

# Local imports
from agent.safety.ethical_guard import EthicalGuard, SafetyVerdict
from agent.safety.audit_log import AuditLogger
from agent.safety.prompt_defense import PromptDefense
from agent.tools.tool_registry import ToolRegistry, ToolResult


# ═══════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass
class OrchestratorResult:
    """Result of a full orchestration cycle."""
    query: str
    response: str
    mode: str = "unknown"        # "reflex", "think", "deep"
    tokens_generated: int = 0
    total_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    doubt_scores: Dict[str, float] = field(default_factory=dict)
    safety_verdict: str = "pass"
    engine_used: str = "python"   # "cpp_core" or "python"


# ═══════════════════════════════════════════════════════════════
# TarsOrchestrator
# ═══════════════════════════════════════════════════════════════

class TarsOrchestrator:
    """
    Central coordinator for TARS inference pipeline.

    Dispatches to:
      - tars_core.TarsEngine (C++ core) if available → 10-20ms/tok
      - Python fallback model if C++ not compiled → 200ms/tok

    Integrates:
      - EthicalGuard (safety)
      - PromptDefense (injection detection)
      - AuditLogger (audit trail)
      - ToolRegistry (tool execution)
    """

    MAX_QUERY_LEN = 4096

    def __init__(
        self,
        workspace: str = ".",
        use_cpp_core: bool = True,
        verbose: bool = True,
    ):
        self.workspace = os.path.abspath(workspace)
        self.verbose = verbose

        # ═══ Safety layer ═══
        self.audit = AuditLogger(session_id=f"tars_{int(time.time())}")
        self.guard = EthicalGuard(audit_logger=self.audit)
        self.prompt_defense = PromptDefense()

        # ═══ Tools layer ═══
        self.tools = ToolRegistry(ethical_guard=self.guard, audit_logger=self.audit)
        self.tools.create_default(workspace)

        # ═══ Inference engine ═══
        self._cpp_engine = None
        self._python_model = None
        self._tokenizer = None

        if use_cpp_core:
            self._try_load_cpp_core()
        if self._cpp_engine is None:
            self._try_load_python_model()

        # ═══ Session state ═══
        self.history: List[Dict[str, str]] = []
        self.total_queries = 0

        # Log startup
        self.audit.log_system("startup", {
            "workspace": workspace,
            "engine": "cpp_core" if self._cpp_engine else "python",
            "tools": self.tools.list_tools(),
        })

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    # ─────────────────────────────────────────
    # Engine loading
    # ─────────────────────────────────────────

    def _try_load_cpp_core(self):
        """Try to load C++/Rust TarsEngine via tars_core module."""
        try:
            import tars_core
            self._cpp_engine = tars_core.TarsEngine()
            self._log("✅ C++ core (tars_core.TarsEngine) loaded")
        except ImportError:
            self._log("⚠️  tars_core not found — will use Python fallback")
        except Exception as e:
            self._log(f"⚠️  tars_core load error: {e}")

    def _try_load_python_model(self):
        """Load Python reference model for inference."""
        try:
            from brain.mamba2.model import TarsMamba2LM
            from brain.tokenizer import TarsTokenizer
            import torch

            device = "cpu"
            model, checkpoint = TarsMamba2LM.load_pretrained(device=device)
            model.eval()
            self._python_model = model
            self._tokenizer = TarsTokenizer()
            self._log(f"✅ Python model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
        except Exception as e:
            self._log(f"⚠️  Python model not loaded: {e}")

    # ─────────────────────────────────────────
    # Main pipeline
    # ─────────────────────────────────────────

    async def process(self, query: str) -> OrchestratorResult:
        """
        Full orchestration cycle.

        1. Input validation + PromptDefense
        2. Mode routing (neural when available)
        3. Inference via C++ core or Python fallback
        4. DoubtEngine verification
        5. Response
        """
        t0 = time.time()
        self.total_queries += 1

        # Input validation
        if len(query) > self.MAX_QUERY_LEN:
            query = query[:self.MAX_QUERY_LEN]

        # ═══ Step 1: Prompt Defense ═══
        defense_result = self.prompt_defense.check(query)
        if not defense_result.is_safe:
            self.audit.log_user(query, intent="blocked_injection")
            return OrchestratorResult(
                query=query,
                response=f"⛔ Запрос заблокирован: {defense_result.threat_type}",
                mode="blocked",
                total_time_ms=(time.time() - t0) * 1000,
                safety_verdict="block",
            )

        # ═══ Step 2: Inference ═══
        result = await self._generate(query)

        # ═══ Step 3: Text safety check ═══
        text_verdict = self.guard.check_text(result.response)
        result.safety_verdict = text_verdict.action
        result.doubt_scores = text_verdict.scores

        # ═══ Step 4: History + Audit ═══
        self.history.append({"user": query, "tars": result.response})
        if len(self.history) > 50:
            self.history = self.history[-50:]

        self.audit.log_user(
            query,
            intent=result.mode,
            response_preview=result.response[:100],
        )

        result.total_time_ms = (time.time() - t0) * 1000
        return result

    async def _generate(self, query: str) -> OrchestratorResult:
        """Generate response using best available engine."""

        # Try C++ core first
        if self._cpp_engine is not None:
            try:
                return await self._generate_cpp(query)
            except Exception as e:
                self._log(f"C++ core error, falling back to Python: {e}")

        # Python fallback
        if self._python_model is not None:
            try:
                return await self._generate_python(query)
            except Exception as e:
                self._log(f"Python model error: {e}")

        return OrchestratorResult(
            query=query,
            response="⚠️ Нет доступного inference engine. "
                     "Обучите модель: python local_train.py",
            mode="error",
        )

    async def _generate_cpp(self, query: str) -> OrchestratorResult:
        """Generate using C++ TarsEngine."""
        t0 = time.time()

        # Tokenize
        if self._tokenizer is None:
            from brain.tokenizer import TarsTokenizer
            self._tokenizer = TarsTokenizer()

        prompt_ids = self._tokenizer.encode(query)

        # Generate via C++ engine
        output_ids = self._cpp_engine.generate(
            prompt_ids=prompt_ids,
            max_tokens=128,
            temperature=0.9,
            top_p=0.92,
        )

        response = self._tokenizer.decode(output_ids)
        duration = time.time() - t0
        n_tokens = len(output_ids)

        # Get doubt scores from C++ engine
        doubt = {}
        try:
            doubt = self._cpp_engine.get_doubt_scores()
        except Exception:
            pass

        return OrchestratorResult(
            query=query,
            response=response,
            mode="think",
            tokens_generated=n_tokens,
            total_time_ms=duration * 1000,
            tokens_per_sec=n_tokens / duration if duration > 0 else 0,
            doubt_scores=doubt,
            engine_used="cpp_core",
        )

    async def _generate_python(self, query: str) -> OrchestratorResult:
        """Generate using Python reference model."""
        import torch

        t0 = time.time()
        input_ids = self._tokenizer.encode(query)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        # Think
        with torch.no_grad():
            think_result = self._python_model.think(input_tensor, query_text=query)

        if isinstance(think_result, tuple):
            logits, stats = think_result
        else:
            logits = think_result
            stats = {}

        # Generate
        from brain.mamba2.generate_mamba import TarsGenerator, GenerationConfig
        gen = TarsGenerator(self._python_model, self._tokenizer)
        config = GenerationConfig(max_tokens=64, temperature=0.9, top_k=40, top_p=0.92)
        gen_result = gen.generate(query, config=config)

        duration = time.time() - t0

        return OrchestratorResult(
            query=query,
            response=gen_result.text,
            mode=stats.get("task_type", "think"),
            tokens_generated=gen_result.tokens_generated,
            total_time_ms=duration * 1000,
            tokens_per_sec=gen_result.tokens_generated / duration if duration > 0 else 0,
            engine_used="python",
        )

    # ─────────────────────────────────────────
    # Tool execution
    # ─────────────────────────────────────────

    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool with safety checks."""
        return await self.tools.execute(tool_name, args)

    # ─────────────────────────────────────────
    # Status
    # ─────────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """System status."""
        return {
            "engine": "cpp_core" if self._cpp_engine else ("python" if self._python_model else "none"),
            "total_queries": self.total_queries,
            "history_len": len(self.history),
            "tools": self.tools.list_tools(),
            "audit_events": self.audit.event_count,
        }
