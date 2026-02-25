"""
generate_mamba.py — TARS v3 Text Generator with Interactive Reasoning.

Supports:
  - <thought>...</thought>  internal monologue (DeepSeek-style)
  - <tool>action: args</tool> RAG / action triggers (pauses generation)
  - IDME Pool rounds for deep thinking
  - Interactive knowledge inject (user can interrupt mid-thought)

Usage:
    from brain.mamba2.generate_mamba import TarsGenerator
    gen = TarsGenerator(model, tokenizer)
    result = gen.generate("Спроектируй архитектуру БД", max_tokens=512)
"""
import torch
import torch.nn.functional as F
import time
import re
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Callable

logger = logging.getLogger("TarsGenerator")


@dataclass
class GenerationResult:
    """Result of a generation call."""
    text: str                          # Final visible answer
    thought: str = ""                  # Internal monologue (hidden from user)
    tool_calls: List[dict] = field(default_factory=list)  # Parsed tool calls
    p_convergence: float = 0.0         # Final p value from Integral Auditor
    idme_rounds: int = 0              # How many IDME rounds were used
    tokens_generated: int = 0
    time_ms: float = 0.0


@dataclass
class GenerationConfig:
    """Generation hyperparameters."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    min_p: float = 0.05
    p_threshold: float = 1.2          # Integral Auditor convergence threshold
    max_idme_rounds: int = 20         # Maximum IDME Pool depth
    enable_thought: bool = True       # Allow <thought> tags
    enable_tools: bool = True         # Allow <tool> tags
    hankel_collapse_threshold: float = 0.15  # Below this = looping


class TarsGenerator:
    """
    3-Tier text generator:
    1. Token-by-token with Mamba-2 base layers
    2. IDME Pool rounds for deep thinking
    3. <thought>/<tool> tag parsing for interactive reasoning
    """

    THOUGHT_OPEN = "<thought>"
    THOUGHT_CLOSE = "</thought>"
    TOOL_PATTERN = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)

    def __init__(self, model, tokenizer, omega_core=None):
        """
        Args:
            model: TarsMamba2LM instance
            tokenizer: tokenizer with encode/decode
            omega_core: OmegaCore instance (optional, for C++ sampling)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.omega_core = omega_core
        self._inject_queue: List[str] = []  # For interactive knowledge inject
        self._tool_handler: Optional[Callable] = None

    def set_tool_handler(self, handler: Callable):
        """Register a callback for <tool> commands.
        handler(action: str, args: str) -> str (result text)
        """
        self._tool_handler = handler

    def inject_knowledge(self, text: str):
        """Inject text into the generation stream (interactive mode).
        Called by the user or by RAG during IDME rounds.
        """
        self._inject_queue.append(text)

    @torch.no_grad()
    def generate(self, prompt: str, config: Optional[GenerationConfig] = None,
                 on_token: Optional[Callable] = None,
                 on_thought: Optional[Callable] = None) -> GenerationResult:
        """
        Generate text from prompt.

        Args:
            prompt: Input text
            config: Generation parameters
            on_token: Callback(token_str) for streaming
            on_thought: Callback(thought_str) for thought visualization

        Returns:
            GenerationResult with text, thought, tool_calls, etc.
        """
        if config is None:
            config = GenerationConfig()

        t0 = time.time()
        device = next(self.model.parameters()).device

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Initialize state
        generated_tokens = []
        thought_tokens = []
        in_thought = False
        tool_calls = []
        text_buffer = ""

        # State for Integral Auditor
        f_history = []
        prev_hidden = None
        idme_rounds = 0
        p_value = 0.0

        # Initial forward pass (prefill) — use forward() for speed
        # think() IDME stats are shown separately in CLI
        prefill_result = self.model(input_ids)
        if isinstance(prefill_result, tuple):
            logits = prefill_result[0]
        else:
            logits = prefill_result

        for step in range(config.max_tokens):
            # Sample next token
            next_logits = logits[:, -1, :]  # [1, vocab]

            if self.omega_core and self.omega_core.available:
                token_id = self.omega_core.sample(
                    next_logits.squeeze(0).cpu().numpy(),
                    temp=config.temperature,
                    min_p=config.min_p,
                    top_p=config.top_p,
                    top_k=config.top_k
                )
            else:
                token_id = self._sample_torch(next_logits, config)

            # Decode token
            token_str = self.tokenizer.decode([token_id])
            text_buffer += token_str

            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id') and token_id == self.tokenizer.eos_token_id:
                break

            # --- Tag parsing ---

            # <thought> open
            if config.enable_thought and self.THOUGHT_OPEN in text_buffer and not in_thought:
                in_thought = True
                text_buffer = text_buffer.replace(self.THOUGHT_OPEN, "")
                if on_thought:
                    on_thought("[Начало размышления]")

            # </thought> close
            if in_thought and self.THOUGHT_CLOSE in text_buffer:
                in_thought = False
                parts = text_buffer.split(self.THOUGHT_CLOSE, 1)
                thought_tokens.append(parts[0])
                text_buffer = parts[1] if len(parts) > 1 else ""
                if on_thought:
                    on_thought("[Завершение размышления]")

            # <tool>...</tool> detection
            if config.enable_tools:
                tool_match = self.TOOL_PATTERN.search(text_buffer)
                if tool_match:
                    tool_content = tool_match.group(1).strip()
                    text_buffer = text_buffer[:tool_match.start()] + text_buffer[tool_match.end():]

                    # Parse action:args
                    if ":" in tool_content:
                        action, args = tool_content.split(":", 1)
                    else:
                        action, args = tool_content, ""

                    tool_call = {"action": action.strip(), "args": args.strip()}
                    tool_calls.append(tool_call)

                    # Execute tool handler if registered
                    if self._tool_handler:
                        tool_result = self._tool_handler(action.strip(), args.strip())
                        if tool_result:
                            self._inject_queue.append(tool_result)

            # Route tokens
            if in_thought:
                thought_tokens.append(token_str)
                if on_thought:
                    on_thought(token_str)
            else:
                generated_tokens.append(token_str)
                if on_token:
                    on_token(token_str)

            # --- Process knowledge inject queue ---
            if self._inject_queue:
                for inject_text in self._inject_queue:
                    inject_ids = self.tokenizer.encode(inject_text)
                    if isinstance(inject_ids, list):
                        inject_ids = torch.tensor([inject_ids], dtype=torch.long, device=device)
                    # Forward pass through injected tokens
                    inject_out = self.model(inject_ids)
                    if isinstance(inject_out, tuple):
                        logits = inject_out[0]
                    else:
                        logits = inject_out
                    if on_thought:
                        on_thought(f"[Inject: {inject_text[:50]}...]")
                self._inject_queue.clear()
                continue  # Skip normal forward, we just updated state

            # --- Integral Auditor (track convergence) ---
            current_hidden = next_logits.detach()
            if prev_hidden is not None:
                delta = (current_hidden - prev_hidden).norm().item()
                f_history.append(delta)

                if len(f_history) >= 4:
                    if self.omega_core and self.omega_core.available:
                        p_value = self.omega_core.integral_audit(f_history)
                    else:
                        p_value = self._integral_audit_py(f_history)

                    # Convergence check
                    if p_value > config.p_threshold and not in_thought:
                        logger.debug(f"Converged at step {step} (p={p_value:.2f})")
                        break
            prev_hidden = current_hidden

            # Next forward step (use forward() — cheaper, no IDME)
            token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
            fwd_result = self.model(token_tensor)
            if isinstance(fwd_result, tuple):
                logits = fwd_result[0]
            else:
                logits = fwd_result

        # Build result
        elapsed = (time.time() - t0) * 1000
        final_text = "".join(generated_tokens).strip()
        final_thought = "".join(thought_tokens).strip()

        return GenerationResult(
            text=final_text,
            thought=final_thought,
            tool_calls=tool_calls,
            p_convergence=p_value,
            idme_rounds=idme_rounds,
            tokens_generated=len(generated_tokens),
            time_ms=elapsed
        )

    @staticmethod
    def _sample_torch(logits, config: GenerationConfig) -> int:
        """Pure PyTorch sampling fallback."""
        logits = logits.squeeze(0).float()
        if config.temperature < 0.05:
            return logits.argmax().item()

        logits = logits / config.temperature

        # Top-K
        if config.top_k > 0:
            top_k = min(config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)

        # Top-P (nucleus)
        if config.top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative - sorted_probs > config.top_p
            sorted_probs[remove_mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            idx = torch.multinomial(sorted_probs, 1).item()
            return sorted_indices[idx].item()

        return torch.multinomial(probs, 1).item()

    @staticmethod
    def _integral_audit_py(f_history, window=8) -> float:
        """Pure Python p-convergence calculation."""
        import numpy as np
        f = np.array(f_history[-window:], dtype=np.float32)
        f = np.clip(f, 1e-12, None)
        if len(f) < 3:
            return 0.0
        x = np.log(np.arange(1, len(f) + 1, dtype=np.float32))
        y = np.log(f)
        n = len(x)
        denom = n * (x**2).sum() - x.sum()**2
        if abs(denom) < 1e-15:
            return 0.0
        b = (n * (x * y).sum() - x.sum() * y.sum()) / denom
        return float(-b)
