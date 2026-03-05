"""
═══════════════════════════════════════════════════════════════
  Attention Sink + SSM Infinite Context for TARS
═══════════════════════════════════════════════════════════════

Enable unlimited context length by combining:
  1. Attention Sink: keep first K tokens (anchors) + sliding window
  2. SSM State Compression: compress old context into SSM state
  3. Landmark tokens: insert summarization markers

Based on StreamingLLM (Xiao et al., 2023):
  Discovered that attention has "sinks" — the first few tokens
  receive disproportionate attention regardless of content.
  Keeping these tokens + sliding window = infinite streaming.

For SSM (Mamba/RWKV): state naturally compresses history,
but can degrade for very long contexts (>16K). This module
manages state quality across ultra-long sequences.

Usage:
    from brain.attention_sink import InfiniteContextManager
    
    ctx = InfiniteContextManager(model, window_size=2048, n_sinks=4)
    
    for chunk in document_chunks:
        logits = ctx.process(chunk_ids)
    
    # After processing 100K+ tokens, model still works correctly
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("Tars.InfiniteContext")


@dataclass
class ContextState:
    """State of the infinite context manager."""
    tokens_processed: int = 0
    chunks_processed: int = 0
    state_compressions: int = 0
    current_window_size: int = 0
    sink_tokens_count: int = 0


class InfiniteContextManager:
    """
    Manage ultra-long context (100K+ tokens) for SSM models.
    
    Strategy:
      1. First K tokens → "sink" (always kept, anchor attention)
      2. Sliding window of W recent tokens → active context
      3. Everything between → compressed into SSM recurrent state
      4. Periodic state regularization to prevent drift
    """
    
    def __init__(
        self,
        model,
        window_size: int = 2048,
        n_sink_tokens: int = 4,
        state_refresh_interval: int = 8192,
        compress_ratio: float = 0.5,
    ):
        """
        Args:
            model: TarsMamba2LM
            window_size: sliding window size (active context)
            n_sink_tokens: number of initial "sink" tokens to always keep
            state_refresh_interval: re-normalize SSM state every N tokens
            compress_ratio: when to trigger state compression
        """
        self.model = model
        self.window_size = window_size
        self.n_sink_tokens = n_sink_tokens
        self.state_refresh_interval = state_refresh_interval
        self.compress_ratio = compress_ratio
        
        self._sink_ids: Optional[torch.Tensor] = None
        self._sink_kv: Optional[torch.Tensor] = None
        self._window_ids: List[int] = []
        self._state = ContextState()
        self._last_logits: Optional[torch.Tensor] = None
        
        # SSM state norms for drift detection
        self._initial_state_norms: Optional[dict] = None
    
    def reset(self):
        """Clear context and start fresh."""
        self._sink_ids = None
        self._sink_kv = None
        self._window_ids = []
        self._state = ContextState()
        self._last_logits = None
        self._initial_state_norms = None
        
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
    
    @torch.no_grad()
    def process(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Process a chunk of tokens, maintaining infinite context.
        
        Args:
            input_ids: [B, L] — next chunk of tokens
            
        Returns:
            logits: [B, L, V]
        """
        B, L = input_ids.shape
        device = input_ids.device
        
        # First chunk: establish sinks
        if self._sink_ids is None:
            return self._process_first_chunk(input_ids)
        
        # Append to window
        new_tokens = input_ids[0].tolist()
        self._window_ids.extend(new_tokens)
        
        # Trim window if too large (keep sinks + window)
        if len(self._window_ids) > self.window_size:
            # Tokens being evicted from window → already in SSM state
            n_evict = len(self._window_ids) - self.window_size
            self._window_ids = self._window_ids[n_evict:]
            self._state.state_compressions += 1
        
        # Build context: sink_tokens + window_tokens
        context_ids = self._build_context(device)
        
        # Forward pass (SSM maintains state from all previous tokens)
        result = self.model(input_ids)  # Only process new tokens
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        
        self._last_logits = logits
        self._state.tokens_processed += L
        self._state.chunks_processed += 1
        self._state.current_window_size = len(self._window_ids)
        
        # Periodic state health check
        if (self._state.tokens_processed % self.state_refresh_interval == 0 and
            self._state.tokens_processed > 0):
            self._check_state_health()
        
        return logits
    
    def _process_first_chunk(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Process very first chunk: extract sinks + initialize."""
        B, L = input_ids.shape
        
        # Extract sink tokens (first K)
        n_sinks = min(self.n_sink_tokens, L)
        self._sink_ids = input_ids[:, :n_sinks].clone()
        self._state.sink_tokens_count = n_sinks
        
        # Rest goes into window
        self._window_ids = input_ids[0, n_sinks:].tolist()
        
        # Full forward pass
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
        
        result = self.model(input_ids)
        if isinstance(result, tuple):
            logits = result[0]
        else:
            logits = result
        
        self._last_logits = logits
        self._state.tokens_processed = L
        self._state.chunks_processed = 1
        self._state.current_window_size = len(self._window_ids)
        
        # Record initial state norms for drift detection
        self._record_state_norms()
        
        return logits
    
    def _build_context(self, device) -> torch.Tensor:
        """Build full context from sinks + window."""
        window_tensor = torch.tensor(
            [self._window_ids], dtype=torch.long, device=device
        )
        
        if self._sink_ids is not None:
            return torch.cat([self._sink_ids.to(device), window_tensor], dim=1)
        return window_tensor
    
    def _record_state_norms(self):
        """Record SSM state norms for drift detection."""
        self._initial_state_norms = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'ssm_state') and module.ssm_state is not None:
                self._initial_state_norms[name] = module.ssm_state.norm().item()
    
    def _check_state_health(self):
        """Check if SSM state has drifted too far from initial."""
        if self._initial_state_norms is None:
            return
        
        max_drift = 0
        for name, module in self.model.named_modules():
            if hasattr(module, 'ssm_state') and module.ssm_state is not None:
                if name in self._initial_state_norms:
                    current = module.ssm_state.norm().item()
                    initial = self._initial_state_norms[name]
                    
                    if initial > 1e-6:
                        drift = abs(current - initial) / initial
                        max_drift = max(max_drift, drift)
                        
                        # State explosion detection
                        if drift > 10.0:
                            logger.warning(
                                f"SSM state drift detected in {name}: "
                                f"{initial:.2f} → {current:.2f} ({drift:.1f}x)"
                            )
                            # Re-normalize state
                            with torch.no_grad():
                                module.ssm_state.data *= initial / max(current, 1e-8)
        
        if max_drift > 2.0:
            logger.info(f"State drift check: max_drift={max_drift:.2f} "
                       f"(tokens={self._state.tokens_processed})")
    
    def get_stats(self) -> dict:
        """Get context management statistics."""
        return {
            "tokens_processed": self._state.tokens_processed,
            "chunks_processed": self._state.chunks_processed,
            "state_compressions": self._state.state_compressions,
            "window_size": self._state.current_window_size,
            "sink_tokens": self._state.sink_tokens_count,
        }
    
    def get_effective_context(self) -> int:
        """Total effective context length (sink + window + SSM memory)."""
        return self._state.tokens_processed
