"""
═══════════════════════════════════════════════════════════════
  Chunked Prefill for Long-Context SSM Inference
═══════════════════════════════════════════════════════════════

Processes long prompts in chunks to avoid GPU OOM while
maintaining full SSM state continuity.

Problem: Processing 50K tokens in one forward pass → OOM
Solution: Process in 2K chunks, carry SSM/WKV state between chunks

Usage:
    from brain.chunked_prefill import chunked_prefill

    # Instead of: model.think(very_long_input)
    logits = chunked_prefill(model, very_long_input_ids, chunk_size=2048)
"""

import torch
import logging
from typing import Optional, Tuple

logger = logging.getLogger("Tars.ChunkedPrefill")


def chunked_prefill(
    model,
    input_ids: torch.Tensor,     # [B, L]
    chunk_size: int = 2048,      # tokens per chunk
    overlap: int = 64,           # overlap for continuity (sliding window)
    verbose: bool = False,
) -> torch.Tensor:
    """
    Process long input in chunks with SSM state carry.
    
    Args:
        model: TarsMamba2LM (must have think/forward method)
        input_ids: [B, L] — full input token IDs
        chunk_size: maximum tokens per forward pass
        overlap: overlapping tokens between chunks for context continuity
        verbose: print progress
    
    Returns:
        logits: [B, L, V] — full logits for entire sequence
    """
    B, L = input_ids.shape
    
    if L <= chunk_size:
        # Short enough for direct processing
        result = model.think(input_ids)
        return result[0] if isinstance(result, tuple) else result
    
    device = input_ids.device
    all_logits = []
    
    # Process in chunks with state carrying
    n_chunks = (L + chunk_size - 1) // chunk_size
    
    if verbose:
        logger.info(f"Chunked prefill: {L} tokens → {n_chunks} chunks of {chunk_size}")
    
    # Reset model state
    if hasattr(model, 'reset_cache'):
        model.reset_cache()
    
    for i in range(n_chunks):
        start = max(0, i * chunk_size - overlap) if i > 0 else 0
        end = min((i + 1) * chunk_size, L)
        
        chunk = input_ids[:, start:end]
        
        # Forward pass (model maintains internal SSM state)
        with torch.no_grad():
            result = model(chunk)
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result
        
        # Only keep logits for non-overlapping portion
        if i > 0 and overlap > 0:
            logits = logits[:, overlap:, :]
        
        all_logits.append(logits)
        
        if verbose and (i + 1) % 5 == 0:
            logger.info(f"  Chunk {i+1}/{n_chunks} processed")
    
    # Concatenate all logits
    full_logits = torch.cat(all_logits, dim=1)
    
    # Trim to exact input length (in case of rounding)
    if full_logits.shape[1] > L:
        full_logits = full_logits[:, :L, :]
    
    if verbose:
        logger.info(f"Chunked prefill complete: {full_logits.shape}")
    
    return full_logits


class StreamingPrefill:
    """
    Streaming prefill for interactive long-context loading.
    
    Useful when loading documents into context:
      1. Start loading large file
      2. Each chunk updates SSM state
      3. After loading, model is ready for generation
    """
    
    def __init__(self, model, chunk_size: int = 2048):
        self.model = model
        self.chunk_size = chunk_size
        self.tokens_processed = 0
        self._last_logits = None
        
        if hasattr(model, 'reset_cache'):
            model.reset_cache()
    
    def feed(self, token_ids: torch.Tensor) -> int:
        """
        Feed tokens into the model, updating internal state.
        
        Args:
            token_ids: [B, L] — next chunk of tokens
        
        Returns:
            total tokens processed so far
        """
        B, L = token_ids.shape
        
        # Process in sub-chunks if needed
        for start in range(0, L, self.chunk_size):
            end = min(start + self.chunk_size, L)
            chunk = token_ids[:, start:end]
            
            with torch.no_grad():
                result = self.model(chunk)
                if isinstance(result, tuple):
                    self._last_logits = result[0]
                else:
                    self._last_logits = result
            
            self.tokens_processed += end - start
        
        return self.tokens_processed
    
    def get_last_logits(self) -> Optional[torch.Tensor]:
        """Get logits from the last chunk (for next-token prediction)."""
        return self._last_logits
    
    def reset(self):
        """Reset state for new context."""
        self.tokens_processed = 0
        self._last_logits = None
        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
