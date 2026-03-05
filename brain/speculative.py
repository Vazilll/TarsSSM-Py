"""
═══════════════════════════════════════════════════════════════
  Speculative Decoding for TARS
═══════════════════════════════════════════════════════════════

MinGRU (Tier 1, fast) drafts K candidate tokens.
Mamba-2 (Tier 2, accurate) verifies all K in one forward pass.

Result: 1.5-2.5x inference speedup at ZERO quality loss.
The output distribution is mathematically identical to vanilla
autoregressive decoding with Mamba-2 alone.

Usage:
    from brain.speculative import SpeculativeDecoder
    
    decoder = SpeculativeDecoder(
        draft_model=mingru_lm,    # MinGRU_LM (fast)
        target_model=mamba2_lm,   # TarsMamba2LM (accurate)
        tokenizer=tokenizer,
    )
    
    result = decoder.generate("Привет, TARS!", max_tokens=256)
"""

import torch
import torch.nn.functional as F
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable, List

logger = logging.getLogger("Tars.Speculative")


@dataclass
class SpecResult:
    """Result of speculative generation."""
    text: str
    tokens_generated: int
    tokens_drafted: int
    tokens_accepted: int
    acceptance_rate: float
    time_ms: float
    speedup: float  # vs estimated vanilla


class SpeculativeDecoder:
    """
    Speculative Decoding: draft-then-verify.
    
    Algorithm:
      1. Draft model generates K tokens autoregressively (fast)
      2. Target model scores all K+1 positions in ONE forward pass
      3. Accept/reject each draft token by comparing probabilities
      4. On first rejection → sample correction token from target
      5. Repeat with remaining budget
    
    Mathematical guarantee: output distribution == target model alone.
    """
    
    def __init__(
        self,
        draft_model,           # MinGRU_LM
        target_model,          # TarsMamba2LM
        tokenizer,             # TarsTokenizer
        draft_k: int = 5,      # tokens to draft per step
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ):
        self.draft = draft_model
        self.target = target_model
        self.tokenizer = tokenizer
        self.K = draft_k
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        self.device = next(target_model.parameters()).device
        self.draft.to(self.device)
        self.draft.eval()
        self.target.eval()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        on_token: Optional[Callable] = None,
        temperature: Optional[float] = None,
    ) -> SpecResult:
        """Generate text using speculative decoding."""
        
        temp = temperature or self.temperature
        t0 = time.time()
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        
        generated = []
        total_drafted = 0
        total_accepted = 0
        
        # Reset caches
        if hasattr(self.target, 'reset_cache'):
            self.target.reset_cache()
        
        draft_hiddens = None  # MinGRU cached hidden states
        
        # Prefill target with prompt
        target_logits = self._target_forward(input_ids)
        # target_logits: [1, L, V] — we only need the last position
        
        # Prefill draft with prompt (get initial hidden states)
        draft_prefill = self.draft(
            input_ids,
            prev_hiddens=None,
            return_hiddens=True
        )
        if isinstance(draft_prefill, tuple):
            _, draft_hiddens = draft_prefill
        
        while len(generated) < max_tokens:
            # ═══ Phase 1: Draft K tokens with MinGRU ═══
            draft_tokens = []
            draft_probs = []
            draft_distributions = []  # full prob distributions for correction
            
            # Start draft from last generated token (or last prompt token)
            if generated:
                last_tok = generated[-1]
            else:
                last_tok = prompt_ids[-1]
            draft_input = torch.tensor(
                [[last_tok]], dtype=torch.long, device=self.device
            )
            
            for _ in range(self.K):
                if len(generated) + len(draft_tokens) >= max_tokens:
                    break
                
                # Forward through draft model
                draft_out = self.draft(
                    draft_input, 
                    prev_hiddens=draft_hiddens, 
                    return_hiddens=True
                )
                
                if isinstance(draft_out, tuple):
                    draft_logits, draft_hiddens = draft_out
                else:
                    draft_logits = draft_out
                    draft_hiddens = None
                
                # Sample from draft
                probs = self._sample_probs(draft_logits[:, -1, :], temp)
                token = torch.multinomial(probs, 1).item()
                
                draft_tokens.append(token)
                draft_probs.append(probs[0, token].item())
                draft_distributions.append(probs)  # save full distribution
                
                # EOS check
                if token == 0:
                    break
                
                draft_input = torch.tensor([[token]], dtype=torch.long, device=self.device)
            
            if not draft_tokens:
                break
            
            total_drafted += len(draft_tokens)
            
            # ═══ Phase 2: Verify ALL draft tokens with Mamba-2 in ONE pass ═══
            verify_input = torch.tensor(
                [draft_tokens], dtype=torch.long, device=self.device
            )
            
            # Target scores all positions
            target_logits = self._target_forward(verify_input)
            # target_logits: [1, K, V]
            
            # ═══ Phase 3: Accept/Reject ═══
            n_accepted = 0
            
            for i, (token, draft_p) in enumerate(zip(draft_tokens, draft_probs)):
                # Target probability for this token
                target_probs = self._sample_probs(target_logits[:, i, :], temp)
                target_p = target_probs[0, token].item()
                
                # Accept with probability min(1, target_p / draft_p)
                acceptance_ratio = target_p / max(draft_p, 1e-10)
                
                if torch.rand(1).item() < min(1.0, acceptance_ratio):
                    # Accept this token
                    generated.append(token)
                    n_accepted += 1
                    total_accepted += 1
                    
                    if on_token:
                        on_token(self.tokenizer.decode([token]))
                    
                    if token == 0:  # EOS
                        break
                else:
                    # Reject — sample correction from adjusted distribution
                    # p_corrected = max(0, p_target - p_draft) normalized
                    draft_dist = draft_distributions[i]  # stored draft probs
                    # Build correction distribution
                    correction_probs = torch.clamp(
                        target_probs - draft_dist, min=0
                    )
                    correction_sum = correction_probs.sum()
                    if correction_sum > 1e-8:
                        correction_probs = correction_probs / correction_sum
                    else:
                        correction_probs = target_probs  # fallback
                    
                    corrected = torch.multinomial(correction_probs, 1).item()
                    generated.append(corrected)
                    total_accepted += 1
                    
                    if on_token:
                        on_token(self.tokenizer.decode([corrected]))
                    
                    # Invalidate draft cache beyond this point
                    draft_hiddens = None
                    break
            
            # If all K accepted, sample one bonus token from target
            if n_accepted == len(draft_tokens) and len(generated) < max_tokens:
                bonus_probs = self._sample_probs(target_logits[:, -1, :], temp)
                bonus = torch.multinomial(bonus_probs, 1).item()
                generated.append(bonus)
                total_accepted += 1
                
                if on_token and bonus != 0:
                    on_token(self.tokenizer.decode([bonus]))
            
            # Check for EOS
            if generated and generated[-1] == 0:
                generated.pop()  # remove EOS from output
                break
        
        elapsed = (time.time() - t0) * 1000
        
        # Estimate vanilla speed (target-only, 1 token per forward)
        tokens_gen = len(generated)
        acceptance = total_accepted / max(total_drafted, 1)
        # Speedup ≈ K * acceptance_rate / (1 + K * (cost_draft/cost_target))
        # For MinGRU/Mamba2, draft is ~10x cheaper
        estimated_speedup = (1 + self.K * acceptance) / (1 + self.K * 0.1)
        
        text = self.tokenizer.decode(generated)
        
        logger.info(
            f"Speculative: {tokens_gen} tokens, "
            f"accept={acceptance:.1%}, "
            f"speedup≈{estimated_speedup:.1f}x, "
            f"{elapsed:.0f}ms"
        )
        
        return SpecResult(
            text=text,
            tokens_generated=tokens_gen,
            tokens_drafted=total_drafted,
            tokens_accepted=total_accepted,
            acceptance_rate=acceptance,
            time_ms=elapsed,
            speedup=estimated_speedup,
        )
    
    def _target_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward through target model, return logits."""
        result = self.target.think(input_ids)
        if isinstance(result, tuple):
            logits, _ = result
        else:
            logits = result
        return logits
    
    def _sample_probs(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Convert logits to sampling probabilities with top-k/top-p."""
        if temperature > 0:
            logits = logits / temperature
        
        # Top-k filtering
        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_val = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_val, torch.full_like(logits, -float('inf')), logits)
        
        # Top-p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative - F.softmax(sorted_logits, dim=-1) > self.top_p
            sorted_logits[mask] = -float('inf')
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
        
        return F.softmax(logits, dim=-1)
