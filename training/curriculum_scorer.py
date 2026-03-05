"""
═══════════════════════════════════════════════════════════════
  Curriculum Learning Scorer for TARS
═══════════════════════════════════════════════════════════════

Scores training data by difficulty and sorts easy-to-hard
for curriculum learning. Research (2025) shows this gives:
  - Faster convergence (10-20% fewer steps)
  - Better final quality (+1-3% on benchmarks)

Difficulty criteria:
  1. Text length (normalized)
  2. Lexical diversity (unique tokens / total tokens)
  3. Avg word length (proxy for vocabulary complexity)
  4. Special character ratio (code, math)
  5. Perplexity under a small model (optional, most accurate)

Usage:
  python training/curriculum_scorer.py --input data/train_corpus.txt
                                       --output data/train_scored.jsonl
                                       --model models/mamba2/brain_best.pt
"""

import os
import sys
import json
import math
import re
import logging
import argparse
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("curriculum")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Curriculum Learning Data Scorer")
    p.add_argument("--input", type=str, required=True,
                   help="Input corpus (txt or jsonl)")
    p.add_argument("--output", type=str, default="data/train_scored.jsonl",
                   help="Output scored JSONL")
    p.add_argument("--model", type=str, default=None,
                   help="Optional: model checkpoint for perplexity scoring")
    p.add_argument("--chunk_size", type=int, default=512,
                   help="Characters per chunk for splitting")
    p.add_argument("--phases", type=int, default=3,
                   help="Number of difficulty phases (2-5)")
    return p.parse_args()


class DifficultyScorer:
    """Multi-criteria difficulty scorer for text chunks."""
    
    def __init__(self, use_model: bool = False, model=None, tokenizer=None, device='cpu'):
        self.use_model = use_model and model is not None
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def score(self, text: str) -> dict:
        """Score text difficulty (0=easy, 1=hard)."""
        scores = {}
        
        # 1. Length score (longer = harder)
        length = len(text)
        scores['length'] = min(length / 2000, 1.0)
        
        # 2. Lexical diversity (more unique words = harder)
        words = text.lower().split()
        if words:
            unique_ratio = len(set(words)) / len(words)
            scores['lexical_diversity'] = unique_ratio
        else:
            scores['lexical_diversity'] = 0.0
        
        # 3. Average word length (longer words = more complex)
        if words:
            avg_word_len = sum(len(w) for w in words) / len(words)
            scores['word_complexity'] = min(avg_word_len / 10, 1.0)
        else:
            scores['word_complexity'] = 0.0
        
        # 4. Special character ratio (code/math = harder)
        special = sum(1 for c in text if c in '{}[]()=+*/<>|&^%$#@!~`\\;:')
        scores['special_chars'] = min(special / max(length, 1) * 10, 1.0)
        
        # 5. Sentence complexity (avg sentence length)
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
            scores['sentence_complexity'] = min(avg_sent_len / 30, 1.0)
        else:
            scores['sentence_complexity'] = 0.0
        
        # 6. Language mixing (non-ASCII ratio — multilingual = harder)
        non_ascii = sum(1 for c in text if ord(c) > 127)
        scores['multilingual'] = min(non_ascii / max(length, 1) * 3, 1.0)
        
        # 7. Model perplexity (most accurate, optional)
        if self.use_model:
            scores['perplexity'] = self._model_perplexity(text)
        
        # Weighted aggregate
        weights = {
            'length': 0.1,
            'lexical_diversity': 0.2,
            'word_complexity': 0.15,
            'special_chars': 0.25,   # code/math gets high difficulty
            'sentence_complexity': 0.15,
            'multilingual': 0.05,
            'perplexity': 0.1,
        }
        
        total = sum(scores.get(k, 0) * w for k, w in weights.items())
        total_weight = sum(w for k, w in weights.items() if k in scores)
        
        scores['aggregate'] = total / max(total_weight, 1e-6)
        
        return scores
    
    def _model_perplexity(self, text: str) -> float:
        """Compute perplexity using the model (normalized 0-1)."""
        import torch
        import torch.nn.functional as F
        
        try:
            ids = self.tokenizer.encode(text)[:512]
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                result = self.model.think(input_ids)
                logits = result[0] if isinstance(result, tuple) else result
                
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean'
                )
                
                ppl = torch.exp(loss).item()
                # Normalize: log(ppl) / 10 → roughly 0-1
                return min(math.log(max(ppl, 1)) / 10, 1.0)
        except Exception as e:
            logger.debug(f"Perplexity error: {e}")
            return 0.5


def chunk_text(text: str, chunk_size: int = 512) -> list:
    """Split text into chunks at sentence boundaries."""
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > chunk_size and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += " " + sent if current else sent
    
    if current.strip():
        chunks.append(current.strip())
    
    return [c for c in chunks if len(c) > 50]  # min length filter


def assign_phase(score: float, n_phases: int = 3) -> int:
    """Assign difficulty phase (0=easiest, n_phases-1=hardest)."""
    return min(int(score * n_phases), n_phases - 1)


def main():
    args = parse_args()
    
    # Optional: load model for perplexity scoring
    model = None
    tokenizer = None
    device = 'cpu'
    
    if args.model and os.path.exists(args.model):
        try:
            import torch
            from brain.mamba2.model import TarsMamba2LM
            from brain.tokenizer import TarsTokenizer
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, _ = TarsMamba2LM.load_pretrained(args.model, device=device)
            model.eval()
            tokenizer = TarsTokenizer()
            logger.info(f"Model loaded for perplexity scoring ({device})")
        except Exception as e:
            logger.warning(f"Skipping model scoring: {e}")
            model = None
    
    scorer = DifficultyScorer(
        use_model=(model is not None),
        model=model, tokenizer=tokenizer, device=device
    )
    
    # Load input
    logger.info(f"Loading: {args.input}")
    with open(args.input, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Chunk
    chunks = chunk_text(text, args.chunk_size)
    logger.info(f"Split into {len(chunks)} chunks")
    
    # Score each chunk
    scored = []
    for i, chunk in enumerate(chunks):
        scores = scorer.score(chunk)
        phase = assign_phase(scores['aggregate'], args.phases)
        
        scored.append({
            'text': chunk,
            'difficulty': round(scores['aggregate'], 4),
            'phase': phase,
            'scores': {k: round(v, 4) for k, v in scores.items()},
        })
        
        if (i + 1) % 500 == 0:
            logger.info(f"  Scored {i+1}/{len(chunks)}...")
    
    # Sort by difficulty (easy → hard)
    scored.sort(key=lambda x: x['difficulty'])
    
    # Stats
    phase_counts = Counter(item['phase'] for item in scored)
    logger.info("Phase distribution:")
    for p in range(args.phases):
        count = phase_counts.get(p, 0)
        pct = count / len(scored) * 100
        label = ['Easy', 'Medium', 'Hard', 'Very Hard', 'Expert'][min(p, 4)]
        logger.info(f"  Phase {p} ({label}): {count} chunks ({pct:.1f}%)")
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in scored:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved: {args.output}")
    logger.info(f"Total: {len(scored)} chunks, sorted easy→hard")


if __name__ == "__main__":
    main()
