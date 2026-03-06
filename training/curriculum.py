"""
═══════════════════════════════════════════════════════════════
  Curriculum Learning with DoubtEngine Feedback
═══════════════════════════════════════════════════════════════

Extended curriculum learning that uses DoubtEngine coherence
as a difficulty proxy. Integrates with train_mamba2.py.

Difficulty estimation (TZ Section 2.2a thresholds):
  1. Heuristic scoring (length, lexical diversity, word complexity, special chars)
  2. DoubtEngine coherence (if available):
     coherence > 0.5 (COHERENCE_FLAG) → easy (model understands)
     0.2 - 0.5                        → medium
     < 0.2 (COHERENCE_BLOCK)          → hard (model struggles)

Schedule:
  Early epochs:  70% easy + 30% medium
  Mid epochs:    40% easy + 40% medium + 20% hard
  Late epochs:   20% easy + 30% medium + 50% hard

Usage:
  python training/curriculum.py --input data/train_corpus.txt --output data/curriculum_plan.json
  python training/curriculum.py --input data/train_corpus.txt --model models/mamba2/mamba2_omega.pt --doubt models/doubt/doubt_engine_best.pt
  python training/curriculum.py --dry_run
"""

import os
import sys
import json
import math
import random
import logging
import argparse
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("curriculum")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Curriculum Learning with Doubt Feedback")
    p.add_argument("--input", type=str, default=None,
                   help="Input corpus (txt)")
    p.add_argument("--output", type=str, default="data/curriculum_plan.json",
                   help="Output curriculum plan JSON")
    p.add_argument("--model", type=str, default=None,
                   help="Brain model checkpoint for doubt-based scoring")
    p.add_argument("--doubt", type=str, default=None,
                   help="DoubtEngine checkpoint for coherence-based difficulty")
    p.add_argument("--chunk_size", type=int, default=512,
                   help="Characters per chunk")
    p.add_argument("--n_phases", type=int, default=3,
                   help="Number of difficulty phases (easy/medium/hard)")
    p.add_argument("--dry_run", action='store_true',
                   help="Quick test with synthetic data")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
#  DoubtEngine threshold constants (import from source of truth)
# ═══════════════════════════════════════════════════════════════

try:
    from brain.doubt_engine import DoubtEngine as _DE
    _COHERENCE_FLAG  = _DE.COHERENCE_FLAG    # 0.5 — below this = "suspicious"
    _COHERENCE_BLOCK = _DE.COHERENCE_BLOCK   # 0.2 — below this = "broken"
except ImportError:
    _COHERENCE_FLAG  = 0.5
    _COHERENCE_BLOCK = 0.2


# ═══════════════════════════════════════════════════════════════
#  Doubt-based Difficulty Estimator
# ═══════════════════════════════════════════════════════════════

class DoubtDifficultyEstimator:
    """
    Uses DoubtEngine coherence as difficulty proxy.
    
    Low coherence = model struggles = hard sample.
    High coherence = model confident = easy sample.
    
    Falls back to heuristic scoring if DoubtEngine unavailable.
    """
    
    def __init__(self, model=None, doubt_engine=None, tokenizer=None, device='cpu'):
        self.model = model
        self.doubt_engine = doubt_engine
        self.tokenizer = tokenizer
        self.device = device
        self.use_doubt = (model is not None and doubt_engine is not None
                         and tokenizer is not None)
        
        if self.use_doubt:
            logger.info("DoubtDifficultyEstimator: using DoubtEngine coherence")
        else:
            logger.info("DoubtDifficultyEstimator: using heuristic scoring (no DoubtEngine)")
    
    def score(self, text: str) -> Dict[str, float]:
        """
        Score text difficulty. Returns dict with 'difficulty' in [0, 1].
        0 = easy, 1 = hard.
        
        Difficulty buckets (aligned with DoubtEngine TZ 2.2a):
          coherence > COHERENCE_FLAG  (0.5) → easy   (difficulty ≈ 0.0–0.3)
          COHERENCE_BLOCK..FLAG (0.2–0.5)   → medium (difficulty ≈ 0.3–0.7)
          coherence < COHERENCE_BLOCK (0.2) → hard   (difficulty ≈ 0.7–1.0)
        """
        scores = {}
        
        # Heuristic features (always available)
        scores['heuristic'] = self._heuristic_score(text)
        
        # DoubtEngine coherence (if available)
        if self.use_doubt:
            coherence = self._doubt_coherence(text)
            scores['coherence'] = coherence
            
            # Bucketed mapping using TZ thresholds
            if coherence > _COHERENCE_FLAG:
                # Model handles this well → easy
                # Map (FLAG..1.0) → (0.0..0.3) linearly
                t = (coherence - _COHERENCE_FLAG) / (1.0 - _COHERENCE_FLAG)
                scores['doubt_difficulty'] = 0.3 * (1.0 - t)
            elif coherence > _COHERENCE_BLOCK:
                # Model struggles somewhat → medium
                # Map (BLOCK..FLAG) → (0.3..0.7)
                t = (coherence - _COHERENCE_BLOCK) / (_COHERENCE_FLAG - _COHERENCE_BLOCK)
                scores['doubt_difficulty'] = 0.7 - 0.4 * t
            else:
                # Model fails → hard
                # Map (0..BLOCK) → (0.7..1.0)
                t = coherence / max(_COHERENCE_BLOCK, 1e-6)
                scores['doubt_difficulty'] = 1.0 - 0.3 * t
            
            # Weighted combination: 70% doubt + 30% heuristic
            # (doubt is more accurate when available)
            scores['difficulty'] = 0.7 * scores['doubt_difficulty'] + 0.3 * scores['heuristic']
        else:
            scores['difficulty'] = scores['heuristic']
        
        return scores
    
    def _heuristic_score(self, text: str) -> float:
        """Heuristic difficulty scoring (no model needed)."""
        length = len(text)
        words = text.lower().split()
        
        if not words:
            return 0.5
        
        # Length score (longer = harder)
        len_score = min(length / 2000, 1.0)
        
        # Lexical diversity
        unique_ratio = len(set(words)) / len(words)
        
        # Average word length
        avg_word_len = sum(len(w) for w in words) / len(words)
        word_score = min(avg_word_len / 10, 1.0)
        
        # Special characters (code/math)
        special = sum(1 for c in text if c in '{}[]()=+*/|&^%$#@!~`\\;:')
        special_score = min(special / max(length, 1) * 10, 1.0)
        
        # Weighted aggregate
        difficulty = (
            0.1 * len_score +
            0.2 * unique_ratio +
            0.15 * word_score +
            0.25 * special_score +
            0.3 * min(avg_word_len / 8, 1.0)  # sentence complexity proxy
        )
        
        return min(max(difficulty, 0.0), 1.0)
    
    def _doubt_coherence(self, text: str) -> float:
        """Get DoubtEngine coherence for text (uses Brain model).
        
        Splits text at midpoint into query/response halves,
        encodes each through Brain embedding, and runs DoubtEngine.
        """
        import torch
        
        try:
            ids = self.tokenizer.encode(text)[:256]
            mid = len(ids) // 2
            if mid < 5:
                return 0.5  # too short for meaningful split
            
            with torch.no_grad():
                q_ids = torch.tensor([ids[:mid]], dtype=torch.long, device=self.device)
                r_ids = torch.tensor([ids[mid:]], dtype=torch.long, device=self.device)
                
                q_emb = self.model.embedding(q_ids).mean(dim=1)
                r_emb = self.model.embedding(r_ids).mean(dim=1)
                
                outputs = self.doubt_engine(q_emb, r_emb)
                return outputs['coherence'].item()
        except Exception as e:
            logger.debug(f"Doubt coherence error: {e}")
            return 0.5


# ═══════════════════════════════════════════════════════════════
#  Curriculum Scheduler
# ═══════════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    Controls difficulty mix over training epochs.
    
    Early: mostly easy samples → stable gradient signal
    Late: mostly hard samples → push performance boundary
    """
    
    # Default schedule: (easy%, medium%, hard%) per phase
    DEFAULT_SCHEDULE = [
        (0.70, 0.25, 0.05),   # Phase 0: early training
        (0.40, 0.40, 0.20),   # Phase 1: mid training
        (0.20, 0.30, 0.50),   # Phase 2: late training
    ]
    
    def __init__(self, n_epochs: int, n_phases: int = 3,
                 schedule: List[Tuple[float, float, float]] = None):
        self.n_epochs = n_epochs
        self.n_phases = n_phases
        self.schedule = schedule or self.DEFAULT_SCHEDULE[:n_phases]
        
        # Ensure schedule length matches n_phases
        while len(self.schedule) < n_phases:
            self.schedule.append(self.schedule[-1])
    
    def get_mix(self, epoch: int) -> Tuple[float, float, float]:
        """
        Get difficulty mix ratios for given epoch.
        
        Returns: (easy_ratio, medium_ratio, hard_ratio)
        """
        # Map epoch to schedule phase
        phase_len = max(1, self.n_epochs // len(self.schedule))
        phase_idx = min(epoch // phase_len, len(self.schedule) - 1)
        
        # Interpolate between phases for smooth transition
        next_idx = min(phase_idx + 1, len(self.schedule) - 1)
        progress = (epoch % phase_len) / max(phase_len, 1)
        
        current = self.schedule[phase_idx]
        next_phase = self.schedule[next_idx]
        
        easy = current[0] * (1 - progress) + next_phase[0] * progress
        medium = current[1] * (1 - progress) + next_phase[1] * progress
        hard = current[2] * (1 - progress) + next_phase[2] * progress
        
        # Normalize
        total = easy + medium + hard
        return (easy / total, medium / total, hard / total)
    
    def __repr__(self):
        phases = []
        for i, (e, m, h) in enumerate(self.schedule):
            phases.append(f"Phase{i}(easy={e:.0%},med={m:.0%},hard={h:.0%})")
        return f"CurriculumScheduler({', '.join(phases)})"


# ═══════════════════════════════════════════════════════════════
#  Dynamic Batch Mixer
# ═══════════════════════════════════════════════════════════════

class DynamicBatchMixer:
    """
    Mixes training data from difficulty pools according to curriculum schedule.
    
    Pools:
      - easy:   difficulty ∈ [0.0, 0.33)
      - medium: difficulty ∈ [0.33, 0.66)
      - hard:   difficulty ∈ [0.66, 1.0]
    """
    
    def __init__(self, scored_data: List[Dict], n_phases: int = 3):
        self.n_phases = n_phases
        
        # Sort into difficulty pools
        thresholds = [i / n_phases for i in range(1, n_phases)]
        
        self.pools = [[] for _ in range(n_phases)]
        for item in scored_data:
            difficulty = item.get('difficulty', 0.5)
            pool_idx = min(int(difficulty * n_phases), n_phases - 1)
            self.pools[pool_idx].append(item)
        
        pool_names = ['easy', 'medium', 'hard', 'very_hard', 'expert']
        for i, pool in enumerate(self.pools):
            name = pool_names[min(i, len(pool_names) - 1)]
            logger.info(f"  Pool '{name}': {len(pool)} samples")
    
    def sample_batch(self, batch_size: int, mix: Tuple[float, ...]) -> List[Dict]:
        """
        Sample a batch according to difficulty mix ratios.
        
        Args:
            batch_size: number of samples
            mix: tuple of ratios per pool (must sum to ~1.0)
        
        Returns:
            list of data items
        """
        batch = []
        
        for i, ratio in enumerate(mix):
            n_from_pool = max(1, int(batch_size * ratio))
            pool = self.pools[i]
            
            if pool:
                samples = random.choices(pool, k=n_from_pool)
                batch.extend(samples)
            elif i > 0 and self.pools[i - 1]:
                # Fallback: borrow from adjacent pool
                samples = random.choices(self.pools[i - 1], k=n_from_pool)
                batch.extend(samples)
        
        # Trim to exact batch_size, shuffle
        random.shuffle(batch)
        batch = batch[:batch_size]
        
        # Pad if rounding gave fewer than batch_size
        while len(batch) < batch_size:
            all_items = [item for pool in self.pools for item in pool]
            if all_items:
                batch.append(random.choice(all_items))
            else:
                break
        
        return batch[:batch_size]
    
    def get_epoch_data(self, mix: Tuple[float, ...], max_samples: int = 0) -> List[Dict]:
        """
        Get full epoch data according to mix ratios.
        
        Returns all available data mixed according to ratios.
        """
        total = sum(len(p) for p in self.pools)
        if max_samples > 0:
            total = min(total, max_samples)
        
        epoch_data = []
        for i, ratio in enumerate(mix):
            n_pool = max(1, int(total * ratio))
            pool = self.pools[i]
            if pool:
                if n_pool >= len(pool):
                    # Use all + repeat
                    repeats = n_pool // len(pool) + 1
                    epoch_data.extend((pool * repeats)[:n_pool])
                else:
                    epoch_data.extend(random.sample(pool, n_pool))
        
        random.shuffle(epoch_data)
        return epoch_data[:total]


# ═══════════════════════════════════════════════════════════════
#  Main: Score and Plan
# ═══════════════════════════════════════════════════════════════

def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """Split text into chunks at paragraph boundaries."""
    import re
    paragraphs = text.split('\n\n')
    
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > chunk_size and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para if current else para
    
    if current.strip():
        chunks.append(current.strip())
    
    return [c for c in chunks if len(c) > 30]


def main():
    args = parse_args()
    
    if args.dry_run:
        # Generate synthetic scored data
        logger.info("Dry run: generating synthetic scored data")
        scored = []
        for i in range(500):
            difficulty = random.random()
            scored.append({
                'text': f"Sample text #{i} " * (10 + int(difficulty * 20)),
                'difficulty': round(difficulty, 4),
                'phase': min(int(difficulty * args.n_phases), args.n_phases - 1),
                'scores': {'heuristic': round(difficulty, 4)},
            })
    else:
        if not args.input:
            logger.error("--input required (or use --dry_run)")
            return
        
        logger.info(f"Loading: {args.input}")
        with open(args.input, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        
        chunks = chunk_text(text, args.chunk_size)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Load models for doubt-based scoring  
        model = None
        doubt_engine = None
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
                logger.info(f"Brain model loaded ({device})")
                
                # Try loading DoubtEngine
                if args.doubt and os.path.exists(args.doubt):
                    try:
                        from brain.doubt_engine import DoubtEngine
                        doubt_engine = DoubtEngine(model.d_model).to(device)
                        state = torch.load(args.doubt, map_location=device, weights_only=True)
                        if 'model_state_dict' in state:
                            doubt_engine.load_state_dict(state['model_state_dict'], strict=False)
                        else:
                            doubt_engine.load_state_dict(state, strict=False)
                        doubt_engine.eval()
                        logger.info("DoubtEngine loaded for coherence-based scoring")
                    except ImportError:
                        logger.warning("brain/doubt_engine.py not found, using heuristic scoring")
                    except Exception as e:
                        logger.warning(f"DoubtEngine load failed: {e}")
            except Exception as e:
                logger.warning(f"Brain model load failed: {e}")
        
        # Score chunks
        estimator = DoubtDifficultyEstimator(
            model=model, doubt_engine=doubt_engine,
            tokenizer=tokenizer, device=device
        )
        
        scored = []
        for i, chunk in enumerate(chunks):
            scores = estimator.score(chunk)
            phase = min(int(scores['difficulty'] * args.n_phases), args.n_phases - 1)
            
            scored.append({
                'text': chunk,
                'difficulty': round(scores['difficulty'], 4),
                'phase': phase,
                'scores': {k: round(v, 4) for k, v in scores.items()},
            })
            
            if (i + 1) % 500 == 0:
                logger.info(f"  Scored {i+1}/{len(chunks)}...")
    
    # Sort by difficulty
    scored.sort(key=lambda x: x['difficulty'])
    
    # Stats
    phase_counts = Counter(item['phase'] for item in scored)
    pool_names = ['Easy', 'Medium', 'Hard', 'Very Hard', 'Expert']
    logger.info("Difficulty distribution:")
    for p in range(args.n_phases):
        count = phase_counts.get(p, 0)
        pct = count / max(len(scored), 1) * 100
        name = pool_names[min(p, len(pool_names) - 1)]
        logger.info(f"  Phase {p} ({name}): {count} chunks ({pct:.1f}%)")
    
    # Demo: show curriculum schedule
    scheduler = CurriculumScheduler(n_epochs=10, n_phases=args.n_phases)
    logger.info(f"\nCurriculum schedule: {scheduler}")
    for epoch in [0, 3, 5, 8]:
        mix = scheduler.get_mix(epoch)
        labels = ['easy', 'medium', 'hard'][:args.n_phases]
        mix_str = ', '.join(f"{l}={r:.0%}" for l, r in zip(labels, mix))
        logger.info(f"  Epoch {epoch}: {mix_str}")
    
    # Save plan
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    plan = {
        'n_chunks': len(scored),
        'n_phases': args.n_phases,
        'phase_distribution': dict(phase_counts),
        'schedule': [
            {'epoch': e, 'mix': scheduler.get_mix(e)}
            for e in range(10)
        ],
        'chunks': scored,
    }
    
    with open(str(output_path), 'w', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nSaved: {output_path}")
    logger.info(f"Total: {len(scored)} chunks, sorted easy→hard")


if __name__ == "__main__":
    main()
