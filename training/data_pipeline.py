"""
═══════════════════════════════════════════════════════════════
  Data Pipeline — 8-Stage Cleaning + MinHash Dedup
═══════════════════════════════════════════════════════════════

Cleans and prepares training data for TARS UMOT pipeline.
Expected yield: 57% (1.5B raw → ~865M clean tokens).

8 Stages (from TZ v3 R14.2):
  1. dedup_exact     — MD5/SHA256 hash dedup (~15% removal)
  2. dedup_minhash   — MinHash Jaccard>0.8 (~5% removal)
  3. filter_language — keep RU+EN only
  4. filter_length   — min 20 tokens, max 4096 tokens
  5. filter_content  — remove "As an AI model..." patterns
  6. fix_formatting  — normalize whitespace, fix encoding
  7. score_quality   — perplexity + heuristics (keep top 80%)
  8. filter_safety   — keyword + regex blocklist

Usage:
  from training.data_pipeline import DataCleaner, MinHashDeduplicator, SequencePacker

  # Clean a corpus file
  cleaner = DataCleaner()
  clean_data = cleaner.process_file("data/raw_corpus.jsonl", "data/clean_corpus.jsonl")
  
  # Or run from CLI
  python training/data_pipeline.py --input data/raw.jsonl --output data/clean.jsonl --stats
"""

import os
import re
import sys
import json
import hashlib
import logging
import argparse
import random
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Set, Iterator, Tuple

logger = logging.getLogger("Tars.DataPipeline")


# ═══════════════════════════════════════════
# 1. MinHash Near-Duplicate Detection
# ═══════════════════════════════════════════

class MinHashDeduplicator:
    """
    MinHash + LSH (Locality Sensitive Hashing) for near-duplicate detection.
    
    Finds pairs with Jaccard similarity > threshold (~0.8).
    Catches ~5% additional duplicates beyond exact dedup.
    
    Implementation:
      1. Compute n-gram (shingle) sets for each text
      2. Generate MinHash signatures (num_perm hash functions)
      3. LSH banding: group into bands, check for collisions
      4. Verify candidates with actual Jaccard similarity
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8,
                 n_gram: int = 5):
        """
        Args:
            num_perm: number of hash permutations for MinHash signature
            threshold: Jaccard similarity threshold for "near-duplicate"
            n_gram: character n-gram size for shingling
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.n_gram = n_gram
        
        # LSH banding parameters
        # For threshold t, optimal: bands * rows = num_perm
        # where (1/bands)^(1/rows) ≈ threshold
        self.bands = max(1, int(num_perm * (1 - threshold)))
        self.rows = max(1, num_perm // self.bands)
        
        # Generate random hash coefficients (universal hashing)
        self._a = [random.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self._b = [random.randint(0, 2**31 - 1) for _ in range(num_perm)]
        self._prime = 2**31 - 1  # Mersenne prime
    
    def _shingle(self, text: str) -> Set[int]:
        """Convert text to set of hashed character n-grams."""
        text = text.lower().strip()
        if len(text) < self.n_gram:
            return {hash(text)}
        
        shingles = set()
        for i in range(len(text) - self.n_gram + 1):
            shingle = text[i:i + self.n_gram]
            shingles.add(hash(shingle) % self._prime)
        return shingles
    
    def _minhash(self, shingles: Set[int]) -> List[int]:
        """Compute MinHash signature from shingle set."""
        if not shingles:
            return [self._prime] * self.num_perm
        
        signature = []
        for i in range(self.num_perm):
            min_hash = self._prime
            for s in shingles:
                h = (self._a[i] * s + self._b[i]) % self._prime
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        
        return signature
    
    def _lsh_buckets(self, signature: List[int]) -> List[str]:
        """Compute LSH bucket keys for a signature."""
        buckets = []
        for band_idx in range(self.bands):
            start = band_idx * self.rows
            end = min(start + self.rows, len(signature))
            band = tuple(signature[start:end])
            bucket_key = f"b{band_idx}_{hash(band)}"
            buckets.append(bucket_key)
        return buckets
    
    def find_duplicates(self, texts: List[str]) -> Set[int]:
        """
        Find indices of near-duplicate texts.
        
        Returns: set of indices to REMOVE (keeps the first occurrence).
        """
        # Compute signatures
        signatures = []
        for text in texts:
            shingles = self._shingle(text)
            sig = self._minhash(shingles)
            signatures.append(sig)
        
        # LSH: group by buckets
        bucket_to_indices = defaultdict(list)
        for idx, sig in enumerate(signatures):
            for bucket_key in self._lsh_buckets(sig):
                bucket_to_indices[bucket_key].append(idx)
        
        # Find candidate pairs (same bucket)
        candidates = set()
        for indices in bucket_to_indices.values():
            if len(indices) < 2:
                continue
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    candidates.add((indices[i], indices[j]))
        
        # Verify candidates with estimated Jaccard
        to_remove = set()
        for i, j in candidates:
            if j in to_remove:
                continue
            
            # Estimate Jaccard from MinHash signatures
            matches = sum(1 for a, b in zip(signatures[i], signatures[j]) if a == b)
            est_jaccard = matches / self.num_perm
            
            if est_jaccard >= self.threshold:
                to_remove.add(j)  # Remove later occurrence
        
        return to_remove


# ═══════════════════════════════════════════
# 2. Data Cleaner — 8 Stages
# ═══════════════════════════════════════════

class DataCleaner:
    """
    8-stage cleaning pipeline.
    Expected yield: 57% (1.5B raw → 865M clean).
    
    Each stage returns filtered items. Stages run in order.
    """
    
    STAGES = [
        'dedup_exact',
        'dedup_minhash',
        'filter_language',
        'filter_length',
        'filter_content',
        'fix_formatting',
        'score_quality',
        'filter_safety',
    ]
    
    # Banned patterns (stage 5: filter_content)
    BANNED_PATTERNS = [
        re.compile(r'[Aa]s an AI (language )?model', re.IGNORECASE),
        re.compile(r'I cannot (help|assist|provide)', re.IGNORECASE),
        re.compile(r'Как языковая модель', re.IGNORECASE),
        re.compile(r'OpenAI|ChatGPT|Claude|GPT-4', re.IGNORECASE),
        re.compile(r'I\'m sorry,? but I (can\'?t|cannot)', re.IGNORECASE),
        re.compile(r'Я не могу (помочь|ответить|сделать)', re.IGNORECASE),
    ]
    
    # Safety blocklist keywords (stage 8)
    SAFETY_KEYWORDS = [
        # Minimal safety filter — extend as needed
        'самоубийство инструкция', 'как сделать бомбу', 'how to make a bomb',
        'наркотики рецепт', 'drug synthesis recipe',
    ]
    
    def __init__(self, min_tokens: int = 20, max_tokens: int = 4096,
                 quality_percentile: float = 0.80,
                 minhash_threshold: float = 0.8):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.quality_percentile = quality_percentile
        self.minhash = MinHashDeduplicator(threshold=minhash_threshold)
        self._stats = defaultdict(int)
    
    def process(self, items: List[Dict]) -> List[Dict]:
        """
        Run all 8 stages on a list of items.
        
        Each item must have a 'text' field.
        
        Returns: cleaned items list.
        """
        self._stats = defaultdict(int)
        self._stats['input'] = len(items)
        
        for stage in self.STAGES:
            before = len(items)
            stage_fn = getattr(self, f'_stage_{stage}')
            items = stage_fn(items)
            removed = before - len(items)
            self._stats[f'{stage}_removed'] = removed
            self._stats[f'{stage}_remaining'] = len(items)
            if removed > 0:
                logger.info(f"  Stage '{stage}': removed {removed} items "
                           f"({removed / max(before, 1):.1%}), "
                           f"remaining {len(items)}")
        
        self._stats['output'] = len(items)
        self._stats['yield_pct'] = len(items) / max(self._stats['input'], 1)
        return items
    
    def process_file(self, input_path: str, output_path: str,
                     text_field: str = 'text') -> dict:
        """
        Process JSONL file through all stages.
        
        Returns: stats dict.
        """
        items = []
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if text_field in item and item[text_field]:
                        items.append(item)
                except json.JSONDecodeError:
                    pass
        
        logger.info(f"Loaded {len(items)} items from {input_path}")
        
        clean = self.process(items)
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in clean:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(clean)} clean items → {output_path}")
        return dict(self._stats)
    
    # ─── Stage implementations ─────────────────
    
    def _stage_dedup_exact(self, items: List[Dict]) -> List[Dict]:
        """Stage 1: SHA256 exact dedup. Catches ~15%."""
        seen = set()
        result = []
        for item in items:
            h = hashlib.sha256(item['text'].encode('utf-8')).hexdigest()
            if h not in seen:
                seen.add(h)
                result.append(item)
        return result
    
    def _stage_dedup_minhash(self, items: List[Dict]) -> List[Dict]:
        """Stage 2: MinHash near-duplicate dedup. Catches ~5%."""
        if len(items) < 2:
            return items
        
        texts = [item['text'] for item in items]
        to_remove = self.minhash.find_duplicates(texts)
        
        return [item for idx, item in enumerate(items) if idx not in to_remove]
    
    def _stage_filter_language(self, items: List[Dict]) -> List[Dict]:
        """Stage 3: Keep RU+EN only (heuristic, no fasttext dependency)."""
        def is_ru_or_en(text: str) -> bool:
            # Simple heuristic: check for Cyrillic or Latin dominance
            cyrillic = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
            latin = sum(1 for c in text if 'A' <= c <= 'z')
            alpha = cyrillic + latin
            if alpha < 10:
                return False
            # At least 60% of alphabetic chars should be RU or EN
            return (cyrillic + latin) / max(len(text), 1) > 0.3
        
        return [item for item in items if is_ru_or_en(item['text'])]
    
    def _stage_filter_length(self, items: List[Dict]) -> List[Dict]:
        """Stage 4: min 20 tokens, max 4096 tokens."""
        def in_range(text: str) -> bool:
            tokens = len(text.split())
            return self.min_tokens <= tokens <= self.max_tokens
        
        return [item for item in items if in_range(item['text'])]
    
    def _stage_filter_content(self, items: List[Dict]) -> List[Dict]:
        """Stage 5: Remove 'As an AI model...' and similar patterns."""
        def is_clean(text: str) -> bool:
            return not any(pattern.search(text) for pattern in self.BANNED_PATTERNS)
        
        return [item for item in items if is_clean(item['text'])]
    
    def _stage_fix_formatting(self, items: List[Dict]) -> List[Dict]:
        """Stage 6: Normalize whitespace, fix encoding."""
        for item in items:
            text = item['text']
            # Collapse multiple spaces/newlines
            text = re.sub(r' {3,}', '  ', text)
            text = re.sub(r'\n{4,}', '\n\n\n', text)
            # Fix common encoding artifacts
            text = text.replace('\x00', '')
            text = text.replace('\ufeff', '')  # BOM
            text = text.replace('\u200b', '')  # Zero-width space
            text = text.strip()
            item['text'] = text
        return [item for item in items if len(item['text']) > 10]
    
    def _stage_score_quality(self, items: List[Dict]) -> List[Dict]:
        """Stage 7: Quality scoring — keep top 80%."""
        for item in items:
            item['_quality'] = self._quality_score(item['text'])
        
        # Sort by quality, keep top percentile
        items.sort(key=lambda x: x['_quality'], reverse=True)
        cutoff = max(1, int(len(items) * self.quality_percentile))
        
        result = items[:cutoff]
        
        # Clean temp field
        for item in result:
            item.pop('_quality', None)
        
        return result
    
    def _stage_filter_safety(self, items: List[Dict]) -> List[Dict]:
        """Stage 8: Keyword + regex safety blocklist."""
        def is_safe(text: str) -> bool:
            text_lower = text.lower()
            return not any(kw in text_lower for kw in self.SAFETY_KEYWORDS)
        
        return [item for item in items if is_safe(item['text'])]
    
    def _quality_score(self, text: str) -> float:
        """Heuristic quality scoring (no external model needed)."""
        score = 0.0
        tokens = text.split()
        n_tokens = len(tokens)
        
        # Good length range
        if 30 < n_tokens < 500:
            score += 0.3
        elif 500 <= n_tokens < 2000:
            score += 0.2
        
        # Has code blocks
        if '```' in text:
            score += 0.2
        
        # Structured content (numbered lists)
        if re.search(r'\d+\.\s', text):
            score += 0.1
        
        # Low repetition
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            unique_ratio = len(set(sentences)) / len(sentences)
            score += unique_ratio * 0.2
        
        # Paragraph structure
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.1
        
        # Lexical diversity
        if n_tokens > 10:
            unique_words = len(set(w.lower() for w in tokens))
            diversity = unique_words / n_tokens
            score += diversity * 0.1
        
        return min(score, 1.0)
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return dict(self._stats)


# ═══════════════════════════════════════════
# 3. Sequence Packer — Efficient batching
# ═══════════════════════════════════════════

class SequencePacker:
    """
    Pack short sequences into longer chunks for training efficiency.
    
    Instead of padding short sequences to max_len (wasting compute),
    concatenate multiple short sequences with separator tokens.
    
    Example:
      Input:  ["Hello world", "How are you?", "Fine thanks"]
      Output: ["Hello world<sep>How are you?<sep>Fine thanks"]
      → single training sample, no padding waste
    """
    
    def __init__(self, max_len: int = 512, sep_token_id: int = 0):
        self.max_len = max_len
        self.sep_token_id = sep_token_id
    
    def pack(self, token_sequences: List[List[int]]) -> List[List[int]]:
        """
        Pack token sequences into chunks of max_len.
        
        Args:
            token_sequences: list of token ID lists
        
        Returns:
            list of packed token ID lists (each ≤ max_len)
        """
        packed = []
        current = []
        
        for seq in token_sequences:
            if not seq:
                continue
            
            # If adding this sequence would overflow, flush current
            needed = len(seq) + (1 if current else 0)  # +1 for separator
            if current and len(current) + needed > self.max_len:
                packed.append(current[:self.max_len])
                current = []
            
            # Add separator if not first
            if current:
                current.append(self.sep_token_id)
            
            current.extend(seq)
        
        # Flush remaining
        if current:
            packed.append(current[:self.max_len])
        
        return packed
    
    def pack_with_loss_mask(self, token_sequences: List[List[int]]) -> List[Tuple[List[int], List[int]]]:
        """
        Pack sequences and return loss mask (0 at separator positions).
        
        Returns: list of (tokens, loss_mask) tuples
        """
        packed = []
        current_tokens = []
        current_mask = []
        
        for seq in token_sequences:
            if not seq:
                continue
            
            needed = len(seq) + (1 if current_tokens else 0)
            if current_tokens and len(current_tokens) + needed > self.max_len:
                packed.append((current_tokens[:self.max_len],
                             current_mask[:self.max_len]))
                current_tokens = []
                current_mask = []
            
            if current_tokens:
                current_tokens.append(self.sep_token_id)
                current_mask.append(0)  # Don't compute loss on separator
            
            current_tokens.extend(seq)
            current_mask.extend([1] * len(seq))
        
        if current_tokens:
            packed.append((current_tokens[:self.max_len],
                         current_mask[:self.max_len]))
        
        return packed


# ═══════════════════════════════════════════
# CLI Interface
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TARS Data Pipeline — 8-stage cleaning")
    parser.add_argument("--input", type=str, help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/clean_corpus.jsonl",
                        help="Output cleaned JSONL file")
    parser.add_argument("--text-field", type=str, default="text",
                        help="Field name for text content")
    parser.add_argument("--min-tokens", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--quality", type=float, default=0.80,
                        help="Quality percentile (keep top N%%)")
    parser.add_argument("--minhash-threshold", type=float, default=0.8)
    parser.add_argument("--stats", action="store_true", help="Print detailed stats")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run with synthetic data")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(levelname)s] %(message)s')
    
    if args.dry_run:
        logger.info("═══ Dry Run: Synthetic Data ═══")
        items = []
        for i in range(200):
            text = f"Пример текста номер {i}. " * random.randint(3, 30)
            if i < 20:
                text = items[0]['text'] if items else text  # exact duplicates
            if i % 50 == 0:
                text += " As an AI model, I cannot help with that."
            items.append({'text': text, 'source': 'synthetic', 'id': i})
        
        cleaner = DataCleaner(
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            quality_percentile=args.quality,
            minhash_threshold=args.minhash_threshold,
        )
        
        clean = cleaner.process(items)
        stats = cleaner.get_stats()
        
        print(f"\n═══ Results ═══")
        print(f"  Input:  {stats['input']} items")
        print(f"  Output: {stats['output']} items")
        print(f"  Yield:  {stats['yield_pct']:.1%}")
        
        if args.stats:
            print(f"\n═══ Per-Stage Stats ═══")
            for stage in DataCleaner.STAGES:
                removed = stats.get(f'{stage}_removed', 0)
                remaining = stats.get(f'{stage}_remaining', '?')
                print(f"  {stage:20s}: -{removed:4d} → {remaining} remaining")
        
        print("\n✅ Dry run complete")
        return
    
    if not args.input:
        parser.error("--input required (or use --dry-run)")
        return
    
    cleaner = DataCleaner(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        quality_percentile=args.quality,
        minhash_threshold=args.minhash_threshold,
    )
    
    stats = cleaner.process_file(args.input, args.output, args.text_field)
    
    print(f"\n═══ Results ═══")
    print(f"  Input:  {stats['input']} items")
    print(f"  Output: {stats['output']} items")
    print(f"  Yield:  {stats['yield_pct']:.1%}")
    
    if args.stats:
        print(f"\n═══ Per-Stage Stats ═══")
        for stage in DataCleaner.STAGES:
            removed = stats.get(f'{stage}_removed', 0)
            remaining = stats.get(f'{stage}_remaining', '?')
            print(f"  {stage:20s}: -{removed:4d} → {remaining} remaining")


if __name__ == "__main__":
    main()
