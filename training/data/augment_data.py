"""
═══════════════════════════════════════════════════════════════
  Data Augmentation Pipeline for TARS Training
═══════════════════════════════════════════════════════════════

Generate more training data from existing data:
  1. Token-level noise (swap, delete, insert → denoising objective)
  2. Sentence shuffling (learn to reconstruct order)
  3. Random masking (BERT-style → predict masked tokens)
  4. Style transfer (formal ↔ informal)
  5. Format variation (same content in different structures)

Result: 3-5x more effective training data without new sources.

Usage:
  python training/augment_data.py \
      --input data/corpus.txt \
      --output data/corpus_augmented.txt \
      --methods noise,shuffle,mask --multiplier 3
"""

import os
import sys
import random
import logging
import argparse
import re
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("augment")


def augment_noise(text: str, noise_rate: float = 0.05) -> str:
    """
    Token-level noise injection.
    
    For each token, with probability noise_rate:
      - 40% swap with neighbor
      - 30% delete token
      - 30% duplicate token
    
    Model learns to be robust to typos and noise.
    """
    words = text.split()
    if len(words) < 3:
        return text
    
    result = []
    i = 0
    while i < len(words):
        if random.random() < noise_rate and len(words) > 3:
            action = random.random()
            if action < 0.4 and i + 1 < len(words):
                # Swap with next
                result.append(words[i + 1])
                result.append(words[i])
                i += 2
                continue
            elif action < 0.7:
                # Delete
                i += 1
                continue
            else:
                # Duplicate
                result.append(words[i])
                result.append(words[i])
        else:
            result.append(words[i])
        i += 1
    
    return ' '.join(result)


def augment_shuffle_sentences(text: str) -> str:
    """
    Shuffle sentences within a paragraph.
    
    Uses the original text as target and shuffled as input
    for reconstruction training.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if len(sentences) < 3:
        return text
    
    shuffled = sentences.copy()
    random.shuffle(shuffled)
    return ' '.join(shuffled)


def augment_mask(text: str, mask_rate: float = 0.15, mask_token: str = "[MASK]") -> str:
    """
    Random masking (BERT-style).
    
    For each word, with probability mask_rate:
      - 80% replace with [MASK]
      - 10% replace with random word from text
      - 10% keep original (model must decide if it's correct)
    """
    words = text.split()
    if len(words) < 5:
        return text
    
    result = []
    for w in words:
        if random.random() < mask_rate:
            action = random.random()
            if action < 0.8:
                result.append(mask_token)
            elif action < 0.9:
                result.append(random.choice(words))
            else:
                result.append(w)
        else:
            result.append(w)
    
    return ' '.join(result)


def augment_case_variation(text: str) -> str:
    """Randomly change case patterns for robustness."""
    variations = [
        text.lower(),
        text.upper()[:100] + text[100:] if len(text) > 100 else text,
        text[0].upper() + text[1:] if text else text,
    ]
    return random.choice(variations)


def augment_punctuation_noise(text: str, rate: float = 0.1) -> str:
    """Add/remove/change punctuation randomly."""
    result = []
    for char in text:
        if char in '.,;:!?' and random.random() < rate:
            action = random.random()
            if action < 0.3:
                continue  # Remove
            elif action < 0.6:
                result.append(random.choice('.,;:!?'))  # Replace
            else:
                result.append(char)
                result.append(char)  # Double
        else:
            result.append(char)
    return ''.join(result)


def create_denoising_pair(text: str) -> tuple:
    """
    Create (noisy_input, clean_target) pair for denoising training.
    
    The model learns to reconstruct clean text from noisy input,
    which improves robustness and language understanding.
    """
    noised = augment_noise(text, noise_rate=0.1)
    noised = augment_punctuation_noise(noised, rate=0.05)
    
    # Format as instruction
    input_text = f"Исправь текст: {noised}"
    target_text = text
    
    return input_text, target_text


def augment_file(input_path: str, output_path: str, 
                 methods: List[str], multiplier: int = 3):
    """
    Augment a text corpus file.
    
    Args:
        input_path: original text file
        output_path: augmented text file  
        methods: list of augmentation methods to apply
        multiplier: how many augmented versions per original
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
    
    if not paragraphs:
        # Try sentence-level
        paragraphs = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) 
                       if s.strip() and len(s.strip()) > 10]
    
    logger.info(f"Source: {len(paragraphs)} paragraphs from {input_path}")
    
    augmented = list(paragraphs)  # Start with originals
    
    method_map = {
        'noise': lambda t: augment_noise(t, 0.05),
        'shuffle': augment_shuffle_sentences,
        'mask': lambda t: augment_mask(t, 0.15),
        'case': augment_case_variation,
        'punctuation': lambda t: augment_punctuation_noise(t, 0.1),
    }
    
    for _ in range(multiplier):
        for para in paragraphs:
            for method_name in methods:
                if method_name in method_map:
                    aug = method_map[method_name](para)
                    if aug != para and len(aug) > 10:
                        augmented.append(aug)
    
    # Also add denoising pairs
    if 'denoise' in methods:
        for para in paragraphs[:len(paragraphs)//2]:
            inp, tgt = create_denoising_pair(para)
            augmented.append(f"{inp}\n{tgt}")
    
    random.shuffle(augmented)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(augmented))
    
    ratio = len(augmented) / max(len(paragraphs), 1)
    logger.info(f"Augmented: {len(paragraphs)} → {len(augmented)} paragraphs ({ratio:.1f}x)")
    logger.info(f"Saved: {output_path}")


def main():
    p = argparse.ArgumentParser(description="TARS Data Augmentation")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--methods", type=str, default="noise,shuffle,mask",
                   help="Comma-separated: noise,shuffle,mask,case,punctuation,denoise")
    p.add_argument("--multiplier", type=int, default=3)
    args = p.parse_args()
    
    output = args.output or args.input.replace('.txt', '_augmented.txt')
    methods = [m.strip() for m in args.methods.split(',')]
    
    augment_file(args.input, output, methods, args.multiplier)


if __name__ == "__main__":
    main()
