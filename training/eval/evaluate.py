"""
═══════════════════════════════════════════════════════════════
  TARS Evaluation Benchmark Suite
═══════════════════════════════════════════════════════════════

Comprehensive benchmarks to measure TARS quality:
  1. Perplexity (language modeling quality)
  2. Text Generation (coherence, diversity)
  3. Inference Speed (tokens/sec, speculative speedup)
  4. Memory Usage (VRAM, model size)
  5. Reasoning (arithmetic, logic, knowledge)

Usage:
  python training/evaluate.py --model models/mamba2/brain_best.pt --all
  python training/evaluate.py --model models/mamba2/brain_best.pt --perplexity
  python training/evaluate.py --model models/mamba2/brain_best.pt --speed
"""

import os
import sys
import json
import time
import logging
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("evaluate")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ═══ Test Data ═══

PERPLEXITY_TEXTS = [
    "Искусственный интеллект — это область компьютерных наук, занимающаяся созданием интеллектуальных машин.",
    "Python является одним из самых популярных языков программирования в мире.",
    "Нейронные сети вдохновлены биологическими нейронными сетями головного мозга.",
    "Машинное обучение позволяет компьютерам учиться на данных без явного программирования.",
    "Трансформеры revolutionized обработку естественного языка благодаря механизму внимания.",
    "Государственные пространственные модели, такие как Mamba, предлагают линейную сложность для длинных последовательностей.",
    "Квантизация позволяет уменьшить размер модели в 4 раза с минимальной потерей качества.",
    "LoRA позволяет дообучить модель, изменяя менее 1% параметров.",
]

REASONING_TESTS = [
    {"q": "2 + 2 = ", "a": "4"},
    {"q": "10 * 5 = ", "a": "50"},
    {"q": "Столица России — ", "a": "Москва"},
    {"q": "Автор 'Война и мир' — ", "a": "Толстой"},
    {"q": "Самая большая планета — ", "a": "Юпитер"},
    {"q": "H2O — это формула ", "a": "воды"},
    {"q": "Python создал ", "a": "Гвидо ван Россум"},
    {"q": "100 - 37 = ", "a": "63"},
]

GENERATION_PROMPTS = [
    "Расскажи о ",
    "Напиши короткий рассказ про ",
    "Объясни, как работает ",
    "Привет! Я хочу узнать о ",
]


def eval_perplexity(model, tokenizer, device, texts=None):
    """Measure perplexity on test texts."""
    if texts is None:
        texts = PERPLEXITY_TEXTS
    
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) < 5:
            continue
        
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            result = model.think(input_ids)
            logits = result[0] if isinstance(result, tuple) else result
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 100))  # cap to avoid overflow
    
    logger.info(f"📊 Perplexity: {perplexity:.2f} (loss={avg_loss:.4f}, {total_tokens} tokens)")
    return {"perplexity": perplexity, "loss": avg_loss, "tokens": total_tokens}


def eval_generation(model, tokenizer, device, max_tokens=100):
    """Measure generation quality: coherence, diversity, repetition."""
    model.eval()
    results = []
    
    for prompt in GENERATION_PROMPTS:
        ids = tokenizer.encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        
        generated = []
        with torch.no_grad():
            result = model.think(input_ids)
            logits = result[0] if isinstance(result, tuple) else result
            
            for _ in range(max_tokens):
                probs = F.softmax(logits[:, -1, :] / 0.7, dim=-1)
                token = torch.multinomial(probs, 1).item()
                
                if token == 0:
                    break
                generated.append(token)
                
                token_t = torch.tensor([[token]], dtype=torch.long, device=device)
                result = model.think(token_t)
                logits = result[0] if isinstance(result, tuple) else result
        
        text = tokenizer.decode(generated)
        
        # Metrics
        tokens = text.split()
        unique_tokens = set(tokens)
        diversity = len(unique_tokens) / max(len(tokens), 1)
        
        # Repetition detection (n-gram overlap)
        ngrams_3 = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
        unique_3grams = set(ngrams_3)
        repetition = 1.0 - len(unique_3grams) / max(len(ngrams_3), 1)
        
        results.append({
            "prompt": prompt[:30],
            "length": len(generated),
            "diversity": diversity,
            "repetition": repetition,
            "text": text[:100],
        })
    
    avg_diversity = np.mean([r['diversity'] for r in results])
    avg_repetition = np.mean([r['repetition'] for r in results])
    avg_length = np.mean([r['length'] for r in results])
    
    logger.info(f"📊 Generation: diversity={avg_diversity:.2f}, "
                f"repetition={avg_repetition:.2f}, avg_len={avg_length:.0f}")
    
    for r in results:
        logger.info(f"  '{r['prompt']}...' → {r['text'][:60]}...")
    
    return {
        "diversity": avg_diversity,
        "repetition": avg_repetition,
        "avg_length": avg_length,
        "samples": results,
    }


def eval_speed(model, tokenizer, device, n_tokens=100, n_runs=3):
    """Measure inference speed (tokens/second)."""
    model.eval()
    
    prompt = "Расскажи о машинном обучении"
    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    
    # Warmup
    with torch.no_grad():
        model.think(input_ids)
    
    speeds = []
    for run in range(n_runs):
        generated = 0
        t0 = time.time()
        
        with torch.no_grad():
            result = model.think(input_ids)
            logits = result[0] if isinstance(result, tuple) else result
            
            for _ in range(n_tokens):
                token = logits[:, -1, :].argmax(dim=-1)
                generated += 1
                
                token_t = token.unsqueeze(0)
                result = model.think(token_t)
                logits = result[0] if isinstance(result, tuple) else result
        
        elapsed = time.time() - t0
        tps = generated / max(elapsed, 1e-6)
        speeds.append(tps)
    
    avg_speed = np.mean(speeds)
    logger.info(f"📊 Speed: {avg_speed:.1f} tokens/sec "
                f"({np.std(speeds):.1f} std, {n_runs} runs)")
    
    return {"tokens_per_sec": avg_speed, "runs": speeds}


def eval_reasoning(model, tokenizer, device):
    """Test basic reasoning/knowledge."""
    model.eval()
    correct = 0
    total = len(REASONING_TESTS)
    
    for test in REASONING_TESTS:
        ids = tokenizer.encode(test['q'])
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        
        generated = []
        with torch.no_grad():
            result = model.think(input_ids)
            logits = result[0] if isinstance(result, tuple) else result
            
            for _ in range(20):
                token = logits[:, -1, :].argmax(dim=-1).item()
                if token == 0:
                    break
                generated.append(token)
                
                token_t = torch.tensor([[token]], dtype=torch.long, device=device)
                result = model.think(token_t)
                logits = result[0] if isinstance(result, tuple) else result
        
        answer = tokenizer.decode(generated).strip().lower()
        expected = test['a'].lower()
        
        is_correct = expected in answer
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        logger.info(f"  {status} '{test['q']}' → '{answer[:40]}' (expected: {expected})")
    
    accuracy = correct / max(total, 1)
    logger.info(f"📊 Reasoning: {correct}/{total} ({accuracy:.0%})")
    
    return {"accuracy": accuracy, "correct": correct, "total": total}


def eval_memory(model):
    """Measure model memory footprint."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in MB
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = param_bytes / 1024 / 1024
    
    # VRAM (if CUDA)
    vram_mb = 0
    if next(model.parameters()).is_cuda:
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    
    logger.info(f"📊 Memory: {total_params:,} params ({size_mb:.1f} MB), "
                f"VRAM={vram_mb:.1f} MB")
    
    return {
        "total_params": total_params,
        "trainable_params": trainable,
        "size_mb": size_mb,
        "vram_mb": vram_mb,
    }


def main():
    p = argparse.ArgumentParser(description="TARS Evaluation Suite")
    p.add_argument("--model", type=str, default="models/mamba2/brain_best.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--all", action="store_true")
    p.add_argument("--perplexity", action="store_true")
    p.add_argument("--generation", action="store_true")
    p.add_argument("--speed", action="store_true")
    p.add_argument("--reasoning", action="store_true")
    p.add_argument("--memory", action="store_true")
    p.add_argument("--output", type=str, default=None,
                   help="Save results as JSON")
    args = p.parse_args()
    
    from brain.mamba2.model import TarsMamba2LM
    from brain.tokenizer import TarsTokenizer
    
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device not in ("auto", "cpu", "cuda"):
        device = args.device
    
    logger.info(f"Loading model: {args.model}")
    model, _ = TarsMamba2LM.load_pretrained(args.model, device=device)
    tokenizer = TarsTokenizer()
    
    logger.info(f"Device: {device}")
    
    results = {}
    
    if args.all or args.memory:
        results['memory'] = eval_memory(model)
    
    if args.all or args.perplexity:
        results['perplexity'] = eval_perplexity(model, tokenizer, device)
    
    if args.all or args.speed:
        results['speed'] = eval_speed(model, tokenizer, device)
    
    if args.all or args.reasoning:
        results['reasoning'] = eval_reasoning(model, tokenizer, device)
    
    if args.all or args.generation:
        results['generation'] = eval_generation(model, tokenizer, device)
    
    # Summary
    logger.info(f"\n{'═' * 50}")
    logger.info(f"  TARS Evaluation Summary")
    logger.info(f"{'═' * 50}")
    for name, data in results.items():
        if isinstance(data, dict):
            key_metrics = {k: v for k, v in data.items() 
                         if isinstance(v, (int, float)) and k != 'tokens'}
            logger.info(f"  {name}: {key_metrics}")
    
    if args.output:
        # Make results JSON-serializable
        clean = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean[k] = {kk: vv for kk, vv in v.items()
                           if isinstance(vv, (int, float, str, list))}
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)
        logger.info(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
