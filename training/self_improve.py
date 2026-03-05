"""
═══════════════════════════════════════════════════════════════
  TARS Self-Improvement Loop
═══════════════════════════════════════════════════════════════

Automated pipeline that makes TARS improve itself:

  1. Generate responses to diverse prompts
  2. Self-judge quality (or use external judge)  
  3. Create preference/feedback data
  4. Train with DPO or KTO
  5. Evaluate improvement
  6. Repeat if improved, rollback if not

This is the "AlphaGo for language models" approach:
self-play → judge → improve → repeat.

Usage:
  python training/self_improve.py \
      --model models/mamba2/brain_best.pt \
      --judge ollama --iterations 5
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("self_improve")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_step(cmd: list, description: str) -> bool:
    """Run a subprocess step. Returns True on success."""
    logger.info(f"  → {description}")
    logger.info(f"    CMD: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, 
            timeout=3600, cwd=str(ROOT),
            encoding='utf-8', errors='replace'
        )
        if result.returncode != 0:
            logger.error(f"    FAILED: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"    TIMEOUT: {description}")
        return False
    except Exception as e:
        logger.error(f"    ERROR: {e}")
        return False


def self_improve(args):
    """Main self-improvement loop."""
    model_path = args.model
    
    logger.info(f"{'═' * 50}")
    logger.info(f"  TARS Self-Improvement Loop")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Judge: {args.judge} ({args.judge_model})")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info(f"{'═' * 50}")
    
    python = sys.executable
    
    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n{'─' * 40}")
        logger.info(f"  Iteration {iteration}/{args.iterations}")
        logger.info(f"{'─' * 40}")
        
        # Backup current model
        backup_path = model_path.replace('.pt', f'_backup_iter{iteration}.pt')
        if os.path.exists(model_path):
            shutil.copy2(model_path, backup_path)
            logger.info(f"  Backup: {backup_path}")
        
        # ═══ Step 1: Evaluate current model ═══
        eval_before = Path(args.save_dir) / f"eval_before_iter{iteration}.json"
        run_step([
            python, "training/evaluate.py",
            "--model", model_path,
            "--perplexity", "--reasoning",
            "--output", str(eval_before),
        ], "Evaluating current model...")
        
        # ═══ Step 2: Generate preference data ═══
        pref_path = Path(args.save_dir) / f"preferences_iter{iteration}.jsonl"
        
        if args.method == "dpo":
            success = run_step([
                python, "training/generate_preferences.py",
                "--judge", args.judge,
                "--judge_model", args.judge_model,
                "--generator", args.generator,
                "--gen_model", args.gen_model,
                "--output", str(pref_path),
                "--n_pairs", str(args.n_pairs),
            ], "Generating preference pairs (RLAIF)...")
            
            if not success:
                logger.error("  Failed to generate preferences, skipping iteration")
                continue
            
            # ═══ Step 3: Train DPO ═══
            success = run_step([
                python, "training/train_dpo.py",
                "--data", str(pref_path),
                "--model", model_path,
                "--save_dir", args.save_dir,
                "--epochs", str(args.train_epochs),
                "--beta", str(args.beta),
            ], "Training DPO alignment...")
            
            improved_path = Path(args.save_dir) / "brain_dpo.pt"
        
        elif args.method == "kto":
            # Generate feedback data
            feedback_path = Path(args.save_dir) / f"feedback_iter{iteration}.jsonl"
            success = run_step([
                python, "training/generate_preferences.py",
                "--judge", args.judge,
                "--judge_model", args.judge_model,
                "--generator", args.generator,
                "--gen_model", args.gen_model,
                "--output", str(pref_path),
                "--n_pairs", str(args.n_pairs),
            ], "Generating feedback data...")
            
            if not success:
                continue
            
            # Convert preferences to KTO format (good/bad labels)
            _convert_prefs_to_kto(str(pref_path), str(feedback_path))
            
            # Train KTO
            success = run_step([
                python, "training/train_kto.py",
                "--data", str(feedback_path),
                "--model", model_path,
                "--save_dir", args.save_dir,
                "--epochs", str(args.train_epochs),
            ], "Training KTO alignment...")
            
            improved_path = Path(args.save_dir) / "brain_kto.pt"
        
        if not success:
            logger.error("  Training failed, rolling back")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, model_path)
            continue
        
        # ═══ Step 4: Evaluate improved model ═══
        eval_after = Path(args.save_dir) / f"eval_after_iter{iteration}.json"
        if improved_path.exists():
            run_step([
                python, "training/evaluate.py",
                "--model", str(improved_path),
                "--perplexity", "--reasoning",
                "--output", str(eval_after),
            ], "Evaluating improved model...")
        
        # ═══ Step 5: Compare and decide ═══
        improved = _compare_evals(str(eval_before), str(eval_after))
        
        if improved:
            logger.info(f"  ✓ Model IMPROVED! Updating best model.")
            if improved_path.exists():
                shutil.copy2(str(improved_path), model_path)
                model_path = str(model_path)
        else:
            logger.info(f"  ✗ No improvement. Rolling back.")
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, model_path)
        
        logger.info(f"  Iteration {iteration} complete.")
    
    logger.info(f"\n{'═' * 50}")
    logger.info(f"  Self-improvement complete! ({args.iterations} iterations)")
    logger.info(f"  Final model: {model_path}")
    logger.info(f"{'═' * 50}")


def _convert_prefs_to_kto(pref_path: str, kto_path: str):
    """Convert preference pairs to KTO format (individual good/bad labels)."""
    with open(pref_path, 'r', encoding='utf-8') as f_in, \
         open(kto_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                # Chosen = good
                f_out.write(json.dumps({
                    "prompt": item['prompt'],
                    "response": item['chosen'],
                    "label": True,
                }, ensure_ascii=False) + '\n')
                # Rejected = bad
                f_out.write(json.dumps({
                    "prompt": item['prompt'],
                    "response": item['rejected'],
                    "label": False,
                }, ensure_ascii=False) + '\n')
            except (json.JSONDecodeError, KeyError):
                pass


def _compare_evals(before_path: str, after_path: str) -> bool:
    """Compare evaluation results. Returns True if improved."""
    try:
        with open(before_path, 'r') as f:
            before = json.load(f)
        with open(after_path, 'r') as f:
            after = json.load(f)
        
        score_before = 0
        score_after = 0
        
        # Lower perplexity = better
        if 'perplexity' in before and 'perplexity' in after:
            ppl_b = before['perplexity'].get('perplexity', 999)
            ppl_a = after['perplexity'].get('perplexity', 999)
            if ppl_a < ppl_b:
                score_after += 1
            else:
                score_before += 1
        
        # Higher reasoning accuracy = better
        if 'reasoning' in before and 'reasoning' in after:
            acc_b = before['reasoning'].get('accuracy', 0)
            acc_a = after['reasoning'].get('accuracy', 0)
            if acc_a >= acc_b:
                score_after += 1
            else:
                score_before += 1
        
        logger.info(f"  Scores: before={score_before}, after={score_after}")
        return score_after > score_before
    
    except Exception as e:
        logger.warning(f"  Could not compare evals: {e}")
        return False


def main():
    p = argparse.ArgumentParser(description="TARS Self-Improvement Loop")
    p.add_argument("--model", type=str, default="models/mamba2/brain_best.pt")
    p.add_argument("--save_dir", type=str, default="models/mamba2")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--method", type=str, default="dpo",
                   choices=["dpo", "kto"])
    p.add_argument("--judge", type=str, default="ollama")
    p.add_argument("--judge_model", type=str, default="qwen2.5:7b")
    p.add_argument("--generator", type=str, default="ollama")
    p.add_argument("--gen_model", type=str, default="qwen2.5:7b")
    p.add_argument("--n_pairs", type=int, default=200, 
                   help="Preference pairs per iteration")
    p.add_argument("--train_epochs", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.1)
    args = p.parse_args()
    
    self_improve(args)


if __name__ == "__main__":
    main()
