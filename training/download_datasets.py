"""
═══════════════════════════════════════════════════════════════
  ТАРС Dataset Downloader — Лучшие датасеты из интернета
═══════════════════════════════════════════════════════════════

Скачивает, фильтрует и сохраняет самые качественные датасеты
для обучения ТАРС Mamba-2 модели.

Использование:
  python training/download_datasets.py                    # всё
  python training/download_datasets.py --only russian     # только русские
  python training/download_datasets.py --only reasoning   # только reasoning
  python training/download_datasets.py --max-samples 50000  # лимит на датасет
  python training/download_datasets.py --data-dir /path/to/data

Категории:
  🇷🇺 Russian — инструкции, диалоги, знания на русском
  🧠 Reasoning — CoT, логика, R1-distill 
  📐 Math — математика, олимпиады
  💻 Code — программирование
  📚 General — энциклопедические знания, инструкции
"""

import os
import sys
import json
import hashlib
import argparse
import re
import time
import subprocess
from pathlib import Path
from typing import List, Optional, Dict

# Fix encoding
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════
#   REGISTRY: Лучшие датасеты (отобраны из 100+ источников)
# ═══════════════════════════════════════════════════════════

DATASETS = [
    # ────── 🇷🇺 RUSSIAN INSTRUCTION / DIALOGUE ──────
    {
        "id": "ru_alpaca",
        "name": "IlyaGusev/ru_turbo_alpaca",
        "category": "russian",
        "description": "50K русских инструкций (ChatGPT). Saiga project, 83-90% quality.",
        "text_fields": ["instruction", "output"],
        "format": "instruction",  # объединяем instruction + output
        "max_samples": 50000,
        "priority": 1,
    },
    {
        "id": "ru_saiga_scored",
        "name": "IlyaGusev/saiga_scored",
        "category": "russian",
        "description": "Scored Russian instruction dataset (Saiga v2). Filtered by quality.",
        "text_fields": ["messages"],
        "format": "messages",  # list of {role, content}
        "max_samples": 50000,
        "priority": 1,
    },
    {
        "id": "ru_turbo_saiga",
        "name": "IlyaGusev/ru_turbo_saiga",
        "category": "russian",
        "description": "Russian multi-turn conversations (Saiga/Baize). ChatGPT-generated.",
        "text_fields": ["messages"],
        "format": "messages",
        "max_samples": 30000,
        "priority": 2,
    },
    {
        "id": "vikhr_instruct",
        "name": "Vikhrmodels/Veles-2.5",
        "category": "russian",
        "description": "Vikhr instruction set. OpenHermes translated to Russian.",
        "text_fields": ["conversations"],
        "format": "conversations",
        "max_samples": 50000,
        "priority": 1,
    },

    # ────── 🧠 REASONING (Chain-of-Thought) ──────
    {
        "id": "openthoughts",
        "name": "open-thoughts/OpenThoughts-114k",
        "category": "reasoning",
        "description": "114K DeepSeek-R1 reasoning traces. STEM, science, coding, puzzles.",
        "text_fields": ["problem", "solution"],
        "format": "instruction",
        "max_samples": 50000,
        "priority": 1,
    },
    {
        "id": "smoltalk",
        "name": "HuggingFaceTB/smoltalk",
        "category": "reasoning",
        "description": "SmolLM3 training data. Reasoning, OpenThoughts3, multilingual.",
        "text_fields": ["messages"],
        "format": "messages",
        "max_samples": 100000,
        "priority": 2,
    },

    # ────── 📐 MATH ──────
    {
        "id": "openmath",
        "name": "nvidia/OpenMathInstruct-2",
        "category": "math",
        "description": "14M math solutions (GSM8K/MATH). Llama-3.1-405B generated.",
        "text_fields": ["problem", "generated_solution"],
        "format": "instruction",
        "max_samples": 100000,
        "priority": 1,
    },
    {
        "id": "numina_math",
        "name": "AI-MO/NuminaMath-CoT",
        "category": "math",
        "description": "859K math problems with CoT. AI Math Olympiad winner.",
        "text_fields": ["problem", "solution"],
        "format": "instruction",
        "max_samples": 50000,
        "priority": 2,
    },

    # ────── 💻 CODE ──────
    {
        "id": "tiny_codes",
        "name": "nampdn-ai/tiny-codes",
        "category": "code",
        "description": "1.6M code snippets. Multi-language, control flow focused.",
        "text_fields": ["prompt", "response"],
        "format": "instruction",
        "max_samples": 50000,
        "priority": 2,
    },

    # ────── 📚 GENERAL KNOWLEDGE ──────
    {
        "id": "infinity_instruct",
        "name": "BAAI/Infinity-Instruct",
        "subset": "7M",
        "category": "general",
        "description": "7.45M gold-standard instructions. Evolutionary techniques.",
        "text_fields": ["conversations"],
        "format": "conversations",
        "max_samples": 100000,
        "priority": 1,
    },
    {
        "id": "cosmopedia",
        "name": "HuggingFaceTB/cosmopedia-100k",
        "category": "general",
        "description": "100K synthetic textbook-quality texts. Wide topic coverage.",
        "text_fields": ["text"],
        "format": "text",
        "max_samples": 50000,
        "priority": 2,
    },
    {
        "id": "wikipedia_ru",
        "name": "wikimedia/wikipedia",
        "subset": "20231101.ru",
        "category": "russian",
        "description": "Russian Wikipedia. General knowledge in Russian.",
        "text_fields": ["text"],
        "format": "text",
        "max_samples": 50000,
        "priority": 2,
    },
]


# ═══════════════════════════════════════════════════════════
#   Text Extraction (supports multiple dataset formats)
# ═══════════════════════════════════════════════════════════

def extract_text(sample: dict, dataset_config: dict) -> Optional[str]:
    """Извлечение текста из sample в зависимости от формата датасета."""
    fmt = dataset_config["format"]
    fields = dataset_config["text_fields"]
    
    try:
        if fmt == "text":
            # Simple text field
            text = sample.get(fields[0], "")
            return text if isinstance(text, str) and len(text) > 30 else None
        
        elif fmt == "instruction":
            # instruction + output → "<input>\n\n<output>"
            parts = []
            for f in fields:
                val = sample.get(f, "")
                if isinstance(val, str) and val.strip():
                    parts.append(val.strip())
            return "\n\n".join(parts) if parts else None
        
        elif fmt == "messages":
            # list of {role, content} → "role: content\n\nrole: content"
            msgs = sample.get(fields[0], [])
            if not isinstance(msgs, list):
                return None
            parts = []
            for msg in msgs:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if content:
                        parts.append(f"{role}: {content}")
            return "\n\n".join(parts) if parts else None
        
        elif fmt == "conversations":
            # list of {from/human, value} → combined text
            convs = sample.get(fields[0], [])
            if not isinstance(convs, list):
                return None
            parts = []
            for turn in convs:
                if isinstance(turn, dict):
                    val = turn.get("value", turn.get("content", ""))
                    role = turn.get("from", turn.get("role", ""))
                    if val:
                        parts.append(f"{role}: {val}")
            return "\n\n".join(parts) if parts else None
    except Exception:
        pass
    return None


# ═══════════════════════════════════════════════════════════
#   Quality Filter (applied to each extracted text)
# ═══════════════════════════════════════════════════════════

def quality_filter(text: str) -> Optional[str]:
    """Фильтр качества текста. Возвращает None если текст мусорный."""
    if not text or len(text) < 50:
        return None
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive URLs
    text = re.sub(r'https?://\S+', '[URL]', text)
    # Collapse whitespace
    text = re.sub(r'[ \t]{4,}', '  ', text)
    text = re.sub(r'\n{5,}', '\n\n\n', text)
    
    # Alpha ratio > 35% (allow some code/math)
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.35:
        return None
    
    # Not too short after cleaning
    if len(text.strip()) < 50:
        return None
    
    # Max length: 10K chars (longer texts are split later by training)
    if len(text) > 10000:
        text = text[:10000]
    
    return text.strip()


# ═══════════════════════════════════════════════════════════
#   Download + Process
# ═══════════════════════════════════════════════════════════

def download_dataset(config: dict, data_dir: Path, max_samples: int = 0) -> int:
    """Скачивает один датасет, фильтрует и сохраняет как .txt."""
    from datasets import load_dataset
    
    ds_id = config["id"]
    ds_name = config["name"]
    subset = config.get("subset", None)
    limit = max_samples or config.get("max_samples", 50000)
    
    out_path = data_dir / f"hf_{ds_id}.txt"
    
    # Skip if already downloaded
    if out_path.exists() and out_path.stat().st_size > 1000:
        mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  ⏭ {ds_id}: already exists ({mb:.1f} MB)")
        return 0
    
    print(f"  📥 {ds_id}: downloading {ds_name}...")
    t0 = time.time()
    
    try:
        # Load with streaming for large datasets
        # ═══ Security: only trust_remote_code for datasets that need it ═══
        kwargs = {}
        if config.get("trust_remote_code", False):
            kwargs["trust_remote_code"] = True
        if subset:
            kwargs["name"] = subset
        
        # Try streaming first (memory-efficient)
        try:
            ds = load_dataset(ds_name, split="train", streaming=True, **kwargs)
            is_streaming = True
        except Exception:
            ds = load_dataset(ds_name, split="train", **kwargs)
            is_streaming = False
        
        # Extract and filter
        texts = []
        seen_hashes = set()
        n_processed = 0
        n_filtered = 0
        
        for sample in ds:
            if len(texts) >= limit:
                break
            n_processed += 1
            
            # Extract text
            raw_text = extract_text(sample, config)
            if not raw_text:
                n_filtered += 1
                continue
            
            # Quality filter
            clean_text = quality_filter(raw_text)
            if not clean_text:
                n_filtered += 1
                continue
            
            # ═══ Fixed: SHA256 on full text (not MD5 on first 200 chars) ═══
            h = hashlib.sha256(clean_text.encode('utf-8', errors='replace')).hexdigest()
            if h in seen_hashes:
                n_filtered += 1
                continue
            seen_hashes.add(h)
            
            texts.append(clean_text)
            
            # Progress
            if n_processed % 10000 == 0:
                print(f"    ... {n_processed:,} processed, {len(texts):,} kept")
        
        if not texts:
            print(f"  ⚠ {ds_id}: no texts extracted")
            return 0
        
        # Save
        os.makedirs(data_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(texts))
        
        mb = out_path.stat().st_size / (1024 * 1024)
        elapsed = time.time() - t0
        print(f"  ✓ {ds_id}: {len(texts):,} texts, {mb:.1f} MB "
              f"(filtered {n_filtered:,}, {elapsed:.0f}s)")
        return len(texts)
    
    except Exception as e:
        print(f"  ✗ {ds_id}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="ТАРС Dataset Downloader")
    parser.add_argument('--data-dir', type=str, default=None,
                       help="Directory to save datasets (default: data/)")
    parser.add_argument('--only', type=str, default=None,
                       choices=['russian', 'reasoning', 'math', 'code', 'general'],
                       help="Download only one category")
    parser.add_argument('--max-samples', type=int, default=0,
                       help="Max samples per dataset (0 = use defaults)")
    parser.add_argument('--list', action='store_true',
                       help="List available datasets without downloading")
    parser.add_argument('--priority', type=int, default=2,
                       help="Max priority to download (1=essential, 2=all)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / "data"
    
    # Filter datasets
    datasets = DATASETS
    if args.only:
        datasets = [d for d in datasets if d["category"] == args.only]
    datasets = [d for d in datasets if d.get("priority", 2) <= args.priority]
    
    if args.list:
        print(f"\n{'='*60}")
        print(f"  Available Datasets ({len(datasets)})")
        print(f"{'='*60}")
        for d in datasets:
            cat_emoji = {"russian": "🇷🇺", "reasoning": "🧠", "math": "📐",
                        "code": "💻", "general": "📚"}.get(d["category"], "?")
            p_mark = "★" if d["priority"] == 1 else "☆"
            print(f"  {p_mark} {cat_emoji} {d['id']}: {d['description']}")
            print(f"      Source: {d['name']} (max {d.get('max_samples', '?')})")
        return
    
    print(f"\n{'='*60}")
    print(f"  ТАРС Dataset Downloader")
    print(f"  Target: {data_dir}")
    print(f"  Datasets: {len(datasets)}")
    print(f"{'='*60}\n")
    
    # Check dependencies
    try:
        from datasets import load_dataset
        print("  ✓ `datasets` library available\n")
    except ImportError:
        print("  ✗ `datasets` library not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"], check=True)
        from datasets import load_dataset
        print("  ✓ `datasets` installed\n")
    
    total_texts = 0
    for config in datasets:
        n = download_dataset(config, data_dir, args.max_samples)
        total_texts += n
    
    # Summary
    print(f"\n{'='*60}")
    total_mb = sum(
        (data_dir / f"hf_{d['id']}.txt").stat().st_size / (1024*1024)
        for d in datasets
        if (data_dir / f"hf_{d['id']}.txt").exists()
    )
    n_files = sum(1 for d in datasets if (data_dir / f"hf_{d['id']}.txt").exists())
    print(f"  DONE: {n_files} datasets, {total_mb:.1f} MB total")
    print(f"  Path: {data_dir}")
    print(f"  Files will be auto-loaded by train_mamba2.py (hf_*.txt)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
