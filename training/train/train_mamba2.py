"""
═══════════════════════════════════════════════════════════════
  Mamba-2 + RWKV-7 Training Script (Deep WuNeng Core)
═══════════════════════════════════════════════════════════════

Обучает TarsMamba2LM (TarsCoreBlock: SSD + WKV + WuNeng Fusion).

Фазы:
  1. Pre-train ВСЁ (TarsCoreBlock + Ω-SSM + MoLE)
  2. Fine-tune WKV + Fusion (заморозить SSD params)
  3. Fine-tune MoLE + MatrixPool
  4. Fine-tune WKV fusion + RAG State Tracking

Использование:
  python training/train_mamba2.py --phase 1 --epochs 3
  python training/train_mamba2.py --phase 4 --epochs 5 --data data/wiki_ru.txt
  python training/train_mamba2.py --phase 2 --pretrained models/mamba2/best.pt
"""

import os
import sys

# Fix Windows cp1252 encoding for Russian output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import time
import json
import copy
import hashlib
import logging
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

logger = logging.getLogger("training.mamba2")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from brain.mamba2.model import TarsMamba2LM


def parse_args():
    p = argparse.ArgumentParser(description="ТАРС Mamba-2 + RWKV-7 Training (Deep WuNeng Core)")
    # ═══ Model params ═══
    p.add_argument('--d_model', type=int, default=2048)   # TARS 1B
    p.add_argument('--n_layers', type=int, default=24)    # 24 rich blocks
    p.add_argument('--vocab_size', type=int, default=4096,
                    help="BPE vocab size (0=byte-level 256)")
    # ═══ Training params ═══
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--accum_steps', type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch × accum)")
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--phase', type=int, default=1, 
                   help="1=pretrain all, 2=WKV+Fusion (freeze SSD), 3=MoLE+Pool, 4=WKV RAG")
    p.add_argument('--curriculum', action='store_true', default=True,
                   help="Enable curriculum learning (short→long sequences)")
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pretrained', type=str, default=None, help="Путь к весам для дообучения")
    p.add_argument('--data', type=str, default=None, help="Путь к текстовому корпусу (.txt)")
    p.add_argument('--device', type=str, default='auto', help="cpu/cuda/auto")
    p.add_argument('--quant', action='store_true', help="Обучать в 1.58-bit режиме (BitNet STE)")
    p.add_argument('--save_dir', type=str, default='models/mamba2')
    p.add_argument('--max_samples', type=int, default=0,
                   help="Максимум обучающих примеров (0 = без ограничений)")
    p.add_argument('--pretrained_emb', type=str, default=None,
                   help="Путь к предобученному эмбеддингу (из MinGRU)")
    p.add_argument('--no_compile', action='store_true',
                   help="Отключить torch.compile")
    # ═══ BitMamba-2 optimizations ═══
    p.add_argument('--bf16', action='store_true',
                   help="Использовать bfloat16 AMP (лучше fp16: шире динамический диапазон)")
    p.add_argument('--grad_ckpt', action='store_true',
                   help="Gradient checkpointing (50-70%% экономия VRAM, чуть медленнее)")
    p.add_argument('--muon', action='store_true',
                   help="Use Muon optimizer (2x faster than AdamW, 50%% less VRAM)")
    p.add_argument('--wsd', action='store_true',
                   help="Warmup-Stable-Decay scheduler (SmolLM2, better for long training)")
    p.add_argument('--mod', action='store_true',
                   help="Mixture of Depths: skip layers for easy tokens (-30%% compute)")
    p.add_argument('--no_wiki', action='store_true',
                   help="Не скачивать Wikipedia (использовать встроенный корпус)")
    p.add_argument('--fp16', action='store_true',
                   help="Использовать float16 AMP + GradScaler")
    p.add_argument('--data_dir', type=str, default=None,
                   help="Папка с данными (hf_*.txt, wiki_ru.txt, tars_personality*.txt)")
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--pin_memory', action='store_true')
    p.add_argument('--compile', action='store_true',
                   help="Force torch.compile")
    # ═══ Data repeat multipliers (configurable, not hardcoded) ═══
    p.add_argument('--identity_repeat', type=int, default=5,
                   help="Кол-во повторов identity корпуса (default: 5)")
    p.add_argument('--personality_repeat', type=int, default=10,
                   help="Кол-во повторов personality корпуса (default: 10)")
    p.add_argument('--mega_repeat', type=int, default=3,
                   help="Кол-во повторов mega personality (default: 3)")
    # ═══ Advanced Optimizations ═══
    p.add_argument('--patience', type=int, default=5,
                   help="Early stopping patience (0=off)")
    p.add_argument('--lora', action='store_true',
                   help="LoRA fine-tuning (parameter-efficient, trains only adapters)")
    p.add_argument('--lora_rank', type=int, default=16,
                   help="LoRA rank (8-32 recommended)")
    p.add_argument('--prune', type=float, default=0.0,
                   help="SparseSSM pruning sparsity (0.3-0.5 recommended, 0=off)")
    p.add_argument('--auto_optimize', action='store_true',
                   help="Auto-detect GPU and apply optimal settings")
    # ═══ T17: DoubtEngine Training Integration ═══
    p.add_argument('--doubt', action='store_true',
                   help="Enable DoubtEngine training alongside Brain")
    p.add_argument('--doubt_lr', type=float, default=1e-3,
                   help="DoubtEngine learning rate (separate from Brain)")
    p.add_argument('--doubt_checkpoint', type=str, default='models/doubt/doubt_engine_best.pt',
                   help="DoubtEngine checkpoint path")
    return p.parse_args()


def load_corpus(data_path=None, download_wiki=True, data_dir=None,
                identity_repeat=5, personality_repeat=10, mega_repeat=3):
    """Загружает обучающий корпус.
    
    Источники (по приоритету):
      1. --data (пользовательский файл)
      2. Встроенный корпус (train_corpus.py: диалоги + тексты)
      3. Wikipedia (из кеша, без auto-download)
      4. TARS memories (data/tars_memories.json)
    
    Args:
        identity_repeat: кол-во повторов identity (default 5)
        personality_repeat: кол-во повторов personality (default 10)  
        mega_repeat: кол-во повторов mega personality (default 3)
    """
    parts = []
    
    # 1. Пользовательский корпус (--data)
    if data_path and os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        parts.append(text)
        print(f"[Data] Custom corpus: {len(text):,} символов ({data_path})")
    
    # 2. Встроенный корпус (диалоги + технические тексты)
    try:
        from training.train_corpus import get_training_text
        built_in = get_training_text()
        parts.append(built_in)
        print(f"[Data] Встроенный корпус: {len(built_in):,} символов")
    except ImportError:
        pass
    
    # 3. Wikipedia (используем локальный файл, НЕ скачиваем)
    _data_root = Path(data_dir) if data_dir else ROOT / "data"
    wiki_path = _data_root / "datasets" / "wiki_ru.txt"
    if wiki_path.exists():
        with open(wiki_path, 'r', encoding='utf-8') as f:
            wiki_text = f.read()
        if len(wiki_text) > 1000:
            parts.append(wiki_text)
            wiki_mb = len(wiki_text.encode('utf-8')) / (1024 * 1024)
            print(f"[Data] Wikipedia: {len(wiki_text):,} символов ({wiki_mb:.1f} MB, кеш)")
    
    # 4. HuggingFace датасеты (data/datasets/hf_*.txt)
    hf_dir = _data_root / "datasets"
    if hf_dir.exists():
        import re as _re
        hf_files = sorted(hf_dir.glob("hf_*.txt"))
        for hf_file in hf_files:
            try:
                with open(hf_file, 'r', encoding='utf-8') as f:
                    hf_text = f.read()
                if len(hf_text) > 1000:
                    # ═══ Quality clean at load time ═══
                    hf_text = _re.sub(r'<[^>]+>', '', hf_text)          # HTML
                    hf_text = _re.sub(r'https?://\S+', '', hf_text)     # URLs
                    hf_text = _re.sub(r'[ \t]{4,}', '  ', hf_text)     # spaces
                    hf_text = _re.sub(r'\n{5,}', '\n\n\n', hf_text)    # newlines
                    hf_text = _re.sub(r'([!?.])\1{4,}', r'\1\1\1', hf_text)  # punct
                    # Remove very short paragraphs
                    paras = hf_text.split('\n\n')
                    paras = [p for p in paras if len(p.strip()) >= 30]
                    hf_text = '\n\n'.join(paras)
                    
                    parts.append(hf_text)
                    hf_mb = len(hf_text.encode('utf-8')) / (1024 * 1024)
                    print(f"[Data] HF {hf_file.stem}: {len(hf_text):,} символов ({hf_mb:.1f} MB)")
            except Exception as e:
                logger.debug(f"Skipped HF file {hf_file.name}: {e}")
    
    # 4b. Sharded datasets (data/shards_*/shard_*.txt) — для 50+ GB
    if hf_dir.exists():
        shard_dirs = sorted(hf_dir.glob("shards_*"))
        for shard_dir in shard_dirs:
            if not shard_dir.is_dir():
                continue
            shard_files = sorted(shard_dir.glob("shard_*.txt"))
            shard_total = 0
            for sf in shard_files:
                try:
                    with open(sf, 'r', encoding='utf-8') as f:
                        shard_text = f.read()
                    if len(shard_text) > 1000:
                        parts.append(shard_text)
                        shard_total += len(shard_text)
                except Exception as e:
                    logger.debug(f"Skipped shard {sf.name}: {e}")
            if shard_total > 0:
                shard_gb = shard_total / (1024**3)
                print(f"[Data] Shards {shard_dir.name}: {len(shard_files)} шардов, {shard_gb:.2f} GB")
    
    # 5. TARS memories
    memories_path = _data_root / "memory" / "tars_memories.json"
    if memories_path.exists():
        try:
            with open(memories_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts = [e.get("text", e.get("memory", "")) for e in data if isinstance(e, dict)]
            parts.append("\n".join(texts))
            print(f"[Data] Memories: {len(texts)} записей")
        except Exception as e:
            logger.debug(f"Memories load error: {e}")
    
    corpus = "\n\n".join(parts)
    
    # 6. TARS Identity (self-description — модель учит свою структуру)
    identity_path = _data_root / "identity" / "tars_identity.txt"
    if identity_path.exists():
        with open(identity_path, 'r', encoding='utf-8') as f:
            identity_text = f.read()
        if len(identity_text) > 100:
            identity_repeated = ("\n\n" + identity_text) * identity_repeat
            corpus = identity_repeated + "\n\n" + corpus
            logger.info(f"[Data] ТАРС Identity: {len(identity_text):,} символов (×{identity_repeat})")
    
    # 7. TARS Personality (стиль общения — юмор, сарказм, прямолинейность)
    personality_path = _data_root / "identity" / "tars_personality.txt"
    if personality_path.exists():
        with open(personality_path, 'r', encoding='utf-8') as f:
            personality_text = f.read()
        if len(personality_text) > 100:
            personality_repeated = ("\n\n" + personality_text) * personality_repeat
            corpus = personality_repeated + "\n\n" + corpus
            logger.info(f"[Data] ТАРС Personality: {len(personality_text):,} символов (×{personality_repeat})")
    
    # 8. TARS Mega Personality (6000+ реплик — основной корпус личности)
    mega_path = _data_root / "identity" / "tars_personality_mega.txt"
    if mega_path.exists():
        with open(mega_path, 'r', encoding='utf-8') as f:
            mega_text = f.read()
        if len(mega_text) > 1000:
            mega_repeated = ("\n\n" + mega_text) * mega_repeat
            corpus = mega_repeated + "\n\n" + corpus
            mega_lines = mega_text.count('\n')
            logger.info(f"[Data] ТАРС Mega Personality: {mega_lines:,} строк ({len(mega_text)//1024} KB, ×{mega_repeat})")
    
    
    # Повторяем маленький корпус
    corpus_bytes = len(corpus.encode('utf-8'))
    if corpus_bytes < 200_000:
        repeat = max(1, 200_000 // corpus_bytes)
        corpus = ("\n\n" + corpus) * repeat
        print(f"[Data] Корпус повторён {repeat}x")
        corpus_bytes = len(corpus.encode('utf-8'))
    
    print(f"[Data] Итого: {len(corpus):,} символов ({corpus_bytes / 1024:.0f} KB)")
    
    # ═══ TRAINING-OPT: Quality filter (dedup + min_len + alpha) ═══
    corpus = filter_corpus_quality(corpus)
    
    return corpus


def filter_corpus_quality(corpus: str) -> str:
    """Фильтрация качества корпуса: дедупликация, мин. длина, alpha ratio.
    
    Исследование (RegMix 2025): качество данных > количество. Фильтрация
    даёт 5-10x ускорение обучения при тех же данных.
    """
    paragraphs = corpus.split('\n\n')
    seen_hashes = set()
    filtered = []
    stats = {'total': len(paragraphs), 'short': 0, 'dup': 0, 'lowforth': 0, 'repeat': 0}
    
    for p in paragraphs:
        p = p.strip()
        # 1. Минимальная длина
        if len(p) < 50:
            stats['short'] += 1
            continue
        # 2. Дедупликация (SHA256 — нет коллизий в отличие от MD5)
        h = hashlib.sha256(p.encode('utf-8', errors='replace')).hexdigest()
        if h in seen_hashes:
            stats['dup'] += 1
            continue
        seen_hashes.add(h)
        # 3. Alpha ratio (>40% alphabetic — отсеивает мусор/код/бинарные данные)
        alpha_count = sum(1 for c in p if c.isalpha())
        if alpha_count / max(len(p), 1) < 0.4:
            stats['lowforth'] += 1
            continue
        # 4. Repetition filter (один символ > 30% текста)
        if len(p) > 10:
            max_char_freq = max(p.count(c) for c in set(p[:100]))  # sample first 100
            if max_char_freq / min(len(p), 100) > 0.3 and max_char_freq > 15:
                stats['repeat'] += 1
                continue
        filtered.append(p)
    
    result = '\n\n'.join(filtered)
    removed = stats['total'] - len(filtered)
    if removed > 0:
        print(f"[Quality] Filtered: -{removed} paragraphs "
              f"(short={stats['short']}, dup={stats['dup']}, "
              f"low_alpha={stats['lowforth']}, repetitive={stats['repeat']})")
        print(f"[Quality] Kept: {len(filtered)}/{stats['total']} paragraphs "
              f"({len(result)//1024} KB)")
    return result


def prepare_data(text: str, seq_len: int, vocab_size: int = 32000, max_samples: int = 0, tokenizer=None):
    """
    Подготовка данных для next-token prediction.
    Использует BPE токенизатор если передан, иначе byte-level CP1251.
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(text)
    else:
        tokens = list(text.encode('cp1251', errors='replace'))
    
    # Если max_samples задан, ограничиваем длину текста
    if max_samples > 0:
        max_toks = max_samples * seq_len
        if len(tokens) > max_toks:
            tokens = tokens[:max_toks]
    
    inputs = []
    targets = []
    stride = seq_len // 2
    
    for i in range(0, len(tokens) - seq_len - 1, stride):
        inp = tokens[i:i + seq_len]
        tgt = tokens[i + 1:i + seq_len + 1]
        if len(inp) == seq_len and len(tgt) == seq_len:
            inputs.append(inp)
            targets.append(tgt)
        if max_samples > 0 and len(inputs) >= max_samples:
            break
    
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long)
    )

def load_hf_weights(model, pretrained_dir):
    """Загрузка базовых слоёв из HuggingFace safetensors/bin"""
    pretrained_path = Path(pretrained_dir)
    found_weights = list(pretrained_path.glob("*.safetensors")) or list(pretrained_path.glob("*.bin"))
    if not found_weights:
        print(f"[Warning] Не найдены веса в {pretrained_dir}")
        return

    print(f"[Model] Загрузка базовых весов Mamba-2 из: {pretrained_dir}")
    try:
        from safetensors.torch import load_file
        state_dict = {}
        for f in found_weights:
            if f.suffix == '.safetensors':
                state_dict.update(load_file(f))
            else:
                state_dict.update(torch.load(f, map_location='cpu', weights_only=True))

        # Очистка ключей HF (примерная адаптация)
        mapped_dict = {}
        for k, v in state_dict.items():
            if 'embeddings' in k or 'backbone.embedding' in k:
                mapped_dict['embedding.weight'] = v
            elif 'layers' in k:
                # Маппинг слоев: model.layers.0 -> blocks.0.core
                new_k = k.replace('backbone.layers.', 'blocks.').replace('mixer.', 'core.')
                mapped_dict[new_k] = v
        
        info = model.load_state_dict(mapped_dict, strict=False)
        print(f"[Model] Переносы успешны. Пропущено слоёв TARS (IDME/MoLE/RWKV): {len(info.missing_keys)}")
    except Exception as e:
        print(f"[Error] Ошибка загрузки pretrained: {e}")

def train(args):
    """Основной цикл обучения."""
    
    # Устройство
    if args.device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = args.device
    device = torch.device(device_str)
    
    if device_str == 'cuda':
        print(f"[GPU] {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
    else:
        print("[CPU] GPU не используется")
    
    # Данные (корпус загружается 1 раз, токенизация — по curriculum)
    corpus = load_corpus(data_path=args.data, download_wiki=not args.no_wiki,
                         data_dir=args.data_dir,
                         identity_repeat=args.identity_repeat,
                         personality_repeat=args.personality_repeat,
                         mega_repeat=args.mega_repeat)
    print(f"[Data] Корпус загружен ({len(corpus):,} символов)")
    
    # ═══ Tokenizer: обучить BPE или использовать byte-level ═══
    from brain.tokenizer import TarsTokenizer, reset_tokenizer
    reset_tokenizer()
    if args.vocab_size > 0:
        print(f"\n[Tokenizer] Обучение SentencePiece BPE (vocab={args.vocab_size})...")
        try:
            tokenizer = TarsTokenizer.train(
                corpus_text=corpus,
                vocab_size=args.vocab_size,
            )
            print(f"[Tokenizer] ✅ BPE: vocab={tokenizer.vocab_size}")
        except Exception as e:
            print(f"[Tokenizer] ⚠ BPE не удалось ({e}), fallback byte-level")
            tokenizer = TarsTokenizer(mode="byte")
    else:
        print(f"[Tokenizer] Byte-level (vocab=256)")
        tokenizer = TarsTokenizer(mode="byte")
    
    actual_vocab = tokenizer.vocab_size
    
    # ═════════════════════════════════════════════════════════
    # Модель: 3-шаговый пайплайн
    #   Step A: Создание модели в FP16 (нормальная инициализация)
    #   Step B: Квантизация в 1.58-bit (если --quant)
    #   Step C: Обучение на данных (Wiki + HF + встроенный корпус)
    # ═════════════════════════════════════════════════════════
    
    # ── Step A: Инициализация в FP16 ──
    print(f"\n[Step A] Создание модели в FP16 (vocab={actual_vocab})...")
    model = TarsMamba2LM(
        d_model=args.d_model,
        n_layers=args.n_layers,
        vocab_size=actual_vocab,
        omega_dim=32,
        pool_size=48,
        n_experts=8,
        quant_mode="fp16",  # Всегда начинаем в FP16
    )
    
    # Load Pretrained or Resume
    save_suffix = "_158bit" if args.quant else ""
    save_path = Path(args.save_dir) / f"mamba2_omega{save_suffix}.pt"
    if args.pretrained:
        load_hf_weights(model, args.pretrained)
        args.phase = 2

    if args.resume and save_path.exists():
        print(f"[Model] Loading checkpoint: {save_path}")
        checkpoint = torch.load(str(save_path), map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif args.resume and args.quant:
        # Quant mode but no 158bit checkpoint → load FP16 as fallback
        fp16_path = Path(args.save_dir) / "mamba2_omega.pt"
        if fp16_path.exists():
            print(f"[Model] Quant: loading FP16 base → {fp16_path}")
            checkpoint = torch.load(str(fp16_path), map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print(f"[!] No checkpoint found — training from scratch")
    
    # ── Gradient Checkpointing (BitMamba-2 style remat) ──
    if args.grad_ckpt:
        model.use_checkpointing = True
        print("  ⚡ Gradient checkpointing enabled (50-70% VRAM savings)")
    
    model.to(device)
    
    # ── Transfer pre-trained embedding from MinGRU ──
    if args.pretrained_emb and os.path.exists(args.pretrained_emb):
        print(f"\n[🔗 Transfer] Загрузка предобученного эмбеддинга из MinGRU...")
        try:
            mingru_emb = torch.load(args.pretrained_emb, map_location=device, weights_only=True)
            if mingru_emb.shape[0] == actual_vocab:
                # MinGRU dim might differ from Mamba-2 d_model
                if mingru_emb.shape[1] == args.d_model:
                    # Perfect match — direct copy
                    model.embedding.weight.data.copy_(mingru_emb)
                    print(f"  ✔ Прямой перенос ({mingru_emb.shape})")
                else:
                    # Dimension mismatch — use projection (truncate or pad)
                    min_dim = min(mingru_emb.shape[1], args.d_model)
                    model.embedding.weight.data[:, :min_dim] = mingru_emb[:, :min_dim]
                    print(f"  ✔ Частичный перенос ({mingru_emb.shape} → {model.embedding.weight.shape})")
        except Exception as e:
            print(f"  ⚠ Embedding transfer failed: {e}")
    
    params = model.count_parameters()
    total = params["total"]
    print(f"[Model] TarsMamba2LM: {total:,} params ({total/1e6:.1f}M), mode=FP16")
    
    # ═══ T17: DoubtEngine (optional, separate from Brain — TZ Section 2.2a) ═══
    # CRITICAL: DoubtEngine НЕ обучается на выходах Brain/MinGRU (per TZ).
    # Обучается на независимом корпусе, использует отдельный optimizer.
    doubt_engine = None
    doubt_optimizer = None
    if getattr(args, 'doubt', False):
        try:
            from brain.doubt_engine import load_doubt_engine
            doubt_ckpt = str(Path(getattr(args, 'doubt_checkpoint', 'models/doubt/doubt_engine_best.pt')))
            doubt_engine = load_doubt_engine(
                d_model=args.d_model,
                checkpoint_path=doubt_ckpt,
                device=str(device),
            )
            if doubt_engine is not None:
                doubt_engine.train()  # switch to training mode
                doubt_lr = getattr(args, 'doubt_lr', 1e-3)
                doubt_optimizer = torch.optim.AdamW(
                    doubt_engine.parameters(), lr=doubt_lr, weight_decay=0.01
                )
                d_params = sum(p.numel() for p in doubt_engine.parameters())
                print(f"[DoubtEngine] {d_params:,} params ({d_params/1e3:.1f}K), LR={doubt_lr}")
        except ImportError:
            print("[DoubtEngine] brain/doubt_engine.py not found — skipping")
            doubt_engine = None
    
    # ── Step B: Квантизация в 1.58-bit (если --quant) ──
    quant_mode = "fp16"
    if args.quant:
        print(f"\n[Step B] Квантизация FP16 -> 1.58-bit...")
        try:
            from brain.mamba2.bitnet import convert_model_to_158bit, model_stats
            convert_model_to_158bit(model)
            stats = model_stats(model)
            quant_mode = "158bit"
            print(f"  Конвертировано: {stats['total_layers']} слоёв UniversalLinear")
            print(f"  Разреженность: {stats['avg_sparsity']:.1%}")
            print(f"  Обучение с STE (Straight-Through Estimator)")
        except Exception as e:
            print(f"  [!] Не удалось квантовать: {e}")
            print(f"  Обучение продолжится в FP16")
    else:
        print(f"\n[Step B] Квантизация пропущена (FP16 режим)")
    
    print(f"\n[Step C] Обучение на данных ({quant_mode} режим)...")
    print(f"  Данные: Wikipedia + HuggingFace + встроенный корпус")
    for k, v in params.items():
        if k != "total" and isinstance(v, int) and v > 0:
            print(f"  {k}: {v:,}")
        elif k == "block_detail (×1)" and isinstance(v, dict):
            for bk, bv in v.items():
                print(f"    {bk}: {bv:,}")
    
    # ═══ Proper weight decay: исключить bias, norm, gamma из decay ═══
    def split_params_wd(params_iter, wd=0.01):
        decay, no_decay = [], []
        for name, p in params_iter:
            if not p.requires_grad:
                continue
            if p.dim() < 2 or 'norm' in name or 'gamma' in name or 'bias' in name:
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {'params': decay, 'weight_decay': wd},
            {'params': no_decay, 'weight_decay': 0.0},
        ]
    
    # ═══ Фазы обучения ═══
    # PersonalityAdapter ЗАМОРОЖЕН в Phase 1-4 (не портит стиль знаниями)
    for p in model.personality.parameters():
        p.requires_grad = False
    
    if args.phase == 1:
        # Phase 1: Pre-train ALL (TarsCoreBlock + Ω-SSM + MoLE)
        print("[Phase 1] Full pre-training: ALL components")
        if getattr(args, 'muon', False):
            # ═══ TRAINING-OPT: Hybrid MuonW (Moonlight 2025) ═══
            # Muon for matrix params (Linear/Conv weights) — 2x faster
            # AdamW for non-matrix params (embeddings, biases, norms)
            from training.muon import Muon
            matrix_params = []
            other_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if p.ndim >= 2:  # matrices: Linear.weight, Conv1d.weight
                    matrix_params.append(p)
                else:  # scalars/vectors: bias, LayerNorm, embeddings
                    other_params.append(p)
            optimizer = Muon(
                [{'params': matrix_params, 'lr': 0.02}],
                lr=0.02, weight_decay=0.01,
            )
            # AdamW for non-matrix params as separate optimizer
            optimizer_aux = torch.optim.AdamW(
                other_params, lr=args.lr, weight_decay=0.1
            ) if other_params else None
            print(f"  ⚡ Hybrid MuonW: Muon({len(matrix_params)} matrices) + "
                  f"AdamW({len(other_params)} vectors)")
        else:
            optimizer = torch.optim.AdamW(split_params_wd(model.named_parameters()), lr=args.lr)
            optimizer_aux = None
    elif args.phase == 2:
        # Phase 2: Freeze SSD-specific params. Train: WKV + Fusion + Ω-SSM + MoLE + NoveltyGate
        print("[Phase 2] Fine-tune: WKV + Fusion + Ω-SSM + MoLE + NoveltyGate (SSD params frozen)")
        for block in model.blocks:
            core = block.core
            # Freeze SSD-specific parameters
            core.A_log.requires_grad = False
            core.D.requires_grad = False
            core.dt_bias.requires_grad = False
            for p in core.conv1d.parameters():
                p.requires_grad = False
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for _, p in trainable):,} params")
        optimizer = torch.optim.AdamW(split_params_wd(trainable), lr=args.lr * 0.1)
    elif args.phase == 3:
        # Phase 3: MoLE + MatrixPool + WaveMerge + WaveGate + NoveltyGate
        print("[Phase 3] Fine-tune: MoLE + MatrixPool + WaveMerge + WaveGate + NoveltyGate")
        for p in model.parameters():
            p.requires_grad = False
        for block in model.blocks:
            for p in block.mole.parameters():
                p.requires_grad = True
            # NoveltyGate — обучаемый порог (Tars.txt §6.1)
            for p in block.novelty_gate.parameters():
                p.requires_grad = True
        for p in model.matrix_pool.parameters():
            p.requires_grad = True
        # WaveConsolidation — обучаемые (Tars.txt §1.4)
        for wc in model.wave_consolidations:
            for p in wc.parameters():
                p.requires_grad = True
        # Model-level NoveltyGate (IDME branch-and-bound в think())
        for p in model.novelty_gate.parameters():
            p.requires_grad = True
        # norm_f — должен адаптироваться при смене внутренних представлений
        for p in model.norm_f.parameters():
            p.requires_grad = True
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for _, p in trainable):,} params")
        optimizer = torch.optim.AdamW(split_params_wd(trainable), lr=args.lr * 0.01)
        optimizer_aux = None
    elif args.phase == 4:
        # Phase 4: WKV fusion + RAG + Dynamic Memory Injection + NoveltyGate + Ω-SSM
        print("[Phase 4] Fine-tune: WKV + RAG + Dynamic Memory + NoveltyGate + Ω-SSM")
        for p in model.parameters():
            p.requires_grad = False
        for block in model.blocks:
            core = block.core
            # WKV-related: fusion gate, wkv_up, time_mix
            for p in core.fusion_gate.parameters():
                p.requires_grad = True
            for p in core.wkv_up.parameters():
                p.requires_grad = True
            core.time_mix.requires_grad = True
            # RAG projections
            for p in block.rag_query.parameters():
                p.requires_grad = True
            for p in block.rag_out.parameters():
                p.requires_grad = True
            # Dynamic Memory Injection (Fused MemoryInjector)
            for p in block.mem_injector.parameters():
                p.requires_grad = True
            # NoveltyGate
            for p in block.novelty_gate.parameters():
                p.requires_grad = True
            # Ω-SSM (Cayley transform parameters)
            for p in block.omega.parameters():
                p.requires_grad = True
        # ═══ Model-level spine projections (Tars.txt §2.4) ═══
        # Без этого to_memory_space/from_memory_space не учатся в Phase 4
        for p in model.to_memory_space.parameters():
            p.requires_grad = True
        for p in model.from_memory_space.parameters():
            p.requires_grad = True
        # WaveConsolidation — суммирующие матрицы между волнами
        for wc in model.wave_consolidations:
            for p in wc.parameters():
                p.requires_grad = True
        # Model-level NoveltyGate (IDME) + norm_f
        for p in model.novelty_gate.parameters():
            p.requires_grad = True
        for p in model.norm_f.parameters():
            p.requires_grad = True
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for _, p in trainable):,} params")
        optimizer = torch.optim.AdamW(split_params_wd(trainable), lr=args.lr * 0.05)
        optimizer_aux = None
    else:
        print(f"[!] Unknown phase {args.phase}, defaulting to Phase 1 (full pre-training)")
        if getattr(args, 'muon', False):
            from training.train_utils import Muon
            optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01)
        else:
            optimizer = torch.optim.AdamW(split_params_wd(model.named_parameters()), lr=args.lr)
        optimizer_aux = None
    
    # Scheduler (estimate total steps from corpus size)
    estimated_samples = max(100, len(tokenizer.encode(corpus[:100000])) * (len(corpus) // max(100000, 1) + 1) // max(args.seq_len // 2, 1))
    total_steps = (estimated_samples // args.batch + 1) * args.epochs
    warmup = total_steps // 10
    
    min_lr_ratio = 0.1  # LR не падает ниже 10% от пика
    
    if getattr(args, 'wsd', False):
        # ═══ TRAINING-OPT: WSD with 80% stable + cosine decay (SmolLM2 2025) ═══
        stable_end = int(total_steps * 0.80)  # 80% stable (was 70%)
        def lr_fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            elif step < stable_end:
                return 1.0  # stable phase — full LR
            else:
                # cosine decay (was linear — cosine gives ~5% better loss)
                progress = (step - stable_end) / max(total_steps - stable_end, 1)
                cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
                return max(min_lr_ratio, cosine)
        print("  ⚡ Scheduler: WSD (10% warmup → 80% stable → cosine decay)")
    else:
        # Standard cosine annealing with min LR floor
        def lr_fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            return max(min_lr_ratio, cosine)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    
    # ═══ Mixture of Depths (-30% compute) ═══
    if getattr(args, 'mod', False):
        from brain.mamba2.mixture_of_depths import add_mod_to_model
        model = add_mod_to_model(model, capacity_factor=0.5)
    
    # torch.compile handled below (line ~531)
    
    # ═══ AMP (BitMamba-2: bfloat16 по умолчанию на TPU/GPU) ═══
    use_amp = device.type == 'cuda'
    if use_amp and args.bf16:
        amp_dtype = torch.bfloat16
        scaler = None  # bfloat16 не нуждается в GradScaler
        print("  ⚡ AMP: bfloat16 (no scaler needed, wider dynamic range)")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler(device.type)
        print("  ⚡ AMP: float16 + GradScaler")
    else:
        amp_dtype = torch.float32
        scaler = None
    
    best_loss = float('inf')
    
    print(f"\n{'═'*60}")
    print(f"  ТАРС Mamba-2 Training — Phase {args.phase}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print(f"{'═'*60}\n")
    
    # ═══ torch.compile / auto-optimize ═══
    compiled_model = None
    if getattr(args, 'auto_optimize', False):
        # Use optimizations module for hardware-adaptive config
        try:
            from brain.mamba2.optimizations import optimize_for_training, get_amp_config
            opt_result = optimize_for_training(model)
            print(f"  ⚡ Auto-optimized for: {opt_result.get('gpu', 'unknown')}")
            print(f"    Compile: {opt_result.get('compile_mode', None)}")
            print(f"    Chunk sizes: SSD={opt_result.get('chunk_size_ssd', '?')}, WKV={opt_result.get('chunk_size_wkv', '?')}")
            if opt_result.get('grad_checkpointing', False):
                print("    Gradient checkpointing: enabled")
            # Override AMP settings from profile
            amp_cfg = get_amp_config()
            if amp_cfg['enabled']:
                amp_dtype = amp_cfg['dtype']
                if amp_cfg['scaler_needed']:
                    scaler = torch.amp.GradScaler(device.type)
                else:
                    scaler = None
                print(f"    AMP dtype: {amp_dtype}")
        except ImportError:
            print("  ⚠ optimizations.py not found, using defaults")
    elif hasattr(torch, 'compile') and not args.no_compile and device.type == 'cuda':
        # Selective compile via model method
        try:
            model.optimize_model(compile_mode="default", gpu_name="auto")
            print("  ⚡ Selective torch.compile enabled (TarsCoreBlock + WaveConsolidation)")
        except Exception as e:
            print(f"  ⚠ optimize_model failed: {e}")
            # Fallback to whole-model compile
            try:
                compiled_model = torch.compile(model, mode="default")
                print("  ⚡ torch.compile fallback (whole model, default mode)")
            except Exception as e2:
                print(f"  ⚠ torch.compile fallback also failed: {e2}")
    forward_model = compiled_model if compiled_model is not None else model
    
    # ═══ LoRA Setup (параметр-эффективная дообучка) ═══
    if getattr(args, 'lora', False):
        try:
            from training.lora import apply_lora
            lora_rank = getattr(args, 'lora_rank', 16)
            lora_stats = apply_lora(
                model,
                rank=lora_rank,
                alpha=lora_rank * 2,
                target_modules=['in_proj', 'out_proj', 'w_gate', 'w_value', 'w_out', 'embed', 'head'],
            )
            print(f"  ⚡ LoRA enabled: rank={lora_rank}, "
                  f"trainable={lora_stats['trainable_params']:,} "
                  f"({lora_stats['trainable_pct']:.2f}%)")
            # Re-create optimizer with only trainable params
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
            print(f"  ⚡ Optimizer re-created for {len(trainable_params)} LoRA param groups")
        except ImportError as e:
            print(f"  ⚠ LoRA import failed: {e}")
    
    # ═══ Curriculum Learning (skip for 1-epoch) ═══
    curriculum_schedule = None
    if args.curriculum and args.epochs >= 3:
        # Постепенно увеличиваем seq_len: 64 → 128 → 256 → target
        max_sl = args.seq_len
        steps = [max(64, max_sl // 4), max(128, max_sl // 2), max_sl]
        epoch_per_step = max(1, args.epochs // len(steps))
        curriculum_schedule = []
        for sl in steps:
            curriculum_schedule.extend([sl] * epoch_per_step)
        while len(curriculum_schedule) < args.epochs:
            curriculum_schedule.append(max_sl)
        print(f"  📚 Curriculum: {' → '.join(str(s) for s in dict.fromkeys(curriculum_schedule))}")
    
    # ═══ TRAINING-OPT: Adaptive gradient accumulation (target=256) ═══
    if args.accum_steps > 1:
        accum_steps = args.accum_steps  # user override
    else:
        target_effective_batch = 256  # sweet spot for SSM (Mamba-2 paper)
        accum_steps = max(1, target_effective_batch // args.batch)
    print(f"  Effective batch: {args.batch} × {accum_steps} = {args.batch * accum_steps}")
    
    # ═══ TRAINING-OPT: EMA model (Polyak averaging, +2-5% eval stability) ═══
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    ema_decay = 0.999
    for p in ema_model.parameters():
        p.requires_grad = False
    print(f"  ⚡ EMA model initialized (decay={ema_decay})")
    
    best_loss = float('inf')
    patience_counter = 0
    
    # ═══ T18: Total steps for adaptive aux loss decay ═══
    steps_per_epoch = max(1, len(c_train_in) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        n_batches = 0
        tokens_processed = 0
        
        # Curriculum: пересоздать данные с новым seq_len
        cur_seq_len = curriculum_schedule[epoch] if curriculum_schedule else args.seq_len
        if epoch == 0 or (curriculum_schedule and cur_seq_len != curriculum_schedule[max(0, epoch-1)]):
            c_inputs, c_targets = prepare_data(corpus, cur_seq_len, actual_vocab, max_samples=args.max_samples, tokenizer=tokenizer)
            n_test_c = max(2, len(c_inputs) // 10)
            c_train_in, c_test_in = c_inputs[:-n_test_c], c_inputs[-n_test_c:]
            c_train_tgt, c_test_tgt = c_targets[:-n_test_c], c_targets[-n_test_c:]
            print(f"  seq_len={cur_seq_len}, samples={len(c_train_in)}")
        
        perm = torch.randperm(len(c_train_in))
        train_in_s = c_train_in[perm]
        train_tgt_s = c_train_tgt[perm]
        
        optimizer.zero_grad(set_to_none=True)
        
        for i in range(0, len(train_in_s), args.batch):
            batch_in = train_in_s[i:i+args.batch].to(device, non_blocking=True)
            batch_tgt = train_tgt_s[i:i+args.batch].to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast(device.type, dtype=amp_dtype):
                    logits = forward_model(batch_in)
                    lm_loss = F.cross_entropy(
                        logits.view(-1, actual_vocab),
                        batch_tgt.view(-1),
                        label_smoothing=args.label_smoothing,
                    )
                    # ═══ T18: MoLE adaptive aux loss balancing ═══
                    _aux = model.mole_aux_loss
                    if not isinstance(_aux, torch.Tensor):
                        _aux = torch.tensor(0.0, device=lm_loss.device)
                    # Adaptive coefficient: linear decay 0.1 → 0.01 over training
                    global_step = (epoch * (len(train_in_s) // args.batch) + n_batches)
                    aux_coeff = max(0.01, 0.1 * (1.0 - global_step / max(total_steps, 1)))
                    _aux = _aux * aux_coeff
                    
                    # ═══ TRAINING-OPT: Multi-objective loss ═══
                    extra_loss = torch.tensor(0.0, device=lm_loss.device)
                    # Predictive coding error (from TarsBlock)
                    try:
                        pred_errors = [b.last_stats.get('pred_error', 0) for b in model.blocks
                                       if isinstance(b.last_stats.get('pred_error'), torch.Tensor)]
                        if pred_errors:
                            extra_loss = extra_loss + 0.001 * torch.stack(pred_errors).mean()
                    except Exception:
                        pass
                    # Novelty regularization: target novelty ≈ 0.5
                    try:
                        novelties = [b.last_stats.get('novelty', 0.5) for b in model.blocks
                                     if isinstance(b.last_stats.get('novelty'), torch.Tensor)]
                        if novelties:
                            novelty_mean = torch.stack(novelties).mean()
                            extra_loss = extra_loss + 0.001 * (novelty_mean - 0.5).pow(2)
                    except Exception:
                        pass
                    
                    loss = (lm_loss + _aux + extra_loss) / accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()  # bfloat16: no scaler needed
            else:
                logits = forward_model(batch_in)
                lm_loss = F.cross_entropy(
                    logits.view(-1, actual_vocab),
                    batch_tgt.view(-1),
                    label_smoothing=args.label_smoothing,
                )
                # ═══ T18: MoLE adaptive aux loss (no-AMP path) ═══
                _aux = model.mole_aux_loss
                if not isinstance(_aux, torch.Tensor):
                    _aux = torch.tensor(0.0, device=lm_loss.device)
                global_step_noamp = (epoch * (len(train_in_s) // args.batch) + n_batches)
                aux_coeff_noamp = max(0.01, 0.1 * (1.0 - global_step_noamp / max(total_steps, 1)))
                _aux = _aux * aux_coeff_noamp
                loss = (lm_loss + _aux) / accum_steps
                loss.backward()
            
            total_loss += loss.item() * accum_steps
            n_batches += 1
            tokens_processed += batch_in.numel()
            
            # Track train accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                train_correct += (preds.view(-1) == batch_tgt.view(-1)).sum().item()
                train_total += batch_tgt.numel()
            
            # Gradient accumulation step
            if n_batches % accum_steps == 0 or i + args.batch >= len(train_in_s):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                # ═══ T18: Gradient norm monitoring for MoLE aux loss ═══
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # ═══ TRAINING-OPT: Step auxiliary optimizer (Hybrid MuonW) ═══
                if optimizer_aux is not None:
                    optimizer_aux.step()
                    optimizer_aux.zero_grad(set_to_none=True)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                # ═══ T17: DoubtEngine training step (separate optimizer) ═══
                if doubt_engine is not None and doubt_optimizer is not None:
                    try:
                        doubt_engine.train()
                        with torch.no_grad():
                            emb = model.embedding(batch_in)
                        # Guard: need at least 2 tokens to split into query/response
                        if emb.shape[1] >= 2:
                            with torch.no_grad():
                                h_q = emb[:, :emb.shape[1]//2].mean(dim=1)  # first half
                                h_r = emb[:, emb.shape[1]//2:].mean(dim=1)  # second half
                            # Positive pairs: genuine (query, response)
                            d_out_pos = doubt_engine(h_q.detach(), h_r.detach())
                            # Negative pairs: shuffled response
                            if h_r.shape[0] > 1:
                                perm_neg = torch.randperm(h_r.shape[0], device=h_r.device)
                            else:
                                perm_neg = torch.zeros(1, dtype=torch.long, device=h_r.device)
                            d_out_neg = doubt_engine(h_q.detach(), h_r[perm_neg].detach())
                            # BCE loss
                            d_loss = (F.binary_cross_entropy(d_out_pos['coherence'], torch.ones_like(d_out_pos['coherence'])) +
                                      F.binary_cross_entropy(d_out_neg['coherence'], torch.zeros_like(d_out_neg['coherence']))) / 2
                            doubt_optimizer.zero_grad(set_to_none=True)
                            d_loss.backward()
                            torch.nn.utils.clip_grad_norm_(doubt_engine.parameters(), 1.0)
                            doubt_optimizer.step()
                    except Exception as e:
                        logger.debug(f"DoubtEngine training step failed: {e}")
                
                # ═══ TRAINING-OPT: EMA update (Polyak averaging) ═══
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        
        # Eval
        model.eval()
        with torch.no_grad():
            eval_losses = []
            eval_correct = 0
            eval_total = 0
            for j in range(0, len(c_test_in), args.batch):
                test_batch = c_test_in[j:j+args.batch].to(device, non_blocking=True)
                test_tgt_batch = c_test_tgt[j:j+args.batch].to(device, non_blocking=True)
                logits = forward_model(test_batch)
                el = F.cross_entropy(
                    logits.view(-1, actual_vocab),
                    test_tgt_batch.view(-1)
                ).item()
                eval_losses.append(el)
                # Eval accuracy
                preds = logits.argmax(dim=-1)
                eval_correct += (preds.view(-1) == test_tgt_batch.view(-1)).sum().item()
                eval_total += test_tgt_batch.numel()
            eval_loss = np.mean(eval_losses)
        
        elapsed = time.time() - t0
        ppl = np.exp(min(eval_loss, 20))
        tok_per_sec = tokens_processed / max(elapsed, 0.01)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        eval_acc = 100.0 * eval_correct / max(eval_total, 1)
        avg_train_loss = total_loss / max(n_batches, 1)
        
        is_best = eval_loss < best_loss
        marker = " ★ BEST" if is_best else ""
        overfit_flag = " ⚠ OVERFIT" if train_acc > eval_acc + 10 else ""
        
        # ═══ T18: Log aux_loss_ratio ═══
        aux_ratio_str = ""
        try:
            _cur_aux = model.mole_aux_loss
            if isinstance(_cur_aux, torch.Tensor):
                aux_ratio = _cur_aux.item() / max(avg_train_loss, 1e-8)
                aux_ratio_str = f" | aux/lm={aux_ratio:.3f}"
        except Exception:
            pass
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {avg_train_loss:.4f} (acc {train_acc:.1f}%) | "
              f"Eval: {eval_loss:.4f} (acc {eval_acc:.1f}%) | "
              f"PPL: {ppl:.1f} | {tok_per_sec:.0f} tok/s | {elapsed:.1f}s{marker}{overfit_flag}{aux_ratio_str}")
        
        # Save checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),  # TRAINING-OPT: EMA weights
            'config': {
                'd_model': args.d_model,
                'n_layers': args.n_layers,
                'vocab_size': actual_vocab,
                'quant_mode': quant_mode,
            },
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'vocab_size': actual_vocab,
            'epoch': epoch,
            'eval_loss': eval_loss,
            'phase': args.phase,
            'quant_mode': quant_mode,
            'tokenizer_mode': tokenizer._mode,
        }
        
        # ═══ T17: Save DoubtEngine state in checkpoint ═══
        if doubt_engine is not None:
            checkpoint['doubt_engine_state_dict'] = doubt_engine.state_dict()
            if doubt_optimizer is not None:
                checkpoint['doubt_optimizer_state_dict'] = doubt_optimizer.state_dict()
            # Also save standalone doubt checkpoint
            doubt_save_dir = Path(getattr(args, 'doubt_checkpoint', 'models/doubt/doubt_engine_best.pt')).parent
            os.makedirs(str(doubt_save_dir), exist_ok=True)
            doubt_save_path = doubt_save_dir / 'doubt_engine_best.pt'
            torch.save({
                'model_state_dict': doubt_engine.state_dict(),
                'd_model': args.d_model,
                'epoch': epoch,
            }, str(doubt_save_path))
            print(f"  [DoubtEngine] Saved: {doubt_save_path}")
        
        # Per-epoch checkpoint (защита от потери прогресса при крашах)
        epoch_path = save_path.with_suffix(f'.epoch{epoch+1}.pt')
        torch.save(checkpoint, str(epoch_path))
        
        # Best checkpoint (use EMA weights — more stable for inference)
        if is_best:
            best_loss = eval_loss
            patience_counter = 0
            # Save EMA version as the best model (more stable)
            ema_checkpoint = dict(checkpoint)
            ema_checkpoint['model_state_dict'] = ema_model.state_dict()
            torch.save(ema_checkpoint, str(save_path))
            print(f"  ✓ Best saved (EMA): {save_path} ({quant_mode})")
        else:
            patience_counter += 1
            print(f"  ↺ Epoch checkpoint: {epoch_path.name} (patience {patience_counter}/{args.patience})")
        
        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  ⏹ Early stopping: {args.patience} эпох без улучшения (best={best_loss:.4f})")
            break
    
    # ═══════════════════════════════════════════════════════════
    # Phase 5: PERSONALITY FINE-TUNING
    # After all phases — fine-tune ONLY on personality data.
    # Recency bias: last gradients stick the most.
    # ═══════════════════════════════════════════════════════════
    personality_epochs = 3
    print(f"\n{'═'*60}")
    print(f"  Phase 5: Personality Fine-tuning ({personality_epochs} epochs)")
    print(f"{'═'*60}")
    
    # Load only personality data
    p_parts = []
    for pfile in ["tars_identity.txt", "tars_personality.txt", "tars_personality_mega.txt"]:
        ppath = ROOT / "data" / "identity" / pfile
        if ppath.exists():
            with open(ppath, 'r', encoding='utf-8') as f:
                ptxt = f.read()
            if len(ptxt) > 100:
                p_parts.append(ptxt)
                print(f"  [P5] {pfile}: {len(ptxt)//1024} KB")
    
    if p_parts:
        p_corpus = "\n\n".join(p_parts)
        
        # ═══ Freeze ALL except PersonalityAdapter ═══
        for p in model.parameters():
            p.requires_grad = False
        for p in model.personality.parameters():
            p.requires_grad = True
        
        p_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        p_total = sum(p.numel() for p in model.parameters())
        print(f"  [P5] PersonalityAdapter: {p_trainable:,} / {p_total:,} params trainable")
        
        # Higher LR for adapter — it's small and needs to learn fast
        p_lr = args.lr * 0.1
        p_optimizer = torch.optim.AdamW(
            [p for p in model.personality.parameters() if p.requires_grad],
            lr=p_lr, weight_decay=0.005
        )
        print(f"  [P5] LR: {p_lr:.2e} | Corpus: {len(p_corpus)//1024} KB")
        
        p_inputs, p_targets = prepare_data(p_corpus, args.seq_len, actual_vocab, tokenizer=tokenizer)
        n_test_p = max(2, len(p_inputs) // 10)
        p_train_in, p_test_in = p_inputs[:-n_test_p], p_inputs[-n_test_p:]
        p_train_tgt, p_test_tgt = p_targets[:-n_test_p], p_targets[-n_test_p:]
        print(f"  [P5] Samples: {len(p_train_in)}")
        
        for p_epoch in range(personality_epochs):
            t0 = time.time()
            model.train()
            p_loss_sum = 0
            p_n = 0
            
            perm = torch.randperm(len(p_train_in))
            for pi in range(0, len(p_train_in), args.batch):
                idx = perm[pi:pi+args.batch]
                b_in = p_train_in[idx].to(device, non_blocking=True)
                b_tgt = p_train_tgt[idx].to(device, non_blocking=True)
                
                if use_amp:
                    with torch.amp.autocast(device.type, dtype=amp_dtype):
                        logits = forward_model(b_in)
                        loss = F.cross_entropy(logits.view(-1, actual_vocab), b_tgt.view(-1))
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(p_optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(p_optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        p_optimizer.step()
                else:
                    logits = forward_model(b_in)
                    loss = F.cross_entropy(logits.view(-1, actual_vocab), b_tgt.view(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    p_optimizer.step()
                
                p_optimizer.zero_grad(set_to_none=True)
                p_loss_sum += loss.item()
                p_n += 1
            
            # Eval on personality data
            model.eval()
            with torch.no_grad():
                p_eval = []
                for pj in range(0, len(p_test_in), args.batch):
                    tb = p_test_in[pj:pj+args.batch].to(device, non_blocking=True)
                    tt = p_test_tgt[pj:pj+args.batch].to(device, non_blocking=True)
                    logits = forward_model(tb)
                    p_eval.append(F.cross_entropy(logits.view(-1, actual_vocab), tt.view(-1)).item())
            
            p_el = np.mean(p_eval) if p_eval else 0
            elapsed = time.time() - t0
            print(f"  P5 Epoch {p_epoch+1}/{personality_epochs} | "
                  f"Train: {p_loss_sum/max(p_n,1):.4f} | Eval: {p_el:.4f} | {elapsed:.1f}s")
        
        # Save final personality-tuned model
        checkpoint['model_state_dict'] = model.state_dict()
        checkpoint['phase'] = 5
        torch.save(checkpoint, str(save_path))
        print(f"  [P5] Personality-tuned model saved: {save_path}")
    
    # ═══════════════════════════════════════════════════════════
    # Post-training: SparseSSM Pruning (if --prune > 0)
    # ═══════════════════════════════════════════════════════════
    if getattr(args, 'prune', 0.0) > 0:
        prune_sparsity = args.prune
        print(f"\n{'═'*60}")
        print(f"  SparseSSM Pruning (sparsity={prune_sparsity:.0%})")
        print(f"{'═'*60}")
        try:
            from brain.mamba2.optimizations import prune_ssm_weights
            prune_stats = prune_ssm_weights(model, sparsity=prune_sparsity)
            print(f"  Pruned: {prune_stats['pruned_params']:,}/{prune_stats['total_params']:,} params")
            print(f"  Actual sparsity: {prune_stats['actual_sparsity']:.1%}")
            
            # Re-save with pruned weights
            checkpoint['model_state_dict'] = model.state_dict()
            checkpoint['pruned'] = True
            checkpoint['prune_sparsity'] = prune_sparsity
            pruned_path = save_path.with_suffix('.pruned.pt')
            torch.save(checkpoint, str(pruned_path))
            print(f"  ✓ Pruned model saved: {pruned_path}")
        except ImportError:
            print("  ⚠ Pruning requires optimizations.py")
        except Exception as e:
            print(f"  ⚠ Pruning failed: {e}")
    
    print(f"\n{'═'*60}")
    print(f"  Done! Best eval loss: {best_loss:.4f}")
    print(f"  Weights: {save_path}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
