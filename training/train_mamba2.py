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
import time
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brain.mamba2.model import TarsMamba2LM


def parse_args():
    p = argparse.ArgumentParser(description="ТАРС Mamba-2 + RWKV-7 Training (Deep WuNeng Core)")
    # ═══ Model params ═══
    p.add_argument('--d_model', type=int, default=2048)   # TARS 1B
    p.add_argument('--n_layers', type=int, default=24)    # 24 rich blocks
    p.add_argument('--vocab_size', type=int, default=256)  # cp1251 bytes
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
    return p.parse_args()


def load_corpus(data_path=None, download_wiki=True):
    """Загружает обучающий корпус.
    
    Источники (по приоритету):
      1. --data (пользовательский файл)
      2. Встроенный корпус (train_corpus.py: диалоги + тексты)
      3. Wikipedia (auto-download 500 статей при первом запуске)
      4. TARS memories (data/tars_memories.json)
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
    
    # 3. Wikipedia (auto-download)
    wiki_path = ROOT / "data" / "wiki_ru.txt"
    if wiki_path.exists():
        with open(wiki_path, 'r', encoding='utf-8') as f:
            wiki_text = f.read()
        if len(wiki_text) > 1000:
            parts.append(wiki_text)
            wiki_mb = len(wiki_text.encode('utf-8')) / (1024 * 1024)
            print(f"[Data] Wikipedia: {len(wiki_text):,} символов ({wiki_mb:.1f} MB, кеш)")
    elif download_wiki:
        print("[Data] Wikipedia корпус не найден — скачиваю 100 000 статей...")
        print("[Data] Это может занять 10-30 минут при первом запуске.")
        try:
            sys.path.insert(0, str(ROOT / "training"))
            from download_wiki import download_corpus
            wiki_text = download_corpus(count=10000)
            if wiki_text:
                parts.append(wiki_text)
                wiki_mb = len(wiki_text.encode('utf-8')) / (1024 * 1024)
                print(f"[Data] Wikipedia: {len(wiki_text):,} символов ({wiki_mb:.1f} MB)")
        except Exception as e:
            print(f"[Data] ⚠ Не удалось скачать Wikipedia: {e}")
            print("[Data]   Запустите вручную: python training/download_wiki.py")
    
    # 4. HuggingFace датасеты (data/hf_*.txt)
    hf_dir = ROOT / "data"
    if hf_dir.exists():
        hf_files = sorted(hf_dir.glob("hf_*.txt"))
        for hf_file in hf_files:
            try:
                with open(hf_file, 'r', encoding='utf-8') as f:
                    hf_text = f.read()
                if len(hf_text) > 1000:
                    parts.append(hf_text)
                    hf_mb = len(hf_text.encode('utf-8')) / (1024 * 1024)
                    print(f"[Data] HF {hf_file.stem}: {len(hf_text):,} символов ({hf_mb:.1f} MB)")
            except Exception:
                pass
    
    # 5. TARS memories
    memories_path = ROOT / "data" / "tars_memories.json"
    if memories_path.exists():
        try:
            with open(memories_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts = [e.get("text", e.get("memory", "")) for e in data if isinstance(e, dict)]
            parts.append("\n".join(texts))
            print(f"[Data] Memories: {len(texts)} записей")
        except Exception:
            pass
    
    corpus = "\n\n".join(parts)
    
    # 6. TARS Identity (self-description — модель учит свою структуру)
    identity_path = ROOT / "data" / "tars_identity.txt"
    if identity_path.exists():
        with open(identity_path, 'r', encoding='utf-8') as f:
            identity_text = f.read()
        if len(identity_text) > 100:
            # Повторяем 5x для усиления — модель должна знать свою архитектуру
            identity_repeated = ("\n\n" + identity_text) * 5
            corpus = identity_repeated + "\n\n" + corpus
            print(f"[Data] ТАРС Identity: {len(identity_text):,} символов (×5 для усиления)")
    
    
    # Повторяем маленький корпус
    corpus_bytes = len(corpus.encode('utf-8'))
    if corpus_bytes < 200_000:
        repeat = max(1, 200_000 // corpus_bytes)
        corpus = ("\n\n" + corpus) * repeat
        print(f"[Data] Корпус повторён {repeat}x")
    
    print(f"[Data] Итого: {len(corpus):,} символов ({corpus_bytes / 1024:.0f} KB)")
    return corpus


def prepare_byte_data(text: str, seq_len: int, vocab_size: int = 32000, max_samples: int = 0):
    """
    Подготовка данных для byte-level обучения.
    Для полной модели нужен SentencePiece tokenizer,
    но для начала используем byte-level (vocab=256, кириллица через cp1251).
    """
    tokens = list(text.encode('cp1251', errors='replace'))
    
    # Если max_samples задан, ограничиваем длину текста
    if max_samples > 0:
        max_chars = max_samples * seq_len
        if len(tokens) > max_chars:
            tokens = tokens[:max_chars]
    
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
                state_dict.update(torch.load(f, map_location='cpu'))

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
    corpus = load_corpus(data_path=args.data, download_wiki=not args.no_wiki)
    print(f"[Data] Корпус загружен ({len(corpus):,} символов)")
    
    # ═══════════════════════════════════════════════════════════
    # Модель: 3-шаговый пайплайн
    #   Step A: Создание модели в FP16 (нормальная инициализация)
    #   Step B: Квантизация в 1.58-bit (если --quant)
    #   Step C: Обучение на данных (Wiki + HF + встроенный корпус)
    # ═══════════════════════════════════════════════════════════
    
    actual_vocab = 256  # byte level
    
    # ── Step A: Инициализация в FP16 ──
    print(f"\n[Step A] Создание модели в FP16...")
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
        checkpoint = torch.load(str(save_path), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif args.resume and args.quant:
        # Quant mode but no 158bit checkpoint → load FP16 as fallback
        fp16_path = Path(args.save_dir) / "mamba2_omega.pt"
        if fp16_path.exists():
            print(f"[Model] Quant: loading FP16 base → {fp16_path}")
            checkpoint = torch.load(str(fp16_path), map_location='cpu', weights_only=False)
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
            mingru_emb = torch.load(args.pretrained_emb, map_location=device)
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
    
    # ═══ Фазы обучения ═══
    if args.phase == 1:
        # Phase 1: Pre-train ALL (TarsCoreBlock + Ω-SSM + MoLE)
        print("[Phase 1] Full pre-training: ALL components")
        if getattr(args, 'muon', False):
            from training.muon import Muon
            optimizer = Muon(model.parameters(), lr=0.02, weight_decay=0.01)
            print("  ⚡ Optimizer: Muon (2x faster, NS orthogonalization)")
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
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
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for p in trainable):,} params")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr * 0.1, weight_decay=0.01)
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
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for p in trainable):,} params")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr * 0.01, weight_decay=0.01)
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
            # Dynamic Memory Injection (Tars.txt §4.3): mem_query, mem_proj, mem_gate
            for p in block.mem_query_proj.parameters():
                p.requires_grad = True
            for p in block.mem_proj.parameters():
                p.requires_grad = True
            for p in block.mem_gate.parameters():
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
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for p in trainable):,} params")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr * 0.05, weight_decay=0.01)
    
    # Scheduler (estimate total steps from corpus size)
    estimated_samples = max(100, len(corpus.encode('cp1251', errors='replace')) // max(args.seq_len // 2, 1))
    total_steps = (estimated_samples // args.batch + 1) * args.epochs
    warmup = total_steps // 10
    
    if getattr(args, 'wsd', False):
        # ═══ WSD: Warmup-Stable-Decay (SmolLM2/MiniCPM) ═══
        stable_end = int(total_steps * 0.7)  # 70% at stable LR
        def lr_fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            elif step < stable_end:
                return 1.0  # stable phase — full LR
            else:
                # decay phase: linear decay to 0.1
                progress = (step - stable_end) / max(total_steps - stable_end, 1)
                return max(0.1, 1.0 - 0.9 * progress)
        print("  ⚡ Scheduler: WSD (warmup → stable → decay)")
    else:
        # Standard cosine annealing
        def lr_fn(step):
            if step < warmup:
                return step / max(warmup, 1)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    
    # ═══ Mixture of Depths (-30% compute) ═══
    if getattr(args, 'mod', False):
        from brain.mamba2.mixture_of_depths import add_mod_to_model
        model = add_mod_to_model(model, capacity_factor=0.5)
    
    # ═══ torch.compile for 30% speedup ═══
    if not args.no_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("  ⚡ torch.compile: enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"  ⚠ torch.compile failed: {e} — continuing without")
    
    # ═══ AMP (BitMamba-2: bfloat16 по умолчанию на TPU/GPU) ═══
    use_amp = device.type == 'cuda'
    if use_amp and args.bf16:
        amp_dtype = torch.bfloat16
        scaler = None  # bfloat16 не нуждается в GradScaler
        print("  ⚡ AMP: bfloat16 (no scaler needed, wider dynamic range)")
    elif use_amp:
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler('cuda')
        print("  ⚡ AMP: float16 + GradScaler")
    else:
        amp_dtype = torch.float32
        scaler = None
    
    best_loss = float('inf')
    
    print(f"\n{'═'*60}")
    print(f"  ТАРС Mamba-2 Training — Phase {args.phase}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print(f"{'═'*60}\n")
    
    # ═══ torch.compile (30-50% ускорение) ═══
    compiled_model = None
    if hasattr(torch, 'compile') and not args.no_compile and device.type == 'cuda':
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("  ⚡ torch.compile enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"  ⚠ torch.compile failed: {e}")
    forward_model = compiled_model if compiled_model is not None else model
    
    # ═══ Curriculum Learning ═══
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
    
    accum_steps = max(1, args.accum_steps)
    print(f"  Effective batch: {args.batch} × {accum_steps} = {args.batch * accum_steps}")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0
        tokens_processed = 0
        
        # Curriculum: пересоздать данные с новым seq_len
        cur_seq_len = curriculum_schedule[epoch] if curriculum_schedule else args.seq_len
        if epoch == 0 or (curriculum_schedule and cur_seq_len != curriculum_schedule[max(0, epoch-1)]):
            c_inputs, c_targets = prepare_byte_data(corpus, cur_seq_len, actual_vocab, max_samples=args.max_samples)
            n_test_c = max(2, len(c_inputs) // 10)
            c_train_in, c_test_in = c_inputs[:-n_test_c], c_inputs[-n_test_c:]
            c_train_tgt, c_test_tgt = c_targets[:-n_test_c], c_targets[-n_test_c:]
            print(f"  seq_len={cur_seq_len}, samples={len(c_train_in)}")
        
        perm = torch.randperm(len(c_train_in))
        train_in_s = c_train_in[perm]
        train_tgt_s = c_train_tgt[perm]
        
        optimizer.zero_grad()
        
        for i in range(0, len(train_in_s), args.batch):
            batch_in = train_in_s[i:i+args.batch].to(device)
            batch_tgt = train_tgt_s[i:i+args.batch].to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = forward_model(batch_in)
                    lm_loss = F.cross_entropy(
                        logits.view(-1, actual_vocab),
                        batch_tgt.view(-1),
                        label_smoothing=args.label_smoothing,
                    )
                    # MoLE auxiliary loss (load balancing + z-loss)
                    loss = (lm_loss + model.mole_aux_loss) / accum_steps
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
                # MoLE auxiliary loss (load balancing + z-loss)
                loss = (lm_loss + model.mole_aux_loss) / accum_steps
                loss.backward()
            
            total_loss += loss.item() * accum_steps
            n_batches += 1
            tokens_processed += batch_in.numel()
            
            # Gradient accumulation step
            if n_batches % accum_steps == 0 or i + args.batch >= len(train_in_s):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            eval_losses = []
            for j in range(0, len(c_test_in), args.batch):
                test_batch = c_test_in[j:j+args.batch].to(device)
                test_tgt_batch = c_test_tgt[j:j+args.batch].to(device)
                logits = forward_model(test_batch)
                el = F.cross_entropy(
                    logits.view(-1, actual_vocab),
                    test_tgt_batch.view(-1)
                ).item()
                eval_losses.append(el)
            eval_loss = np.mean(eval_losses)
        
        elapsed = time.time() - t0
        ppl = np.exp(min(eval_loss, 20))
        tok_per_sec = tokens_processed / max(elapsed, 0.01)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {total_loss/max(n_batches,1):.4f} | "
              f"Eval: {eval_loss:.4f} | PPL: {ppl:.1f} | "
              f"{tok_per_sec:.0f} tok/s | {elapsed:.1f}s")
        
        # Save checkpoint
        os.makedirs(args.save_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
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
        }
        
        # Per-epoch checkpoint (защита от потери прогресса при крашах)
        epoch_path = save_path.with_suffix(f'.epoch{epoch+1}.pt')
        torch.save(checkpoint, str(epoch_path))
        
        # Best checkpoint
        if eval_loss < best_loss:
            best_loss = eval_loss
            torch.save(checkpoint, str(save_path))
            print(f"  ✓ Best saved: {save_path} ({quant_mode})")
        else:
            print(f"  ↺ Epoch checkpoint: {epoch_path.name}")
    
    print(f"\n{'═'*60}")
    print(f"  Done! Best eval loss: {best_loss:.4f}")
    print(f"  Weights: {save_path}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
