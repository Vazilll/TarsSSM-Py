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
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--phase', type=int, default=1, 
                   help="1=pretrain all, 2=WKV+Fusion (freeze SSD), 3=MoLE+Pool, 4=WKV RAG")
    p.add_argument('--resume', action='store_true')
    p.add_argument('--pretrained', type=str, default=None, help="Путь к весам для дообучения")
    p.add_argument('--data', type=str, default=None, help="Путь к текстовому корпусу (.txt)")
    p.add_argument('--device', type=str, default='auto', help="cpu/cuda/auto")
    p.add_argument('--quant', action='store_true', help="Обучать в 1.58-bit режиме (BitNet STE)")
    p.add_argument('--save_dir', type=str, default='models/mamba2')
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
        from brain.min_gru.train_corpus import get_training_text
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
            wiki_text = download_corpus(count=100000)
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
    
    # Повторяем маленький корпус
    corpus_bytes = len(corpus.encode('utf-8'))
    if corpus_bytes < 200_000:
        repeat = max(1, 200_000 // corpus_bytes)
        corpus = ("\n\n" + corpus) * repeat
        print(f"[Data] Корпус повторён {repeat}x")
    
    print(f"[Data] Итого: {len(corpus):,} символов ({corpus_bytes / 1024:.0f} KB)")
    return corpus


def prepare_byte_data(text: str, seq_len: int, vocab_size: int = 32000):
    """
    Подготовка данных для byte-level обучения.
    Для полной модели нужен SentencePiece tokenizer,
    но для начала используем byte-level (vocab=256, pad до vocab_size).
    """
    tokens = list(text.encode('utf-8'))
    
    inputs = []
    targets = []
    stride = seq_len // 2
    
    for i in range(0, len(tokens) - seq_len - 1, stride):
        inp = tokens[i:i + seq_len]
        tgt = tokens[i + 1:i + seq_len + 1]
        if len(inp) == seq_len and len(tgt) == seq_len:
            inputs.append(inp)
            targets.append(tgt)
    
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
              f"({torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB)")
    else:
        print("[CPU] GPU не используется")
    
    # Данные
    corpus = load_corpus(data_path=args.data)
    inputs, targets = prepare_byte_data(corpus, args.seq_len, args.vocab_size)
    print(f"[Data] {len(inputs)} примеров (seq_len={args.seq_len})")
    
    # Разделение
    n_test = max(2, len(inputs) // 10)
    train_in, test_in = inputs[:-n_test], inputs[-n_test:]
    train_tgt, test_tgt = targets[:-n_test], targets[-n_test:]
    
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
    
    model.to(device)
    
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
            print(f"  Конвертировано: {stats['universal_linear_count']} слоёв UniversalLinear")
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    elif args.phase == 2:
        # Phase 2: Freeze SSD-specific params. Train: WKV + Fusion + Ω-SSM + MoLE
        # В TarsCoreBlock SSD и WKV делят общую in_proj, 
        # но мы можем заморозить SSD-специфичные: A_log, D, dt_bias, conv1d
        print("[Phase 2] Fine-tune: WKV + Fusion + Ω-SSM + MoLE (SSD params frozen)")
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
        # Phase 3: Freeze everything. Train: MoLE (per-block) + MatrixPool
        print("[Phase 3] Fine-tune: MoLE + MatrixPool (rest frozen)")
        for p in model.parameters():
            p.requires_grad = False
        for block in model.blocks:
            for p in block.mole.parameters():
                p.requires_grad = True
        for p in model.matrix_pool.parameters():
            p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for p in trainable):,} params")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr * 0.01, weight_decay=0.01)
    elif args.phase == 4:
        # Phase 4: Fine-tune WKV fusion gate + time_mix only (для RAG State Tracking)
        print("[Phase 4] Fine-tune: WKV fusion + time_mix only (для RAG State Tracking)")
        for p in model.parameters():
            p.requires_grad = False
        for block in model.blocks:
            core = block.core
            # Train WKV-related: fusion gate, wkv_up, time_mix
            for p in core.fusion_gate.parameters():
                p.requires_grad = True
            for p in core.wkv_up.parameters():
                p.requires_grad = True
            core.time_mix.requires_grad = True
            # Train RAG projections
            for p in block.rag_query.parameters():
                p.requires_grad = True
            for p in block.rag_out.parameters():
                p.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable: {sum(p.numel() for p in trainable):,} params")
        optimizer = torch.optim.AdamW(trainable, lr=args.lr * 0.05, weight_decay=0.01)
    
    # Scheduler
    total_steps = (len(train_in) // args.batch + 1) * args.epochs
    warmup = total_steps // 10
    
    def lr_fn(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    
    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    best_loss = float('inf')
    
    print(f"\n{'═'*60}")
    print(f"  ТАРС Mamba-2 Training — Phase {args.phase}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print(f"{'═'*60}\n")
    
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        total_loss = 0
        n_batches = 0
        
        perm = torch.randperm(len(train_in))
        train_in_s = train_in[perm]
        train_tgt_s = train_tgt[perm]
        
        for i in range(0, len(train_in_s), args.batch):
            batch_in = train_in_s[i:i+args.batch].to(device)
            batch_tgt = train_tgt_s[i:i+args.batch].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(batch_in)
                    loss = F.cross_entropy(
                        logits.view(-1, actual_vocab),
                        batch_tgt.view(-1)
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_in)
                loss = F.cross_entropy(
                    logits.view(-1, actual_vocab),
                    batch_tgt.view(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            total_loss += loss.item()
            n_batches += 1
        
        # Eval
        model.eval()
        with torch.no_grad():
            eval_losses = []
            for j in range(0, len(test_in), args.batch):
                test_batch = test_in[j:j+args.batch].to(device)
                test_tgt_batch = test_tgt[j:j+args.batch].to(device)
                logits = model(test_batch)
                el = F.cross_entropy(
                    logits.view(-1, actual_vocab),
                    test_tgt_batch.view(-1)
                ).item()
                eval_losses.append(el)
            eval_loss = np.mean(eval_losses)
        
        elapsed = time.time() - t0
        ppl = np.exp(min(eval_loss, 20))
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {total_loss/max(n_batches,1):.4f} | "
              f"Eval: {eval_loss:.4f} | PPL: {ppl:.1f} | "
              f"{elapsed:.1f}s")
        
        # Save best
        if eval_loss < best_loss:
            best_loss = eval_loss
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save({
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
            }, str(save_path))
            print(f"  ✓ Saved: {save_path} ({quant_mode})")
    
    print(f"\n{'═'*60}")
    print(f"  Done! Best eval loss: {best_loss:.4f}")
    print(f"  Weights: {save_path}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
