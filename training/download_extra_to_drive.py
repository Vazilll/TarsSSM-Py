"""
═══════════════════════════════════════════════════════════════
  Скачать ДОПОЛНИТЕЛЬНЫЕ датасеты на Google Drive
  + Квантизация 1.58-bit (отдельный файл для MAX обучения)
═══════════════════════════════════════════════════════════════

Запуск в Colab (после основного обучения):
  !python training/download_extra_to_drive.py

Датасеты скачиваются НАПРЯМУЮ на Drive → не теряются!
После скачивания — дообучение модели на extra-данных в 1.58-bit.

Результат:
  Medium: mamba2_omega.pt        (FP16, base data)
  Top:    mamba2_omega_158bit_extra.pt  (1.58-bit, extra data)
"""

import os
import sys
import time
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Проверка Drive
DRIVE_DATA = Path("/content/drive/MyDrive/TarsData")
DRIVE_MODELS = Path("/content/drive/MyDrive/TarsModels")

if not DRIVE_DATA.exists():
    print("❌ Google Drive не подключён!")
    print("   Запусти сначала: drive.mount('/content/drive')")
    sys.exit(1)

DRIVE_MODELS.mkdir(parents=True, exist_ok=True)

# Дополнительные датасеты для дообучения
EXTRA_DATASETS = [
    # ─── Больше кода ───
    {
        "name": "m-a-p/CodeFeedback-Filtered-Instruction",
        "desc": "157K отфильтрованных инструкций по коду",
        "count": 20000,
        "format": "instruct",
    },
    {
        "name": "codeparrot/github-code-clean",
        "desc": "Чистый код с GitHub (Python, JS, Rust)",
        "count": 10000,
        "format": "code",
        "subsets": ["Python-all"],
    },
    
    # ─── Больше русского ───
    {
        "name": "d0rj/OpenOrca-ru",
        "desc": "Русские инструкции (расширенный набор)",
        "count": 50000,
        "format": "instruct",
    },
    {
        "name": "d0rj/OpenHermes-2.5-ru",
        "desc": "GPT-4 качество на русском (расширенный)",
        "count": 50000,
        "format": "sharegpt",
    },
    
    # ─── Больше математики ───
    {
        "name": "meta-math/MetaMathQA",
        "desc": "MetaMath расширенный",
        "count": 20000,
        "format": "instruct",
    },
    {
        "name": "TIGER-Lab/MathInstruct",
        "desc": "Математика с CoT (расширенный)",
        "count": 20000,
        "format": "instruct",
    },
    
    # ─── Больше reasoning ───
    {
        "name": "open-thoughts/OpenThoughts-114k",
        "desc": "Chain-of-Thought расширенный",
        "count": 30000,
        "format": "sharegpt",
    },
    {
        "name": "OpenAssistant/oasst2",
        "desc": "OpenAssistant расширенный",
        "count": 30000,
        "format": "chat",
    },
    
    # ─── Диалоги ───
    {
        "name": "Den4ikAI/russian_instructions_2",
        "desc": "Русские инструкции (расширенный)",
        "count": 50000,
        "format": "instruct",
    },
    {
        "name": "IlyaGusev/ru_turbo_alpaca",
        "desc": "GPT-4 русские инструкции (все)",
        "count": 30000,
        "format": "instruct",
    },
]

print("═" * 60)
print("  📥 Скачивание дополнительных данных на Drive")
print(f"  📂 Папка: {DRIVE_DATA}")
print("═" * 60)
print()

# Используем основной загрузчик
from training.download_hf_dataset import download_one_dataset

total_new = 0
for ds in EXTRA_DATASETS:
    output_dir = str(DRIVE_DATA)
    safe_name = ds["name"].replace("/", "_")
    output_file = os.path.join(output_dir, f"hf_{safe_name}.txt")
    
    # Если файл уже есть и достаточно большой — пропускаем
    if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ {ds['name']}: уже есть ({size_mb:.1f} MB)")
        continue
    
    text = download_one_dataset(ds, output_dir)
    if text:
        total_new += 1

print()
print("═" * 60)
if total_new > 0:
    print(f"  ✅ Скачано {total_new} новых датасетов на Drive")
else:
    print(f"  ✅ Все датасеты уже на Drive")

# Статистика
all_files = list(DRIVE_DATA.glob("hf_*.txt"))
total_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
print(f"  📊 Всего на Drive: {len(all_files)} датасетов, {total_mb:.0f} MB")
print("═" * 60)


# ═══════════════════════════════════════════════════════════════
#  КВАНТИЗАЦИЯ: Дообучение на extra-данных → 1.58-bit
# ═══════════════════════════════════════════════════════════════

def quantize_extra_data():
    """
    Дообучение модели на extra-данных в 1.58-bit режиме.
    
    Берёт обученную базовую модель и fine-tune'ит на дополнительных
    датасетах. Результат — отдельный файл для MAX варианта.
    
    Входные модели (по приоритету):
      1. TarsModels/mamba2/mamba2_omega_158bit.pt (уже квантованная)
      2. TarsModels/mamba2/mamba2_omega.pt (FP16)
      3. models/mamba2/mamba2_omega.pt (локальная)
    
    Выход:
      TarsModels/mamba2/mamba2_omega_158bit_extra.pt
    """
    print()
    print("═" * 60)
    print("  🔧 Квантизация: extra-данные → 1.58-bit TOP модель")
    print("═" * 60)
    
    try:
        import torch
    except ImportError:
        print("  ❌ PyTorch не установлен")
        return False
    
    # ═══ 1. Найти базовую модель ═══
    search_paths = [
        DRIVE_MODELS / "mamba2" / "mamba2_omega_158bit.pt",
        DRIVE_MODELS / "mamba2" / "mamba2_omega.pt",
        ROOT / "models" / "mamba2" / "mamba2_omega_158bit.pt",
        ROOT / "models" / "mamba2" / "mamba2_omega.pt",
    ]
    
    base_path = None
    for p in search_paths:
        if p.exists() and p.stat().st_size > 1000:
            base_path = p
            break
    
    if base_path is None:
        print("  ⚠ Базовая модель не найдена — квантизация пропущена")
        print("  ℹ Сначала обучи модель: !python colab_train.py")
        print("  Допустимые пути:")
        for p in search_paths:
            print(f"    {p}")
        return False
    
    print(f"  📦 Базовая модель: {base_path}")
    base_mb = base_path.stat().st_size / (1024 * 1024)
    print(f"     Размер: {base_mb:.1f} MB")
    
    # ═══ 2. Загрузить extra-данные ═══
    print("\n  📚 Загрузка extra-данных для дообучения...")
    extra_texts = []
    extra_files = sorted(DRIVE_DATA.glob("hf_*.txt"))
    total_chars = 0
    MAX_EXTRA_MB = 100  # Лимит extra-данных для квантизации
    MAX_EXTRA_BYTES = MAX_EXTRA_MB * 1024 * 1024
    
    for f in extra_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                text = fh.read()
            if len(text) > 1000:
                if total_chars + len(text) > MAX_EXTRA_BYTES:
                    remaining = MAX_EXTRA_BYTES - total_chars
                    if remaining > 10000:
                        text = text[:remaining]
                        extra_texts.append(text)
                        total_chars += remaining
                    print(f"  ⚠ Лимит {MAX_EXTRA_MB} MB — хватит для квантизации")
                    break
                extra_texts.append(text)
                total_chars += len(text)
        except Exception:
            pass
    
    if not extra_texts:
        print("  ⚠ Нет текстовых данных для дообучения")
        return False
    
    corpus = "\n\n".join(extra_texts)
    corpus_mb = len(corpus.encode('cp1251', errors='replace')) / (1024 * 1024)
    print(f"  📊 Extra корпус: {len(extra_texts)} файлов, {corpus_mb:.1f} MB")
    del extra_texts
    gc.collect()
    
    # ═══ 3. Загрузить модель ═══
    print("\n  🧠 Загрузка модели...")
    checkpoint = torch.load(str(base_path), map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    
    d_model = config.get("d_model", 512)
    n_layers = config.get("n_layers", 8)
    vocab_size = config.get("vocab_size", 256)
    
    print(f"     d_model={d_model}, n_layers={n_layers}, vocab={vocab_size}")
    
    from brain.mamba2.model import TarsMamba2LM
    
    model = TarsMamba2LM(
        d_model=d_model,
        n_layers=n_layers,
        vocab_size=vocab_size,
        quant_mode="fp16",
    )
    
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    del checkpoint
    gc.collect()
    
    # ═══ 4. Конвертация в 1.58-bit ═══
    print("\n  ⚡ Конвертация → 1.58-bit...")
    from brain.mamba2.bitnet import convert_model_to_158bit, model_stats
    
    convert_model_to_158bit(model)
    stats = model_stats(model)
    print(f"     Слоёв конвертировано: {stats['universal_linear_count']}")
    print(f"     Разреженность: {stats['avg_sparsity']:.1%}")
    print(f"     Сжатие: ~{stats['compression_ratio']:.1f}x")
    
    # ═══ 5. Дообучение на extra-данных ═══
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"\n  🔥 Дообучение на extra-данных ({device})...")
    
    # Токенизация (byte-level cp1251)
    tokens = list(corpus[:5_000_000].encode('cp1251', errors='replace'))
    token_tensor = torch.tensor(tokens, dtype=torch.long)
    del corpus, tokens
    gc.collect()
    
    seq_len = 256
    batch_size = 8  # Безопасно для T4
    epochs = 2
    lr = 5e-5  # Маленький LR для fine-tune
    
    print(f"     Токенов: {len(token_tensor):,}")
    print(f"     batch={batch_size}, seq_len={seq_len}, epochs={epochs}, lr={lr}")
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )
    
    # AMP
    use_amp = device == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    model.train()
    t0 = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0
        
        indices = list(range(0, len(token_tensor) - seq_len - 1, seq_len))
        if len(indices) > 2000:
            import random
            random.shuffle(indices)
            indices = indices[:2000]
        
        for start in indices:
            x = token_tensor[start:start + seq_len].unsqueeze(0).to(device)
            y = token_tensor[start + 1:start + seq_len + 1].unsqueeze(0).to(device)
            
            if use_amp:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"     Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.4f} ({elapsed:.0f}s)")
    
    # ═══ 6. Сохранение ═══
    output_dir = DRIVE_MODELS / "mamba2"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mamba2_omega_158bit_extra.pt"
    
    save_data = {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "quant_mode": "158bit",
        },
        "stats": stats,
        "extra_data_files": len(extra_files),
        "extra_corpus_mb": corpus_mb,
        "fine_tune_epochs": epochs,
    }
    torch.save(save_data, str(output_path))
    
    output_mb = output_path.stat().st_size / (1024 * 1024)
    total_time = time.time() - t0
    
    print(f"\n  ✅ TOP модель сохранена!")
    print(f"     Файл: {output_path}")
    print(f"     Размер: {output_mb:.1f} MB")
    print(f"     Время: {total_time:.0f} сек")
    
    # Сравнение
    print(f"\n  📊 Сравнение моделей:")
    print(f"     Medium (FP16):        {base_mb:.1f} MB — {base_path.name}")
    print(f"     Top (1.58-bit+extra): {output_mb:.1f} MB — {output_path.name}")
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return True


# ═══ Запуск квантизации ═══
print()
quantize_extra_data()

print()
print("═" * 60)
print("  📁 Данные на Drive:")

# Показать все файлы
for f in sorted(DRIVE_DATA.glob("hf_*.txt")):
    mb = f.stat().st_size / (1024 * 1024)
    print(f"     📄 {f.name}: {mb:.1f} MB")

quant_file = DRIVE_MODELS / "mamba2" / "mamba2_omega_158bit_extra.pt"
if quant_file.exists():
    qmb = quant_file.stat().st_size / (1024 * 1024)
    print(f"     🧠 {quant_file.name}: {qmb:.1f} MB (TOP)")

print("═" * 60)
print()
print("  Для дообучения medium: !python colab_train.py --skip-download")
print("  TOP модель готова:     MyDrive/TarsModels/mamba2/mamba2_omega_158bit_extra.pt")
