"""
═══════════════════════════════════════════════════════════════
  TARS v3 — Kaggle Training Notebook
═══════════════════════════════════════════════════════════════

Запуск на Kaggle с GPU (P100/T4/A100):

1. Загрузи весь репозиторий как Dataset на Kaggle
   (назови его, например: "tarsssm-py")
2. Создай новый Notebook → Add Data → выбери свой датасет
3. Включи GPU: Settings → Accelerator → GPU P100 / T4×2
4. Скопируй этот скрипт в первую ячейку и запусти

Или добавь этот файл прямо в ноутбук:
  !python /kaggle/input/tarsssm-py/kaggle_train.py

═══════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
#  1. ОПРЕДЕЛЕНИЕ ОКРУЖЕНИЯ
# ═══════════════════════════════════════════════════════════════

IS_KAGGLE = os.path.exists("/kaggle")
IS_COLAB = os.path.exists("/content")

if IS_KAGGLE:
    INPUT_DIR = Path("/kaggle/input")
    WORK_DIR = Path("/kaggle/working/TarsSSM-Py")
    OUTPUT_DIR = Path("/kaggle/working/output")
elif IS_COLAB:
    INPUT_DIR = Path("/content/drive/MyDrive")
    WORK_DIR = Path("/content/TarsSSM-Py")
    OUTPUT_DIR = Path("/content/output")
else:
    # Локальный запуск
    INPUT_DIR = Path(__file__).resolve().parent
    WORK_DIR = INPUT_DIR
    OUTPUT_DIR = INPUT_DIR / "output"

print("=" * 65)
print("  ТАРС v3 — Kaggle/Colab Training Pipeline")
print("=" * 65)
print(f"  Environment: {'Kaggle' if IS_KAGGLE else 'Colab' if IS_COLAB else 'Local'}")
print(f"  Input:  {INPUT_DIR}")
print(f"  Work:   {WORK_DIR}")
print(f"  Output: {OUTPUT_DIR}")


# ═══════════════════════════════════════════════════════════════
#  2. КОПИРОВАНИЕ РЕПОЗИТОРИЯ В РАБОЧУЮ ДИРЕКТОРИЮ
# ═══════════════════════════════════════════════════════════════

def setup_workspace():
    """
    Kaggle монтирует датасеты в read-only /kaggle/input/.
    Копируем в /kaggle/working/ чтобы можно было писать модели.
    """
    if WORK_DIR.exists() and (WORK_DIR / "mega_train.py").exists():
        print("\n✅ Рабочая директория уже настроена")
        return True
    
    # Ищем репозиторий в датасетах Kaggle
    repo_src = None
    if IS_KAGGLE:
        for d in INPUT_DIR.iterdir():
            if d.is_dir():
                # Ищем mega_train.py как маркер нашего репо
                if (d / "mega_train.py").exists():
                    repo_src = d
                    break
                # Или может быть вложен на 1 уровень
                for sd in d.iterdir():
                    if sd.is_dir() and (sd / "mega_train.py").exists():
                        repo_src = sd
                        break
                if repo_src:
                    break
    elif IS_COLAB:
        # В Colab пользователь может клонировать через git
        if not WORK_DIR.exists():
            print("\n📥 Клонируй репозиторий:")
            print("  !git clone https://github.com/Vazilll/TarsSSM-Py /content/TarsSSM-Py")
            return False
        return True
    else:
        # Локальный запуск — работаем in-place
        return True
    
    if repo_src is None:
        print("\n❌ Репозиторий не найден в /kaggle/input/!")
        print("   Добавь датасет с репозиторием через Add Data")
        print("   (файл mega_train.py должен быть в корне)")
        return False
    
    print(f"\n📂 Копирование {repo_src} → {WORK_DIR}...")
    t0 = time.time()
    
    # Копируем всё, кроме тяжёлых директорий
    SKIP = {'.git', '__pycache__', 'venv', '.venv', 'node_modules', '.mypy_cache'}
    
    def copy_tree(src, dst):
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            if item.name in SKIP:
                continue
            target = dst / item.name
            if item.is_dir():
                copy_tree(item, target)
            else:
                shutil.copy2(str(item), str(target))
    
    copy_tree(repo_src, WORK_DIR)
    elapsed = time.time() - t0
    print(f"  ✅ Скопировано за {elapsed:.1f}s")
    return True


# ═══════════════════════════════════════════════════════════════
#  3. УСТАНОВКА ЗАВИСИМОСТЕЙ
# ═══════════════════════════════════════════════════════════════

def install_deps():
    """Устанавливает зависимости (torch уже есть в Kaggle)."""
    print("\n" + "=" * 65)
    print("  Фаза 0: Установка зависимостей")
    print("=" * 65)
    
    # Проверяем torch + CUDA
    try:
        import torch
        print(f"\n  PyTorch: {torch.__version__}")
        print(f"  CUDA:    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU:     {torch.cuda.get_device_name(0)}")
            print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("  ⚠ PyTorch не найден — устанавливаю...")
        subprocess.run([sys.executable, "-m", "pip", "install", "torch", "-q"])
    
    # Основные пакеты (без torch — он уже есть)
    packages = [
        "einops", "tqdm", "psutil",
        "sentencepiece", "tokenizers",
        "sentence-transformers",
        "datasets",
        "transformers",
    ]
    
    # Проверяем что уже установлено
    to_install = []
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            to_install.append(pkg)
    
    if to_install:
        print(f"\n  📦 Установка: {', '.join(to_install)}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + to_install + ["-q"],
            check=False
        )
    else:
        print("\n  ✅ Все пакеты уже установлены")
    
    return True


# ═══════════════════════════════════════════════════════════════
#  4. СКАЧИВАНИЕ ДАННЫХ
# ═══════════════════════════════════════════════════════════════

def download_data():
    """Скачивает обучающие данные (Wikipedia + HF)."""
    print("\n" + "=" * 65)
    print("  Фаза 1: Скачивание данных")
    print("=" * 65)
    
    os.chdir(str(WORK_DIR))
    sys.path.insert(0, str(WORK_DIR))
    
    data_dir = WORK_DIR / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 1. Wikipedia
    wiki_path = data_dir / "wiki_ru.txt"
    if wiki_path.exists() and wiki_path.stat().st_size > 100_000:
        wiki_mb = wiki_path.stat().st_size / 1024 / 1024
        print(f"\n  📚 Wikipedia: уже есть ({wiki_mb:.1f} MB)")
    else:
        print("\n  📚 Скачивание Wikipedia (10 000 статей)...")
        try:
            result = subprocess.run(
                [sys.executable, str(WORK_DIR / "training" / "download_wiki.py"),
                 "--count", "10000"],
                cwd=str(WORK_DIR), timeout=1800
            )
            if result.returncode == 0:
                print("  ✅ Wikipedia скачана")
            else:
                print("  ⚠ Wikipedia не скачана (не критично)")
        except Exception as e:
            print(f"  ⚠ Wikipedia: {e}")
    
    # 2. HuggingFace datasets
    hf_files = list(data_dir.glob("hf_*.txt"))
    if len(hf_files) >= 1:
        total_mb = sum(f.stat().st_size for f in hf_files) / 1024 / 1024
        print(f"  🤗 HuggingFace: уже есть ({len(hf_files)} файлов, {total_mb:.0f} MB)")
    else:
        print("  🤗 Скачивание HuggingFace датасетов...")
        try:
            result = subprocess.run(
                [sys.executable, str(WORK_DIR / "training" / "download_hf_dataset.py"),
                 "--preset", "all"],
                cwd=str(WORK_DIR), timeout=1800
            )
            if result.returncode == 0:
                print("  ✅ HF данные скачаны")
            else:
                print("  ⚠ HF данные не скачаны (не критично)")
        except Exception as e:
            print(f"  ⚠ HF: {e}")
    
    # Итого
    total = sum(f.stat().st_size for f in data_dir.glob("*") if f.is_file())
    print(f"\n  📊 Итого данных: {total / 1024 / 1024:.0f} MB")
    return True


# ═══════════════════════════════════════════════════════════════
#  5. ОБУЧЕНИЕ
# ═══════════════════════════════════════════════════════════════

def train_reflex():
    """Фаза 2: Рефлексы (MinGRU classifier, ~1 мин)."""
    print("\n" + "=" * 65)
    print("  Фаза 2: Рефлексы (MinGRU Classifier)")
    print("=" * 65)
    
    result = subprocess.run(
        [sys.executable, str(WORK_DIR / "training" / "train_reflex.py"),
         "--epochs", "100", "--lr", "0.002"],
        cwd=str(WORK_DIR), timeout=600
    )
    return result.returncode == 0


def train_mingru():
    """Фаза 3: MinGRU LM (System 1, ~15 мин GPU)."""
    print("\n" + "=" * 65)
    print("  Фаза 3: MinGRU Language Model (System 1)")
    print("=" * 65)
    
    result = subprocess.run(
        [sys.executable, str(WORK_DIR / "training" / "train_mingru.py"),
         "--epochs", "25",
         "--lr", "3e-3",
         "--dim", "512",
         "--layers", "6",
         "--batch", "32",
         "--seq_len", "256",
         "--augment",
        ],
        cwd=str(WORK_DIR), timeout=3600
    )
    return result.returncode == 0


def train_mamba2():
    """
    Фаза 4: Mamba-2 Brain (основное обучение, ~2-4ч GPU).
    
    12 слоёв × 768d, 4 под-фазы обучения:
      Phase 1: Full pretrain (все компоненты, 5 эпох)
      Phase 2: Fine-tune WKV + Fusion (SSD frozen, 3 эпохи)
      Phase 3: Fine-tune MoLE + MatrixPool (2 эпохи)
      Phase 4: Fine-tune WKV + RAG + Memory (2 эпохи)
    """
    print("\n" + "=" * 65)
    print("  Фаза 4: Mamba-2 Brain (12×768d, Full Architecture)")
    print("=" * 65)
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Определяем batch size по VRAM
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if vram_gb >= 40:      # A100
            batch = "32"
            accum = "2"
        elif vram_gb >= 16:    # P100 / T4
            batch = "16"
            accum = "4"
        else:                  # P4 / K80
            batch = "8"
            accum = "8"
        print(f"\n  GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.0f} GB)")
        print(f"  Batch: {batch} × {accum} = {int(batch) * int(accum)} effective")
    else:
        batch = "4"
        accum = "4"
        print("\n  ⚠ Нет GPU — обучение будет ОЧЕНЬ медленным!")
    
    # Transfer embedding MinGRU → Mamba-2 (если есть)
    emb_args = []
    mingru_path = WORK_DIR / "models" / "mingru_weights.pt"
    if mingru_path.exists():
        print(f"  🔗 Transfer embedding: {mingru_path}")
        try:
            cp = torch.load(str(mingru_path), map_location='cpu', weights_only=False)
            state = cp.get('model_state_dict', cp)
            for k in state:
                if 'shared_embedding' in k or 'emb.weight' in k:
                    emb_path = WORK_DIR / "models" / "tars_v3" / "_transfer_embedding.pt"
                    emb_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(state[k], str(emb_path))
                    emb_args = ["--pretrained_emb", str(emb_path)]
                    print(f"  ✅ Embedding ({state[k].shape}) saved")
                    break
        except Exception as e:
            print(f"  ⚠ Transfer failed: {e}")
    
    base_args = [
        "--d_model", "768",
        "--n_layers", "12",
        "--vocab_size", "256",
        "--batch", batch,
        "--accum_steps", accum,
        "--device", device,
        "--curriculum",
        "--label_smoothing", "0.1",
    ] + emb_args
    
    train_script = str(WORK_DIR / "training" / "train_mamba2.py")
    
    phases = [
        # (phase, epochs, lr, seq_len, description)
        ("1", "5", "3e-4", "256", "Full pretrain (SSD + WKV + Ω-SSM + MoLE)"),
        ("2", "3", "1e-4", "512", "Fine-tune WKV + Fusion (SSD frozen)"),
        ("3", "2", "3e-5", "512", "Fine-tune MoLE + MatrixPool"),
        ("4", "2", "1.5e-5", "512", "Fine-tune WKV + RAG + Memory"),
    ]
    
    results = {}
    for phase, epochs, lr, seq_len, desc in phases:
        print(f"\n  ── Phase {phase}/4: {desc} ──")
        
        phase_args = base_args + [
            "--epochs", epochs,
            "--lr", lr,
            "--phase", phase,
            "--seq_len", seq_len,
        ]
        if phase != "1":
            phase_args.append("--resume")
        
        result = subprocess.run(
            [sys.executable, train_script] + phase_args,
            cwd=str(WORK_DIR), timeout=7200  # 2ч на фазу макс
        )
        
        results[f"p{phase}"] = result.returncode == 0
        if result.returncode != 0:
            print(f"  ⚠ Phase {phase} finished with errors")
    
    all_ok = all(results.values())
    if all_ok:
        print("\n  ✅ Все 4 фазы Mamba-2 завершены!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  ⚠ Фазы с ошибками: {failed}")
    
    return all_ok


def train_quantize():
    """Фаза 5: Квантизация FP16 → 1.58-bit + дообучение."""
    print("\n" + "=" * 65)
    print("  Фаза 5: Квантизация BitNet 1.58-bit")
    print("=" * 65)
    
    fp16_path = WORK_DIR / "models" / "mamba2" / "mamba2_omega.pt"
    if not fp16_path.exists():
        print(f"  ⚠ FP16 модель не найдена: {fp16_path}")
        print("  Пропускаем квантизацию")
        return False
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    result = subprocess.run(
        [sys.executable, str(WORK_DIR / "training" / "train_mamba2.py"),
         "--d_model", "768", "--n_layers", "12",
         "--batch", "16", "--accum_steps", "4",
         "--epochs", "3", "--lr", "5e-5",
         "--phase", "1", "--quant",
         "--resume", "--device", device,
         "--seq_len", "256", "--label_smoothing", "0.1",
        ],
        cwd=str(WORK_DIR), timeout=7200
    )
    return result.returncode == 0


def validate_model():
    """Фаза 7: Тестовая генерация."""
    print("\n" + "=" * 65)
    print("  Фаза 7: Валидация")
    print("=" * 65)
    
    result = subprocess.run(
        [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '{WORK_DIR}')
import torch
from brain.tokenizer import TarsTokenizer
from brain.mamba2.model import TarsMamba2LM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = TarsTokenizer()

model, ckpt = TarsMamba2LM.load_pretrained(device=device)
model.eval()

if ckpt is None:
    print("  No trained weights found")
else:
    print(f"  Model: {{ckpt}}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {{params:,}}")
    
    for prompt in ["привет", "как дела", "что такое"]:
        tokens = tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top5 = torch.topk(probs, 5)
        preds = []
        for idx, prob in zip(top5.indices.tolist(), top5.values.tolist()):
            char = tokenizer.decode([idx])
            preds.append(f"'{{char}}'({{prob:.2%}})")
        print(f"  '{{prompt}}' → {{', '.join(preds)}}")
    
    print("  ✅ Model works!")
"""],
        cwd=str(WORK_DIR), timeout=120
    )
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════════
#  6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ═══════════════════════════════════════════════════════════════

def save_outputs():
    """Копирует обученные модели в output для скачивания."""
    print("\n" + "=" * 65)
    print("  Сохранение результатов")
    print("=" * 65)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Список моделей для сохранения
    model_files = [
        WORK_DIR / "models" / "mamba2" / "mamba2_omega.pt",
        WORK_DIR / "models" / "mamba2" / "mamba2_omega_158bit.pt",
        WORK_DIR / "models" / "mingru_weights.pt",
        WORK_DIR / "models" / "reflex" / "reflex_classifier.pt",
    ]
    
    # Также сохраняем тренировочные логи
    log_files = [
        WORK_DIR / "mega_train.log",
    ]
    
    saved = []
    for src in model_files + log_files:
        if src.exists():
            dst = OUTPUT_DIR / src.name
            shutil.copy2(str(src), str(dst))
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"  📦 {src.name} → output/ ({size_mb:.1f} MB)")
            saved.append(src.name)
        else:
            print(f"  ⏭ {src.name} — не найден")
    
    # Собираем tars_v3 если есть все
    tars_v3_out = OUTPUT_DIR / "tars_v3"
    tars_v3_out.mkdir(exist_ok=True)
    
    copies = {
        "reflex.pt": WORK_DIR / "models" / "reflex" / "reflex_classifier.pt",
        "mingru.pt": WORK_DIR / "models" / "mingru_weights.pt",
        "mamba2.pt": WORK_DIR / "models" / "mamba2" / "mamba2_omega.pt",
        "mamba2_158bit.pt": WORK_DIR / "models" / "mamba2" / "mamba2_omega_158bit.pt",
    }
    
    for dst_name, src in copies.items():
        if src.exists():
            shutil.copy2(str(src), str(tars_v3_out / dst_name))
    
    # Config
    config = {
        "encoding": "cp1251",
        "vocab_size": 256,
        "d_model": 768,
        "n_layers": 12,
        "n_experts": 8,
        "omega_dim": 32,
        "pool_size": 48,
    }
    import json
    (tars_v3_out / "config.json").write_text(
        json.dumps({"models": {"mamba2": {"params": config}}}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"\n  📁 Результаты в: {OUTPUT_DIR}")
    print(f"  📁 tars_v3 сборка: {tars_v3_out}")
    
    if IS_KAGGLE:
        print(f"\n  💡 Скачай результаты:")
        print(f"     Notebook → Output → Download All")
    
    return saved


# ═══════════════════════════════════════════════════════════════
#  7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    results = {}
    
    # ── Setup ──
    if not setup_workspace():
        print("\n❌ Workspace setup failed!")
        return
    
    os.chdir(str(WORK_DIR))
    sys.path.insert(0, str(WORK_DIR))
    
    # ── Phase 0: Dependencies ──
    results["install"] = install_deps()
    
    # ── Phase 1: Data ──
    results["download"] = download_data()
    
    # ── Phase 2: Reflex ──
    try:
        results["reflex"] = train_reflex()
    except Exception as e:
        print(f"  ⚠ Reflex error: {e}")
        results["reflex"] = False
    
    # ── Phase 3: MinGRU ──
    try:
        results["mingru"] = train_mingru()
    except Exception as e:
        print(f"  ⚠ MinGRU error: {e}")
        results["mingru"] = False
    
    # ── Phase 4: Mamba-2 (MAIN) ──
    try:
        results["mamba2"] = train_mamba2()
    except Exception as e:
        print(f"  ⚠ Mamba-2 error: {e}")
        results["mamba2"] = False
    
    # ── Phase 5: Quantization ──
    try:
        results["quantize"] = train_quantize()
    except Exception as e:
        print(f"  ⚠ Quantize error: {e}")
        results["quantize"] = False
    
    # ── Phase 7: Validation ──
    try:
        results["validate"] = validate_model()
    except Exception as e:
        print(f"  ⚠ Validate error: {e}")
        results["validate"] = False
    
    # ── Save outputs ──
    saved = save_outputs()
    
    # ═══ ИТОГИ ═══
    total_time = time.time() - t0
    hours = total_time / 3600
    
    print("\n" + "=" * 65)
    print("  ИТОГИ ОБУЧЕНИЯ")
    print("=" * 65)
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print(f"\n  ⏱  Время: {hours:.1f} часов ({total_time:.0f} сек)")
    
    if all(results.values()):
        print("\n  🎯 ВСЕ ФАЗЫ ЗАВЕРШЕНЫ УСПЕШНО!")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"\n  ⚠ Фазы с ошибками: {', '.join(failed)}")
    
    print("=" * 65)


if __name__ == "__main__":
    main()
