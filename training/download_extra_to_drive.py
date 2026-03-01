"""
═══════════════════════════════════════════════════════════════
  Скачать ДОПОЛНИТЕЛЬНЫЕ датасеты на Google Drive
═══════════════════════════════════════════════════════════════

Запуск в Colab (после основного обучения):
  !python training/download_extra_to_drive.py

Датасеты скачиваются НАПРЯМУЮ на Drive → не теряются!
Можно использовать для дообучения на большем объёме данных.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Проверка Drive
DRIVE_DATA = Path("/content/drive/MyDrive/TarsData")
if not DRIVE_DATA.exists():
    print("❌ Google Drive не подключён!")
    print("   Запусти сначала: drive.mount('/content/drive')")
    sys.exit(1)

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
print()
print("  Для дообучения: !python colab_train.py --skip-download")
