"""
═══════════════════════════════════════════════════════════════
  ТАРС — Скачать ВСЕ данные для обучения одной командой
═══════════════════════════════════════════════════════════════

Скачивает:
  1. Wikipedia (100 000 статей)
  2. HuggingFace датасеты (код, диалоги, агенты)
  3. Загрузка в LEANN память (RAG для вспоминания знаний)

Использование:
  python training/download_all.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))


def main():
    print("═" * 60)
    print("  ТАРС — Загрузка ВСЕХ обучающих данных")
    print("═" * 60)
    print()

    # ═══ 1. Wikipedia ═══
    print("━" * 60)
    print("  📚 Фаза 1/3: Русская Wikipedia (100 000 статей)")
    print("━" * 60)
    try:
        from download_wiki import download_corpus
        download_corpus(count=10000)
    except Exception as e:
        print(f"  ⚠ Wikipedia: {e}")
    print()

    # ═══ 2. HuggingFace ═══
    print("━" * 60)
    print("  🤗 Фаза 2/3: HuggingFace датасеты (код + чат + агенты)")
    print("━" * 60)
    try:
        from download_hf_dataset import download_preset
        download_preset("all")
    except ImportError:
        print("  ⚠ Библиотека 'datasets' не установлена.")
        print("  Выполните: pip install datasets")
    except Exception as e:
        print(f"  ⚠ HuggingFace: {e}")
    print()

    # ═══ 3. LEANN память ═══
    print("━" * 60)
    print("  🧠 Фаза 3/3: Загрузка в LEANN (векторная память)")
    print("━" * 60)
    try:
        from ingest_to_leann import ingest_all
        ingest_all()
    except Exception as e:
        print(f"  ⚠ LEANN: {e}")
        print("  Запустите вручную: python training/ingest_to_leann.py")
    print()

    print("═" * 60)
    print("  ✅ Загрузка завершена!")
    print("  Обучение: python training/train_mamba2.py --phase 1")
    print("═" * 60)


if __name__ == "__main__":
    main()
