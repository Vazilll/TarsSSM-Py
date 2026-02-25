# TARS v3 — Обучение моделей (Deep WuNeng Core)

Все скрипты для обучения и дообучения моделей TARS.

---

## Быстрый старт

```powershell
cd C:\Users\Public\Tarsfull\TarsSSM-Py

# Всё за один раз (reflex + mingru + mamba2)
venv\Scripts\python.exe training\train_all.py

# Или по отдельности:
venv\Scripts\python.exe training\train_reflex.py     # 30 сек, CPU
venv\Scripts\python.exe training\train_mingru.py     # 5 мин, CPU
venv\Scripts\python.exe training\train_mamba2.py     # 30 мин+, GPU

# С GPU:
venv\Scripts\python.exe training\train_all.py --device cuda
```

---

## Структура файлов

| Файл | Что обучает | Время | Устройство |
|------|-------------|-------|-----------|
| `train_reflex.py` | ReflexClassifier (MinGRU intent) | ~30 сек | CPU |
| `train_mingru.py` | MinGRU Language Model (System 1) | ~5 мин | CPU/GPU |
| `train_mamba2.py` | **TarsMamba2LM** (Mamba-2 + RWKV-7 + Ω-SSM + MoLE) | 30 мин+ | GPU |
| `train_all.py` | Все компоненты последовательно | ~40 мин | auto |
| `train_corpus.py` | Утилита генерации корпуса | - | - |
| `quantize_models.py` | BitNet квантизация (1.58-bit) | ~5 мин | CPU |

---

## Mamba-2 + RWKV-7 (Deep WuNeng Core)

### Архитектура
```
Каждый TarsBlock содержит TarsCoreBlock (ssd.py):
  x → Shared In-Proj → ┌─ Conv1D → SSD scan (recall)
                        └─ WKV scan (state tracking)
                        → Deep Gated Fusion (WuNeng)
                        → Shared Out-Proj → y
```

### 4 фазы обучения

| Фаза | Что обучается | Что заморожено | LR |
|------|--------------|----------------|----|
| 1 | ВСЁ (TarsCoreBlock + Ω-SSM + MoLE) | — | 3e-4 |
| 2 | WKV + Fusion + Ω-SSM + MoLE | SSD (A_log, D, dt_bias, conv1d) | 3e-5 |
| 3 | MoLE + MatrixPool | Всё остальное | 3e-6 |
| 4 | WKV fusion + time_mix + RAG proj | Всё остальное | 1.5e-5 |

### Команды

```powershell
# Фаза 1: Полный претрейн
venv\Scripts\python.exe training\train_mamba2.py --phase 1 --epochs 3

# Фаза 1 с кастомным корпусом (например, Википедия):
venv\Scripts\python.exe training\train_mamba2.py --phase 1 --data data\wiki_ru.txt --epochs 5

# Фаза 2: Дообучение WKV + Fusion (на кастомном датасете)
venv\Scripts\python.exe training\train_mamba2.py --phase 2 --pretrained models\mamba2\best.pt --epochs 3

# Фаза 4: Дообучение WKV для RAG State Tracking
venv\Scripts\python.exe training\train_mamba2.py --phase 4 --epochs 5

# Полноразмерная модель (768d, 12 слоёв, ~130M параметров)
venv\Scripts\python.exe training\train_mamba2.py --d_model 768 --n_layers 12 --device cuda
```

---

## Подготовка данных для обучения

### Формат корпуса
Обычный текстовый файл `.txt` в UTF-8:
```
Любой текст на русском языке.
Каждая строка — отдельный абзац.
Рекомендуемый размер: 10+ МБ для качественного результата.
```

### Скачивание Википедии (русской)
```powershell
# 1. Скачать дамп (только текст статей, ~4 GB сжатый)
Invoke-WebRequest -Uri "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles.xml.bz2" -OutFile data\ruwiki.xml.bz2

# 2. Извлечь текст через WikiExtractor
pip install wikiextractor
python -m wikiextractor.WikiExtractor data\ruwiki.xml.bz2 -o data\wiki_extracted --json

# 3. Собрать в один файл
python -c "
import json, glob, os
texts = []
for f in sorted(glob.glob('data/wiki_extracted/**/*', recursive=True)):
    if os.path.isfile(f):
        for line in open(f, encoding='utf-8'):
            try:
                d = json.loads(line)
                if len(d.get('text','')) > 100:
                    texts.append(d['text'])
            except: pass
with open('data/wiki_ru.txt', 'w', encoding='utf-8') as out:
    out.write('\n\n'.join(texts))
print(f'Saved {len(texts)} articles')
"

# 4. Обучить на Википедии
venv\Scripts\python.exe training\train_mamba2.py --phase 1 --data data\wiki_ru.txt --epochs 3
```

---

## Unified Pipeline (train_all.py)

```powershell
# Полный пайплайн (Omega → Reflex → MinGRU → Mamba-2 → Quantize)
venv\Scripts\python.exe training\train_all.py

# Только одна модель
venv\Scripts\python.exe training\train_all.py --only reflex
venv\Scripts\python.exe training\train_all.py --only mingru
venv\Scripts\python.exe training\train_all.py --only mamba2
venv\Scripts\python.exe training\train_all.py --only mamba2 --phase 4

# С кастомным корпусом и GPU
venv\Scripts\python.exe training\train_all.py --data data\wiki_ru.txt --device cuda --mamba-epochs 10

# Пропустить OmegaCore компиляцию
venv\Scripts\python.exe training\train_all.py --skip-omega
```

### Аргументы train_all.py

| Аргумент | По умолчанию | Описание |
|----------|-------------|----------|
| `--only` | ALL | reflex / mingru / mamba2 / quantize |
| `--phase` | 1 | Фаза обучения Mamba-2 (1-4) |
| `--device` | auto | cpu / cuda / auto |
| `--data` | built-in | Путь к текстовому корпусу |
| `--d_model` | 256 | Размерность модели (256=demo, 768=full) |
| `--n_layers` | 4 | Число TarsBlock (4=demo, 12=full) |
| `--mamba-epochs` | 3 | Эпохи Mamba-2 |
| `--reflex-epochs` | 100 | Эпохи Reflex |
| `--skip-omega` | false | Пропустить C++ компиляцию |

---

## Веса моделей

После обучения веса сохраняются в `models/`:
```
models/
  mamba2/
    mamba2_omega.pt    # Основной мозг (Mamba-2 + RWKV-7)
    best.pt            # Лучший чекпоинт
  reflex/
    reflex.pt          # ReflexClassifier
  mingru/
    mingru_lm.pt       # MinGRU Language Model
```
