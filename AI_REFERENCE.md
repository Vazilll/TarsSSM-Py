# 🤖 TARS HELIX LITE — Справочник для ИИ-агентов

> **Цель:** Этот документ — стартовая точка для любого ИИ, работающего с проектом.
> **Дата:** 2026-03-08 | **Архитектура:** HELIX LITE v6 | **Модель:** 116M–296M params

---

## 📁 СТРУКТУРА ПРОЕКТА

```
TarsSSM-Py/
├── config.py                     # TarsConfig — ВСЕ гиперпараметры модели
├── local_train.py                # 🎯 ГЛАВНЫЙ скрипт обучения (Colab + Local)
├── train_lite.py                 # Альтернативный тренировочный скрипт
├── launch_tars.py                # Запуск инференса
├── requirements.txt              # Зависимости
│
├── brain/                        # 🧠 ЯДРО АРХИТЕКТУРЫ
│   ├── tokenizer.py              #   Токенизатор (BPE + UTF-8 byte fallback)
│   ├── doubt_engine.py           #   Движок сомнений (safety/coherence/repeat)
│   ├── speculative.py            #   EAGLE-3 спекулятивная генерация
│   ├── rrn.py                    #   Reasoning Network
│   ├── mamba2/core/
│   │   ├── ssd.py                #   ⭐ TarsCoreBlock (SSD + WKV + Fusion)
│   │   ├── model_lite.py         #   ⭐ TarsHelixLite (основная модель)
│   │   ├── brain_core.py         #   Полное ядро Brain (Phase 2+)
│   │   ├── model.py              #   Полная модель (Phase 2+)
│   │   ├── bitnet.py             #   BitNet квантизация (1.58-bit)
│   │   └── tars_block.py         #   HelixBlock обёртка
│   ├── min_gru/                  #   MinGRU (Mode Router: REFLEX/THINK/DEEP)
│   ├── omega_core/               #   Omega контекст-менеджер
│   ├── reflexes/                 #   Рефлекторные быстрые ответы
│   └── spiking/                  #   SNN экспериментальное
│
├── training/                     # 📚 ОБУЧЕНИЕ
│   ├── data/
│   │   └── download_hf_dataset.py  # Скачивание HF датасетов (9 пресетов)
│   ├── train/
│   │   └── train_mamba2.py       # ⚠️ LEGACY (старая модель TarsMamba2LM)
│   └── umot_loss.py              # UMOT Loss (CE/SFT/DPO/Safety/Personality)
│
├── memory/                       # Память (SDM, Kanerva, контекст)
├── core/                         # C++ ядра (матрицы, SSD scan, RMSNorm)
├── agent/                        # Agent OS (тулы, безопасность)
├── sensory/                      # Ввод/вывод (парсинг, рендер)
├── tools/                        # Внешние инструменты (file, code, web)
├── tests/                        # Тесты
├── data/                         # Данные для обучения (.txt файлы)
├── models/                       # Чекпоинты (symlink → Drive)
└── ui/                           # Пользовательский интерфейс
```

---

## ⭐ КЛЮЧЕВЫЕ ФАЙЛЫ (читать в первую очередь)

| Приоритет | Файл | Что содержит |
|:---------:|:------|:-------------|
| 1 | `config.py` | `TarsConfig` — все гиперпараметры: d_model, n_layers, vocab_size, d_state… |
| 2 | `brain/mamba2/core/ssd.py` | `TarsCoreBlock` — SSD scan + WKV-7 + WuNeng Fusion. **Сердце архитектуры.** |
| 3 | `brain/mamba2/core/model_lite.py` | `TarsHelixLite` — Embedding → N×HelixBlock → RMSNorm → LM Head |
| 4 | `local_train.py` | Обучение: auto HF download, UTF-8 byte tokenizer, checkpointing, Drive backup |
| 5 | `TARS_HELIX_GOLDEN.md` | Полная спецификация архитектуры (700 строк) |

---

## 🧬 АРХИТЕКТУРА HELIX LITE

### Формула одного блока:

```
Input x (d_model=1024)
  │
  ├─ RMSNorm(x)
  │
  ├─ SSD Path:   s_{t+1} = γ_t · s_t + B_t · x_t       (Mamba-2)
  │               y_ssd   = C_t^T · s_{t+1} + D · x_t
  │
  ├─ WKV Path:   β_t = σ(W_β · k_t)                       (Gated DeltaNet)
  │               w_gated = β_t ⊙ w_t, α_gated = (1-β_t) ⊙ α_t
  │               Δ_t = v_t − S_{t-1} · k_t
  │               S_t = w_gated ⊙ S_{t-1} + α_gated · k_t ⊗ Δ_t
  │               y_wkv = S_t · r_t
  │
  ├─ Fusion:     gate = σ(W_up · GELU(W_down · [y_ssd; y_wkv]))   (2048→192→d_inner→σ)
  │               y = gate ⊙ y_ssd + (1-gate) ⊙ y_wkv             (гейтированное слияние)
  │
  ├─ SwiGLU FFN: h = SiLU(W₁·x) ⊙ W₂·x; out = W₃·h
  │
  └─ Residual:   output = x + out
```

### Конфигурация по умолчанию (TarsConfig):

```python
d_model      = 1024    # Скрытая размерность
n_layers     = 20      # 20 HelixBlocks
vocab_size   = 48256   # 48K текст + 256 управление (Phase 2+)
d_state      = 64      # SSM state dimension
headdim      = 64      # 16 голов × 64
expand       = 2       # SSD expansion (d_inner = 2048)
dim_ff       = 2816    # SwiGLU FFN hidden (~2.75×d_model)
wkv_low_rank = 24      # WKV Low-Rank
```

### Режим LITE тренировки (текущий):

```python
vocab_size   = 256     # UTF-8 byte-level (без BPE)
d_model      = 768     # На T4 (15GB VRAM)
n_layers     = 16      # На T4
# Итого: ~116M параметров
```

---

## ✅ КОМАНДЫ ВЕРИФИКАЦИИ

### 1. Smoke Test (быстрая проверка):
```bash
python local_train.py --test-only
```
**Ожидаемый результат:**
- `SMOKE TEST PASSED`
- Forward: loss ~5.5 (для vocab=256)
- Backward: без ошибок
- Generate: тензор расширяется

### 2. Подсчёт параметров:
```bash
python local_train.py --count-params
```
**Ожидаемый результат:**
- Полный конфиг (1024d, 20L, V=48256): ~296M params
- Lite конфиг (768d, 16L, V=256): ~116M params

### 3. Обучение на реальных данных:
```bash
# Скачать данные
python training/data/download_hf_dataset.py --preset chat

# Обучение (small = ~15 мин)
python local_train.py --level small

# Обучение (medium = ~2-3 часа)
python local_train.py --level medium --drive colab
```
**Ожидаемый результат:**
- HF скачивает 2 русских датасета (~22 MB)
- Loss стартует ~5.5, падает до ~3-4 за medium
- Чекпоинты сохраняются каждые 30 мин

### 4. Тест tokenizer:
```python
from brain.tokenizer import TarsTokenizer
tok = TarsTokenizer(mode="utf8")
encoded = tok.encode("Привет, ТАРС!")
decoded = tok.decode(encoded)
assert decoded == "Привет, ТАРС!"
```

---

## 🔧 ТЕКУЩНЕ ПРОБЛЕМЫ (Март 2026)

| # | Проблема | Статус | Описание |
|:-:|:---------|:------:|:---------|
| 1 | NaN при загрузке старых чекпоинтов | 🔧 Фикс | Старые чекпоинты (vocab=48256) несовместимы с byte-level (vocab=256). Авто-детекция добавлена. |
| 2 | `train_mamba2.py` использует старую модель | ⚠️ Legacy | Использует `TarsMamba2LM`, не `TarsHelixLite`. Не трогать, это legacy. |
| 3 | BPE tokenizer не обучен | 📋 Planned | Сейчас UTF-8 byte-level (vocab=256). BPE (vocab=48K) — Phase 2. |
| 4 | Graduated 8-8-8 layout | 📋 Planned | Сейчас ВСЕ блоки uniform (SSD+WKV). Graduated layout — Phase 2. |

---

## 📊 ДАННЫЕ ДЛЯ ОБУЧЕНИЯ

### Автоматическое скачивание (HF Presets):

| Preset | Описание | Размер |
|:-------|:---------|:-------|
| `chat` | Русские диалоги (GrandMaster, Russian Instructions) | ~22 MB |
| `code` | Код (StarCoder, CodeAlpaca) | ~50 MB |
| `math` | Математика (GSM8K, MATH) | ~15 MB |
| `reasoning` | Рассуждения (Chain-of-Thought) | ~30 MB |
| `safety` | Безопасность (Red/Green responses) | ~10 MB |
| `general` | Общие знания (Wikipedia, OpenAssistant) | ~100 MB |

```bash
# Скачать конкретный пресет:
python training/data/download_hf_dataset.py --preset chat --count 5000

# Все пресеты:
python training/data/download_hf_dataset.py --preset all
```

Файлы сохраняются в `data/hf_*.txt`.

---

## 🏗️ ПРАВИЛА ДЛЯ ИИ-АГЕНТОВ

### DO ✅

1. **Всегда используй `TarsHelixLite`** для обучения и инференса HELIX LITE
2. **Всегда используй `TarsConfig`** — все параметры модели определены там
3. **Для тренировки используй `local_train.py`** — это единственный актуальный скрипт
4. **Проверяй forward pass** перед запуском обучения (smoke test)
5. **Проверяй NaN** в чекпоинтах при загрузке
6. **UTF-8 byte-level** (vocab=256) для текущего обучения
7. **Запускай `--test-only`** после любых изменений в модели

### DON'T ❌

1. **НЕ используй `TarsMamba2LM`** — это устаревшая модель
2. **НЕ используй `train_mamba2.py`** — это legacy скрипт для старой архитектуры
3. **НЕ меняй формулы в `ssd.py`** без понимания SSD/WKV математики
4. **НЕ ставь vocab_size=48256** для LITE тренировки (используй 256)
5. **НЕ загружай optimizer state** из старых чекпоинтов при смене архитектуры
6. **НЕ используй dummy optimizer.step()** перед scheduler (портит веса!)
7. **НЕ трогай** `TARS_HELIX_GOLDEN.md` — это спецификация, не код

---

## 🧪 ЧЕКЛИСТ ПЕРЕД ОБУЧЕНИЕМ

```
[ ] config.py: TarsConfig параметры корректны
[ ] model_lite.py: TarsHelixLite использует TarsCoreBlock из ssd.py
[ ] local_train.py --test-only: SMOKE TEST PASSED
[ ] data/*.txt: текстовые файлы для обучения существуют
[ ] models/tars_lite/: нет старых/несовместимых чекпоинтов
[ ] GPU/VRAM: достаточно для выбранного уровня
[ ] AMP: fp16 для T4, bf16 для A100/H100
```

---

## 📐 МАТЕМАТИКА (ключевые формулы)

### SSD (Mamba-2):
```
γ_t = FlashSigmoid(W_γ · x_t) = 0.5 + 0.25 · (W_γ · x_t)
s_{t+1} = γ_t ⊙ s_t + B_t · x_t
y_ssd = C_t^T · s_{t+1} + D · x_t
```

### WKV-7 (RWKV-7 GatedDeltaNet):
```
w_t = exp(-exp(W_w · x_t + w_gate))   — per-dim vector decay (double-exp)
β_t = σ(W_β · k_t)                    — gated balance
w_gated = β_t ⊙ w_t                   — gated decay (high β → preserve)
α_gated = (1-β_t) ⊙ α_t               — gated learning rate (low β → learn)
Δ_t = v_t − S_{t-1} · k_t             — prediction error
S_t = w_gated ⊙ S_{t-1} + α_gated · k_t ⊗ Δ_t  — state update
y_wkv = S_t · r_t                      — readout
```

### WuNeng Fusion (Gated Bottleneck Blend):
```
gate = σ(W_up · GELU(W_down · [y_ssd; y_wkv_up]))
       d_inner←192     192←2*d_inner    (информационное горлышко)
y_fused = gate ⊙ y_ssd + (1 - gate) ⊙ y_wkv_up
```

### RMSNorm:
```
x̂ = γ · x / √(mean(x²) + ε)
```

### Embedding:
```
E[token_id] × √d_model    (Vaswani scaling)
```

---

## 🔗 СВЯЗАННЫЕ ДОКУМЕНТЫ

| Документ | Описание |
|:---------|:---------|
| `TARS_HELIX_GOLDEN.md` | Полная архитектура (режимы, память, ночной цикл) |
| `Tars_TZ_v3.md` | Техническое задание v3 (88 дебатов экспертов) |
| `TARS_5_AGENTS_PLAN.md` | План 5 агентов параллельной разработки |
| `TARS_FORMULA_INTEGRATION_MAP.md` | Карта интеграции 78 формул |
| `TARS_HELIX_v6_DEFINITIVE.md` | Финальная версия архитектуры v6 |

---

> 🧬 **TARS HELIX LITE: Dual-SSM (SSD + WKV), WuNeng Fusion, UTF-8 byte-level.**
> **Один скрипт `local_train.py`. Одна модель `TarsHelixLite`. Один конфиг `TarsConfig`.**
