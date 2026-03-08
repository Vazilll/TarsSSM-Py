# 🧬 TARS HELIX — ЗОЛОТОЙ СТАНДАРТ АРХИТЕКТУРЫ

> **Версия:** Golden v1 — синтез лучшего из TZ v3 (88 дебатов), DEFINITIVE v6, FORMULA MAP (78 формул), 5 AGENTS PLAN
> **Дата:** 2026-03-08
> **Статус:** ЕДИНСТВЕННЫЙ ВАЛИДНЫЙ ДОКУМЕНТ для имплементации
> **Принцип:** Нейронка = ВСЯ логика. Код = ТОЛЬКО вычисления. 0 if/else на hot path.

---

## ⚡ ПАРАМЕТРЫ СИСТЕМЫ

```
╔══════════════════════════════════════════════════════════════════╗
║  🧬 TARS HELIX — GOLDEN ARCHITECTURE                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ═══ МОДЕЛЬ ═══                                                  ║
║  Phase 1-2:  TARS-Base   380M dense, 20 HelixBlocks             ║
║  Phase 3+:   TARS-Full   450M dense, 24 HelixBlocks             ║
║  d_model:    1024                                                ║
║  n_heads:    16 (head_dim = 64)                                  ║
║  Precision:  Ternary {-1, 0, +1} (1.58-bit weights)            ║
║  Activations: INT8 (THINK), INT4 (REFLEX)                      ║
║  Vocab:      48,256 (48K text + 256 tool/control tokens)        ║
║  Context:    ∞ (SSM linear recurrence, NO KV cache)             ║
║  Embedding:  INT8, weight-tied with LM head                     ║
║                                                                  ║
║  ═══ SSM ДВОЙНАЯ СПИРАЛЬ ═══                                    ║
║  SSD (Mamba-2):  Structured State-Space Duality, d_state=64     ║
║  WKV (RWKV-7):   GatedDeltaNet, Low-Rank r=24                  ║
║  Layout:         Graduated 8-8-8                                 ║
║    Blocks 0-7:   SSD ONLY (быстрые паттерны)                    ║
║    Blocks 8-15:  Full Dual (SSD + WKV параллельно)              ║
║    Blocks 16-19: WKV ONLY (глубокий контекст)                   ║
║  Fusion:         WuNeng Bottleneck (2048→192→1024, GELU)        ║
║                                                                  ║
║  ═══ СПЕЦИАЛИЗАЦИЯ ═══                                           ║
║  MoLE LoRA:  Phase 1: 4 experts × rank=16                       ║
║              Phase 3: 8 experts × rank=8                         ║
║              Personality expert: always-on α=0.3                 ║
║  SwiGLU FFN: dim_ff = 2816, Double Sparsity (87% zeros)        ║
║                                                                  ║
║  ═══ СКОРОСТЬ ═══                                                ║
║  REFLEX:  50-70 tok/s (INT4, skip 50% blocks)                   ║
║  THINK:   30-45 tok/s (INT8, all blocks)                        ║
║  DEEP:    15-25 tok/s (INT8, full memory access)                ║
║  TTFT:    <2ms (SSM State Cache for system prompt)              ║
║  Engine:  C++/Rust ядро (AVX2/VNNI), Python = orchestration     ║
║                                                                  ║
║  ═══ RAM ═══                                                     ║
║  Hard limit:     700 MB                                          ║
║  Model (mmap):   ~100 MB (ternary packed)                       ║
║  Embeddings:     ~48 MB (INT8)                                   ║
║  SSM states:     ~9 MB (FP16)                                    ║
║  Memory (SDM+):  ~80 MB                                          ║
║  Runtime + OS:   ~80 MB                                          ║
║  Average active: ~400 MB                                         ║
║  Headroom:       ~130 MB                                         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🔄 КАК РАБОТАЕТ TARS — ПОЛНЫЙ ЦИКЛ ПО БЛОКАМ

---

### БЛОК 0: СЛУШАТЕЛЬ (всегда активен, ~150 MB RAM)

```
╔═══════════════════════════════════════════════════╗
║  🔇 SLEEP MODE — B = 0                           ║
║                                                   ║
║  ТАРС "спит", но listener работает:              ║
║  ┌─────────────────────────────┐                  ║
║  │  stdin / IME / Wake Word    │                  ║
║  │  Polling: 100ms interval    │                  ║
║  │  RAM: ~150 MB (mmap cold)   │                  ║
║  └─────────────┬───────────────┘                  ║
║                │                                  ║
║    Пользователь начал печатать...                 ║
║                │                                  ║
║                ▼                                  ║
║  ┌─────────────────────────────┐                  ║
║  │  madvise(WILLNEED)         │  ← keyboard      ║
║  │  Prefetch model pages      │     event         ║
║  │  18ms async (NVMe)         │     triggers      ║
║  └─────────────┬───────────────┘     warm-up      ║
║                │                                  ║
║    К моменту когда юзер допечатает (500-2000ms)  ║
║    все страницы модели уже в RAM.                ║
║                ▼                                  ║
╚═══════════════════════════════════════════════════╝
```

---

### БЛОК 1: ВХОД — Санитизация + Токенизация

```
USER: "Напиши unittest для calculate_tax()"
         │
         ▼
┌─────────────────────────────────────────┐
│  INPUT SANITIZER                        │
│                                         │
│  1. NFC нормализация (Unicode)          │
│  2. Удаление zero-width символов        │
│  3. Emoji → text (если нужно)           │
│  4. Binary detection (magic bytes)      │
│                                         │
│  Результат: чистая строка UTF-8         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  TOKENIZER (Qwen BPE, 48,256 vocab)    │
│                                         │
│  "Напиши" "unit" "test" "для"          │
│  "calculate" "_" "tax" "()"             │
│  → [4521, 892, 1037, 203, ...]         │
│                                         │
│  C++/Rust: ~1-2ms                       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  EMBEDDING LOOKUP (INT8, weight-tied)   │
│                                         │
│  E[token_id] ∈ ℝ^1024                  │
│  × √d_model = × 32 (Vaswani scaling)   │
│                                         │
│  48,256 × 1024 × 1 byte = 48 MB        │
│  Shared with LM head (weight tying)     │
└────────────────┬────────────────────────┘
                 │
                 ▼
```

---

### БЛОК 2: MODE ROUTER — Рефлекторная классификация (0.05ms)

```
┌══════════════════════════════════════════════════════════════════┐
│  🧠 MODE ROUTER — MinGRU classifier                            │
│  ═══════════════════════════════════════════════                 │
│                                                                  │
│  НЕЙРОНКА. Не if/else. Обучена на примерах.                     │
│                                                                  │
│  MinGRU (d=512, 2 layers, ~1M params):                          │
│    z = σ(W_z · x + U_z · h_{t-1})          — update gate       │
│    h̃ = W_h · x + U_h · (z ⊙ h_{t-1})      — candidate        │
│    h_t = (1-z) ⊙ h_{t-1} + z ⊙ h̃           — new state       │
│                                                                  │
│  → mode_logits = W_mode · h_final                               │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │  "привет"              → REFLEX  (0.05ms exit)   │           │
│  │  "напиши unittest..."  → THINK   (full Brain)    │           │
│  │  "проанализируй код    → DEEP    (full Brain     │           │
│  │   и найди все баги"               + full memory)  │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
│  REFLEX → shared LM head (proj 512→1024) → ответ ~2000 tok/s   │
│  THINK/DEEP → продолжаем в основной Brain ▼                     │
└══════════════════════════════════════════════════════════════════┘
         │
         ▼
  (if REFLEX → skip Brain entirely, emit via MinGRU LM head)
  (if THINK/DEEP → continue below)
```

---

### БЛОК 3: CONTEXT MANAGER — Управление окном

```
┌─────────────────────────────────────────────────────┐
│  CONTEXT MANAGER (8K token window)                  │
│                                                     │
│  if len(context) > 0.75 × 8192:                     │
│    summary = model.summarize(context[:half])         │
│    SDM.store(key=embed(summary), val=old_turns)      │
│    context = [system_prompt, summary, recent_turns]  │
│                                                     │
│  SSM State Cache: если system_prompt не изменился:  │
│    → загрузить cached SSM states                    │
│    → TTFT ≈ 0 (не нужен prefill!)                   │
│                                                     │
│  SDM Memory Recall:                                  │
│    query = embed(user_input)                         │
│    Matryoshka 2-stage:                               │
│      Fast: e[:64] scan 30K slots → top-50           │
│      Full: e[:384] re-rank top-50 → top-5           │
│    memories = SDM.read(query, top_k=5)               │
│    → inject into context: [system, memory, history]  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
```

---

### БЛОК 4: BRAIN — 20 HelixBlocks (основной цикл, ~10-25ms/tok)

```
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║  ARENA ALLOCATOR: 80 MB pre-allocated. Zero malloc. Zero frag.      ║
║  alloc(size) = pointer += size  →  O(1)                             ║
║  reset() = pointer = 0          →  O(1) per generation              ║
║                                                                      ║
║  ══════════════════════════════════════════════════                   ║
║  ►►  GRADUATED 8-8-8 HELIX LOOP  (20 blocks, TARS-Base)            ║
║  ══════════════════════════════════════════════════                   ║
║                                                                      ║
║  ┌────── BLOCKS 0-7: SSD ONLY (быстрые паттерны) ─────────────┐   ║
║  │                                                              │   ║
║  │  Для каждого block:                                          │   ║
║  │                                                              │   ║
║  │  1. RMSNorm:  x̂ = γ · x / √(mean(x²) + ε)                │   ║
║  │               Fused C++ kernel, one pass.                    │   ║
║  │                                                              │   ║
║  │  2. SSD SCAN (Mamba-2):                                      │   ║
║  │     ┌──────────────────────────────────────────────┐        │   ║
║  │     │  γ_t = FlashSigmoid(W_γ · x_t)             │        │   ║
║  │     │      = 0.5 + 0.25 · (W_γ · x_t)  ← no exp!│        │   ║
║  │     │                                              │        │   ║
║  │     │  s_{t+1} = γ_t · s_t + B_t · x_t           │        │   ║
║  │     │  ~~~~~~~~  ~~~~  ~~~   ~~~  ~~~             │        │   ║
║  │     │  new      decay old   input  data           │        │   ║
║  │     │  state    gate  state proj   token          │        │   ║
║  │     │                                              │        │   ║
║  │     │  y_ssd = C_t^T · s_{t+1} + D · x_t         │        │   ║
║  │     │  ~~~~~   ~~~~    ~~~~~~~   ~   ~~~          │        │   ║
║  │     │  output  read    state    skip  direct      │        │   ║
║  │     │          proj            conn   path        │        │   ║
║  │     │                                              │        │   ║
║  │     │  d_state = 64. Scalar-Identity A.           │        │   ║
║  │     │  16 heads × 64 state = 1024×64 FP16 = 128K │        │   ║
║  │     │  BitNet weights: {-1,0,+1} → ADD/SUB only! │        │   ║
║  │     └──────────────────────────────────────────────┘        │   ║
║  │                                                              │   ║
║  │  3. SwiGLU FFN + Double Sparsity:                            │   ║
║  │     h = SiLU(W₁·x) ⊙ W₂·x       ← gated activation       │   ║
║  │     mask = (|h| > ε)               ← 87% zeros!             │   ║
║  │     W₃_active = W₃[:, nonzero]    ← sparse matmul          │   ║
║  │     → 87% операций ПРОПУЩЕНО (BitNet 67% × act 60%)        │   ║
║  │                                                              │   ║
║  │  4. MoLE LoRA Injection:                                     │   ║
║  │     scores = softmax(W_r · [h; personality_emb])            │   ║
║  │     top2 = argmax_2(scores)                                  │   ║
║  │     h += Σ (score_i · B_i · A_i · h)    ← LoRA rank=16    │   ║
║  │     Personality expert: ALWAYS α=0.3, not routed.            │   ║
║  │                                                              │   ║
║  │  5. Residual: output = h + x_input                           │   ║
║  │                                                              │   ║
║  │  NO WKV. NO fusion. PURE SSD. Быстро. Локально.            │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ┌────── BLOCKS 8-15: FULL DUAL (интеграция SSD + WKV) ────────┐  ║
║  │                                                              │   ║
║  │  *** ЗДЕСЬ РОЖДАЕТСЯ HELIX — две спирали работают вместе ***│   ║
║  │                                                              │   ║
║  │  1. RMSNorm                                                  │   ║
║  │                                                              │   ║
║  │  2. ПАРАЛЛЕЛЬНО:                                             │   ║
║  │     ┌────────────────────┐  ┌────────────────────────┐      │   ║
║  │     │  SSD PATH          │  │  WKV-7 PATH (RWKV-7)  │      │   ║
║  │     │  s = γ·s + B·x    │  │                        │      │   ║
║  │     │  y_ssd = C^T·s+Dx │  │  w = σ(W_w · x_t)     │      │   ║
║  │     │                    │  │  per-dim vector decay  │      │   ║
║  │     │  Захватывает:      │  │                        │      │   ║
║  │     │  • синтаксис       │  │  S = w⊙S + α·k⊗Δ     │      │   ║
║  │     │  • локальный       │  │  Δ = v − β·S·k        │      │   ║
║  │     │    порядок          │  │  ^^^^^^^^^^^^^^^^^^^^^│      │   ║
║  │     │  • паттерны кода   │  │  Covariance Delta Rule│      │   ║
║  │     │                    │  │  (second-order update!)│      │   ║
║  │     │                    │  │                        │      │   ║
║  │     │                    │  │  Захватывает:          │      │   ║
║  │     │                    │  │  • долгий контекст     │      │   ║
║  │     │                    │  │  • ассоциации          │      │   ║
║  │     │                    │  │  • тему разговора      │      │   ║
║  │     │                    │  │                        │      │   ║
║  │     │                    │  │  Low-Rank r=24         │      │   ║
║  │     │                    │  │  16h × 64 × 24 × 2    │      │   ║
║  │     └────────┬───────────┘  └────────────┬───────────┘      │   ║
║  │              │                            │                  │   ║
║  │              └──────────┬─────────────────┘                  │   ║
║  │                         │                                    │   ║
║  │                         ▼                                    │   ║
║  │  3. WuNeng FUSION — информационное горлышко:                │   ║
║  │     ┌────────────────────────────────────────────────┐      │   ║
║  │     │                                                │      │   ║
║  │     │  concat = [y_ssd; y_wkv]    ← 2048 dims      │      │   ║
║  │     │        ↓                                       │      │   ║
║  │     │  compressed = W_down · concat  ← 2048→192     │      │   ║
║  │     │        ↓                    ГОРЛЫШКО!          │      │   ║
║  │     │  activated = GELU(compressed)  ← нелинейность │      │   ║
║  │     │        ↓                                       │      │   ║
║  │     │  y_fused = W_up · activated    ← 192→1024     │      │   ║
║  │     │                                                │      │   ║
║  │     │  СУТЬ: модель ВЫНУЖДЕНА извлечь               │      │   ║
║  │     │  квинтэссенцию из обоих потоков, сжав          │      │   ║
║  │     │  2048→192. Как два глаза дают объём —           │      │   ║
║  │     │  два SSM дают ОБЪЁМНОЕ ПОНИМАНИЕ.              │      │   ║
║  │     │                                                │      │   ║
║  │     │  Params: 2048×192 + 192×1024 = 590K/block     │      │   ║
║  │     └────────────────────────────────────────────────┘      │   ║
║  │                                                              │   ║
║  │  4. SwiGLU FFN + Double Sparsity (same as SSD blocks)      │   ║
║  │  5. MoLE LoRA Injection                                     │   ║
║  │  6. Residual                                                 │   ║
║  │                                                              │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ┌────── BLOCKS 16-19: WKV ONLY (глубокий контекст) ────────────┐  ║
║  │                                                              │   ║
║  │  Только WKV-7 (RWKV-7 GatedDeltaNet).                      │   ║
║  │  NO SSD. NO fusion. PURE контекстуальное понимание.         │   ║
║  │                                                              │   ║
║  │  Здесь модель "вспоминает" всё что знает:                   │   ║
║  │  • WKV state интегрирует ВЕСЬ разговор                      │   ║
║  │  • Covariance Delta Rule = second-order memory              │   ║
║  │  • Low-Rank r=24 → компактно, но ёмко                      │   ║
║  │                                                              │   ║
║  │  + SwiGLU FFN + MoLE + Residual (как обычно)               │   ║
║  │                                                              │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║  ══════════════════════════════════════════════════                   ║
║  EARLY EXIT (после block 8 и 16):                                   ║
║    confidence = max(softmax(E^T · h))                               ║
║    if confidence > 0.9: EXIT → emit token                           ║
║    "привет" → exits at block 8 (0 compute wasted!)                  ║
║    "напиши QuickSort" → all 20 blocks (максимум думания)           ║
║  ══════════════════════════════════════════════════                   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
         │
         ▼
```

---

### БЛОК 5: LM HEAD + SAMPLING — Генерация следующего токена

```
┌══════════════════════════════════════════════════════════════════┐
│  LM HEAD (weight-tied, INT8)                                    │
│                                                                  │
│  logits = E^T · h_final     ← те же embeddings, transposed     │
│  logits ∈ ℝ^48,256          ← один скор на каждый токен        │
│                                                                  │
│  EAGLE-3 Speculative Decoding:                                   │
│  ┌───────────────────────────────────────────────┐              │
│  │  Из block 10: predict 3 EXTRA токена          │              │
│  │  Verify all 4 в ОДНОМ forward pass            │              │
│  │  Accept rate ~75% → 2.5× speedup              │              │
│  │                                                │              │
│  │  Draft head: ~4M params, 0.05ms per draft      │              │
│  │  vs DFlash (8 passes × 1.2ms = 9.6ms)         │              │
│  │  → EAGLE-3 в 200× дешевле на CPU              │              │
│  └───────────────────────────────────────────────┘              │
│                                                                  │
│  SAMPLING:                                                       │
│    Top-p (nucleus): keep until cumsum(sorted_p) > 0.9           │
│    Temperature: T=0.7 (creative) or T=0.1 (code)               │
│    Repetition Penalty: logits[seen] /= 1.1                      │
│                                                                  │
│  → next_token_id                                                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
```

---

### БЛОК 6: DOUBT ENGINE — Проверка качества (НЕЙРОНКА)

```
┌══════════════════════════════════════════════════════════════════┐
│  🔍 DOUBT ENGINE — 3 нейронные головы на hidden state          │
│                                                                  │
│  НЕ if/else! Обучено контрастивно на хороших/плохих ответах.   │
│                                                                  │
│  coherence = σ(W_coh · h)   ∈ [0,1]  ← связность ответа       │
│  safety    = σ(W_saf · h)   ∈ [0,1]  ← безопасность            │
│  repeat    = σ(W_rep · h)   ∈ [0,1]  ← повторяемость           │
│                                                                  │
│  doubt = 1 − min(coherence, safety, 1−repeat)                   │
│                                                                  │
│  ┌────────────────────────────────────────┐                     │
│  │  doubt < 0.3  → EMIT token ✅         │                     │
│  │  doubt 0.3-0.7 → EMIT + log warning  │                     │
│  │  doubt > 0.7  → RETRY with T×0.5     │  ← resample!        │
│  │                  max 3 retries         │                     │
│  └────────────────────────────────────────┘                     │
│                                                                  │
│  Ensemble (Phase 2+):                                            │
│    CriticHead + DoubtEngine vote.                                │
│    Both agree = confident. Disagree = conservative.              │
│                                                                  │
│  Cost: 3 × Linear(1024, 1) = 3072 params. ~0.001ms.            │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
```

---

### БЛОК 7: TOOL DETECTION + EXECUTION (if applicable)

```
┌══════════════════════════════════════════════════════════════════┐
│  🔧 TOOL EXECUTION — Agent OS                                   │
│                                                                  │
│  Output tokens проверяются на tool_call pattern:                │
│  <tool_call>{"name": "file_search", "args": {...}}</tool_call>  │
│                                                                  │
│  ┌─── 4-LAYER SAFETY ─────────────────────────────────────┐    │
│  │                                                         │    │
│  │  L1: AGENT GUARDIAN (1MB classifier, НЕЙРОНКА)         │    │
│  │      → classify: READ/WRITE/EXECUTE/DESTRUCTIVE        │    │
│  │                                                         │    │
│  │  L2: SANDBOX CHECK                                      │    │
│  │      → whitelist folders. System32 = BLOCKED.          │    │
│  │                                                         │    │
│  │  L3: USER CONFIRMATION (if WRITE/DESTRUCTIVE)           │    │
│  │      → "⚠️ Удалить 47 файлов? [Да] [Нет] [Показать]"  │    │
│  │                                                         │    │
│  │  L4: AUDIT LOG (all actions, 7-day ring buffer)         │    │
│  │      → timestamp + action + file list + undo info      │    │
│  │                                                         │    │
│  │  KILL SWITCH: "ТАРС СТОП" → мгновенная остановка      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Tool result → inject back into context → continue generation   │
│  Tool timeout: 30s chain budget (consecutive calls)              │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
```

---

### БЛОК 8: MEMORY UPDATE — ТАРС запоминает

```
┌══════════════════════════════════════════════════════════════════┐
│  💾 MEMORY UPDATE (после каждого ответа)                        │
│                                                                  │
│  1. Kanerva SDM Write (30K slots, INT8, 50MB):                  │
│     key = embed(response_summary)                                │
│     cos_sim = key · addresses / ||...||                          │
│     top5 = argmax_5(cos_sim)                                     │
│     contents[top5] += α · (value − contents[top5])              │
│     strengths[top5] += 1.0                                       │
│     → Recall ≥ 93% for topics with 3+ entries                   │
│                                                                  │
│  2. Conversation Genome (90 days, compressed):                   │
│     append(user_intent, tool_results, emotion, timestamp)        │
│                                                                  │
│  3. User Twin Update (16-dim vector):                            │
│     EMA update: twin = 0.95·twin + 0.05·observed                │
│     Tracks: tech_level, interests, schedule, mood_pattern...     │
│                                                                  │
│  4. ADI Emotional Update (16-dim vector):                        │
│     satisfaction += observation                                   │
│     helpfulness += observation                                    │
│     → Adapts ТАРС personality over time                          │
│                                                                  │
│  Phase 2: + LEANN (25K docs, HNSW index, RETRO cross-attn)     │
│  Phase 3: + Titans (surprise-based, rank-1 SGD, async)          │
└══════════════════════════════════════════════════════════════════┘
```

---

### БЛОК 9: НОЧНОЙ ЦИКЛ — ТАРС учится сам (2:00 AM, ~2 часа)

```
╔══════════════════════════════════════════════════════════════════╗
║  🌙 NIGHT CYCLE — самообучение с гарантией не-деградации       ║
║                                                                  ║
║  2:00 AM. Юзер спит. ТАРС работает.                            ║
║  RAM: 650-690 MB (freed: caches, speculative buffers)           ║
║                                                                  ║
║  ═══ PHASE 1: ANALYSE (2 min) ═══                               ║
║  • Собрать все дневные conversations                             ║
║  • DoubtEngine.score(each_response) → quality                   ║
║  • User signals: corrections, re-asks, thumbs                    ║
║  • Sort: best → worst                                            ║
║                                                                  ║
║  ═══ PHASE 2: PRIVACY FILTER (2 min) ═══                        ║
║  • NER scan: имена, телефоны, адреса → TAG as SECRET            ║
║  • Regex: emails, credit cards, SSN → TAG as PRIVATE            ║
║  • ONLY PUBLIC-tagged data → training. NEVER private.            ║
║  • Memorization Audit: 50 private prompts → check no leak       ║
║                                                                  ║
║  ═══ PHASE 3: DREAM REPLAY (20 min) ═══                         ║
║  • Replay 200 best interactions through model                    ║
║  • Contrastive: good vs bad response pairs                       ║
║  • STC (Short-Term Consolidation) updates                        ║
║                                                                  ║
║  ═══ PHASE 4: DREAM TRAINING (15 min) ═══                       ║
║  • Generate 100 new dreams (temp=0.9, creative)                  ║
║  • DoubtEngine filter: keep 70% best                             ║
║  • Train on good dreams (LoRA only!)                             ║
║                                                                  ║
║  ═══ PHASE 5: SPIN SELF-PLAY (25 min) ═══                       ║
║  • 4 iterations:                                                  ║
║    1. Generate response                                           ║
║    2. DoubtEngine ranks: chosen vs rejected                      ║
║    3. DPO update: L = -log σ(β·(log_ratio))                     ║
║    4. LoRA update ONLY (base weights FROZEN!)                    ║
║  • η = 3.6e-4 (3e-4 × 1.2 STE compensation)                    ║
║                                                                  ║
║  ═══ PHASE 6: ALAS CURRICULUM (45 min) ═══                       ║
║  • Evaluate all MoLE experts: 500 test cases × 4 experts        ║
║  • Identify TOP-3 weakest experts                                ║
║  • Focus 70% compute on weakest → equalize quality              ║
║                                                                  ║
║  ═══ PHASE 7: PoI GATE (10 min) ═══                              ║
║  ┌────────────────────────────────────────────────────┐          ║
║  │  Proof-of-Improvement:                             │          ║
║  │  score_after = eval(golden_test_set)               │          ║
║  │                                                     │          ║
║  │  if score_after >= score_before - 0.05:            │          ║
║  │    → DEPLOY new LoRA ✅                             │          ║
║  │  else:                                              │          ║
║  │    → ROLLBACK to pre-night state ⛔                 │          ║
║  │    → ТАРС НЕ МОЖЕТ стать хуже. Никогда.           │          ║
║  └────────────────────────────────────────────────────┘          ║
║                                                                  ║
║  ═══ QUARANTINE TRUST (Phase 2+) ═══                             ║
║  LoRA updates → 3-night quarantine buffer                        ║
║  Extended validation before promotion to active                  ║
║  Safety >> speed of learning                                      ║
║                                                                  ║
║  ═══ WEIGHT DNA (Phase 3) ═══                                    ║
║  Nightly SVD fingerprint (800 bytes, 0.1s)                       ║
║  Compare vs yesterday: if drift > 0.3 → WARNING                 ║
║  If drift > 0.6 → BLOCK Night Cycle → manual restart             ║
║                                                                  ║
║  ~2:00 → ~4:00 AM. ТАРС завтра УМНЕЕ. Гарантировано.           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🛡️ ЗАЩИТА — ИНВАРИАНТЫ (всегда истинны)

```
╔══════════════════════════════════════════════════════════════════╗
║  PROVEN INVARIANTS — ГАРАНТИИ СИСТЕМЫ                           ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. |SSM state| < 32,000     Assert every 100 tokens            ║
║  2. RSS < 700 MB             Assert every 1000 generations      ║
║  3. TTFT < 15ms              SSM State Cache + mmap warmup      ║
║  4. Night non-regression     PoI gate: score_after ≥ floor      ║
║  5. SDM recall ≥ 93%         For topics with 3+ stored entries  ║
║  6. Ternary quality ≤ 10%    Max accuracy drop vs FP16 teacher  ║
║  7. Arena frag = 0           Bump allocator, mathematically     ║
║  8. Atomic writes            write(tmp) → fsync → rename        ║
║  9. MoLE stability           Personality expert always-on α=0.3 ║
║  10. Privacy guarantee       SECRET data never enters training   ║
║                                                                  ║
║  NaN Guard: RingBuffer of 3 checkpoints.                         ║
║    NaN detected → rollback to ring[-1]. Automatic recovery.     ║
║                                                                  ║
║  Checksummed LoRA: SHA256 on every load.                         ║
║    Corrupt → fallback: latest → previous → base. 3 layers.     ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 🏗️ RUNTIME: 3-LAYER ARCHITECTURE

```
╔══════════════════════════════════════════════════════════════════╗
║                    3 СЛОЯ RUNTIME                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  СЛОЙ 1: C++/Rust ЯДРО (hot path, 10-20ms/tok)                 ║
║  ═══════════════════════════════════════════                     ║
║  • ssd_scan.cpp      s = γ·s + B·x           AVX2/VNNI         ║
║  • wkv7_update.cpp   S = w⊙S + α·k⊗Δ        Rust SIMD         ║
║  • bitnet_matmul.cpp ADD/SUB only, no FPU     ASM ternary       ║
║  • swiglu_fused.cpp  SiLU(W₁x)⊙W₂x + sparse sparse BLAS      ║
║  • rmsnorm_fused.cpp γ·x/√(mean(x²)+ε)       fused kernel      ║
║  • rope_fused.cpp    RoPE θ=500K              fused kernel      ║
║  • arena.rs          bump allocator           zero-frag          ║
║  • engine.rs         main inference loop      Rust               ║
║  • eagle_decode.rs   speculative loop         Rust               ║
║                                                                  ║
║  СЛОЙ 2: НЕЙРОНКА = ВСЯ ЛОГИКА (learned W matrices)            ║
║  ═══════════════════════════════════════════════                 ║
║  Mode Router:    MinGRU(x) → [REFLEX, THINK, DEEP]             ║
║  WuNeng Fusion:  W_down·concat → GELU → W_up   = learned gate ║
║  MoLE routing:   softmax(W_g · x) → top-2      = learned       ║
║  MoD decision:   σ(W_r · x) > threshold        = learned       ║
║  Early Exit:     max(softmax(h @ W_lm)) > 0.9  = learned       ║
║  DoubtEngine:    3 × Linear(d, 1)              = learned       ║
║                                                                  ║
║  ВСЁ = W · x + activation. Нет if/else. Всё обучаемо.         ║
║  C++ просто ПЕРЕМНОЖАЕТ матрицы.                                ║
║                                                                  ║
║  СЛОЙ 3: PYTHON ОРКЕСТРАЦИЯ (не hot path)                       ║
║  ═══════════════════════════════════════                         ║
║  • Training (Muon, DPO, Night Cycle)         PyTorch            ║
║  • Memory (SDM write/read, compaction)       Python             ║
║  • Tools (file, code, web)                    Python             ║
║  • UI (chat, CLI)                             Python             ║
║  • Config, logging, disk guardian             Python             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📊 ОБУЧЕНИЕ: UMOT + Alternating Batches

```
╔══════════════════════════════════════════════════════════════════╗
║  TRAINING PIPELINE (Phase 0.5, 3 weeks, $150-300 GPU)          ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Data: 1.25B tokens (500M teacher + 750M self-gen)              ║
║  Teacher: Llama-3.1 8B or Qwen 7B (cloud GPU)                  ║
║  Student: TARS-Base 380M ternary                                ║
║                                                                  ║
║  UMOT Loss (Alternating Batches — NOT mixed!):                  ║
║  ┌──────────────────────────────────────────────────┐           ║
║  │  Each step = ONE loss with FULL gradient:        │           ║
║  │                                                   │           ║
║  │  CE:          100% → 30%  (main LM objective)    │           ║
║  │  SFT:         0%   → 30%  (instruction following)│           ║
║  │  DPO:         0%   → 14%  (preference alignment) │           ║
║  │  Safety:      0%   → 7.5% (safe responses)       │           ║
║  │  CoT:         0%   → 8%   (reasoning chains)     │           ║
║  │  Personality: 0%   → 20%  (ТАРС character)       │           ║
║  │  FC:          0%   → 12%  (function calling)      │           ║
║  │                                                   │           ║
║  │  → NO PCGrad/CAGrad (0% overhead)                │           ║
║  │  → Each DPO step = 100% DPO gradient (not 5%)    │           ║
║  └──────────────────────────────────────────────────┘           ║
║                                                                  ║
║  Optimizer:                                                      ║
║    2D params → Muon (Newton-Schulz₅ orthogonal)                ║
║    1D params → AdamW (standard)                                  ║
║                                                                  ║
║  Quantization:                                                   ║
║    Cooperative STE (19% more efficient than standard STE)        ║
║    Adaptive QA-KD: τ: 2→3→4 (adaptive temperature)             ║
║    Gradual α ramp: 0.01→0.5 over last 5% of training           ║
║    TALS: Ternary-Aware Label Smoothing (zero runtime cost)      ║
║                                                                  ║
║  LR Schedule: WSD (Warmup-Stable-Decay)                         ║
║    5% warmup → 75% stable → 20% cosine decay                   ║
║                                                                  ║
║  Memory-efficient:                                               ║
║    GaLore (SVD rank-64) → -65% optimizer memory                 ║
║    Gradient checkpointing → -50% activation memory              ║
║    Sequence packing (FFD) → 99% GPU utilization                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📅 TIMELINE — РЕАЛИСТИЧНЫЙ

```
Phase 0   (Day 1):       Config + SSM State Cache + Entropy temp
Phase 0.5 (Weeks 1-3):   Data gen (GPU, $150) + UMOT + QA-KD
Phase 1a  (Weeks 4-5):   Python prototype (15 tok/s) + A/B ablation
Phase 1b  (Weeks 4-7):   C++/Rust core (parallel with 1a!)
Phase 2   (Weeks 8-10):  Full C++ integration + EAGLE-3 + LEANN
Phase 3   (Month 3-4):   Night Cycle + SPIN + Privacy Guard
Phase 4   (Month 5+):    TARS-Full 450M + autonomous evolution

MANDATORY ABLATIONS (Phase 1a):
  A: SSD-only (20 blocks)  — baseline
  B: WKV-only (20 blocks)  — baseline
  C: Dual 8-8-8            — hypothesis
  → Pick winner or confirm Dual advantage

5 AGENTS parallel:
  🔴 Agent 1: C++/Rust core (kernels + runtime)
  🟢 Agent 2: Memory (SDM, context, recall)
  🔵 Agent 3: Training (UMOT, Night Cycle)
  🟡 Agent 4: Python model + config
  🟣 Agent 5: Agent OS + safety + tests
```

---

> 🧬 **TARS HELIX: Two strands. One mind. 700 megabytes.**
> **Нейронка = ВСЯ логика. Код = ТОЛЬКО вычисления.**
> **Рефлекторный вход → двойная спираль → объёмное понимание → действие.**
> **Каждую ночь умнее. Гарантировано.** 🧬
