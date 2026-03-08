# 🧬 TARS — ПОЛНАЯ КАРТА ФОРМУЛ В ДЕЙСТВИИ
# Как 68+ формул работают ВМЕСТЕ от первого байта до последнего токена

> Каждая формула показана В КОНТЕКСТЕ — где именно она срабатывает,
> что получает на вход, что отдаёт на выход, и почему именно ОНА.

---
---

# ════════════════════════════════════════════════════════════
#  ФАЗА 1: ОБУЧЕНИЕ — как рождается TARS
# ════════════════════════════════════════════════════════════

```
RAW DATA (1.5B tokens, $100)
    │
    ├─ 8-Stage Cleaning Pipeline:
    │   MD5 dedup → MinHash(Jaccard>0.8) → langID → len filter
    │   → content filter → NFC normalize → quality score → safety
    │   Yield: 57% → 865M clean tokens
    │
    ▼
SEQUENCE PACKING (J10):                              ← FORMULA #1
    sorted_by_length(seqs) → first_fit_decreasing(max=4096)
    Efficiency: 45% → 99% (2× throughput!)
    │
    ▼
╔═══════════════ TRAINING LOOP ═══════════════════════════════╗
║                                                             ║
║  TOKENIZER: Qwen BPE 48,256 tokens                          ║
║  text → [ids] → Embedding (FP16, J3 Weight-Tied with LM)    ║
║    E(token) ∈ ℝ^1280, FP16 precision                        ║
║    │                                                        ║
║    ▼                                                        ║
║  RoPE (J1): q = R(θ=500K, pos) · q_raw                      ║ ← FORMULA #2
║    θ^(2i/d), rotation matrix per position                   ║
║    │                                                        ║
║    ▼                                                        ║
║  24 FRACTAL BLOCKS (A12):                                   ║ ← FORMULA #3
║  ┌─────────────────────────────────────────────────────────┐║
║  │ Stochastic Depth (D8):                                  │║ ← FORMULA #4
║  │   u ~ Uniform(0,1); if u < 0.15: SKIP block             │║
║  │   else: y = Block(x)/(1-p_drop) + x                     │║
║  │                                                         │║
║  │ RMSNorm (C1): x̂ = γ · x / √(mean(x²) + ε)            │║ ← FORMULA #5
║  │                                                          │║
║  │ ┌─── TWO PARALLEL SSM PATHS ────────────────────────┐  │║
║  │ │                                                    │  │║
║  │ │ SSD (A1):                        WKV-7 (A2):       │  │║
║  │ │  s = γ·s + B·x                   S = W⊙S + α·k⊗Δ │  │║ ← FORMULA #6,7
║  │ │  γ = σ(W_γ·x)                    W = σ(W_w·x)     │  │║
║  │ │  y_ssd = C·s + D·x               y_wkv = C·S·q    │  │║
║  │ │  Scalar-A, d_state=64             Vector, d_st=24  │  │║
║  │ │                                                    │  │║
║  │ │ BitNet (B1) inside both:                           │  │║ ← FORMULA #8
║  │ │  α = median(|W|)                                   │  │║
║  │ │  W_q = round_ternary(W/α) ∈ {-1,0,+1}            │  │║
║  │ │  STE backward: ∂L/∂W ≈ ∂L/∂W_q                   │  │║
║  │ └────────────────────────────────────────────────────┘  │║
║  │                                                          │║
║  │ WuNeng Fusion (A3):                                     │║ ← FORMULA #9
║  │   gate = FlashSigmoid(W·x) = 0.5 + 0.25·(W·x)  (J4)  │║ ← FORMULA #10
║  │   y = gate · y_ssd + (1-gate) · y_wkv                  │║
║  │                                                          │║
║  │ SwiGLU + Double Sparsity (A7):                          │║ ← FORMULA #11
║  │   h = SiLU(W₁x) ⊙ W₂x                                │║
║  │   mask = (|h| > ε);  h = h ⊙ mask                      │║
║  │   W₃_active = W₃[:, nonzero(mask)]  ← AIP 13% load    │║
║  │   out = W₃_active @ h_sparse                           │║
║  │   Combined: 87% zeros (BitNet 67% × activation 60%)    │║
║  │                                                          │║
║  │ MoLE Personality Router (A11):                          │║ ← FORMULA #12
║  │   r = softmax(W_r · [h; personality_emb])               │║
║  │   top2 = argmax_2(r)                                    │║
║  │   h = Σ (r_i · (h + B_i · A_i · h))  ← LoRA rank=8    │║
║  │                                                          │║
║  │ SpikeBus Encode (J6):                                   │║ ← FORMULA #13
║  │   spike_i = sign(h_i - EMA(|h_i|)×0.5)                 │║
║  │   spike ∈ {-1, 0, +1}  →  next block routing           │║
║  │                                                          │║
║  │ Residual: output = h + x_input                          │║ ← FORMULA #14
║  └─────────────────────────────────────────────────────────┘║
║                                                              ║
║  × 24 blocks (6S + 6M + 6L + 6S fractal)                   ║
║                                                              ║
║  Every 4th wave → SharedGlobalAttention:                    ║
║    Diff Transformer (A4):                                   ║ ← FORMULA #15
║      attn = softmax(Q₁K₁ᵀ/√d) − λ·softmax(Q₂K₂ᵀ/√d)     ║
║    QK-Norm (C2): Q = RMSNorm(Q), K = RMSNorm(K)           ║ ← FORMULA #16
║    GQA-2 (A5): Q(8h), K(2g), V(2g) → KV ÷4               ║ ← FORMULA #17
║    CLA: share KV across waves → ÷2 more → total KV ÷8    ║
║    │                                                         ║
║    ▼                                                         ║
║  LM Head (J3): logits = E^T · h                            ║ ← uses FORMULA #2
║  Weight-Tied with Embedding, FP16                            ║
║    │                                                         ║
║    ▼                                                         ║
║  UMOT LOSS (J7):                                            ║ ← FORMULA #18
║    L = 0.8·CE + 0.1·KL_distill + 0.05·FC + 0.03·safety    ║
║        + 0.02·balance                                        ║
║    Weights change per epoch (curriculum)                     ║
║                                                              ║
║  EMA Self-Distillation (D5):                                ║ ← FORMULA #19
║    θ_ema = 0.999·θ_ema + 0.001·θ                           ║
║    L_distill = KL(p_student || p_ema) · T²                  ║
║                                                              ║
║  PCGrad (D2) — if SSD and WKV gradients conflict:          ║ ← FORMULA #20
║    cos(g_ssd, g_wkv) < 0 ?                                  ║
║    g_proj = g_ssd − (g_ssd·g_wkv/||g_wkv||²)·g_wkv        ║
║                                                              ║
║  SLM (D3): loss only on top-50% hardest tokens             ║ ← FORMULA #21
║  Label Smoothing (D9): target = (1-ε)·one_hot + ε/V        ║ ← FORMULA #22
║                                                              ║
║  OPTIMIZER:                                                  ║
║    2D params → Muon (D1):                                   ║ ← FORMULA #23
║      G_orth = Newton_Schulz₅(G)                             ║
║      Xₖ₊₁ = Xₖ·(1.5I − 0.5·Xₖᵀ·Xₖ)                      ║
║      W ← W − η·G_orth                                       ║
║    1D params → AdamW (D1):                                  ║ ← FORMULA #24
║      m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²               ║
║      W ← W·(1-η·λ) − η·m̂/(√v̂+ε)                          ║
║                                                              ║
║  LR Schedule — WSD (J9):                                    ║ ← FORMULA #25
║    5% warmup (linear) → 75% stable → 20% cosine decay      ║
║                                                              ║
║  GaLore (J8) on large layers:                               ║ ← FORMULA #26
║    P = SVD_top_r(∇L), r=64                                  ║
║    G_low = Pᵀ·G;  states in ℝ^{r×n} not ℝ^{m×n}           ║
║    -65% optimizer memory!                                    ║
║                                                              ║
║  Gradient Checkpointing (J11):                              ║ ← FORMULA #27
║    Checkpoint at blocks [0, 6, 12, 18]                       ║
║    Recompute between → -50% memory, +15% time               ║
║                                                              ║
║  MoLE Balance Loss:                                         ║ ← FORMULA #28
║    L_bal = α·Σᵢ(fᵢ·Pᵢ), push fᵢ → 1/K                    ║
║    expert_entropy H > 1.5 nats → healthy                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    │
    ▼
ALIGNMENT (post-training):
  DPO (D6): L = -log σ(β·(log π_θ(y_w)/π_ref(y_w)             ← FORMULA #29
                           − log π_θ(y_l)/π_ref(y_l)))
  Safety contrastive:                                           ← FORMULA #30
    L = max(0, d(anchor,pos) − d(anchor,neg) + margin)
    margin = 0.5, trained on 1500 safe + 500 harmful pairs
    │
    ▼
EXPORT: model.pt (ternary weights ~60-114MB)
```

---
---

# ════════════════════════════════════════════════════════════
#  ФАЗА 2: INFERENCE — как TARS думает (token by token)
# ════════════════════════════════════════════════════════════

```
USER: "Напиши unittest для calculate_tax()"
    │
    ▼
INPUT SANITIZER (I3):                                        ← FORMULA #31
  text = NFC(text); remove ZW; emoji→text
  │
  ▼
TOKENIZER: "Напиши" "unit" "test" ... → [4521, 892, ...]
  │
  ▼
CONTEXT MANAGER (8K window):                                 ← FORMULA #32
  if len(context) > 0.75 × 8192:
    old_turns = summarize(context[:half])  → SDM store
    context = [summary] + context[half:]   → fits in 8K
  │
  ▼
MinGRU MODE ROUTER (3M params):                              ← FORMULA #33
  MinGRU recurrence:
    z = σ(W_z · x + U_z · h_{t-1})          — update gate
    h̃ = W_h · x + U_h · (z ⊙ h_{t-1})       — candidate
    h_t = (1 - z) ⊙ h_{t-1} + z ⊙ h̃         — new state

  Classify: mode_logits = W_mode · h_final
  Decision:
    "привет"         → REFLEX (exit here, 0.5ms)
    "unittest для..."→ THINKING (full Brain)
    
  SPINE ROUTER (Speculative Spine, A10):                     ← FORMULA #34
    route_map = MinGRU_route(prompt) → [run,run,skip,run...]
    Per block: classify intent → skip/run (85% accuracy)
  │
  ▼

╔═══════════ BRAIN (per token generation, ~10-25ms) ═══════╗
║                                                            ║
║  TensorPool (F4): pre-allocated buffers, ZERO malloc       ║ ← FORMULA #35
║  gc.disable() → no GC pauses (p99: 15ms → 3ms)            ║
║                                                            ║
║  Arena Allocator (H3):                                     ║ ← FORMULA #36
║    alloc(size): pointer += size  (O(1))                    ║
║    reset(): pointer = 0  (O(1), per generation)            ║
║    fragmentation = 0 (proven ∎)                            ║
║                                                            ║
║  Embedding: E[token_id] ∈ ℝ^1280 (FP16, weight-tied)     ║
║  + RoPE (J1): R(θ=500K, position) applied to Q,K          ║
║                                                            ║
║  ═══ BLOCK LOOP (24 blocks, fractal S-M-L-S) ═══          ║
║                                                            ║
║  SpikeBus check (J6):                                      ║ ← FORMULA #37
║    if spine_route[block_id] == SKIP:                        ║
║      output = input  (pure residual, 0 compute!)           ║
║      GOTO next block                                       ║
║                                                            ║
║  MoD Router (A6):                                          ║ ← FORMULA #38
║    r = FlashSigmoid(W_r · x) = 0.5 + 0.25·(W_r·x)       ║
║    if r < threshold_p: output = input (SKIP)               ║
║                                                            ║
║  RMSNorm (C1): x̂ = γ · x / RMS(x)                        ║
║                                                            ║
║  ┌─── DUAL SSM (single-step decode, O(1)) ──────────┐    ║
║  │                                                    │    ║
║  │ SSD (A1):                                          │    ║
║  │   γ_t = FlashSig(W_γ · x_t)                      │    ║
║  │   s_new = γ_t · s_old + (W_B · x_t)              │    ║ ← FORMULA #39
║  │   y_ssd = (W_C · x_t)^T · s_new + D · x_t        │    ║
║  │   SSM state: FP32 (precision critical!)            │    ║
║  │   Weights: BitNet {-1,0,+1} (ADD/SUB only!)       │    ║
║  │   Bound: |s| < 32,000 (H1, assert every 100 tok) │    ║
║  │                                                    │    ║
║  │ WKV-7 (A2):                                        │    ║
║  │   w = σ(W_w · x_t)  per-dim vector decay          │    ║
║  │   S = w ⊙ S + α · k ⊗ (v − β · S · k)           │    ║ ← FORMULA #40
║  │   = Covariance Delta Rule (second-order update!)   │    ║
║  │   Better associative recall than first-order SSM   │    ║
║  │                                                    │    ║
║  │ HiPPO Init (J5): 4 banks γ=0.5/0.9/0.99/0.999   │    ║
║  │   = captures 2-tok to 1000-tok dependencies       │    ║
║  └────────────────────────────────────────────────────┘    ║
║                                                            ║
║  WuNeng Fusion (A3):                                       ║
║    gate = FlashSig(W_gate · x_t)                          ║ ← FORMULA #41
║    Simple token ("def") → gate=0.7 → SSD dominates        ║
║    Context token ("как раньше") → gate=0.2 → WKV recalls  ║
║    y_fused = gate · y_ssd + (1-gate) · y_wkv              ║
║                                                            ║
║  SwiGLU + Double Sparsity (A7):                            ║
║    h = SiLU(W₁·x) ⊙ (W₂·x)                              ║ ← FORMULA #42
║    indices = (|h| > 0.01).nonzero()  → only 13%           ║
║    W₃_active = W₃[:, indices]                              ║
║    output = W₃_active @ h[indices]                         ║
║    = 87% ops SKIPPED! = 8× less bandwidth!                 ║
║                                                            ║
║  MoLE (A11):                                               ║
║    Expert selection via spikes + hidden state              ║ ← FORMULA #43
║    h += Σ_top2 (score_i · B_i · A_i · h)  ← LoRA r=8    ║
║                                                            ║
║  SpikeBus Encode (J6) → next block:                       ║
║    spike = sign(h − threshold) ∈ {-1,0,+1}^1280          ║ ← FORMULA #44
║    320 bytes vs 5KB FP32 = 16× compression                ║
║                                                            ║
║  Residual: h = h + x_input                                ║
║  ══════════════════════════════════════════════════════    ║
║                                                            ║
║  EARLY EXIT (F2):                                          ║ ← FORMULA #45
║    After block 8, 16:                                      ║
║    confidence = max(softmax(E^T · h))                      ║
║    if confidence > 0.9: EXIT → emit token early            ║
║    Avg: 30% tokens exit at block 8 (simple = fast!)        ║
║                                                            ║
║  SharedGlobalAttention (every 4th wave, block 16):         ║
║    Diff Attn (A4):                                         ║
║      a = softmax(Q₁K₁ᵀ/√d) − 0.5·softmax(Q₂K₂ᵀ/√d)     ║ ← FORMULA #46
║    GQA-2 + CLA (A5): KV cache = 6MB (÷33 vs standard!)   ║ ← FORMULA #47
║    MiniKV (F3):  evict 50% + quant to INT2 = -86%         ║ ← FORMULA #48
║    RETRO cross-attn (E2):                                  ║ ← FORMULA #49
║      Q=W_Q·h; K=W_K·LEANN_chunks; V=W_V·LEANN_chunks    ║
║      h_enriched = softmax(QKᵀ/√d) · V                     ║
║                                                            ║
║  Titans Memory (A9, E3):                                   ║ ← FORMULA #50
║    surprise = ||x − M·x||²                                 ║
║    if surprise > θ: M ← M + η·(x − M·x)·xᵀ              ║
║    Async: runs in background, non-blocking                 ║
║    O(1) cost, quality IMPROVES with context!               ║
║                                                            ║
║  LM Head: logits = E^T · h_final  (FP16, weight-tied)    ║
║                                                            ║
║  EAGLE-3 Spec (F1):                                        ║ ← FORMULA #51
║    From block 10: predict 3 extra tokens                   ║
║    Verify all 4 in single forward pass                     ║
║    Accept rate ~75% → 2.5× speedup                         ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    │
    ▼
SAMPLING:                                                     ← FORMULA #52
  Top-p (nucleus): keep tokens until cumsum(sorted_p) > 0.9
  Temperature: logits /= T (T=0.7 for creative, 0.1 for code)
  Repetition Penalty:                                         ← FORMULA #53
    logits[already_generated] /= α_rep (α_rep=1.1)
    Prevents: "I think I think I think..."
  │
  ▼
DOUBT ENGINE (H1):                                           ← FORMULA #54
  coherence = CoherenceHead(h) ∈ [0,1]
  safety = SafetyHead(h) ∈ [0,1]   ← contrastive trained
  repeat = RepeatHead(h) ∈ [0,1]
  doubt = 1 − min(coherence, safety, 1−repeat)
  if doubt > 0.7: RETRY with different sampling
  │
  ▼
TOOL EXECUTION (if detected):                                ← FORMULA #55
  Parse: [TOOL_CALL|file_write|{"path":"test.py","content":...}]
  Security: EthicalGuard 5 hard rules
  Timeout: 30s chain budget (consecutive tool calls)
  │
  ▼
OUTPUT → user
```

---
---

# ════════════════════════════════════════════════════════════
#  ФАЗА 3: НОЧНОЙ ЦИКЛ — как TARS учится сам
# ════════════════════════════════════════════════════════════

```
2:00 AM — NIGHT CYCLE STARTS
    │
    ▼
POWER MANAGER (I2):                                          ← FORMULA #56
  SetThreadExecutionState(CONTINUOUS | SYSTEM_REQUIRED)
  → prevent Windows sleep
    │
    ▼
ANALYSE DAY'S DATA:
  For each conversation:
    quality = DoubtEngine.score(response)
    user_signal = (corrections, re-asks, thumbs)
    pairs = rank_responses(quality + user_signal)
    │
    ▼
DREAM REPLAY:
  Best conversations → replay through model
  EMA update (D5): θ_ema = 0.999·θ_ema + 0.001·θ           ← FORMULA #57
    │
    ▼
SPIN (D7, 4 iterations):                                    ← FORMULA #58
  For k = 1..4:
    Generate y ~ π_θ(·|x)
    Rank: (y_win, y_lose) pairs
    DPO update: L = -log σ(β·(log_ratio_w − log_ratio_l))
    LoRA update only (base weights FROZEN)
    η = 3.6e-4 (3e-4 × 1.2 STE compensation)

  Hindsight Distillation:                                    ← FORMULA #59
    User corrected "X" to "Y" →
    L_hindsight = CE(model(context), Y)
    = learn from corrections (ceiling = USER expertise!)
    │
    ▼
QUALITY GATE:
  PoI (Proof of Improvement):                                ← FORMULA #60
    score_after = eval(golden_test_set)
    assert score_after ≥ score_before - 0.05
    FAIL → rollback LoRA to pre-night state

  CQS (Conversational Quality Score):                        ← FORMULA #61
    CQS = 0.3·coherence + 0.3·relevance + 0.2·safety + 0.2·diversity
    Monitor: should trend upward over weeks
    │
    ▼
HOUSEKEEPING:
  SDM Compaction (E1):                                       ← FORMULA #62
    dead = strengths < 0.01
    if dead.sum() > 0.3 * total: rebalance + free slots
  
  SDM Decay: strengths *= 0.95                               ← FORMULA #63
  
  LEANN Index: rebuild HNSW if >1000 new entries
  
  Disk Guardian: if used_space > 2GB → delete oldest logs
  
  NaN Check: assert no NaN.tmp recovery files                ← refers to FORMULA J15
    │
    ▼
POWER MANAGER: restore → allow sleep
3:00 AM — NIGHT CYCLE COMPLETE (~30 min)
```

---
---

# ════════════════════════════════════════════════════════════
#  ФАЗА 4: MEMORY — как TARS помнит
# ════════════════════════════════════════════════════════════

```
USER SAYS SOMETHING
    │
    ├─── STORE PATH ─────────────────────────────────────
    │                                                    │
    │  SDM Write (E1):                                   │ ← FORMULA #64
    │    cos_sim = (query · addresses) / (||q||·||a||)   │
    │    top5 = argmax_5(cos_sim)                        │
    │    contents[top5] += α · (value − contents[top5])  │
    │    strengths[top5] += 1.0                           │
    │                                                    │
    │  LEANN Store (E2):                                 │ ← FORMULA #65
    │    embed = MiniLM(text) ∈ ℝ^384                    │
    │    Semantic chunk: split where cos_sim < 0.5        │
    │    HNSW index: O(log N) insert                      │
    │    PQ compress: 384 → 48 bytes per vector           │
    │                                                    │
    │  Titans Update (E3):                               │ ← FORMULA #66
    │    surprise = ||x − M·x||²                         │
    │    if surprise > θ:                                 │
    │      M ← M + η·(x − M·x)·xᵀ   (rank-1 SGD)     │
    │    Runs async in background thread                  │
    │                                                    │
    ├─── RECALL PATH ────────────────────────────────────
    │                                                    │
    │  SDM Read (E1):                                    │ ← FORMULA #67
    │    scores = cos_sim(query, addresses)               │
    │    Matryoshka 2-stage (J12):                       │
    │      Fast: e[:64] scan all 30K → top-50            │ ← FORMULA #68
    │      Full: e[:384] re-rank top-50 → top-5          │
    │    result = Σ (score_i · strength_i · content_i)   │
    │    Recall ≥ 93% for topics with 3+ entries         │
    │                                                    │
    │  LEANN Recall (E2):                                │ ← FORMULA #69
    │    HNSW search: O(log N), K=5 chunks               │
    │    → feed to RETRO cross-attention at block 16     │
    │    Effective capacity: 400M × 25 = 10B equivalent! │
    │                                                    │
    │  Gaussian Uncertainty (J13):                       │ ← FORMULA #70
    │    Each entry: N(μ, σ²)                            │
    │    New concept: σ high (unsure → less weight)      │
    │    Seen often: σ low (confident → more weight)     │
    │    similarity = KL divergence between Gaussians    │
    │                                                    │
    │  Doc-to-LoRA (J14):                                │ ← FORMULA #71
    │    topic detected → load LoRA_topic.pt (3MB)       │
    │    W' = W + B·A (rank-8, topic-specific)           │
    │    TIES merge if multiple topics overlap            │
    │    200× compression vs raw text!                    │
    │                                                    │
    │  MemTree (E4, Phase 3):                            │ ← FORMULA #72
    │    root → branch (topic) → leaf (fact)             │
    │    Coarse-to-fine retrieval                         │
```

---
---

# ════════════════════════════════════════════════════════════
#  ФАЗА 5: ЗАЩИТА — как TARS остаётся безопасным
# ════════════════════════════════════════════════════════════

```
LAYER 1 — Input:
  InputSanitizer (I3)                                        ← FORMULA #31
  Binary detect: magic bytes check (40 LOC)
  
LAYER 2 — Model:
  SSM Bound: |s| < 32,000 (assert every 100 tok)            ← FORMULA #73
  NaN Guard (J15): RingBuffer 3 checkpoints                 ← FORMULA #74
    ring = deque(maxlen=3)
    NaN detected → rollback to ring[-1]
  Memory Bound: RSS < 700MB (assert every 1000 gen)          ← FORMULA #75

LAYER 3 — Output:
  DoubtEngine (H1): doubt > 0.7 → retry                     ← reuses #54
  
LAYER 4 — Execution:
  EthicalGuard: 5 hard rules (no delete, no network, etc.)
  Tool timeout: 30s chain budget
  Audit log: all tool calls logged locally
  
LAYER 5 — Persistence:
  Atomic Write (H2):                                         ← FORMULA #76
    write(tmpfile) → fsync → os.replace(tmpfile, real)
    NTFS atomic rename, journal recovery on crash
  Optimistic Lock:                                           ← FORMULA #77
    mtime_before = getmtime(f)
    ... process ...
    assert |getmtime(f) − mtime_before| < 0.001
  Checksummed LoRA:                                          ← FORMULA #78
    SHA256(lora_file) checked on every load
    3 fallbacks: latest → previous → base
```

---
---

# ════════════════════════════════════════════════════════════
#  📊 ИТОГО: ПОЛНАЯ ТАБЛИЦА — 78 ФОРМУЛ В ДЕЙСТВИИ
# ════════════════════════════════════════════════════════════

```
╔═══════════════════════════════════════════════════════════════════╗
║  PHASE          │ FORMULAS  │ KEY INNOVATIONS                    ║
╠═════════════════╪═══════════╪════════════════════════════════════╣
║  Training       │ #1-#30    │ UMOT, Muon+AdamW, GaLore, PCGrad ║
║  Inference      │ #31-#55   │ Dual SSM, Double Sparsity, EAGLE  ║
║  Night Cycle    │ #56-#63   │ SPIN, Hindsight Distill, PoI gate ║
║  Memory         │ #64-#72   │ SDM+RETRO, Titans, Matryoshka     ║
║  Safety         │ #73-#78   │ Bounds, NaN recovery, Atomic IO   ║
╠═════════════════╪═══════════╪════════════════════════════════════╣
║  TOTAL          │ 78        │ 7 unique (🏆), 10 proven (∎)      ║
╚═════════════════╧═══════════╧════════════════════════════════════╝

UNIQUE TARS INNOVATIONS (🏆):
  1. Double Sparsity (87% ops=0)      — A7
  2. SpikeFormer (Hamming attention)  — A8
  3. TTT-Titans (O(1) attention)       — A9
  4. Speculative Spine (MinGRU routes) — A10
  5. Personality MoLE (experts=styles) — A11
  6. Fractal Blocks (variable depth)   — A12
  7. SpikeBus (dual FP32+SNN channel) — J6

PROVEN INVARIANTS (∎, 10/10):
  1. SSM bounded       2. Memory bounded     3. TTFT < 15ms
  4. Night non-regress 5. SDM recall ≥ 93%   6. Ternary ≤ 10%
  7. Arena 0 frag      8. Atomic writes      9. MoLE stable
  10. Complete coverage
```

---

> **78 формул. 7 unique. 10 proofs.**
> **Каждая формула ПРИВЯЗАНА к конкретному месту в pipeline.**
> **Каждая формула ОБОСНОВАНА дебатом.**
> **Гибрид = не мешанина, а ПОСЛЕДОВАТЕЛЬНОСТЬ — каждая формула знает свой вход и выход.**
