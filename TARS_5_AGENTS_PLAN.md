# 🏭 TARS — ПЛАН ДЛЯ 5 ПАРАЛЛЕЛЬНЫХ АГЕНТОВ
# C++/Rust ядро + Python обвязка + нейросеть вместо логики

> **Дата:** 2026-03-08 v2
> **Принцип:** Ядро думает на C++/Rust/ASM. Python = клей. Всё что можно — в нейронку.
> **Длительность:** 4 недели (Phase 0+1 параллельно)

---

# ═══════════════════════════════════════
#  🧠 ФИЛОСОФИЯ: 3 СЛОЯ TARS
# ═══════════════════════════════════════

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          3 СЛОЯ TARS                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌────────────────────────────────────────────────────────────────────────┐ ║
║  │  СЛОЙ 1: C++/Rust/ASM — БЫСТРОЕ ЯДРО (inference hot path)            │ ║
║  │  ═══════════════════════════════════════════════════════════          │ ║
║  │  • SSD scan:     s = γ·s + B·x              ← AVX2/VNNI intrinsics  │ ║
║  │  • WKV-7 update: S = S·(diag(w)+aᵀb)+vᵀk   ← Rust SIMD            │ ║
║  │  • BitNet matmul: ADD/SUB only, no FPU       ← ASM ternary kernel   │ ║
║  │  • SwiGLU + Double Sparsity                  ← sparse BLAS          │ ║
║  │  • RMSNorm:      y = γ·x/√(mean(x²)+ε)     ← fused C++ kernel     │ ║
║  │  • Tokenizer BPE:  split+merge               ← C++ (1-2ms)          │ ║
║  │  • EAGLE-3 draft:  spec decode loop           ← Rust (tight loop)    │ ║
║  │  • Arena allocator: zero-frag bump            ← C pure (brk-like)    │ ║
║  │                                                                       │ ║
║  │  Всего: ~3,000-5,000 LOC C++/Rust                                    │ ║
║  │  Скорость: 10-20ms/tok (vs 200ms Python)                             │ ║
║  └──────────────────────────────┬─────────────────────────────────────────┘ ║
║                                 │ pybind11 / ctypes / PyO3                  ║
║  ┌──────────────────────────────▼─────────────────────────────────────────┐ ║
║  │  СЛОЙ 2: НЕЙРОНКА = ВСЯ ЛОГИКА (learned, not coded!)                 │ ║
║  │  ═══════════════════════════════════════════════════════              │ ║
║  │  • Mode Router:     MinGRU 3M → reflex/think/deep    ← НЕЙРОНКА    │ ║
║  │  • MoD decision:    σ(W_r·x) > p → skip/run          ← НЕЙРОНКА    │ ║
║  │  • WuNeng fusion:   gate = σ(W·x)                    ← НЕЙРОНКА    │ ║
║  │  • MoLE routing:    softmax(W_g·x) → top-2 experts   ← НЕЙРОНКА    │ ║
║  │  • Early Exit:      confidence = max(softmax(h@W))    ← НЕЙРОНКА    │ ║
║  │  • DoubtEngine:     3 linear heads на hidden state    ← НЕЙРОНКА    │ ║
║  │  • Safety check:    contrastive SafetyHead            ← НЕЙРОНКА    │ ║
║  │  • Spine Router:    MinGRU predicts block skip        ← НЕЙРОНКА    │ ║
║  │  • Context summary: model сам решает что сжимать      ← НЕЙРОНКА    │ ║
║  │                                                                       │ ║
║  │  ВСЁ это = просто матрицы × вектор → inference на C++ ядре!          │ ║
║  │  Нет if/else логики. Нет rules. Всё ОБУЧЕНО.                         │ ║
║  └──────────────────────────────┬─────────────────────────────────────────┘ ║
║                                 │                                           ║
║  ┌──────────────────────────────▼─────────────────────────────────────────┐ ║
║  │  СЛОЙ 3: PYTHON — ОРКЕСТРАЦИЯ (не на hot path!)                      │ ║
║  │  ═══════════════════════════════════════════════════════              │ ║
║  │  • Training loop (Muon, DPO, Night Cycle)             ← Python      │ ║
║  │  • Data pipeline (clean, dedup, curriculum)           ← Python      │ ║
║  │  • Tools execution (file, code, web)                  ← Python      │ ║
║  │  • Memory management (SDM, LEANN, compaction)         ← Python      │ ║
║  │  • UI (chat, CLI)                                     ← Python      │ ║
║  │  • Config, logging, disk guardian                     ← Python      │ ║
║  │                                                                       │ ║
║  │  Ничего из этого НЕ на hot path. Python OK.                          │ ║
║  └────────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

# ═══════════════════════════════════════
#  📊 ЧТО ПЕРЕНОСИТСЯ В НЕЙРОНКУ
# ═══════════════════════════════════════

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  БЫЛО (if/else код)             →    СТАЛО (нейронка, learned weights)      ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  Mode routing:                                                                ║
║  ❌ if len(tokens)<5: REFLEX    →  ✅ MinGRU(x) → [p_reflex, p_think, p_deep]║
║  ❌ elif question: THINKING     →     Обучено на примерах, не на правилах!   ║
║                                                                               ║
║  Block skipping:                                                              ║
║  ❌ if block > 16: skip         →  ✅ Spine: MinGRU(h) → p_skip per block    ║
║  ❌ hardcoded rules              →     Model УЧИТСЯ какие блоки лишние       ║
║                                                                               ║
║  MoD token routing:                                                           ║
║  ❌ top-K(difficulty)            →  ✅ r = σ(W_r·x) > threshold → skip/run   ║
║  ❌ fixed K                      →     Adaptive per-token decision            ║
║                                                                               ║
║  SSD↔WKV fusion:                                                             ║
║  ❌ 50/50 hardcoded              →  ✅ gate = σ(W·x), y = gate·SSD+(1-g)·WKV ║
║  ❌ constant ratio               →     Data-dependent, per-token adaptive    ║
║                                                                               ║
║  Expert selection:                                                            ║
║  ❌ if topic=="code": expert=3   →  ✅ softmax(W_gate·x) → top-2 experts     ║
║  ❌ keyword matching              →     Learned domain+personality routing    ║
║                                                                               ║
║  Early exit:                                                                  ║
║  ❌ if layer==8: check_conf()    →  ✅ max(softmax(h @ W_lm)) > θ → exit     ║
║  ❌ hardcoded checkpoints         →     Confidence from MODEL itself          ║
║                                                                               ║
║  Safety check:                                                                ║
║  ❌ if "опасно" in text: block   →  ✅ SafetyHead(h) → p_safe (contrastive)  ║
║  ❌ keyword blacklist             →     Learned from 1500 examples + negs     ║
║                                                                               ║
║  Coherence check:                                                             ║
║  ❌ if repeat_ratio > 0.5: retry →  ✅ RepeatHead(h) → p_repeat (learned)    ║
║  ❌ rule-based                    →     Trained on good/bad responses          ║
║                                                                               ║
║  Context compression:                                                         ║
║  ❌ if turns > 10: truncate      →  ✅ Model auto-summary via SDM write      ║
║  ❌ fixed window                  →     Learned importance scoring            ║
║                                                                               ║
║  ИТОГ: 0 if/else на inference path.                                          ║
║  Всё = W·x + sigmoid/softmax. Всё на C++ ядре.                              ║
║  Всё ОБУЧАЕМО. Всё АДАПТИРУЕТСЯ к данным.                                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---
---

# ═══════════════════════════════════════
#  ВИЗУАЛЬНАЯ КАРТА 5 АГЕНТОВ v2
# ═══════════════════════════════════════

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    TARS 5-AGENT MAP v2 (C++/Rust Core)                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🔴 AGENT 1: C++/Rust ЯДРО         🟢 AGENT 2: MEMORY + RETRIEVAL          ║
║  ═══════════════════════════       ═════════════════════════════             ║
║  core/                             memory/                                   ║
║  ├── kernels/                      ├── store.py (SDM INT8)                  ║
║  │   ├── ssd_scan.cpp   (AVX2)    ├── leann.py (LEANN+RETRO)               ║
║  │   ├── wkv7_update.cpp          ├── titans.py (surprise SGD)              ║
║  │   ├── bitnet_matmul.cpp (ASM)  ├── memo.py (MemTree)                    ║
║  │   ├── swiglu_fused.cpp         ├── matryoshka.py                         ║
║  │   ├── rmsnorm_fused.cpp        ├── context_manager.py                    ║
║  │   ├── rope_fused.cpp           └── gaussian_embed.py                     ║
║  │   ├── attention_diff.cpp                                                  ║
║  │   └── softmax_fused.cpp                                                   ║
║  ├── runtime/                                                                ║
║  │   ├── engine.rs (Rust)   ← main inference loop                           ║
║  │   ├── arena.rs           ← bump allocator                                ║
║  │   ├── tokenizer.rs       ← BPE                                           ║
║  │   ├── eagle_decode.rs    ← speculative loop                              ║
║  │   └── state_cache.rs     ← SSM state management                          ║
║  ├── bindings/                                                               ║
║  │   ├── tars_core.pyi      ← Python type stubs                            ║
║  │   └── lib.rs / bind.cpp  ← PyO3 / pybind11                              ║
║  └── build.zig / CMakeLists.txt                                              ║
║                                                                              ║
║  🔵 AGENT 3: TRAINING (Python)    🟡 AGENT 4: INFRA + PYTHON MODEL         ║
║  ══════════════════════════       ═══════════════════════════════            ║
║  training/                         brain/ (Python reference model)           ║
║  ├── muon.py (optimizer)          ├── omega_core/model.py                   ║
║  ├── train_utils.py               ├── omega_core/block.py                   ║
║  ├── curriculum.py                ├── mamba2/ssd.py (Python ref)            ║
║  ├── lora.py                      ├── mamba2/bitnet.py (Python ref)         ║
║  ├── distill_from_teacher.py      ├── speculative.py                        ║
║  ├── night_cycle.py               ├── doubt_engine.py                       ║
║  ├── dpo.py                       ├── tokenizer.py                          ║
║  ├── umot_loss.py                 config.py + utils/ + sensory/             ║
║  └── data_pipeline.py             requirements.txt                           ║
║                                                                              ║
║  🟣 AGENT 5: AGENT + SAFETY + UI                                           ║
║  ═══════════════════════════════                                             ║
║  agent/tools/ + agent/safety/ + ui/ + tests/ + launch_tars.py               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---
---

# ═══════════════════════════════════════
# 🔴 AGENT 1: C++/Rust ЯДРО INFERENCE
# ═══════════════════════════════════════

## Зона ответственности
**Весь hot path inference на C++/Rust/ASM.** Ни одной Python строки в критическом пути.
Ядро = ЧИСТЫЕ вычисления. Вся "логика" — это просто матрицы (нейронка).

## Файлы (ЭКСКЛЮЗИВНЫЕ)
```
core/
├── kernels/                    ← C++ горячие ядра (AVX2/VNNI)
│   ├── ssd_scan.cpp            ← s = γ·s + B·x (vectorized)
│   ├── ssd_scan.h              ← header
│   ├── wkv7_update.cpp         ← S = S·(diag(w)+aᵀb)+vᵀk
│   ├── bitnet_matmul.cpp       ← Ternary: ADD/SUB only, no FPU
│   ├── bitnet_matmul_avx.S     ← ASM: AVX2 ternary kernel (peak perf)
│   ├── swiglu_fused.cpp        ← SiLU(W₁x)⊙W₂x + sparsity mask
│   ├── rmsnorm_fused.cpp       ← γ·x/√(mean(x²)+ε), fused
│   ├── rope_fused.cpp          ← RoPE θ=500K, QK-Norm then rotate
│   ├── attention_diff.cpp      ← Diff Transformer: attn₁ − λ·attn₂  
│   ├── softmax_fused.cpp       ← online softmax (numerically stable)
│   ├── embedding_lookup.cpp    ← embed × √d_model scaling
│   └── smoothquant.cpp         ← Y = (X·S⁻¹)·(S·W) INT8 path
│
├── runtime/                    ← Rust: inference engine + memory
│   ├── engine.rs               ← TarsEngine: load model → generate
│   ├── model_loader.rs         ← Load .safetensors → ternary arrays
│   ├── forward_pass.rs         ← 24 blocks: call C++ kernels
│   ├── generate.rs             ← Autoregressive loop + sampling
│   ├── eagle_decode.rs         ← EAGLE-3 speculative decoding
│   ├── arena.rs                ← 80MB bump allocator (zero-frag)
│   ├── tokenizer.rs            ← BPE encode/decode (fast)
│   ├── state_cache.rs          ← SSM states (SSD + WKV) management
│   └── tensor.rs               ← Tensor struct: data ptr + shape + stride
│
├── bindings/                   ← Python ↔ C++/Rust мост
│   ├── bind.cpp                ← pybind11 обёртки для C++ kernels
│   ├── lib.rs                  ← PyO3 обёртки для Rust runtime
│   └── tars_core.pyi           ← Python type stubs для IDE
│
├── build.zig                   ← Zig build system (кросс-платформа)
├── CMakeLists.txt              ← CMake fallback
└── tests/
    ├── test_kernels.cpp        ← Google Test для C++ kernels
    └── test_engine.rs          ← Rust tests для runtime
```

## Задачи
```
Неделя 1 (фундамент):
  [ ] tensor.rs — базовая Tensor struct (data, shape, stride, dtype)
  [ ] arena.rs — bump allocator: alloc(size)→ptr, reset()→ptr=0
  [ ] rmsnorm_fused.cpp — γ·x/√(mean(x²)+ε) с AVX2 vectorization
  [ ] embedding_lookup.cpp — lookup + × √d_model scaling
  [ ] build.zig — настройка кросс-компиляции

Неделя 2 (ядра):
  [ ] bitnet_matmul.cpp — ternary matmul: ADD/SUB only
  [ ] bitnet_matmul_avx.S — ASM: AVX2 ternary (peak speed)
  [ ] ssd_scan.cpp — s_{t+1} = γ·s_t + B·x_t (SIMD vectorized)
  [ ] wkv7_update.cpp — S = S·(diag(w)+aᵀb)+vᵀk (non-diagonal!)
  [ ] swiglu_fused.cpp — SiLU(W₁x)⊙W₂x + |h|>ε sparsity mask
  [ ] rope_fused.cpp — QK-Norm → RoPE θ=500K (correct order K6!)
  [ ] softmax_fused.cpp — online softmax (max-subtract trick)

Неделя 3 (runtime + spec decode):
  [ ] attention_diff.cpp — Diff Transformer: softmax(Q₁K₁ᵀ) − λ·softmax(Q₂K₂ᵀ)
  [ ] smoothquant.cpp — Y=(X·S⁻¹)·(S·W) INT8 matmul path
  [ ] model_loader.rs — parse .safetensors → ternary weight arrays
  [ ] forward_pass.rs — 24 blocks calling C++ kernels via FFI
  [ ] generate.rs — autoregressive: sample → decode → loop
  [ ] tokenizer.rs — BPE encode/decode (Qwen 48K vocab)

Неделя 4 (интеграция):
  [ ] eagle_decode.rs — EAGLE-3 speculative loop (accept/reject)
  [ ] state_cache.rs — SSM state carry + 4-bank HiPPO init
  [ ] engine.rs — TarsEngine: full pipeline, Python-callable
  [ ] bind.cpp — pybind11 wrappers для всех C++ kernels
  [ ] lib.rs — PyO3 wrappers для Rust runtime → Python module
  [ ] tars_core.pyi — type stubs для Python IDE support
  [ ] test_kernels.cpp — unit tests: accuracy vs Python reference
  [ ] test_engine.rs — integration: full generation test
```

## Ключевое правило: НЕЙРОНКА = ЛОГИКА
```
Все "решения" модели — это ПРОСТО матрицы в C++ ядре:

  MoD routing:    score = bitnet_matmul(W_router, x)    → C++ kernel
                  if score > threshold → skip (1 compare, no Python!)
  
  WuNeng fusion:  gate = sigmoid(bitnet_matmul(W_gate, x))  → C++ kernel
                  y = gate * ssd_out + (1-gate) * wkv_out   → fused kernel
  
  MoLE routing:   scores = bitnet_matmul(W_mole, x)     → C++ kernel
                  top2 = argpartition(scores)             → std::nth_element
  
  Early Exit:     logits = matmul(h, W_lm)               → C++ kernel  
                  conf = max(softmax(logits))             → softmax kernel
                  if conf > 0.9 → EXIT (1 compare!)
  
  DoubtEngine:    doubt = sigmoid(matmul(h, W_doubt))    → C++ kernel
                  3 heads = 3 small matmuls, DONE!

  → НИКАКИХ Python вызовов на inference hot path!
  → Вся "магия" = обученные матрицы W
  → C++ просто перемножает и сравнивает
```

## Интерфейс с Python (пересечение с Agent 4):
```python
# Agent 1 ЭКСПОРТИРУЕТ через PyO3/pybind11:
import tars_core  # compiled C++/Rust module

# Низкоуровневые kernels (для Agent 4 — Python model reference):
tars_core.ssd_scan(state, gamma, B, x)        → new_state, output
tars_core.wkv7_update(S, w, a, b, v, k)       → new_S
tars_core.bitnet_matmul(W_ternary, x, alpha)   → output
tars_core.rmsnorm(x, gamma, eps)               → output
tars_core.swiglu_fused(W1, W2, x, sparsity)   → output
tars_core.rope(Q, K, theta, positions)          → Q_rot, K_rot
tars_core.diff_attention(Q1, Q2, K1, K2, V, lam) → output

# Высокоуровневый runtime (для Agent 5 — orchestrator):
class TarsEngine:
    def load(path: str, config: dict)
    def generate(prompt_ids: list[int], max_tokens: int,
                 temperature: float, top_p: float) → list[int]
    def get_hidden_state(layer: int) → numpy.ndarray
    def load_lora(path: str)
    def get_doubt_scores() → dict  # coherence, safety, repeat
```

---
---

# ═══════════════════════════════════════
# 🟢 AGENT 2: MEMORY & RETRIEVAL
# ═══════════════════════════════════════

## Зона ответственности
Все системы памяти (Python — не на hot path, допустимо).
SDM cosine_sim → может использовать C++ ядро от Agent 1 для matmul.

## Файлы (ЭКСКЛЮЗИВНЫЕ)
```
memory/
├── store.py                ← SDM INT8: write/read/compaction (E1)
├── leann.py                ← LEANN + RETRO cross-attention (E2)
├── titans.py               ← Titans surprise SGD update (E3)
├── memo.py                 ← MemTree hierarchical storage (E4)
├── matryoshka.py           ← 2-stage retrieval (J12)
├── gaussian_embed.py       ← N(μ,σ²) uncertainty (J13)
├── doc_to_lora.py          ← Document → LoRA (J14)
└── context_manager.py      ← Sliding window + auto-summary
```

## Оптимизация: C++ ядра для тяжёлых операций
```python
# Memory может вызывать C++ kernels от Agent 1 для скорости:
from tars_core import bitnet_matmul  # для SDM cosine similarity
from tars_core import rmsnorm        # для нормализации embeddings

# Но сама логика памяти = Python (не hot path, ~5ms OK)
```

## Задачи (без изменений, всё на Python — memory не hot path)
```
Неделя 1: SDM store (write/read/compaction)
Неделя 2: LEANN + RETRO cross-attention + Matryoshka
Неделя 3: Titans + Gaussian + Doc-to-LoRA + MemTree
```

---
---

# ═══════════════════════════════════════
# 🔵 AGENT 3: TRAINING PIPELINE
# ═══════════════════════════════════════

## Зона ответственности
Тренировка = Python (PyTorch). НЕ hot path, batch processing.
Использует C++ kernels от Agent 1 через custom torch.autograd.Function.

## Файлы (ЭКСКЛЮЗИВНЫЕ)
```
training/
├── muon.py                 ← Muon: quintic NS 3.4445x−4.775x³+2.0315x⁵
├── train_utils.py          ← PCGrad, gradient clipping, Z-loss
├── curriculum.py           ← 3-epoch + sequence packing
├── lora.py                 ← LoRA r=8, α=16, TIES merge
├── distill_from_teacher.py ← EMA ×T² + QA-KD
├── night_cycle.py          ← SPIN 4-iter + PoI gate
├── data_pipeline.py        ← 8-stage cleaning + MinHash
├── umot_loss.py            ← 5-component UMOT loss
├── dpo.py                  ← DPO β=0.1
└── custom_kernels.py       ← torch.autograd.Function → C++ kernels
local_train.py              ← Main training script
```

## Синхронизация с C++ ядром:
```python
# training/custom_kernels.py — МОСТ между PyTorch и C++ ядрами:

class BitNetMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W_ternary, x, alpha):
        ctx.save_for_backward(W_ternary, x, alpha)
        return tars_core.bitnet_matmul(W_ternary, x, alpha)  # C++ kernel!
    
    @staticmethod
    def backward(ctx, grad_output):
        W, x, alpha = ctx.saved_tensors
        # STE: ∂L/∂W = ∂L/∂W_q · 1 (pretend quantize didn't happen)
        grad_x = grad_output @ W.float() * alpha
        grad_W = grad_output.T @ x  # through STE
        return None, grad_x, None

# Таким образом training ИСПОЛЬЗУЕТ те же C++ ядра что и inference!
# Нет расхождения между train и inference → нет bugs!
```

## Задачи (без изменений + custom_kernels.py на неделе 1)
```
Неделя 1: muon.py, PCGrad, grad_clip, Z-loss + custom_kernels.py
Неделя 2: UMOT, EMA distill, curriculum, data_pipeline
Неделя 3: LoRA, DPO, night_cycle, local_train.py
```

---
---

# ═══════════════════════════════════════
# 🟡 AGENT 4: PYTHON MODEL + CONFIG + INFRA
# ═══════════════════════════════════════

## Зона ответственности
**Python reference model** (для training + fallback) + config + utils.
Модель на Python полностью синхронна с C++ ядром — те же формулы, другой язык.

## Файлы (ЭКСКЛЮЗИВНЫЕ)
```
brain/                          ← Python reference (for training!)
├── omega_core/
│   ├── model.py                ← TarsModel: 24 fractal blocks (PyTorch)
│   ├── block.py                ← TarsCoreBlock (Python reference)
│   ├── mole_router.py          ← MoLE 8 experts ← НЕЙРОНКА (3 матрицы)
│   ├── wuneng_fusion.py        ← FlashSigmoid ← НЕЙРОНКА (1 матрица)
│   └── mixture_of_depths.py    ← MoD ← НЕЙРОНКА (1 матрица)
├── mamba2/
│   ├── ssd.py                  ← SSD scan Python (mirrors ssd_scan.cpp!)
│   └── bitnet.py               ← BitNet Python (mirrors bitnet_matmul.cpp!)
├── min_gru/
│   ├── min_gru.py              ← MinGRU ← НЕЙРОНКА (small RNN)
│   └── spine_router.py         ← Spine ← НЕЙРОНКА (block skip predictor)
├── speculative.py              ← EAGLE-3 Python
├── doubt_engine.py             ← DoubtEngine ← НЕЙРОНКА (3 heads)
└── tokenizer.py                ← Qwen BPE wrapper

config.py                       ← Grand Unified Config
utils/
├── safe_file.py, checksummed_lora.py, disk_guardian.py
├── power_manager.py, tensor_pool.py, nan_guard.py
sensory/
├── input_sanitizer.py
requirements.txt
```

## КЛЮЧЕВОЕ: Синхронизация Python ↔ C++
```python
# brain/mamba2/ssd.py — Python reference:
def ssd_scan_python(state, gamma, B, x):
    """REFERENCE implementation — MUST match ssd_scan.cpp exactly!"""
    return gamma * state + B * x

# При inference:
#   if USE_CPP_CORE:
#       output = tars_core.ssd_scan(state, gamma, B, x)  # C++ 10×faster
#   else:
#       output = ssd_scan_python(state, gamma, B, x)      # Python fallback

# Тест синхронизации (Agent 5 пишет test):
#   assert allclose(python_output, cpp_output, atol=1e-5)
```

## Задачи
```
Неделя 1 (ПЕРВЫЙ ПРИОРИТЕТ — config!):
  [ ] config.py — ALL hyperparams (85 params)
  [ ] requirements.txt
  [ ] safe_file.py, tensor_pool.py, input_sanitizer.py

Неделя 2 (Python reference model):
  [ ] bitnet.py — Python UniversalLinear (reference для C++ kernel)
  [ ] ssd.py — Python SSD scan (reference для C++ kernel)
  [ ] block.py — TarsCoreBlock (uses Python kernels, switchable to C++)
  [ ] model.py — TarsModel: 24 fractal blocks

Неделя 3 (neural logic modules):
  [ ] wuneng_fusion.py — gate = σ(W·x), 1 матрица ← НЕЙРОНКА
  [ ] mole_router.py — softmax(W_g·x) → top-2, 3 матрицы ← НЕЙРОНКА
  [ ] mixture_of_depths.py — r = σ(W_r·x), 1 матрица ← НЕЙРОНКА
  [ ] doubt_engine.py — 3 × Linear(d_model, 1), 3 матрицы ← НЕЙРОНКА
  [ ] spine_router.py — MinGRU(h) → p_skip, RNN ← НЕЙРОНКА
  [ ] min_gru.py — mode router 3M params ← НЕЙРОНКА

Неделя 4:
  [ ] speculative.py — EAGLE-3 Python ref
  [ ] nan_guard.py, checksummed_lora.py, disk_guardian.py, power_manager.py
```

---
---

# ═══════════════════════════════════════
# 🟣 AGENT 5: AGENT + SAFETY + UI + TESTS
# ═══════════════════════════════════════

## Зона ответственности
Оболочка + tests (включая C++ ↔ Python синхронизацию!).

## Файлы (ЭКСКЛЮЗИВНЫЕ)
```
agent/tools/ + agent/safety/ + agent/orchestrator.py
ui/chat_ui.py + ui/cli.py
tests/
├── test_model.py           ← Python model tests
├── test_kernels.py         ← C++ kernel accuracy vs Python reference!
├── test_memory.py          ← Memory tests
├── test_training.py        ← Training pipeline tests
├── test_agent.py           ← Agent + tools tests
└── test_integration.py     ← Full pipeline: input → C++ engine → output
launch_tars.py
```

## КРИТИЧЕСКИЙ тест: C++ ↔ Python синхронизация
```python
# tests/test_kernels.py — Agent 5 ОБЯЗАН написать:

def test_ssd_scan_sync():
    """C++ kernel MUST match Python reference."""
    state = torch.randn(64)
    gamma, B, x = torch.rand(64), torch.randn(64), torch.randn(1280)
    
    py_result = ssd_scan_python(state, gamma, B, x)       # Python ref
    cpp_result = tars_core.ssd_scan(state, gamma, B, x)    # C++ kernel
    assert torch.allclose(py_result, cpp_result, atol=1e-5)

def test_bitnet_matmul_sync():
    """Ternary matmul MUST match Python reference."""
    W = torch.randint(-1, 2, (1280, 1280), dtype=torch.int8)
    x = torch.randn(1280)
    alpha = 0.5
    
    py_result = bitnet_forward_python(W, x, alpha)
    cpp_result = tars_core.bitnet_matmul(W, x, alpha)
    assert torch.allclose(py_result, cpp_result, atol=1e-4)

# Все 12 kernels — по одному тесту. Если расхождение → BUG!
```

---
---

# ═══════════════════════════════════════
# ⏰ TIMELINE v2 — 4 НЕДЕЛИ
# ═══════════════════════════════════════

```
              ДЕНЬ 1-2      ДЕНЬ 3-7         НЕДЕЛЯ 2          НЕДЕЛЯ 3         НЕДЕЛЯ 4
             ┌──────────┬───────────────┬─────────────────┬─────────────────┬────────────────┐
🟡 AGENT 4   │ config   │ safe_file     │ Py model: ssd   │ Neural modules  │ nan_guard      │
(Py Model)   │ ★ПЕРВЫЙ  │ tensor_pool   │ bitnet, block   │ MoLE, MoD, Doubt│ disk, power    │
             │ reqs.txt │ sanitizer     │ model.py 24blk  │ spine, mingru   │ speculative.py │
             ├──────────┼───────────────┼─────────────────┼─────────────────┼────────────────┤
🔴 AGENT 1   │ tensor.rs│ rmsnorm.cpp   │ bitnet_avx.S    │ attention_diff  │ engine.rs      │
(C++ Core)   │ arena.rs │ embed.cpp     │ ssd_scan.cpp    │ smoothquant.cpp │ eagle_decode.rs│
             │ build.zig│ softmax.cpp   │ wkv7.cpp        │ forward_pass.rs │ bindings (PyO3)│
             │          │               │ swiglu.cpp      │ generate.rs     │ test_kernels   │
             │          │               │ rope.cpp        │ tokenizer.rs    │ state_cache.rs │
             ├──────────┼───────────────┼─────────────────┼─────────────────┼────────────────┤
🟢 AGENT 2   │ wait     │ SDM write     │ LEANN+RETRO     │ Titans          │ doc_to_lora    │
(Memory)     │ config   │ SDM read      │ matryoshka      │ gaussian        │ MemTree        │
             │          │ compaction    │ context_mgr     │                 │                │
             ├──────────┼───────────────┼─────────────────┼─────────────────┼────────────────┤
🔵 AGENT 3   │ wait     │ muon+NS      │ UMOT loss       │ LoRA+DPO        │ night_cycle    │
(Training)   │ config   │ PCGrad       │ EMA distill     │ custom_kernels  │ local_train    │
             │          │ grad clip    │ curriculum      │ data_pipeline   │                │
             ├──────────┼───────────────┼─────────────────┼─────────────────┼────────────────┤
🟣 AGENT 5   │ ethical  │ tool_registry │ tools (5 files) │ orchestrator    │ launch_tars    │
(Agent+UI)   │ guard    │ timeout       │ prompt_defense  │ chat_ui + cli   │ ALL TESTS      │
             │ audit_log│               │                 │                 │ C++↔Py sync!   │
             └──────────┴───────────────┴─────────────────┴─────────────────┴────────────────┘

★ КРИТИЧЕСКИЙ ПУТЬ:
  Day 1:  Agent 4 config → ALL
  Day 3:  Agent 1 starts C++ kernels ║ Agent 4 starts Python model (PARALLEL!)
  Week 2: Agent 1 core kernels done  ║ Agent 4 Python model done
  Week 3: Agent 1 Rust runtime       ║ Agent 3 custom_kernels.py (bridge)
  Week 4: Agent 1 bindings           → Agent 5 integration + sync tests
```

---

# ═══════════════════════════════════════
# 🔗 СИНХРОНИЗАЦИЯ C++↔Python
# ═══════════════════════════════════════

```
ГАРАНТИЯ СИНХРОННОСТИ:

  1. ОДНИ И ТЕ ЖЕ формулы:
     Python (Agent 4):  s = gamma * s + B * x        ← для training
     C++ (Agent 1):     s = gamma * s + B * x        ← для inference
     
  2. ОДНИ И ТЕ ЖЕ веса:
     Python: model.save("weights.safetensors")        ← после training
     C++:    tars_core.load("weights.safetensors")     ← для inference
     
  3. ТЕСТЫ СИНХРОННОСТИ (Agent 5):
     assert allclose(python_forward(x), cpp_forward(x), atol=1e-5)
     для КАЖДОГО из 12 kernels!
     
  4. RUNTIME SWITCH:
     config.py: use_cpp_core = True    ← production: C++ ядро
                use_cpp_core = False   ← debugging: Python fallback
     
     if config.use_cpp_core:
         output = tars_core.generate(prompt_ids)     # 10-20ms/tok
     else:
         output = python_model.generate(prompt_ids)  # 200ms/tok (debug)
```

---

# ═══════════════════════════════════════
# 📋 ПРАВИЛА v2
# ═══════════════════════════════════════

```
ПРАВИЛО 1: НИКОГДА не редактировать файлы другого агента.

ПРАВИЛО 2: Agent 4 config → ПЕРВЫЙ. Agent 1 build system → ПЕРВЫЙ.

ПРАВИЛО 3: Каждый C++ kernel ОБЯЗАН иметь Python reference (Agent 4).
           Каждый Python ref ОБЯЗАН иметь C++ mirror (Agent 1).
           
ПРАВИЛО 4: Нейронка = логика. Код = вычисления. Нет if/else на hot path.

ПРАВИЛО 5: Git ветки: agent-1/cpp-core, agent-2/memory, 
           agent-3/training, agent-4/py-model, agent-5/agent-ui

ПРАВИЛО 6: Формат весов: .safetensors (единый для Python и C++).

ПРАВИЛО 7: Все routers/gates/heads — это МАТРИЦЫ (Linear layers).
           Они обучаются Agent 3. Выполняются Agent 1 (C++ kernel).
           Определяются Agent 4 (Python code). 3 агента, 1 формула.
```

---

> 🏭 **5 агентов. 3 слоя. 0 if/else на inference.**
> **C++/Rust ядро: 10-20ms/tok. Python: training + tools + UI.**
> **Нейронка = ВСЯ логика. Код = ТОЛЬКО вычисления.**
> **12 C++ kernels ↔ 12 Python references ↔ 12 sync tests.**
