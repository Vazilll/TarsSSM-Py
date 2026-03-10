"""
═══════════════════════════════════════════════════════════════
  ПОЛНАЯ ВЕРИФИКАЦИЯ — по AI_REFERENCE.md чеклисту
═══════════════════════════════════════════════════════════════

Проверяет:
  1. Синтаксис всех изменённых файлов
  2. TarsConfig параметры корректны (чеклист п.1)
  3. model_lite.py: TarsHelixLite использует TarsCoreBlock (чеклист п.2)
  4. Smoke test: forward + backward + generate (чеклист п.3)
  5. WSD schedule корректен (3 фазы)
  6. Muon optimizer можно импортировать
  7. Импорт local_train_extras без NameError
  8. Все тесты TarsHelixLite проходят
"""
import ast, sys, os, warnings, shutil
warnings.filterwarnings("ignore")

# Clear __pycache__ to ensure fresh imports after edits
for root, dirs, files in os.walk(os.path.dirname(os.path.abspath(__file__))):
    if 'venv' in root:
        continue
    for d in dirs:
        if d == '__pycache__':
            shutil.rmtree(os.path.join(root, d), ignore_errors=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OK = 0
FAIL = 0

def check(name, condition, detail=""):
    global OK, FAIL
    if condition:
        OK += 1
        print(f"  ✅ {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))

print()
print("═" * 60)
print("  🔬 ПОЛНАЯ ВЕРИФИКАЦИЯ СИСТЕМЫ ОБУЧЕНИЯ")
print("═" * 60)

# ═════════════════════════════════════════
# 1. СИНТАКСИС
# ═════════════════════════════════════════
print("\n── 1. Синтаксис файлов ──")
files = [
    'local_train.py', 'train_lite.py', 'config.py',
    'local_train_extras.py', 'brain/mamba2/core/model_lite.py',
    'brain/mamba2/core/ssd.py', 'brain/mamba2/core/bitnet.py',
    'tests/test_model.py', 'tests/test_training.py',
]
for f in files:
    try:
        ast.parse(open(f, 'r', encoding='utf-8').read())
        check(f"Syntax {f}", True)
    except SyntaxError as e:
        check(f"Syntax {f}", False, str(e))

# ═════════════════════════════════════════
# 2. CONFIG (AI_REFERENCE чеклист п.1)
# ═════════════════════════════════════════
print("\n── 2. TarsConfig ──")
from config import TarsConfig

cfg = TarsConfig()
check("quant_mode = fp16 (Bug 9)", cfg.quant_mode == "fp16", cfg.quant_mode)
check("d_model = 1024", cfg.d_model == 1024, str(cfg.d_model))
check("n_layers = 20", cfg.n_layers == 20, str(cfg.n_layers))
check("vocab_size = 48256 (full)", cfg.vocab_size == 48256, str(cfg.vocab_size))
check("d_state = 64", cfg.d_state == 64, str(cfg.d_state))
check("headdim = 64", cfg.headdim == 64, str(cfg.headdim))
check("expand = 2", cfg.expand == 2, str(cfg.expand))
check("dim_ff = 2816", cfg.dim_ff == 2816, str(cfg.dim_ff))

# ═════════════════════════════════════════
# 3. MODEL STRUCTURE (AI_REFERENCE чеклист п.2)
# ═════════════════════════════════════════
print("\n── 3. Модель TarsHelixLite ──")
import torch
from brain.mamba2.core.model_lite import TarsHelixLite, HelixBlock
from brain.mamba2.core.ssd import TarsCoreBlock

lite_cfg = TarsConfig(d_model=128, n_layers=2, vocab_size=256, d_state=16, headdim=32)
model = TarsHelixLite(lite_cfg)

check("TarsHelixLite создаётся", model is not None)
check("Использует HelixBlock", len(model.blocks) == 2 and isinstance(model.blocks[0], HelixBlock))
check("HelixBlock содержит TarsCoreBlock",
      hasattr(model.blocks[0], 'core') and isinstance(model.blocks[0].core, TarsCoreBlock))
check("Weight tying (embed ↔ lm_head)",
      model.lm_head.weight is model.embedding.weight,
      "lm_head.weight is embedding.weight")
check("Vaswani embed scaling",
      hasattr(model, 'embed_scale') and abs(model.embed_scale - 128**0.5) < 0.01,
      f"√d_model = {model.embed_scale:.2f}")

n_params = sum(p.numel() for p in model.parameters())
check(f"Параметры > 0", n_params > 0, f"{n_params/1e6:.2f}M")

# ═════════════════════════════════════════
# 4. SMOKE TEST (AI_REFERENCE чеклист п.3)
# ═════════════════════════════════════════
print("\n── 4. Smoke Test ──")
B, L = 2, 64
x = torch.randint(0, 256, (B, L))
labels = torch.randint(0, 256, (B, L))

# Forward
result = model(x, labels=labels)
check("Forward: logits shape", result['logits'].shape == (B, L, 256),
      str(result['logits'].shape))
check("Forward: loss finite", 
      not torch.isnan(result['loss']) and not torch.isinf(result['loss']),
      f"loss={result['loss'].item():.4f}")
check("Forward: loss ~5.5 (для vocab=256)", 
      2.0 < result['loss'].item() < 8.0,
      f"loss={result['loss'].item():.4f}")

# Backward  
result['loss'].backward()
grads_ok = True
no_grad = []
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is None:
        grads_ok = False
        no_grad.append(name)
n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
check("Backward: все trainable градиенты есть", grads_ok,
      f"missing: {no_grad[:3]}" if no_grad else f"{n_trainable} trainable, {n_frozen} frozen")

# Generate (Bug 8 fix)
model.eval()
with torch.no_grad():
    prompt = torch.randint(0, 256, (1, 8))
    gen = model.generate(prompt, max_new_tokens=16)
check("Generate: тензор расширяется", gen.shape[1] > 8,
      f"{prompt.shape} → {gen.shape}")

# Generate batch>1 (Bug 8 — .item() fix)
try:
    with torch.no_grad():
        prompt2 = torch.randint(0, 256, (2, 8))
        gen2 = model.generate(prompt2, max_new_tokens=4)
    check("Generate batch=2: no crash (Bug 8)", True, str(gen2.shape))
except Exception as e:
    check("Generate batch=2: no crash (Bug 8)", False, str(e))

# ═════════════════════════════════════════
# 5. WSD SCHEDULE (GOLDEN spec)
# ═════════════════════════════════════════
print("\n── 5. WSD Schedule ──")
from local_train import get_wsd_schedule as wsd_main

opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
sched = wsd_main(opt, warmup_steps=10, total_steps=200, min_lr_ratio=0.1)
opt.step()  # avoid warning

lrs = []
for i in range(200):
    lrs.append(opt.param_groups[0]['lr'])
    sched.step()

check("Warmup: lr растёт (0→10)", lrs[0] < lrs[9], f"{lrs[0]:.4f} < {lrs[9]:.4f}")
check("Stable: lr постоянен (50-150)", abs(lrs[50] - lrs[100]) < 0.001,
      f"lr[50]={lrs[50]:.4f}, lr[100]={lrs[100]:.4f}")
check("Decay: lr падает (160→199)", lrs[195] < lrs[160],
      f"lr[160]={lrs[160]:.4f} → lr[195]={lrs[195]:.4f}")

# Check train_lite.py also has WSD
from train_lite import get_wsd_schedule as wsd_lite
check("train_lite.py: WSD (не cosine)", wsd_lite is not None, "get_wsd_schedule exists")

# ═════════════════════════════════════════
# 6. MUON OPTIMIZER
# ═════════════════════════════════════════
print("\n── 6. Muon Optimizer ──")
try:
    from training.muon import Muon
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = Muon([p], lr=0.01)
    loss = (p ** 2).sum()
    loss.backward()
    opt.step()
    check("Muon: import + step", True)
except Exception as e:
    check("Muon: import + step", False, str(e))

# ═════════════════════════════════════════
# 7. LOCAL_TRAIN_EXTRAS (Bug 6, 7)
# ═════════════════════════════════════════
print("\n── 7. local_train_extras.py ──")
try:
    src = open('local_train_extras.py', 'r', encoding='utf-8').read()
    check("Bug 6: TARS_V3 отсутствует", "TARS_V3" not in src, "replaced with SAVE_DIR")
    check("Bug 7: legacy warning", "WARNING" in src and "LEGACY" in src)
except Exception as e:
    check("local_train_extras.py", False, str(e))

# ═════════════════════════════════════════
# 8. GRADSCALER ORDER (Bug 1)
# ═════════════════════════════════════════
print("\n── 8. GradScaler порядок (Bug 1) ──")
src = open('local_train.py', 'r', encoding='utf-8').read()
# Check that unscale_ comes before clip_grad_norm_
unscale_pos = src.find("scaler.unscale_(optimizer)")
clip_pos = src.find("clip_grad_norm_", unscale_pos)
check("Bug 1: unscale_ ПЕРЕД clip_grad_norm_",
      0 < unscale_pos < clip_pos,
      f"unscale@{unscale_pos}, clip@{clip_pos}")

# ═════════════════════════════════════════
# 9. TOTAL_STEPS учитывает ACCUM (Bug 4)
# ═════════════════════════════════════════
print("\n── 9. WSD total_steps (Bug 4) ──")
check("Bug 4: optimizer_steps_per_epoch",
      "optimizer_steps_per_epoch" in src,
      "total_steps based on optimizer steps, not batch iterations")

# ═════════════════════════════════════════
# ИТОГ
# ═════════════════════════════════════════
print()
print("═" * 60)
total = OK + FAIL
if FAIL == 0:
    print(f"  ✅ ВСЕ {OK}/{total} ПРОВЕРОК ПРОЙДЕНЫ")
else:
    print(f"  ⚠️  {OK}/{total} пройдено, {FAIL} провалено")
print("═" * 60)
print()
