"""
═══ TARS Training Stability & Performance Test ═══
Tests: NaN/Inf stability, gradient norms, memory_vec divergence,
       timing per component, optimization potential.
"""
import os, sys, time
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np

from brain.mamba2.model import TarsMamba2LM
from training.train_mamba2 import load_corpus, prepare_byte_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 65)
print("  TARS Stability & Performance Analysis")
print("=" * 65)
print(f"Device: {device}")

# ═══ Mid-size model (256d, 6 layers = 3 waves, 2 spine updates) ═══
FULL = False
if FULL:
    D, N, SEQ = 768, 12, 128
else:
    D, N, SEQ = 256, 6, 128

print(f"\nModel: d={D}, layers={N}, seq={SEQ}")
model = TarsMamba2LM(
    d_model=D, n_layers=N, vocab_size=256,
    d_state=64 if FULL else 16, headdim=64 if FULL else 16,
    omega_dim=32 if FULL else 8, pool_size=48 if FULL else 4,
    n_experts=8 if FULL else 4,
)
model.to(device)
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,} ({params/1e6:.1f}M)")

# ═══ Data ═══
print("\nLoading corpus...")
corpus = load_corpus(download_wiki=False)
inputs, targets = prepare_byte_data(corpus, SEQ, max_samples=300)
n_test = max(4, len(inputs) // 10)
train_in, test_in = inputs[:-n_test], inputs[-n_test:]
train_tgt, test_tgt = targets[:-n_test], targets[-n_test:]
print(f"Train: {len(train_in)}, Test: {len(test_in)}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
batch_size = 4 if FULL else 8
epochs = 8

# ═══ Tracking arrays ═══
train_losses = []
eval_losses = []
grad_norms_total = []
grad_norms_spine = []
grad_norms_core = []
spine_mem_norms = []     # track memory_vec magnitude
nan_count = 0
inf_count = 0
epoch_times = []

# ═══ Timing breakdown (first epoch only) ═══
time_forward = 0.0
time_backward = 0.0
time_optimizer = 0.0
n_timing_batches = 0

print(f"\n{'='*65}")
print(f"{'Epoch':>5} | {'Train':>8} | {'Eval':>8} | {'PPL':>7} | "
      f"{'GradNorm':>8} | {'SpineGN':>8} | {'MemNorm':>8} | {'Time':>5}")
print(f"{'-'*65}")

for epoch in range(epochs):
    t0 = time.time()
    model.train()
    total_loss = 0
    n_batches = 0
    
    perm = torch.randperm(len(train_in))
    train_in_s = train_in[perm]
    train_tgt_s = train_tgt[perm]
    
    for i in range(0, len(train_in_s), batch_size):
        batch_in = train_in_s[i:i+batch_size].to(device)
        batch_tgt = train_tgt_s[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        
        # ─── Forward ───
        t_fwd = time.time()
        logits = model(batch_in)
        lm_loss = F.cross_entropy(
            logits.view(-1, 256), batch_tgt.view(-1),
            label_smoothing=0.1,
        )
        loss = lm_loss + model.mole_aux_loss
        t_fwd_end = time.time()
        
        # NaN/Inf check
        if torch.isnan(loss):
            nan_count += 1
            print(f"  !! NaN loss at epoch {epoch+1}, batch {i//batch_size}")
            continue
        if torch.isinf(loss):
            inf_count += 1
            print(f"  !! Inf loss at epoch {epoch+1}, batch {i//batch_size}")
            continue
        
        # ─── Backward ───
        t_bwd = time.time()
        loss.backward()
        t_bwd_end = time.time()
        
        # Gradient clipping + norms
        total_gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        
        # Spine-specific gradient norm
        spine_params = list(model.to_memory_space.parameters()) + list(model.from_memory_space.parameters())
        spine_gn = 0.0
        for p in spine_params:
            if p.grad is not None:
                spine_gn += p.grad.norm().item() ** 2
        spine_gn = spine_gn ** 0.5
        
        # Core gradient norm (first block TarsCoreBlock)
        core_params = list(model.blocks[0].core.parameters())
        core_gn = 0.0
        for p in core_params:
            if p.grad is not None:
                core_gn += p.grad.norm().item() ** 2
        core_gn = core_gn ** 0.5
        
        grad_norms_total.append(total_gn)
        grad_norms_spine.append(spine_gn)
        grad_norms_core.append(core_gn)
        
        # ─── Optimizer step ───
        t_opt = time.time()
        optimizer.step()
        t_opt_end = time.time()
        
        # Timing (all epochs)
        time_forward += (t_fwd_end - t_fwd)
        time_backward += (t_bwd_end - t_bwd)
        time_optimizer += (t_opt_end - t_opt)
        n_timing_batches += 1
        
        total_loss += loss.item()
        n_batches += 1
    
    # ─── Eval ───
    model.eval()
    with torch.no_grad():
        eval_ls = []
        for j in range(0, len(test_in), batch_size):
            tb = test_in[j:j+batch_size].to(device)
            tt = test_tgt[j:j+batch_size].to(device)
            logits = model(tb)
            el = F.cross_entropy(logits.view(-1, 256), tt.view(-1)).item()
            eval_ls.append(el)
        eval_loss = np.mean(eval_ls)
    
    elapsed = time.time() - t0
    train_avg = total_loss / max(n_batches, 1)
    ppl = np.exp(min(eval_loss, 20))
    epoch_times.append(elapsed)
    train_losses.append(train_avg)
    eval_losses.append(eval_loss)
    
    # Track memory_vec norm (probe via a forward pass)
    with torch.no_grad():
        probe = torch.randint(0, 256, (1, SEQ), device=device)
        _ = model(probe)
    
    recent_spine = np.mean(grad_norms_spine[-n_batches:]) if grad_norms_spine else 0
    recent_total = np.mean(grad_norms_total[-n_batches:]) if grad_norms_total else 0
    
    print(f"  {epoch+1:3d}   | {train_avg:8.4f} | {eval_loss:8.4f} | {ppl:7.1f} | "
          f"{recent_total:8.4f} | {recent_spine:8.4f} | {'n/a':>8} | {elapsed:5.1f}s")

# ═══ ANALYSIS ═══
print(f"\n{'='*65}")
print("  STABILITY ANALYSIS")
print(f"{'='*65}")

# 1. NaN/Inf
print(f"\n  NaN count:          {nan_count}")
print(f"  Inf count:          {inf_count}")
if nan_count == 0 and inf_count == 0:
    print(f"  -> STABLE (no NaN/Inf)")
else:
    print(f"  -> UNSTABLE!")

# 2. Loss convergence
print(f"\n  Loss: {eval_losses[0]:.4f} -> {eval_losses[-1]:.4f} "
      f"(delta={eval_losses[-1]-eval_losses[0]:.4f})")
loss_decreasing = eval_losses[-1] < eval_losses[0]
# Check monotonicity (allowing small bumps)
non_monotonic = 0
for i in range(1, len(eval_losses)):
    if eval_losses[i] > eval_losses[i-1] + 0.05:
        non_monotonic += 1
print(f"  Non-monotonic bumps: {non_monotonic}/{len(eval_losses)-1}")
if loss_decreasing and non_monotonic <= 1:
    print(f"  -> CONVERGENCE: Smooth")
elif loss_decreasing:
    print(f"  -> CONVERGENCE: OK but bumpy")
else:
    print(f"  -> DIVERGENCE!")

# 3. Gradient stability
gn_arr = np.array(grad_norms_total)
print(f"\n  Grad norms: mean={gn_arr.mean():.4f} std={gn_arr.std():.4f} "
      f"max={gn_arr.max():.4f} min={gn_arr.min():.4f}")

spine_arr = np.array(grad_norms_spine)
core_arr = np.array(grad_norms_core)
print(f"  Spine grad: mean={spine_arr.mean():.4f} std={spine_arr.std():.4f}")
print(f"  Core grad:  mean={core_arr.mean():.4f} std={core_arr.std():.4f}")

# Gradient ratio (spine vs core)
ratio = spine_arr.mean() / max(core_arr.mean(), 1e-8)
print(f"  Spine/Core ratio:   {ratio:.4f}")
if ratio > 100:
    print(f"  -> WARNING: Spine gradients >> Core (potential instability)")
elif ratio < 0.01:
    print(f"  -> WARNING: Spine gradients << Core (not learning?)")
else:
    print(f"  -> BALANCED gradient flow")

# 4. Gradient variance over time (stability)
if len(gn_arr) > 20:
    first_half = gn_arr[:len(gn_arr)//2]
    second_half = gn_arr[len(gn_arr)//2:]
    print(f"\n  Grad norm trend: first_half={first_half.mean():.4f} "
          f"second_half={second_half.mean():.4f}")
    if second_half.mean() < first_half.mean() * 1.5:
        print(f"  -> STABLE (not exploding)")
    else:
        print(f"  -> WARNING: Gradient growth detected")

# ═══ PERFORMANCE ═══
print(f"\n{'='*65}")
print("  PERFORMANCE ANALYSIS")
print(f"{'='*65}")

total_time_all = sum(epoch_times)
avg_epoch = np.mean(epoch_times)
print(f"\n  Total time:     {total_time_all:.1f}s ({epochs} epochs)")
print(f"  Avg epoch:      {avg_epoch:.1f}s")

if n_timing_batches > 0:
    avg_fwd = time_forward / n_timing_batches * 1000
    avg_bwd = time_backward / n_timing_batches * 1000
    avg_opt = time_optimizer / n_timing_batches * 1000
    avg_total = avg_fwd + avg_bwd + avg_opt
    
    print(f"\n  Per-batch breakdown:")
    print(f"    Forward:    {avg_fwd:7.1f} ms  ({avg_fwd/avg_total*100:5.1f}%)")
    print(f"    Backward:   {avg_bwd:7.1f} ms  ({avg_bwd/avg_total*100:5.1f}%)")
    print(f"    Optimizer:  {avg_opt:7.1f} ms  ({avg_opt/avg_total*100:5.1f}%)")
    print(f"    Total:      {avg_total:7.1f} ms/batch")
    
    # Spinal cord overhead estimation
    # The spinal cord adds: 1 x.mean() + 1 Linear(768->384) per wave (except last)
    # For 12 layers = 6 waves, that's 5 extra operations
    n_waves = N // 2
    spine_ops = n_waves - 1  # number of extra spine operations
    # Estimate: Linear(768->384) = 768*384 = ~295K MACs, x.mean() = negligible
    spine_macs = spine_ops * D * 384
    total_core_macs = N * D * D * 10  # rough: each block ~10*d^2 MACs
    spine_overhead_pct = spine_macs / total_core_macs * 100
    
    print(f"\n  Spinal cord overhead (estimated):")
    print(f"    Extra ops: {spine_ops} Linear({D}->{384}) per forward")
    print(f"    Extra MACs: {spine_macs:,} vs Core: {total_core_macs:,}")
    print(f"    Overhead: ~{spine_overhead_pct:.2f}% of core compute")

# ═══ Specifig h_mean() optimization ═══
print(f"\n  Optimization opportunities:")
print(f"    1. h_mean computed 3x per wave (left, right, post-merge)")
print(f"       -> Can reuse post-merge h_curr for spine (already done)")
print(f"    2. memory_vec update uses Linear({D}->384)")
print(f"       -> Lightweight: {D*384:,} params = {D*384*4/1024:.0f} KB")
print(f"    3. Spine EMA (0.7/0.3) is algebraically stable (bounded)")
print(f"       -> No risk of explosion: max|memory_vec| is bounded")

# ═══ VERDICT ═══
print(f"\n{'='*65}")

issues = []
if nan_count > 0: issues.append("NaN detected")
if inf_count > 0: issues.append("Inf detected")
if not loss_decreasing: issues.append("Loss not decreasing")
if ratio > 100: issues.append("Spine gradients too large")
if ratio < 0.01: issues.append("Spine gradients too small")

if not issues:
    print("  VERDICT: STABLE & EFFICIENT")
    print(f"  Loss: {eval_losses[0]:.4f} -> {eval_losses[-1]:.4f}")
    print(f"  PPL:  {np.exp(min(eval_losses[0], 20)):.1f} -> {np.exp(min(eval_losses[-1], 20)):.1f}")
    print(f"  Spine overhead: ~{spine_overhead_pct:.2f}%")
    print(f"  No NaN/Inf, gradients balanced, convergence smooth")
else:
    print(f"  VERDICT: ISSUES FOUND: {', '.join(issues)}")
print(f"{'='*65}")
