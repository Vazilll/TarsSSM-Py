"""Quick SNN verification test."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

print("Test 1: Import...", end=" ")
from brain.spiking.spiking_synapse import SI_LIF, SpikingMinGRUBlock
print("OK")

print("Test 2: SI-LIF...", end=" ")
lif = SI_LIF(64)
s, m = lif(torch.randn(1, 64))
vals = sorted(s.unique().tolist())
assert all(v in [-1.0, 0.0, 1.0] for v in vals), f"Bad values: {vals}"
print(f"OK (vals={vals})")

print("Test 3: Sequence...", end=" ")
s2, m2 = lif(torch.randn(1, 5, 64))
assert s2.shape == (1, 5, 64)
print(f"OK ({s2.shape})")

print("Test 4: Backward...", end=" ")
x = torch.randn(1, 32, requires_grad=True)
s3, _ = SI_LIF(32)(x)
s3.sum().backward()
assert x.grad is not None
print(f"OK (grads={(x.grad!=0).sum().item()}/{x.grad.numel()})")

print("Test 5: SpikingMinGRUBlock...", end=" ")
block = SpikingMinGRUBlock(128, num_heads=2)
o, st = block(torch.randn(1, 3, 128))
assert o.shape == (1, 3, 128)
p = sum(p.numel() for p in block.parameters())
print(f"OK (shape={o.shape}, params={p:,})")

print("Test 6: Block backward...", end=" ")
o2, _ = block(torch.randn(1, 2, 128))
loss = o2.sum()
loss.backward()
gc = sum(1 for p in block.parameters() if p.grad is not None)
print(f"OK ({gc} params have grads)")

print("\n=== ALL 6 TESTS PASSED ===")
