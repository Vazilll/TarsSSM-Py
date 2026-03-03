"""
═══════════════════════════════════════════════════════════════
  Export PyTorch Model → TarsSSM(rs) Format
═══════════════════════════════════════════════════════════════

Exports trained Mamba-2 model weights from PyTorch (.pt) to a flat
binary format (.tars) that can be memory-mapped by the Rust engine.

Usage:
  python export_to_rust.py --model models/mamba2/brain_best.pt
                           --output ../TarsSSM(rs)/models/brain.tars

Format (.tars):
  Header (256 bytes):
    magic:       [u8; 4]    = "TARS"
    version:     u32        = 1
    d_model:     u32        = 2048
    n_layers:    u32        = 24
    vocab_size:  u32        = 256
    quant_mode:  u32        = 158 (1.58-bit)
    n_tensors:   u32
    pad:         [u8; ...]

  Tensor Index (n_tensors × 64 bytes each):
    name:        [u8; 48]   = "blocks.0.ssd.in_proj.weight"
    dtype:       u8         = 0=f32, 1=f16, 2=i8(1.58-bit)
    ndim:        u8
    shape:       [u32; 4]
    offset:      u64        = byte offset in data section

  Data Section:
    Raw tensor data, aligned to 64 bytes (cache-line aligned)
"""

import os
import sys
import struct
import argparse
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


TARS_MAGIC = b"TARS"
TARS_VERSION = 1

# Dtype codes
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_I8 = 2   # 1.58-bit quantized


def quantize_to_158bit(weight: torch.Tensor) -> np.ndarray:
    """
    Quantize fp32 weight → 1.58-bit ({-1, 0, +1}) as int8.
    
    Method: abs-mean scaling + round to nearest trit.
    """
    w = weight.float()
    scale = w.abs().mean()
    if scale < 1e-8:
        return np.zeros(w.shape, dtype=np.int8)
    
    w_scaled = w / scale
    # Round to {-1, 0, +1}
    w_trit = w_scaled.round().clamp(-1, 1).to(torch.int8)
    return w_trit.numpy()


def export_model(model_path: str, output_path: str, quantize: bool = True):
    """Export PyTorch .pt model to .tars format."""
    print(f"Loading {model_path}...")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract state dict
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    
    # Filter to actual tensor parameters
    tensors = {}
    for name, param in state.items():
        if isinstance(param, torch.Tensor) and param.numel() > 0:
            tensors[name] = param
    
    print(f"Found {len(tensors)} tensors")
    
    # Detect model config from weight shapes
    d_model = 2048
    n_layers = 24
    vocab_size = 256
    
    for name, t in tensors.items():
        if 'embedding' in name and t.dim() == 2:
            vocab_size, d_model = t.shape
        if 'blocks.' in name:
            layer_idx = int(name.split('.')[1])
            n_layers = max(n_layers, layer_idx + 1)
    
    print(f"Config: d_model={d_model}, n_layers={n_layers}, vocab={vocab_size}")
    
    # ── Build tensor index ──
    index_entries = []
    data_parts = []
    current_offset = 0
    
    total_original = 0
    total_quantized = 0
    
    for name, param in sorted(tensors.items()):
        shape = list(param.shape)
        ndim = len(shape)
        
        # Determine dtype and convert
        if quantize and 'weight' in name and param.dim() >= 2 and 'norm' not in name and 'embedding' not in name:
            # Quantize large weight matrices to 1.58-bit
            data = quantize_to_158bit(param)
            dtype_code = DTYPE_I8
            total_original += param.numel() * 4  # fp32
            total_quantized += data.nbytes
        else:
            # Keep as f32 (norms, biases, embeddings)
            data = param.float().numpy()
            dtype_code = DTYPE_F32
            total_original += data.nbytes
            total_quantized += data.nbytes
        
        data_bytes = data.tobytes()
        
        # Align to 64 bytes (cache line)
        pad_needed = (64 - (len(data_bytes) % 64)) % 64
        data_bytes += b'\x00' * pad_needed
        
        # Pad shape to 4 dims
        shape_padded = shape + [1] * (4 - len(shape))
        
        index_entries.append({
            'name': name[:48],
            'dtype': dtype_code,
            'ndim': ndim,
            'shape': shape_padded[:4],
            'offset': current_offset,
        })
        
        data_parts.append(data_bytes)
        current_offset += len(data_bytes)
    
    # ── Write output ──
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'wb') as f:
        # Header (256 bytes)
        header = struct.pack(
            '<4sIIIII',
            TARS_MAGIC,
            TARS_VERSION,
            d_model,
            n_layers,
            vocab_size,
            158 if quantize else 32,  # quant_mode
        )
        header += struct.pack('<I', len(index_entries))  # n_tensors
        header += b'\x00' * (256 - len(header))  # pad to 256
        f.write(header)
        
        # Tensor Index (64 bytes per entry)
        for entry in index_entries:
            name_bytes = entry['name'].encode('utf-8')[:48]
            name_bytes += b'\x00' * (48 - len(name_bytes))
            
            idx = struct.pack(
                '<48sBB4IQ',
                name_bytes,
                entry['dtype'],
                entry['ndim'],
                *entry['shape'],
                entry['offset'],
            )
            f.write(idx)
        
        # Data Section
        for part in data_parts:
            f.write(part)
    
    file_size = os.path.getsize(output_path)
    ratio = total_quantized / max(total_original, 1) * 100
    
    print(f"\n✓ Exported: {output_path}")
    print(f"  Tensors: {len(tensors)}")
    print(f"  Original: {total_original / 1024 / 1024:.1f} MB (fp32)")
    print(f"  Quantized: {total_quantized / 1024 / 1024:.1f} MB (1.58-bit)")
    print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
    print(f"  Compression: {ratio:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch → TarsSSM(rs)")
    parser.add_argument("--model", type=str, default="models/mamba2/brain_best.pt")
    parser.add_argument("--output", type=str, default="../TarsSSM(rs)/models/brain.tars")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()
    
    export_model(args.model, args.output, quantize=not args.no_quantize)
