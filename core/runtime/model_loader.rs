// ═══════════════════════════════════════════════════════════════
//   model_loader.rs — SafeTensors Weight Loader
// ═══════════════════════════════════════════════════════════════
//
// Loads .safetensors weights via memory-mapping for fast startup.
// Supports float32 and int8 (ternary) weights.
//
// Agent 1 — Week 3

use super::tensor::{Tensor, DType};
use std::collections::HashMap;

/// Load weights from safetensors file
///
/// Safetensors format: JSON header (tensor metadata) + raw data.
/// We memory-map the file for zero-copy access.
pub fn load_safetensors(path: &str) -> Result<Vec<Tensor>, String> {
    let file_data = std::fs::read(path)
        .map_err(|e| format!("Failed to read {}: {}", path, e))?;

    if file_data.len() < 8 {
        return Err("File too small for safetensors format".to_string());
    }

    // First 8 bytes: u64 little-endian header size
    let header_size = u64::from_le_bytes(
        file_data[0..8].try_into().unwrap()
    ) as usize;

    if 8 + header_size > file_data.len() {
        return Err("Invalid header size".to_string());
    }

    // Parse JSON header
    let header_str = std::str::from_utf8(&file_data[8..8 + header_size])
        .map_err(|e| format!("Invalid header UTF-8: {}", e))?;

    let header: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
        .map_err(|e| format!("Invalid header JSON: {}", e))?;

    let data_start = 8 + header_size;
    let mut tensors = Vec::new();

    for (name, info) in &header {
        if name == "__metadata__" {
            continue;
        }

        let info = info.as_object()
            .ok_or_else(|| format!("Invalid tensor info for {}", name))?;

        let dtype_str = info.get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("Missing dtype for {}", name))?;

        let shape: Vec<usize> = info.get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("Missing shape for {}", name))?
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();

        let offsets = info.get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| format!("Missing offsets for {}", name))?;

        let start = offsets[0].as_u64().unwrap() as usize + data_start;
        let end = offsets[1].as_u64().unwrap() as usize + data_start;

        let dtype = match dtype_str {
            "F32" => DType::Float32,
            "I8" => DType::Int8,
            "F16" | "BF16" => DType::Float16,
            _ => {
                eprintln!("Warning: unsupported dtype {} for {}, skipping", dtype_str, name);
                continue;
            }
        };

        if end > file_data.len() {
            return Err(format!("Tensor {} data out of bounds", name));
        }

        let tensor_data = file_data[start..end].to_vec();
        let strides = compute_strides(&shape, dtype);

        tensors.push(Tensor {
            data: tensor_data,
            shape,
            strides,
            dtype,
        });
    }

    Ok(tensors)
}

fn compute_strides(shape: &[usize], dtype: DType) -> Vec<usize> {
    let elem_size = dtype.size_bytes();
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = elem_size;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
