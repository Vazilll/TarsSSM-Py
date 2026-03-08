// ═══════════════════════════════════════════════════════════════
//   tensor.rs — Lightweight Tensor for TARS Rust runtime
// ═══════════════════════════════════════════════════════════════
//
// Minimal tensor struct: raw data pointer + shape + stride + dtype.
// No autograd — inference only. Zero-copy views via slicing.
//
// Agent 1 — Week 1

use std::fmt;

/// Supported data types for tensors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    Float32,
    Int8,      // BitNet ternary / INT8 quantized
    Float16,   // Half precision (stored as u16)
}

impl DType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Int8 => 1,
            DType::Float16 => 2,
        }
    }
}

/// Lightweight inference tensor — owns its data or borrows via slice
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<u8>,       // Raw bytes
    pub shape: Vec<usize>,   // Dimensions
    pub strides: Vec<usize>, // Byte strides per dimension
    pub dtype: DType,
}

impl Tensor {
    /// Create a new zero-initialized tensor
    pub fn zeros(shape: &[usize], dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let nbytes = numel * dtype.size_bytes();
        let strides = Self::compute_strides(shape, dtype);
        Tensor {
            data: vec![0u8; nbytes],
            shape: shape.to_vec(),
            strides,
            dtype,
        }
    }

    /// Create tensor from float32 data
    pub fn from_f32(data: &[f32], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "Data length mismatch");
        let bytes: Vec<u8> = data.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let strides = Self::compute_strides(shape, DType::Float32);
        Tensor {
            data: bytes,
            shape: shape.to_vec(),
            strides,
            dtype: DType::Float32,
        }
    }

    /// Create tensor from int8 data (BitNet ternary weights)
    pub fn from_i8(data: &[i8], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "Data length mismatch");
        let bytes: Vec<u8> = data.iter().map(|&x| x as u8).collect();
        let strides = Self::compute_strides(shape, DType::Int8);
        Tensor {
            data: bytes,
            shape: shape.to_vec(),
            strides,
            dtype: DType::Int8,
        }
    }

    /// Get float32 slice (panics if wrong dtype)
    pub fn as_f32(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::Float32);
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.data.len() / 4,
            )
        }
    }

    /// Get mutable float32 slice
    pub fn as_f32_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype, DType::Float32);
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut f32,
                self.data.len() / 4,
            )
        }
    }

    /// Get int8 slice
    pub fn as_i8(&self) -> &[i8] {
        assert_eq!(self.dtype, DType::Int8);
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const i8,
                self.data.len(),
            )
        }
    }

    /// Number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Compute contiguous strides (row-major / C order)
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

    /// Reshape (must preserve numel)
    pub fn reshape(&mut self, new_shape: &[usize]) {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_numel, "Reshape: numel mismatch");
        self.shape = new_shape.to_vec();
        self.strides = Self::compute_strides(&self.shape, self.dtype);
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Tensor({:?}, {:?}, {}B)",
               self.shape, self.dtype, self.data.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros(&[3, 4], DType::Float32);
        assert_eq!(t.numel(), 12);
        assert_eq!(t.data.len(), 48);
        assert_eq!(t.as_f32().len(), 12);
    }

    #[test]
    fn test_from_f32() {
        let data: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let t = Tensor::from_f32(&data, &[2, 3]);
        assert_eq!(t.as_f32(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_ternary() {
        let data: Vec<i8> = vec![-1, 0, 1, 1, -1, 0];
        let t = Tensor::from_i8(&data, &[2, 3]);
        assert_eq!(t.as_i8(), &[-1, 0, 1, 1, -1, 0]);
    }
}
