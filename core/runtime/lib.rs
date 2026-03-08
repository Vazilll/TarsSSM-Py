// ═══════════════════════════════════════════════════════════════
//   lib.rs — PyO3 Bindings: Rust ↔ Python Bridge
// ═══════════════════════════════════════════════════════════════
//
// Exposes TarsEngine and all kernels to Python via `import tars_core`.
//
// Usage from Python:
//   import tars_core
//   engine = tars_core.TarsEngine()
//   engine.load("model.safetensors")
//   tokens = engine.generate([1, 2, 3], max_tokens=100)
//
// Agent 1 — Week 4

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict};

// Re-export internal modules
mod tensor;
mod arena;
mod state_cache;
mod tokenizer;
mod generate;
mod model_loader;
mod forward_pass;
mod eagle_decode;

use tensor::DType;

// ═══════════════════════════════════════
// Python-exposed TarsEngine class
// ═══════════════════════════════════════

/// Model configuration
#[derive(Debug, Clone)]
struct ModelConfig {
    d_model: usize,
    n_layers: usize,
    vocab_size: usize,
    d_state: usize,
    headdim: usize,
    n_heads: usize,
    d_inner: usize,
    d_conv: usize,
    chunk_size: usize,
    n_experts: usize,
    top_k_experts: usize,
    rope_base: f64,
    max_seq_len: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        let d_model = 2048;
        let expand = 2;
        let headdim = 64;
        ModelConfig {
            d_model,
            n_layers: 24,
            vocab_size: 32000,
            d_state: 128,
            headdim,
            n_heads: (d_model * expand) / headdim,
            d_inner: d_model * expand,
            d_conv: 4,
            chunk_size: 64,
            n_experts: 8,
            top_k_experts: 2,
            rope_base: 1_000_000.0,
            max_seq_len: 32768,
        }
    }
}

/// TARS Inference Engine (Python-accessible)
#[pyclass]
struct TarsEngine {
    config: ModelConfig,
    weights: Vec<tensor::Tensor>,
    arena_inner: arena::Arena,
    state_cache_inner: state_cache::StateCache,
    rope_freqs: Vec<f32>,
    loaded: bool,
}

#[pymethods]
impl TarsEngine {
    #[new]
    fn new() -> Self {
        let config = ModelConfig::default();
        let arena_inner = arena::Arena::default_80mb();
        let state_cache_inner = state_cache::StateCache::new(
            config.n_layers,
            config.n_heads,
            config.headdim,
            config.d_state,
        );
        TarsEngine {
            config,
            weights: Vec::new(),
            arena_inner,
            state_cache_inner,
            rope_freqs: Vec::new(),
            loaded: false,
        }
    }

    /// Load model weights from safetensors file
    fn load(&mut self, path: &str) -> PyResult<()> {
        self.weights = model_loader::load_safetensors(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e))?;

        // Precompute RoPE
        let half_d = self.config.headdim / 2;
        let max_seq = self.config.max_seq_len;
        self.rope_freqs = vec![0.0f32; max_seq * half_d];
        for pos in 0..max_seq {
            for i in 0..half_d {
                let theta = pos as f64 /
                    self.config.rope_base.powf((2 * i) as f64 / self.config.headdim as f64);
                self.rope_freqs[pos * half_d + i] = theta as f32;
            }
        }

        self.loaded = true;
        Ok(())
    }

    /// Generate tokens from prompt IDs
    #[pyo3(signature = (prompt_ids, max_tokens=2048, temperature=0.8, top_p=0.95, top_k=50))]
    fn generate(
        &mut self,
        prompt_ids: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> PyResult<Vec<u32>> {
        if !self.loaded {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Model not loaded. Call .load() first."
            ));
        }

        // For now, return placeholder (full implementation connects to forward_pass)
        Ok(vec![0u32; 0])
    }

    /// Reset all internal states (new conversation)
    fn reset_state(&mut self) {
        self.state_cache_inner.reset();
        self.arena_inner.reset();
    }

    /// Get memory usage stats
    fn memory_stats(&self) -> PyResult<(usize, usize)> {
        Ok((self.arena_inner.used(), self.arena_inner.peak()))
    }

    /// Check if model is loaded
    fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "TarsEngine(d={}, L={}, vocab={}, loaded={})",
            self.config.d_model,
            self.config.n_layers,
            self.config.vocab_size,
            self.loaded,
        )
    }
}

// ═══════════════════════════════════════
// Standalone kernel functions
// ═══════════════════════════════════════

/// BitNet ternary matmul (direct kernel access)
#[pyfunction]
fn bitnet_matmul(w_ternary: Vec<i8>, x: Vec<f32>, alpha: f32) -> PyResult<Vec<f32>> {
    let cols = x.len();
    let rows = w_ternary.len() / cols;
    let mut out = vec![0.0f32; rows];
    forward_pass::bitnet_matmul_ffi(&w_ternary, &x, alpha, rows, cols, &mut out);
    Ok(out)
}

/// RMSNorm (direct kernel access)
#[pyfunction]
#[pyo3(signature = (x, gamma, eps=1e-6))]
fn rmsnorm(x: Vec<f32>, gamma: Vec<f32>, eps: f32) -> PyResult<Vec<f32>> {
    let mut out = vec![0.0f32; x.len()];
    forward_pass::rmsnorm_ffi(&x, &gamma, eps, &mut out);
    Ok(out)
}

/// SSD scan step (direct kernel access)
#[pyfunction]
fn ssd_scan(state: Vec<f32>, gamma: Vec<f32>, b: Vec<f32>, x: Vec<f32>)
    -> PyResult<(Vec<f32>, Vec<f32>)>
{
    let mut out_state = vec![0.0f32; state.len()];
    forward_pass::ssd_scan_step_ffi(&state, &gamma, &b, &x, &mut out_state);
    let output = out_state.clone();
    Ok((out_state, output))
}

/// WKV-7 update (direct kernel access)
#[pyfunction]
fn wkv7_update(s: Vec<f32>, w: Vec<f32>, a: Vec<f32>, b: Vec<f32>,
               v: Vec<f32>, k: Vec<f32>) -> PyResult<Vec<f32>>
{
    let dim = w.len();
    let mut s_out = vec![0.0f32; dim * dim];
    let mut y_out = vec![0.0f32; dim];
    let r = vec![1.0f32; dim]; // Default receptance = 1
    forward_pass::wkv7_step_ffi(&s, &w, &a, &b, &v, &k, &r, dim, &mut s_out, &mut y_out);
    Ok(s_out)
}

/// SwiGLU fused (direct kernel access)
#[pyfunction]
#[pyo3(signature = (w1, w2, x, sparsity_threshold=0.0))]
fn swiglu_fused(w1: Vec<f32>, w2: Vec<f32>, x: Vec<f32>,
                sparsity_threshold: f32) -> PyResult<Vec<f32>>
{
    // Pre-compute gate = W1 @ x and value = W2 @ x (simplified 1D)
    let mut out = vec![0.0f32; w1.len()];
    let _nnz = forward_pass::swiglu_fused_ffi(&w1, &w2, sparsity_threshold, &mut out);
    Ok(out)
}

// ═══════════════════════════════════════
// Module initialization
// ═══════════════════════════════════════

/// TARS Core — C++/Rust Inference Engine
#[pymodule]
fn tars_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TarsEngine>()?;
    m.add_function(wrap_pyfunction!(bitnet_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(rmsnorm, m)?)?;
    m.add_function(wrap_pyfunction!(ssd_scan, m)?)?;
    m.add_function(wrap_pyfunction!(wkv7_update, m)?)?;
    m.add_function(wrap_pyfunction!(swiglu_fused, m)?)?;
    Ok(())
}
