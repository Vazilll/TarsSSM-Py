// ═══════════════════════════════════════════════════════════════
//   engine.rs — TARS Inference Engine (main entry point)
// ═══════════════════════════════════════════════════════════════
//
// TarsEngine: load model weights → run forward pass → generate tokens.
// Calls C++ kernels via FFI for all hot-path operations.
//
// Python-accessible via PyO3 (see lib.rs).
//
// Agent 1 — Week 4

pub mod tensor;
pub mod arena;
pub mod state_cache;
pub mod tokenizer;
pub mod generate;
pub mod model_loader;
pub mod forward_pass;
pub mod eagle_decode;

use std::path::Path;
use arena::Arena;
use state_cache::StateCache;
use tensor::{Tensor, DType};

/// Model configuration (mirrors config.py TarsConfig)
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub d_state: usize,
    pub headdim: usize,
    pub n_heads: usize,
    pub d_inner: usize,
    pub d_conv: usize,
    pub chunk_size: usize,
    pub n_experts: usize,
    pub top_k_experts: usize,
    pub rope_base: f64,
    pub max_seq_len: usize,
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

/// Generation parameters
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams {
            max_tokens: 2048,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
            min_p: 0.05,
            repetition_penalty: 1.1,
        }
    }
}

/// Main inference engine
pub struct TarsEngine {
    pub config: ModelConfig,
    pub weights: Vec<Tensor>,         // All model weights
    pub arena: Arena,                 // Scratch allocator
    pub state_cache: StateCache,      // SSM states
    pub rope_freqs: Vec<f32>,         // Precomputed RoPE frequencies
    pub loaded: bool,
}

impl TarsEngine {
    /// Create engine with default config
    pub fn new() -> Self {
        let config = ModelConfig::default();
        let arena = Arena::default_80mb();
        let state_cache = StateCache::new(
            config.n_layers,
            config.n_heads,
            config.headdim,
            config.d_state,
        );
        TarsEngine {
            config,
            weights: Vec::new(),
            arena,
            state_cache,
            rope_freqs: Vec::new(),
            loaded: false,
        }
    }

    /// Load model from safetensors file
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        if !Path::new(path).exists() {
            return Err(format!("Model file not found: {}", path));
        }

        self.weights = model_loader::load_safetensors(path)?;

        // Precompute RoPE frequencies
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

    /// Generate tokens from prompt
    pub fn generate(&mut self, prompt_ids: &[u32], params: &GenerateParams)
        -> Result<Vec<u32>, String>
    {
        if !self.loaded {
            return Err("Model not loaded".to_string());
        }
        generate::generate(self, prompt_ids, params)
    }

    /// Get hidden state at specific layer (for DoubtEngine etc.)
    pub fn get_hidden_state(&self, _layer: usize) -> Option<&Tensor> {
        // TODO: implement hidden state extraction
        None
    }

    /// Reset all caches (for new conversation)
    pub fn reset_state(&mut self) {
        self.state_cache.reset();
        self.arena.reset();
    }
}
