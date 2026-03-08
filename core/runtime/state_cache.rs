// ═══════════════════════════════════════════════════════════════
//   state_cache.rs — SSM State Management
// ═══════════════════════════════════════════════════════════════
//
// Manages SSD and WKV recurrent states across generation steps.
// Each layer has: SSD state [n_heads, headdim, d_state]
//                 WKV state [d_state, d_state]
//                 Conv1d state [conv_dim, d_conv]
//
// Agent 1 — Week 4

use super::tensor::{Tensor, DType};

/// Per-layer state container
#[derive(Clone)]
pub struct LayerState {
    pub ssd_state: Tensor,    // [n_heads, headdim, d_state]
    pub wkv_state: Tensor,    // [d_state, d_state]
    pub conv_state: Tensor,   // [conv_dim, d_conv]
    pub x_prev: Tensor,       // [1, d_model] — last token for time-shift
}

/// State cache for all layers
pub struct StateCache {
    pub layers: Vec<LayerState>,
    pub n_layers: usize,
    pub position: usize,  // Current generation position
}

impl StateCache {
    pub fn new(n_layers: usize, n_heads: usize, headdim: usize, d_state: usize) -> Self {
        let d_inner = n_heads * headdim;
        let d_conv = 4;
        let ngroups = 1;
        let conv_dim = d_inner + 2 * ngroups * d_state;

        let layers = (0..n_layers).map(|_| {
            LayerState {
                ssd_state: Tensor::zeros(&[n_heads, headdim, d_state], DType::Float32),
                wkv_state: Tensor::zeros(&[d_state, d_state], DType::Float32),
                conv_state: Tensor::zeros(&[conv_dim, d_conv], DType::Float32),
                x_prev: Tensor::zeros(&[1, n_heads * headdim], DType::Float32),
            }
        }).collect();

        StateCache {
            layers,
            n_layers,
            position: 0,
        }
    }

    /// Reset all states (new conversation)
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            // Zero all state tensors
            for byte in layer.ssd_state.data.iter_mut() { *byte = 0; }
            for byte in layer.wkv_state.data.iter_mut() { *byte = 0; }
            for byte in layer.conv_state.data.iter_mut() { *byte = 0; }
            for byte in layer.x_prev.data.iter_mut() { *byte = 0; }
        }
        self.position = 0;
    }

    /// Advance position counter
    pub fn advance(&mut self, n_tokens: usize) {
        self.position += n_tokens;
    }
}
