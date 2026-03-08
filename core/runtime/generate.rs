// ═══════════════════════════════════════════════════════════════
//   generate.rs — Autoregressive Token Generation
// ═══════════════════════════════════════════════════════════════
//
// Sampling loop: forward → sample → append → repeat.
// Supports temperature, top-p, top-k, min-p, and repetition penalty.
//
// Agent 1 — Week 3

use super::TarsEngine;
use super::GenerateParams;
use super::forward_pass;

/// Top-p (nucleus) + top-k + min-p sampling
fn sample_token(logits: &mut [f32], params: &GenerateParams, generated: &[u32]) -> u32 {
    let vocab_size = logits.len();

    // 1. Repetition penalty
    if params.repetition_penalty != 1.0 {
        for &tok in generated.iter().rev().take(64) {
            let idx = tok as usize;
            if idx < vocab_size {
                if logits[idx] > 0.0 {
                    logits[idx] /= params.repetition_penalty;
                } else {
                    logits[idx] *= params.repetition_penalty;
                }
            }
        }
    }

    // 2. Temperature scaling
    if params.temperature > 0.0 && params.temperature != 1.0 {
        let inv_temp = 1.0 / params.temperature;
        for l in logits.iter_mut() {
            *l *= inv_temp;
        }
    }

    // 3. Softmax
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for l in logits.iter_mut() {
        *l = (*l - max_val).exp();
        sum += *l;
    }
    let inv_sum = 1.0 / (sum + 1e-8);
    for l in logits.iter_mut() {
        *l *= inv_sum;
    }

    // 4. Sort indices by probability (descending)
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    // 5. Top-k filter
    let k = params.top_k.min(vocab_size);
    let candidates = &indices[..k];

    // 6. Top-p filter
    let mut cumsum = 0.0f32;
    let mut cutoff = candidates.len();
    for (i, &idx) in candidates.iter().enumerate() {
        cumsum += logits[idx];
        if cumsum >= params.top_p {
            cutoff = i + 1;
            break;
        }
    }
    let candidates = &candidates[..cutoff];

    // 7. Min-p filter
    let max_prob = logits[candidates[0]];
    let min_threshold = max_prob * params.min_p;
    let candidates: Vec<usize> = candidates.iter()
        .filter(|&&idx| logits[idx] >= min_threshold)
        .cloned()
        .collect();

    if candidates.is_empty() {
        return indices[0] as u32;  // fallback: argmax
    }

    // 8. Sample from filtered distribution
    let total: f32 = candidates.iter().map(|&idx| logits[idx]).sum();
    let r = rand_f32() * total;  // pseudo-random
    let mut acc = 0.0f32;
    for &idx in &candidates {
        acc += logits[idx];
        if acc >= r {
            return idx as u32;
        }
    }

    candidates[candidates.len() - 1] as u32
}

/// Simple pseudo-random f32 in [0, 1)
fn rand_f32() -> f32 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    // LCG-based quick random
    let x = nanos.wrapping_mul(1103515245).wrapping_add(12345);
    (x >> 16) as f32 / 32768.0
}

/// Main generation loop
pub fn generate(
    engine: &mut TarsEngine,
    prompt_ids: &[u32],
    params: &GenerateParams,
) -> Result<Vec<u32>, String> {
    let mut output = Vec::with_capacity(params.max_tokens);

    // Prefill: process all prompt tokens
    for &token_id in prompt_ids {
        let _logits = forward_pass::forward_full(token_id, engine);
        engine.state_cache.advance(1);
    }

    // Generate: autoregressive loop
    let mut last_token = *prompt_ids.last().unwrap_or(&0);

    for _step in 0..params.max_tokens {
        let mut logits = forward_pass::forward_full(last_token, engine);
        engine.state_cache.advance(1);

        let next_token = sample_token(&mut logits, params, &output);
        output.push(next_token);
        last_token = next_token;

        // EOS check (token 0 or 2 typically)
        if next_token == 0 || next_token == 2 {
            break;
        }
    }

    Ok(output)
}
