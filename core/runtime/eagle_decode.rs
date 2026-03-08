// ═══════════════════════════════════════════════════════════════
//   eagle_decode.rs — EAGLE-3 Speculative Decoding
// ═══════════════════════════════════════════════════════════════
//
// Speculative decoding: draft K tokens with a lightweight head,
// then verify all K in one forward pass → accept/reject.
//
// Expected speedup: 2-3× for greedy/low-temperature generation.
//
// Agent 1 — Week 4

use super::TarsEngine;
use super::GenerateParams;
use super::forward_pass;

/// EAGLE-3 speculative decoding parameters
pub struct EagleConfig {
    pub draft_tokens: usize,  // K: number of speculative tokens (default 3)
    pub accept_threshold: f32, // Minimum probability for acceptance
}

impl Default for EagleConfig {
    fn default() -> Self {
        EagleConfig {
            draft_tokens: 3,
            accept_threshold: 0.0, // Accept all that match
        }
    }
}

/// Draft K tokens using the current model state
///
/// In EAGLE-3, the draft head is a lightweight MLP on top of hidden states.
/// For now, we use greedy decoding from the full model as a placeholder.
fn draft_tokens(
    engine: &mut TarsEngine,
    last_token: u32,
    k: usize,
) -> Vec<u32> {
    let mut drafts = Vec::with_capacity(k);
    let mut token = last_token;

    for _ in 0..k {
        let logits = forward_pass::forward_full(token, engine);
        // Greedy: argmax
        let next = logits.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        drafts.push(next);
        token = next;
    }

    drafts
}

/// Verify drafted tokens: run all K through the full model,
/// accept longest prefix that matches draft.
fn verify_drafts(
    engine: &mut TarsEngine,
    drafts: &[u32],
    _config: &EagleConfig,
) -> usize {
    // In full implementation: run all draft tokens in parallel through
    // the model and compare output distributions.
    //
    // For now: accept all drafts (placeholder).
    // Real implementation would:
    //   1. Save state checkpoint
    //   2. Forward all K draft tokens
    //   3. For each position, check if draft matches model's top prediction
    //   4. Accept longest matching prefix
    //   5. Rollback state to end of accepted prefix

    drafts.len()  // Accept all (placeholder)
}

/// EAGLE-3 speculative generation loop
pub fn generate_speculative(
    engine: &mut TarsEngine,
    prompt_ids: &[u32],
    params: &GenerateParams,
    eagle_config: &EagleConfig,
) -> Result<Vec<u32>, String> {
    let mut output = Vec::with_capacity(params.max_tokens);

    // Prefill
    for &token_id in prompt_ids {
        let _logits = forward_pass::forward_full(token_id, engine);
        engine.state_cache.advance(1);
    }

    let mut last_token = *prompt_ids.last().unwrap_or(&0);
    let mut total_drafted = 0usize;
    let mut total_accepted = 0usize;

    while output.len() < params.max_tokens {
        // 1. Draft K tokens
        let drafts = draft_tokens(engine, last_token, eagle_config.draft_tokens);
        total_drafted += drafts.len();

        // 2. Verify
        let n_accepted = verify_drafts(engine, &drafts, eagle_config);
        total_accepted += n_accepted;

        // 3. Accept prefix
        for &tok in &drafts[..n_accepted] {
            output.push(tok);
            engine.state_cache.advance(1);

            if tok == 0 || tok == 2 {
                // EOS
                return Ok(output);
            }
        }

        if n_accepted > 0 {
            last_token = drafts[n_accepted - 1];
        } else {
            // No drafts accepted — fall back to standard decoding
            let mut logits = forward_pass::forward_full(last_token, engine);
            let next = logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            output.push(next);
            engine.state_cache.advance(1);
            last_token = next;

            if next == 0 || next == 2 {
                break;
            }
        }
    }

    Ok(output)
}
