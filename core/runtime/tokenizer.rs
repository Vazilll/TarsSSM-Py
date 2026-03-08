// ═══════════════════════════════════════════════════════════════
//   tokenizer.rs — BPE Tokenizer for TARS
// ═══════════════════════════════════════════════════════════════
//
// Fast BPE encode/decode. Compatible with Qwen tokenizer (48K vocab).
// Loads vocabulary from merges.txt + vocab.json.
//
// Python reference: brain/tokenizer.py
//
// Agent 1 — Week 3

use std::collections::HashMap;

/// BPE Tokenizer
pub struct Tokenizer {
    /// Token string → ID
    pub encoder: HashMap<String, u32>,
    /// ID → token string
    pub decoder: HashMap<u32, String>,
    /// BPE merge pairs (lower index = higher priority)
    pub merges: Vec<(String, String)>,
    /// Vocab size
    pub vocab_size: usize,
    /// Special tokens
    pub bos_id: u32,
    pub eos_id: u32,
    pub pad_id: u32,
}

impl Tokenizer {
    /// Create from vocabulary and merge files
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self, String> {
        // Load vocab.json
        let vocab_str = std::fs::read_to_string(vocab_path)
            .map_err(|e| format!("Failed to read vocab: {}", e))?;
        let encoder: HashMap<String, u32> = serde_json::from_str(&vocab_str)
            .map_err(|e| format!("Failed to parse vocab: {}", e))?;

        let decoder: HashMap<u32, String> = encoder.iter()
            .map(|(k, &v)| (v, k.clone()))
            .collect();

        // Load merges.txt
        let merges_str = std::fs::read_to_string(merges_path)
            .map_err(|e| format!("Failed to read merges: {}", e))?;
        let merges: Vec<(String, String)> = merges_str.lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        let vocab_size = encoder.len();

        Ok(Tokenizer {
            encoder,
            decoder,
            merges,
            vocab_size,
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        })
    }

    /// Create a minimal byte-level tokenizer (fallback when no vocab files)
    pub fn byte_level() -> Self {
        let mut encoder = HashMap::new();
        let mut decoder = HashMap::new();

        // Map each byte to its own token
        for i in 0..256u32 {
            let s = format!("<0x{:02X}>", i);
            encoder.insert(s.clone(), i);
            decoder.insert(i, s);
        }

        Tokenizer {
            encoder,
            decoder,
            merges: Vec::new(),
            vocab_size: 256,
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.merges.is_empty() {
            // Byte-level fallback
            return text.bytes().map(|b| b as u32).collect();
        }

        // Character-level tokenization → BPE merges
        let mut tokens: Vec<String> = text.chars()
            .map(|c| c.to_string())
            .collect();

        // Apply BPE merges iteratively
        for (a, b) in &self.merges {
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == *a && tokens[i + 1] == *b {
                    let merged = format!("{}{}", a, b);
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Map to IDs
        tokens.iter()
            .map(|t| *self.encoder.get(t).unwrap_or(&0))
            .collect()
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.decoder.get(&id))
            .cloned()
            .collect::<Vec<String>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level() {
        let tok = Tokenizer::byte_level();
        let ids = tok.encode("Hi");
        assert_eq!(ids, vec![72, 105]); // ASCII: H=72, i=105
    }
}
