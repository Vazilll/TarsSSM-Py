"""
reflex_classifier.py — Tier 1 MinGRU Confidence Classifier.

Determines if a query is simple enough for Reflex to handle,
or if it needs the full Mamba-2 pipeline.

Outputs:
  P_conf > 0.85 → Reflex handles it (Tier 1)
  P_conf < 0.85 → Pass to Mamba-2 SSM (Tier 2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger("ReflexClassifier")


class ReflexClassifier(nn.Module):
    """
    Ultra-lightweight MinGRU-based classifier.
    Input: token IDs → Embedding → MinGRU → Linear → sigmoid(P_conf)
    Also outputs an intent class for quick actions.
    """

    # Intent classes for Tier-1 handling
    INTENTS = [
        "greeting",       # 0: "Привет", "Здравствуй"
        "farewell",       # 1: "Пока", "До свидания" 
        "status",         # 2: "Как дела?", "Что делаешь?"
        "time",           # 3: "Который час?", "Какая дата?"
        "quick_action",   # 4: "Открой браузер", "Выключи музыку"
        "complex",        # 5: Needs Tier 2+ (catch-all)
    ]

    def __init__(self, vocab_size=256, embed_dim=64, hidden_dim=64, n_intents=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mingru_gate = nn.Linear(embed_dim, hidden_dim)
        self.mingru_hidden = nn.Linear(embed_dim, hidden_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.intent_head = nn.Linear(hidden_dim, n_intents)
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: [batch, seq_len]
        Returns:
            confidence: [batch] float in (0, 1)
            intent: [batch] int (intent class index)
        """
        x = self.embed(input_ids)  # [B, L, E]

        # MinGRU scan (sequential for inference, parallel for training)
        B, L, E = x.shape
        h = torch.zeros(B, self.hidden_dim, device=x.device)

        for t in range(L):
            xt = x[:, t, :]
            gate = torch.sigmoid(self.mingru_gate(xt))
            h_tilde = self.mingru_hidden(xt)  # No tanh (MinGRU style)
            h = (1 - gate) * h + gate * h_tilde

        # Heads
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        intent_logits = self.intent_head(h)
        intent = intent_logits.argmax(dim=-1)

        return confidence, intent

    def classify(self, text: str, tokenizer, device="cpu") -> dict:
        """High-level API: classify a text query."""
        tokens = tokenizer.encode(text)
        if isinstance(tokens, list):
            tokens = torch.tensor([tokens], dtype=torch.long, device=device)

        self.eval()
        with torch.no_grad():
            conf, intent_idx = self.forward(tokens)

        conf_val = conf.item()
        intent_name = self.INTENTS[intent_idx.item()] if intent_idx.item() < len(self.INTENTS) else "complex"

        return {
            "confidence": conf_val,
            "intent": intent_name,
            "tier": 1 if conf_val > 0.85 else 2,
            "needs_mamba": conf_val < 0.85,
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
