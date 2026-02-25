"""
train_reflex.py — Train the Tier-1 MinGRU Reflex Classifier.

Trains on intent classification + confidence estimation.
Data: simple patterns (greeting, farewell, status, commands) vs complex queries.

Usage:
    python train_reflex.py --epochs 50 --lr 0.001
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainReflex")


# ---- Training Data ----
# Simple patterns (Tier 1 can handle)
SIMPLE_DATA = {
    "greeting": [
        "привет", "здравствуй", "здравствуйте", "хай", "хэй",
        "доброе утро", "добрый день", "добрый вечер", "приветик",
        "hello", "hi", "hey",
    ],
    "farewell": [
        "пока", "до свидания", "до встречи", "бай", "прощай",
        "спокойной ночи", "увидимся",
    ],
    "status": [
        "как дела", "что делаешь", "как ты", "ты тут", "ты здесь",
        "что нового", "как настроение",
    ],
    "time": [
        "который час", "какая дата", "сколько время", "какой день",
        "сегодня какое число",
    ],
    "quick_action": [
        "открой браузер", "выключи музыку", "поставь на паузу",
        "включи свет", "выключи компьютер", "открой файл",
        "запусти программу", "закрой окно", "сделай скриншот",
    ],
}

# Complex queries (Need Tier 2+)
COMPLEX_DATA = [
    "спроектируй архитектуру базы данных для стартапа",
    "проанализируй последние новости о квантовых компьютерах",
    "напиши код нейронной сети на pytorch",
    "сравни преимущества rust и c++ для системного программирования",
    "объясни теорию относительности простыми словами",
    "как работает мезонинное финансирование",
    "создай бизнес-план для SaaS продукта",
    "реализуй алгоритм поиска пути A* с визуализацией",
    "проведи анализ конкурентов в нише AI ассистентов",
    "какие тренды в машинном обучении ожидаются в 2027",
    "напиши научную статью по теории сходимости интегралов",
    "оптимизируй этот SQL запрос для базы с миллионом записей",
]


class SimpleTokenizer:
    """Character-level tokenizer for training."""
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size

    def encode(self, text: str, max_len=64):
        tokens = [min(ord(c), self.vocab_size - 1) for c in text.lower()[:max_len]]
        # Pad to max_len
        tokens += [0] * (max_len - len(tokens))
        return tokens


def build_dataset(tokenizer, max_len=64):
    """Build training tensors from pattern data."""
    inputs = []
    conf_labels = []   # 1.0 for simple, 0.0 for complex
    intent_labels = [] # 0-4 for simple intents, 5 for complex

    # Simple (high confidence)
    intent_map = {name: i for i, name in enumerate(SIMPLE_DATA.keys())}
    for intent_name, phrases in SIMPLE_DATA.items():
        intent_id = intent_map[intent_name]
        for phrase in phrases:
            tokens = tokenizer.encode(phrase, max_len)
            inputs.append(tokens)
            conf_labels.append(1.0)
            intent_labels.append(intent_id)

    # Complex (low confidence)
    for phrase in COMPLEX_DATA:
        tokens = tokenizer.encode(phrase, max_len)
        inputs.append(tokens)
        conf_labels.append(0.0)
        intent_labels.append(5)  # "complex" intent

    X = torch.tensor(inputs, dtype=torch.long)
    Y_conf = torch.tensor(conf_labels, dtype=torch.float32)
    Y_intent = torch.tensor(intent_labels, dtype=torch.long)
    return X, Y_conf, Y_intent


def train(args):
    from brain.reflex_classifier import ReflexClassifier

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Device: {device}")

    tokenizer = SimpleTokenizer(vocab_size=256)
    X, Y_conf, Y_intent = build_dataset(tokenizer)
    X, Y_conf, Y_intent = X.to(device), Y_conf.to(device), Y_intent.to(device)

    model = ReflexClassifier(vocab_size=256, embed_dim=64, hidden_dim=64, n_intents=6).to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    conf_loss_fn = nn.BCELoss()
    intent_loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        conf_pred, intent_pred = model(X)

        loss_c = conf_loss_fn(conf_pred, Y_conf)
        loss_i = intent_loss_fn(model.intent_head(
            # Re-run to get intent logits
            torch.zeros(1)  # placeholder
        ), Y_intent)

        # Simpler: just run forward again for logits
        optimizer.zero_grad()
        conf_pred, _ = model(X)
        # Get raw intent logits by running through the model internals
        x = model.embed(X)
        B, L, E = x.shape
        h = torch.zeros(B, model.hidden_dim, device=device)
        for t in range(L):
            xt = x[:, t, :]
            gate = torch.sigmoid(model.mingru_gate(xt))
            h_tilde = model.mingru_hidden(xt)
            h = (1 - gate) * h + gate * h_tilde
        intent_logits = model.intent_head(h)
        conf_out = torch.sigmoid(model.confidence_head(h)).squeeze(-1)

        loss_c = conf_loss_fn(conf_out, Y_conf)
        loss_i = intent_loss_fn(intent_logits, Y_intent)
        loss = loss_c + loss_i

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            acc_conf = ((conf_out > 0.5).float() == Y_conf).float().mean().item()
            acc_intent = (intent_logits.argmax(-1) == Y_intent).float().mean().item()
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Conf Acc: {acc_conf:.1%} | "
                f"Intent Acc: {acc_intent:.1%}"
            )

    # Save
    os.makedirs("models/reflex", exist_ok=True)
    save_path = "models/reflex/reflex_classifier.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reflex MinGRU Classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
