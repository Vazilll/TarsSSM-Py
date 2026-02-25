"""
train_reflex.py — Train the Tier-1 MinGRU Reflex Classifier.

Trains on intent classification + confidence estimation.
Data: 200+ simple patterns + augmentation vs complex queries.

Usage:
    python train_reflex.py --epochs 100 --lr 0.001
"""
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import os
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainReflex")


# ════════════════════════════════════════════
# Training Data — 200+ фраз
# ════════════════════════════════════════════

SIMPLE_DATA = {
    "greeting": [
        "привет", "здравствуй", "здравствуйте", "хай", "хэй",
        "доброе утро", "добрый день", "добрый вечер", "приветик",
        "hello", "hi", "hey", "приветствую", "здарова", "ку",
        "доброго времени суток", "приветики", "хеллоу", "хелло",
        "прив", "хаюшки", "дратути", "здрасте", "приветствую вас",
        "добрейшего дня", "доброго утра", "доброго дня",
    ],
    "farewell": [
        "пока", "до свидания", "до встречи", "бай", "прощай",
        "спокойной ночи", "увидимся", "всего доброго", "удачи",
        "до скорого", "бай бай", "пока пока", "всего хорошего",
        "счастливо", "до завтра", "покеда", "чао", "адьё",
        "покедова", "ну всё пока", "я пошёл", "я ушёл",
    ],
    "status": [
        "как дела", "что делаешь", "как ты", "ты тут", "ты здесь",
        "что нового", "как настроение", "как поживаешь", "чем занят",
        "что происходит", "как жизнь", "что слышно", "как сам",
        "ты работаешь", "ты функционируешь", "жив ещё", "ты онлайн",
        "ты на связи", "есть кто", "ты тут ещё", "как оно",
    ],
    "time": [
        "который час", "какая дата", "сколько время", "какой день",
        "сегодня какое число", "что за день", "какой сегодня день",
        "скажи время", "дата сегодня", "покажи часы", "время покажи",
        "текущая дата", "текущее время", "сколько сейчас времени",
    ],
    "quick_action": [
        "открой браузер", "выключи музыку", "поставь на паузу",
        "включи свет", "выключи компьютер", "открой файл",
        "запусти программу", "закрой окно", "сделай скриншот",
        "открой проводник", "запусти калькулятор", "покажи рабочий стол",
        "закрой все окна", "перезагрузись", "обнови страницу",
        "увеличь громкость", "уменьши громкость", "выключи звук",
        "поставь таймер", "открой настройки", "запусти терминал",
        "открой папку", "создай файл", "удали файл",
        "скопируй текст", "вставь текст", "отмени действие",
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
    "расскажи подробно как устроена нейронная сеть трансформер",
    "напиши мне эссе на тему искусственного интеллекта",
    "проведи полный код ревью этого python проекта",
    "сделай рефакторинг всего модуля авторизации",
    "объясни разницу между supervised и unsupervised learning",
    "напиши unit тесты для этого класса на pytest",
    "помоги разобраться в этом стектрейсе ошибки",
    "создай REST API для управления задачами",
    "расскажи как работает квантовый компьютер",
    "напиши парсер для html страницы на python",
    "объясни что такое блокчейн и зачем он нужен",
    "помоги оптимизировать производительность приложения",
]


# ════════════════════════════════════════════
# Аугментация данных
# ════════════════════════════════════════════

def augment_phrase(text: str) -> list:
    """Генерирует вариации фразы для расширения датасета."""
    variants = [text]
    variants.append(text + "!")
    variants.append(text + "?")
    variants.append(text.upper())
    variants.append(text.capitalize())
    if len(text) > 3:
        pos = random.randint(1, len(text) - 2)
        typo = text[:pos] + random.choice('абвгдежзиклмнопрстуфхцчшщэюя') + text[pos+1:]
        variants.append(typo)
    if len(text) > 5:
        variants.append(text[:len(text)//2 + 2])
    return variants


# ════════════════════════════════════════════
# Майнинг данных из баз
# ════════════════════════════════════════════

def mine_complex_from_hf(data_dir: Path, max_samples: int = 300) -> list:
    """
    Извлекает сложные запросы из скачанных HF датасетов.
    Файлы: data/hf_Den4ikAI_russian_instructions_2.txt и другие hf_*.txt
    """
    complex_phrases = []
    
    for hf_file in sorted(data_dir.glob("hf_*.txt")):
        try:
            with open(hf_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line) < 15 or len(line) > 200:
                        continue
                    # Ищем вопросы/инструкции (потенциально сложные)
                    lower = line.lower()
                    if any(kw in lower for kw in [
                        'напиши', 'объясни', 'расскажи', 'создай', 'реализуй',
                        'проанализируй', 'спроектируй', 'оптимизируй', 'сравни',
                        'как работает', 'что такое', 'почему', 'каким образом',
                        'write', 'explain', 'create', 'implement', 'analyze',
                        'design', 'compare', 'how does', 'what is',
                    ]):
                        complex_phrases.append(line[:200])
                    
                    if len(complex_phrases) >= max_samples:
                        break
        except Exception as e:
            logger.warning(f"Ошибка чтения {hf_file}: {e}")
        
        if len(complex_phrases) >= max_samples:
            break
    
    logger.info(f"[Mine] HF datasets: {len(complex_phrases)} сложных запросов")
    return complex_phrases


def mine_complex_from_wiki(data_dir: Path, max_samples: int = 100) -> list:
    """Извлекает сложные фразы из wiki_ru.txt (вопросы о мире)."""
    wiki_file = data_dir / "wiki_ru.txt"
    complex_phrases = []
    
    if not wiki_file.exists():
        return complex_phrases
    
    try:
        with open(wiki_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) < 30 or len(line) > 200:
                    continue
                # Берём информативные предложения (не заголовки, не списки)
                if '.' in line and not line.startswith(('#', '-', '*', '=')):
                    # Превращаем утверждение в вопрос "расскажи про X"
                    complex_phrases.append(f"расскажи: {line[:150]}")
                    if len(complex_phrases) >= max_samples:
                        break
    except Exception as e:
        logger.warning(f"Ошибка чтения wiki_ru.txt: {e}")
    
    logger.info(f"[Mine] wiki_ru.txt: {len(complex_phrases)} фраз")
    return complex_phrases


def mine_from_memories(data_dir: Path, max_samples: int = 50) -> list:
    """Извлекает данные из tars_memories.json (пользовательский контекст)."""
    mem_file = data_dir / "tars_memories.json"
    phrases = []
    
    if not mem_file.exists():
        return phrases
    
    try:
        import json
        with open(mem_file, 'r', encoding='utf-8') as f:
            memories = json.load(f)
        
        for entry in memories[:max_samples]:
            text = entry.get("text", "") if isinstance(entry, dict) else str(entry)
            if len(text) > 10:
                # Пользовательские факты = потенциально сложные
                phrases.append(text[:200])
    except Exception as e:
        logger.warning(f"Ошибка чтения memories: {e}")
    
    logger.info(f"[Mine] tars_memories.json: {len(phrases)} записей")
    return phrases


def mine_from_leann(max_samples: int = 50) -> list:
    """Извлекает данные из LEANN индекса (семантическая память)."""
    try:
        from memory.leann import LeannIndex
        leann = LeannIndex()
        # LEANN хранит тексты — берём их как сложные контексты
        phrases = [t[:200] for t in leann.texts[:max_samples] if len(t) > 15]
        logger.info(f"[Mine] LEANN: {len(phrases)} документов")
        return phrases
    except Exception as e:
        logger.debug(f"LEANN не доступен: {e}")
        return []


def build_dataset(max_len=64, augment=True):
    """
    Строит тренировочный датасет из ВСЕХ доступных источников:
      - Hardcoded simple patterns (200+ фраз по 5 интентам)
      - Hardcoded complex patterns (24 фразы)
      - Mined complex from HF datasets (до 300 фраз)
      - Mined complex from wiki_ru.txt (до 100 фраз)
      - Mined from tars_memories.json (до 50 фраз)
      - Mined from LEANN index (до 50 фраз)
    """
    from brain.tokenizer import TarsTokenizer
    tokenizer = TarsTokenizer()
    
    inputs = []
    conf_labels = []
    intent_labels = []

    intent_map = {name: i for i, name in enumerate(SIMPLE_DATA.keys())}
    
    # 1. Простые паттерны (hardcoded + аугментация)
    for intent_name, phrases in SIMPLE_DATA.items():
        intent_id = intent_map[intent_name]
        for phrase in phrases:
            all_variants = augment_phrase(phrase) if augment else [phrase]
            for variant in all_variants:
                tokens = tokenizer.encode(variant.lower())[:max_len]
                tokens += [0] * (max_len - len(tokens))
                inputs.append(tokens)
                conf_labels.append(1.0)
                intent_labels.append(intent_id)

    complex_intent = len(SIMPLE_DATA)  # "complex" = последний интент
    
    # 2. Hardcoded complex
    for phrase in COMPLEX_DATA:
        all_variants = augment_phrase(phrase) if augment else [phrase]
        for variant in all_variants:
            tokens = tokenizer.encode(variant.lower())[:max_len]
            tokens += [0] * (max_len - len(tokens))
            inputs.append(tokens)
            conf_labels.append(0.0)
            intent_labels.append(complex_intent)
    
    # 3. Mined complex из баз данных
    data_dir = ROOT / "data"
    mined_complex = []
    mined_complex += mine_complex_from_hf(data_dir, max_samples=300)
    mined_complex += mine_complex_from_wiki(data_dir, max_samples=100)
    mined_complex += mine_from_memories(data_dir, max_samples=50)
    mined_complex += mine_from_leann(max_samples=50)
    
    for phrase in mined_complex:
        tokens = tokenizer.encode(phrase.lower())[:max_len]
        tokens += [0] * (max_len - len(tokens))
        inputs.append(tokens)
        conf_labels.append(0.0)  # complex → confidence LOW
        intent_labels.append(complex_intent)

    X = torch.tensor(inputs, dtype=torch.long)
    Y_conf = torch.tensor(conf_labels, dtype=torch.float32)
    Y_intent = torch.tensor(intent_labels, dtype=torch.long)
    
    n_simple = sum(1 for c in conf_labels if c == 1.0)
    n_complex = sum(1 for c in conf_labels if c == 0.0)
    n_mined = len(mined_complex)
    
    logger.info(
        f"Dataset: {len(X)} примеров | "
        f"simple={n_simple} | complex={n_complex} (из них mined={n_mined}) | "
        f"{len(SIMPLE_DATA) + 1} классов"
    )
    
    return X, Y_conf, Y_intent


def train(args):
    from brain.reflex_classifier import ReflexClassifier

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Device: {device}")

    X, Y_conf, Y_intent = build_dataset(augment=True)
    
    # Train/Test split (80/20)
    n = len(X)
    perm = torch.randperm(n)
    X, Y_conf, Y_intent = X[perm], Y_conf[perm], Y_intent[perm]
    
    split = int(n * 0.8)
    X_train, X_test = X[:split].to(device), X[split:].to(device)
    Y_conf_train, Y_conf_test = Y_conf[:split].to(device), Y_conf[split:].to(device)
    Y_intent_train, Y_intent_test = Y_intent[:split].to(device), Y_intent[split:].to(device)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    n_intents = len(SIMPLE_DATA) + 1  # +1 для "complex"
    model = ReflexClassifier(vocab_size=256, embed_dim=64, hidden_dim=64, n_intents=n_intents).to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    conf_loss_fn = nn.BCELoss()
    intent_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Cosine LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    
    best_test_acc = 0.0

    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        
        # Forward через внутренние слои модели
        x = model.embed(X_train)
        B, L, E = x.shape
        h = torch.zeros(B, model.hidden_dim, device=device)
        for t in range(L):
            xt = x[:, t, :]
            gate = torch.sigmoid(model.mingru_gate(xt))
            h_tilde = model.mingru_hidden(xt)
            h = (1 - gate) * h + gate * h_tilde
        intent_logits = model.intent_head(h)
        conf_out = torch.sigmoid(model.confidence_head(h)).squeeze(-1)

        loss_c = conf_loss_fn(conf_out, Y_conf_train)
        loss_i = intent_loss_fn(intent_logits, Y_intent_train)
        loss = loss_c + loss_i

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % max(1, args.epochs // 10) == 0 or epoch == 0:
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                x_t = model.embed(X_test)
                B_t, L_t, E_t = x_t.shape
                h_t = torch.zeros(B_t, model.hidden_dim, device=device)
                for t in range(L_t):
                    xt_t = x_t[:, t, :]
                    gate_t = torch.sigmoid(model.mingru_gate(xt_t))
                    h_tilde_t = model.mingru_hidden(xt_t)
                    h_t = (1 - gate_t) * h_t + gate_t * h_tilde_t
                test_intent = model.intent_head(h_t)
                test_conf = torch.sigmoid(model.confidence_head(h_t)).squeeze(-1)
                
                test_acc_conf = ((test_conf > 0.5).float() == Y_conf_test).float().mean().item()
                test_acc_intent = (test_intent.argmax(-1) == Y_intent_test).float().mean().item()
            model.train()
            
            train_acc_conf = ((conf_out > 0.5).float() == Y_conf_train).float().mean().item()
            train_acc_intent = (intent_logits.argmax(-1) == Y_intent_train).float().mean().item()
            
            logger.info(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Loss: {loss.item():.4f} | "
                f"Train: conf={train_acc_conf:.0%} int={train_acc_intent:.0%} | "
                f"Test: conf={test_acc_conf:.0%} int={test_acc_intent:.0%}"
            )
            
            if test_acc_intent > best_test_acc:
                best_test_acc = test_acc_intent

    # Save
    os.makedirs("models/reflex", exist_ok=True)
    save_path = "models/reflex/reflex_classifier.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Saved to {save_path} | Best test intent acc: {best_test_acc:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Reflex MinGRU Classifier")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--cpu", action="store_true")
    train(parser.parse_args())
