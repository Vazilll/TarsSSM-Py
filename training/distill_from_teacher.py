"""
═══════════════════════════════════════════════════════════════
  Knowledge Distillation for TARS
═══════════════════════════════════════════════════════════════

Generate synthetic training data from a teacher model (local or API)
and train TARS on it. Teacher can be:
  - Local: Ollama (Qwen2.5, Llama3, Mistral)
  - API: OpenAI (GPT-4o), Anthropic (Claude)

Modes:
  1. generate — create synthetic dataset from teacher
  2. distill  — train TARS on generated dataset with KD loss

Usage:
  # Step 1: Generate data (local teacher via Ollama)
  python training/distill_from_teacher.py generate \\
      --teacher ollama --teacher_model qwen2.5:72b \\
      --output data/synthetic_teacher.jsonl --n_samples 5000

  # Step 2: Train TARS on teacher data
  python training/distill_from_teacher.py distill \\
      --data data/synthetic_teacher.jsonl \\
      --model models/mamba2/brain_best.pt
"""

import os
import sys
import json
import time
import logging
import argparse
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("distill")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ═══ Prompt Templates for Synthetic Data ═══
PROMPT_TEMPLATES = {
    "chat": [
        "Расскажи о {topic}",
        "Что такое {topic}?",
        "Объясни простыми словами: {topic}",
        "Привет! Как ты можешь помочь с {topic}?",
        "Мне нужна помощь с {topic}",
    ],
    "instruct": [
        "Напиши пошаговую инструкцию: {topic}",
        "Составь список из 5 пунктов о {topic}",
        "Сравни преимущества и недостатки {topic}",
        "Дай краткое описание: {topic}",
        "Приведи 3 примера {topic}",
    ],
    "reasoning": [
        "Реши задачу: {topic}",
        "Проанализируй: {topic}",
        "Какие аргументы за и против {topic}?",
        "Почему {topic} важно?",
        "Какие последствия {topic}?",
    ],
    "code": [
        "Напиши Python функцию для {topic}",
        "Исправь ошибку в коде: {topic}",
        "Оптимизируй алгоритм: {topic}",
        "Напиши тест для {topic}",
        "Объясни код: {topic}",
    ],
}

TOPICS = [
    "машинное обучение", "нейронные сети", "Python программирование",
    "алгоритмы сортировки", "базы данных SQL", "веб-разработка",
    "история компьютеров", "искусственный интеллект", "обработка текста",
    "кибербезопасность", "облачные вычисления", "мобильная разработка",
    "математический анализ", "линейная алгебра", "теория вероятностей",
    "физика", "химия", "биология", "философия", "экономика",
    "управление проектами", "дизайн интерфейсов", "тестирование ПО",
    "Docker и контейнеризация", "Git и контроль версий",
    "рекурсия", "динамическое программирование", "граф и дерево",
    "REST API", "микросервисы", "DevOps практики",
    "генетические алгоритмы", "компьютерное зрение", "NLP",
    "рекомендательные системы", "временные ряды", "кластеризация",
]


def generate_prompts(n_samples: int) -> list:
    """Generate diverse prompts from templates."""
    prompts = []
    categories = list(PROMPT_TEMPLATES.keys())
    
    for i in range(n_samples):
        cat = categories[i % len(categories)]
        template = random.choice(PROMPT_TEMPLATES[cat])
        topic = random.choice(TOPICS)
        prompt = template.format(topic=topic)
        prompts.append({"prompt": prompt, "category": cat, "topic": topic})
    
    random.shuffle(prompts)
    return prompts


# ═══ Teacher Backends ═══

def query_ollama(prompt: str, model: str = "qwen2.5:7b",
                 base_url: str = "http://localhost:11434") -> str:
    """Query local Ollama model."""
    import urllib.request
    
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 512}
    }).encode('utf-8')
    
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result.get("response", "")
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return ""


def query_openai(prompt: str, model: str = "gpt-4o-mini",
                 api_key: str = None) -> str:
    """Query OpenAI API."""
    import urllib.request
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    data = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.7,
    }).encode('utf-8')
    
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"OpenAI error: {e}")
        return ""


def cmd_generate(args):
    """Generate synthetic dataset from teacher model."""
    prompts = generate_prompts(args.n_samples)
    
    # Select teacher backend
    if args.teacher == "ollama":
        query_fn = lambda p: query_ollama(p, model=args.teacher_model)
    elif args.teacher == "openai":
        query_fn = lambda p: query_openai(p, model=args.teacher_model)
    else:
        raise ValueError(f"Unknown teacher: {args.teacher}")
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    success = 0
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, item in enumerate(prompts):
            # Rate limiting
            if i > 0 and args.teacher == "openai":
                time.sleep(0.5)
            
            response = query_fn(item["prompt"])
            
            if response and len(response) > 20:
                entry = {
                    "prompt": item["prompt"],
                    "response": response,
                    "category": item["category"],
                    "topic": item["topic"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                success += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Generated {i+1}/{args.n_samples} ({success} success)")
    
    logger.info(f"Done! {success}/{args.n_samples} samples → {args.output}")


def cmd_distill(args):
    """Train TARS on teacher-generated data with KD loss."""
    import torch
    import torch.nn.functional as F
    from brain.mamba2.model import TarsMamba2LM
    from brain.tokenizer import TarsTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    logger.info(f"Loading TARS from {args.model}...")
    model, _ = TarsMamba2LM.load_pretrained(args.model, device=device)
    model.train()
    
    tokenizer = TarsTokenizer()
    
    # Load teacher data
    data = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    logger.info(f"Loaded {len(data)} teacher samples")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    save_path = Path(args.save_dir) / "brain_distilled.pt"
    os.makedirs(args.save_dir, exist_ok=True)
    
    total_loss = 0
    n_steps = 0
    
    for epoch in range(args.epochs):
        random.shuffle(data)
        epoch_loss = 0
        
        for i, item in enumerate(data):
            # Concat prompt + response
            text = item["prompt"] + "\n" + item["response"]
            ids = tokenizer.encode(text)[:args.max_len]
            
            if len(ids) < 10:
                continue
            
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            
            # Forward
            result = model.think(input_ids)
            logits = result[0] if isinstance(result, tuple) else result
            
            # CE loss (teacher response as ground truth)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_steps += 1
            
            if (i + 1) % 100 == 0:
                avg = epoch_loss / (i + 1)
                logger.info(f"  [{epoch+1}/{args.epochs}] step {i+1}/{len(data)} loss={avg:.4f}")
        
        avg_loss = epoch_loss / max(len(data), 1)
        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        # Save
        checkpoint = {'model_state_dict': model.state_dict()}
        torch.save(checkpoint, str(save_path))
        logger.info(f"  Saved: {save_path}")
    
    logger.info(f"Distillation complete! {n_steps} steps")


def main():
    p = argparse.ArgumentParser(description="TARS Knowledge Distillation")
    sub = p.add_subparsers(dest="command")
    
    # Generate subcommand
    gen = sub.add_parser("generate", help="Generate synthetic data from teacher")
    gen.add_argument("--teacher", type=str, default="ollama",
                     choices=["ollama", "openai"])
    gen.add_argument("--teacher_model", type=str, default="qwen2.5:7b")
    gen.add_argument("--output", type=str, default="data/synthetic_teacher.jsonl")
    gen.add_argument("--n_samples", type=int, default=5000)
    
    # Distill subcommand
    dist = sub.add_parser("distill", help="Train TARS on teacher data")
    dist.add_argument("--data", type=str, default="data/synthetic_teacher.jsonl")
    dist.add_argument("--model", type=str, default="models/mamba2/brain_best.pt")
    dist.add_argument("--save_dir", type=str, default="models/mamba2")
    dist.add_argument("--epochs", type=int, default=3)
    dist.add_argument("--lr", type=float, default=2e-5)
    dist.add_argument("--max_len", type=int, default=512)
    
    args = p.parse_args()
    
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "distill":
        cmd_distill(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
