"""
═══════════════════════════════════════════════════════════════
  RLAIF Preference Data Generator for TARS
═══════════════════════════════════════════════════════════════

Generates preference pairs (chosen vs rejected) using AI feedback.
No human annotation needed → scalable DPO/KTO training.

Pipeline:
  1. Generate prompt → get 2 responses from same model
  2. Use judge model (Ollama/OpenAI) to pick winner
  3. Output JSONL: {"prompt": ..., "chosen": ..., "rejected": ...}

Judge criteria:
  - Helpfulness, accuracy, coherence, safety
  - Comparison format (explicit winner selection)

Usage:
  # Auto-generate preferences with local judge
  python training/generate_preferences.py \
      --model models/mamba2/brain_best.pt \
      --judge ollama --judge_model qwen2.5:7b \
      --output data/preferences.jsonl --n_pairs 1000

  # Then train DPO
  python training/train_dpo.py --data data/preferences.jsonl
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
logger = logging.getLogger("rlaif")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ═══ Prompts for generating diverse preference data ═══
PROMPTS_RU = [
    "Объясни, что такое машинное обучение, простыми словами.",
    "Напиши короткий рассказ про робота, который учится быть человеком.",
    "Какие есть способы оптимизировать Python код?",
    "Сравни PostgreSQL и MongoDB — когда что использовать?",
    "Как работает внимание (attention) в трансформерах?",
    "Напиши функцию сортировки слиянием на Python с комментариями.",
    "Объясни теорему Байеса с примером из реальной жизни.",
    "Что лучше: REST API или GraphQL? Аргументы за и против.",
    "Как настроить CI/CD пайплайн для Python проекта?",
    "Расскажи о принципах SOLID с примерами на Python.",
    "Как работает garbage collector в Python?",
    "Напиши асинхронный HTTP клиент на aiohttp.",
    "Объясни разницу между процессами и потоками.",
    "Что такое CAP теорема? Приведи примеры.",
    "Как защитить веб-приложение от SQL инъекций?",
    "Реши задачу: найти два числа в массиве с суммой K.",
    "Как правильно спроектировать REST API?",
    "Объясни Raft consensus алгоритм простыми словами.",
    "Напиши Dockerfile для Python FastAPI приложения.",
    "Как работает блокчейн? Объясни архитектуру.",
    "Расскажи про паттерн Observer с примером.",
    "Как оптимизировать SQL запросы?",
    "Объясни разницу между UTF-8 и cp1251.",
    "Напиши unit тест для функции парсинга JSON.",
    "Как работает виртуальная память?",
    "Что такое event loop в asyncio?",
    "Сравни Docker и Kubernetes.",
    "Как настроить логирование в Python проекте?",
    "Напиши регулярное выражение для парсинга email.",
    "Объясни принцип работы нейронной сети.",
]

JUDGE_PROMPT_TEMPLATE = """Ты — судья. Сравни два ответа на вопрос и выбери лучший.

ВОПРОС: {prompt}

===ОТВЕТ A===
{response_a}
===КОНЕЦ A===

===ОТВЕТ B===
{response_b}
===КОНЕЦ B===

Критерии оценки:
1. Точность и правильность информации
2. Полнота ответа
3. Ясность и структура
4. Практическая полезность

Ответь ТОЛЬКО одной буквой: A или B (какой ответ лучше).
Ответ:"""


def query_model(prompt: str, model_name: str = "ollama",
                model_id: str = "qwen2.5:7b", temperature: float = 0.8) -> str:
    """Query a model for response generation."""
    import urllib.request
    
    if model_name == "ollama":
        data = json.dumps({
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 400}
        }).encode('utf-8')
        
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
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
    
    elif model_name == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        data = json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400,
            "temperature": temperature,
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
    
    return ""


def generate_tars_response(prompt: str, model, tokenizer, 
                            device: str, temperature: float = 0.9) -> str:
    """Generate response using TARS model itself."""
    import torch
    
    ids = tokenizer.encode(prompt)[:256]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    
    model.eval()
    with torch.no_grad():
        result = model.think(input_ids)
        logits = result[0] if isinstance(result, tuple) else result
        
        generated = []
        for _ in range(200):
            next_logits = logits[:, -1, :]
            probs = torch.softmax(next_logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1).item()
            
            if token == 0:
                break
            generated.append(token)
            
            token_t = torch.tensor([[token]], dtype=torch.long, device=device)
            result = model.think(token_t)
            logits = result[0] if isinstance(result, tuple) else result
    
    return tokenizer.decode(generated)


def judge_pair(prompt: str, resp_a: str, resp_b: str,
               judge_name: str, judge_model: str) -> str:
    """Ask judge model which response is better. Returns 'A' or 'B'."""
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        response_a=resp_a,
        response_b=resp_b,
    )
    
    # Lower temperature for more deterministic judging
    verdict = query_model(judge_prompt, judge_name, judge_model, temperature=0.1)
    verdict = verdict.strip().upper()
    
    if 'A' in verdict[:3]:
        return 'A'
    elif 'B' in verdict[:3]:
        return 'B'
    else:
        # If judge can't decide, flip a coin (signals poor pair)
        return random.choice(['A', 'B'])


def main():
    p = argparse.ArgumentParser(description="RLAIF Preference Data Generator")
    p.add_argument("--judge", type=str, default="ollama",
                   choices=["ollama", "openai"])
    p.add_argument("--judge_model", type=str, default="qwen2.5:7b")
    p.add_argument("--generator", type=str, default="ollama",
                   choices=["ollama", "openai", "tars"])
    p.add_argument("--gen_model", type=str, default="qwen2.5:7b")
    p.add_argument("--tars_model", type=str, default="models/mamba2/brain_best.pt",
                   help="TARS checkpoint (if --generator tars)")
    p.add_argument("--output", type=str, default="data/preferences.jsonl")
    p.add_argument("--n_pairs", type=int, default=500)
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Custom prompts file (one per line)")
    args = p.parse_args()
    
    # Load custom prompts or use built-in
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = PROMPTS_RU
    
    # If using TARS as generator, load model
    tars_model = None
    tars_tokenizer = None
    tars_device = "cpu"
    if args.generator == "tars":
        import torch
        from brain.mamba2.model import TarsMamba2LM
        from brain.tokenizer import TarsTokenizer
        
        tars_device = "cuda" if torch.cuda.is_available() else "cpu"
        tars_model, _ = TarsMamba2LM.load_pretrained(args.tars_model, device=tars_device)
        tars_tokenizer = TarsTokenizer()
        logger.info(f"TARS model loaded ({tars_device})")
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    success = 0
    skipped = 0
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for i in range(args.n_pairs):
            prompt = prompts[i % len(prompts)]
            
            # Generate 2 responses (different temperatures → diversity)
            if args.generator == "tars":
                resp_a = generate_tars_response(prompt, tars_model, tars_tokenizer, tars_device, 0.7)
                resp_b = generate_tars_response(prompt, tars_model, tars_tokenizer, tars_device, 1.1)
            else:
                resp_a = query_model(prompt, args.generator, args.gen_model, 0.7)
                resp_b = query_model(prompt, args.generator, args.gen_model, 1.1)
            
            # Skip if either response is too short
            if len(resp_a) < 20 or len(resp_b) < 20:
                skipped += 1
                continue
            
            # Skip if responses are too similar (no learning signal)
            if resp_a.strip() == resp_b.strip():
                skipped += 1
                continue
            
            # Rate limit for API calls
            if args.judge == "openai":
                time.sleep(0.5)
            
            # Judge
            winner = judge_pair(prompt, resp_a, resp_b,
                              args.judge, args.judge_model)
            
            if winner == 'A':
                chosen, rejected = resp_a, resp_b
            else:
                chosen, rejected = resp_b, resp_a
            
            entry = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
            
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            success += 1
            
            if (i + 1) % 25 == 0:
                logger.info(f"  [{i+1}/{args.n_pairs}] {success} pairs, {skipped} skipped")
    
    logger.info(f"Done! {success} preference pairs → {args.output}")
    logger.info(f"  Skipped: {skipped} (too short or duplicate)")
    logger.info(f"\nNext: python training/train_dpo.py --data {args.output}")


if __name__ == "__main__":
    main()
