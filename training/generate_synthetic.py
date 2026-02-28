"""
═══════════════════════════════════════════════════════════════
  Генерация синтетических STEM данных для TARS
═══════════════════════════════════════════════════════════════

Техника из Phi-4 (400B synthetic STEM tokens) и SmolLM2 (FineMath).

3 типа данных:
  1. Math Reasoning — пошаговые решения задач  
  2. Code Reasoning — объяснения алгоритмов
  3. Logic Chains   — цепочки рассуждений (если A→B и B→C, то A→C)

Использование:
  # Через Qwen API (рекомендуется):
  python training/generate_synthetic.py --provider qwen --output data/synthetic_stem.txt --n_samples 10000

  # Через OpenAI-compatible API:
  python training/generate_synthetic.py --provider openai --api_key "..." --output data/synthetic_stem.txt

  # Offline с простым генератором (без API):
  python training/generate_synthetic.py --provider offline --output data/synthetic_stem.txt --n_samples 50000
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ═══ Offline генератор (без API) ═══

MATH_TEMPLATES_RU = [
    ("Задача: Найди {op} чисел {a} и {b}.\n"
     "Решение:\n"
     "Шаг 1: {a} {sign} {b}\n"
     "Шаг 2: = {result}\n"
     "Ответ: {result}"),
    
    ("Вопрос: Если у Маши {a} яблок и она купила ещё {b}, сколько у неё всего?\n"
     "Рассуждение:\n"
     "1. Начальное количество: {a}\n"
     "2. Добавлено: {b}\n"
     "3. Итого: {a} + {b} = {result}\n"
     "Ответ: {result} яблок."),
    
    ("Задача: Поезд проехал {a} км за {b} часов. Какова средняя скорость?\n"
     "Решение:\n"
     "Средняя скорость = расстояние / время\n"
     "v = {a} / {b} = {result} км/ч\n"
     "Ответ: {result} км/ч."),
]

LOGIC_TEMPLATES_RU = [
    ("Посылка 1: Все {A} являются {B}.\n"
     "Посылка 2: {X} является {A}.\n"
     "Вопрос: Является ли {X} {B}?\n"
     "Рассуждение:\n"
     "1. По посылке 1: если нечто — {A}, то оно — {B}.\n"
     "2. По посылке 2: {X} — это {A}.\n"
     "3. Следовательно, {X} является {B}.\n"
     "Ответ: Да, {X} является {B}."),
    
    ("Если {cond1}, то {result1}.\n"
     "Если {result1}, то {result2}.\n"
     "Дано: {cond1}.\n"
     "Вопрос: верно ли, что {result2}?\n"
     "Рассуждение по транзитивности:\n"
     "1. {cond1} → {result1} (по первому правилу)\n"
     "2. {result1} → {result2} (по второму правилу)\n"
     "3. Значит, {cond1} → {result2}\n"
     "Ответ: Да, {result2} — верно."),
]

CODE_TEMPLATES_RU = [
    ("Задача: Напиши функцию на Python для {task}.\n"
     "Решение:\n"
     "```python\n"
     "{code}\n"
     "```\n"
     "Объяснение: {explanation}"),
]

# Data pools for template filling
OBJECTS = ["кошки", "собаки", "птицы", "рыбы", "деревья", "цветы",
           "студенты", "учителя", "программисты", "музыканты"]
PROPERTIES = ["живые существа", "теплокровные", "позвоночные",
              "разумные", "трудолюбивые", "творческие"]
NAMES = ["Тарс", "Алиса", "Борис", "Вика", "Грач", "Дана"]

CONDITIONS = [
    ("идёт дождь", "земля мокрая", "растения получают воду"),
    ("температура ниже нуля", "вода замерзает", "лёд образуется"),
    ("солнце встало", "стало светло", "птицы запели"),
    ("код скомпилирован", "программа запущена", "результат получен"),
]

CODE_TASKS = [
    ("суммы элементов списка",
     "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return total",
     "Проходим по каждому элементу и складываем в total."),
    ("факториала числа",
     "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
     "Рекурсивно умножаем n на факториал (n-1)."),
    ("поиска максимума в списке",
     "def find_max(lst):\n    if not lst:\n        return None\n    mx = lst[0]\n    for x in lst[1:]:\n        if x > mx:\n            mx = x\n    return mx",
     "Проходим по списку, обновляя максимум."),
    ("проверки палиндрома",
     "def is_palindrome(s):\n    s = s.lower()\n    return s == s[::-1]",
     "Сравниваем строку с её перевёрнутой версией."),
]


def generate_math_sample():
    ops = [
        ("сумму", "+", lambda a, b: a + b),
        ("разность", "-", lambda a, b: a - b),
        ("произведение", "×", lambda a, b: a * b),
    ]
    op_name, sign, func = random.choice(ops)
    a = random.randint(1, 999)
    b = random.randint(1, 999)
    result = func(a, b)
    
    template = random.choice(MATH_TEMPLATES_RU)
    return template.format(op=op_name, a=a, b=b, sign=sign, result=result)


def generate_logic_sample():
    template = random.choice(LOGIC_TEMPLATES_RU)
    if "{A}" in template:
        A = random.choice(OBJECTS)
        B = random.choice(PROPERTIES)
        X = random.choice(NAMES)
        return template.format(A=A, B=B, X=X)
    else:
        cond1, result1, result2 = random.choice(CONDITIONS)
        return template.format(cond1=cond1, result1=result1, result2=result2)


def generate_code_sample():
    task, code, explanation = random.choice(CODE_TASKS)
    template = random.choice(CODE_TEMPLATES_RU)
    return template.format(task=task, code=code, explanation=explanation)


def generate_offline(n_samples):
    """Generate synthetic STEM data without API."""
    samples = []
    generators = [generate_math_sample, generate_logic_sample, generate_code_sample]
    weights = [0.4, 0.35, 0.25]  # Math-heavy (Phi-4 style)
    
    for _ in range(n_samples):
        gen = random.choices(generators, weights=weights, k=1)[0]
        samples.append(gen())
    
    return samples


def generate_with_api(provider, api_key, n_samples, base_url=None):
    """Generate synthetic reasoning data using LLM API."""
    try:
        import openai
    except ImportError:
        print("[!] pip install openai required for API generation")
        return []
    
    prompts = [
        "Придумай математическую задачу для школьника и реши её пошагово на русском. "
        "Покажи каждый шаг рассуждения.",
        
        "Придумай логическую задачу с посылками и выводом. "
        "Покажи цепочку рассуждений step-by-step на русском.",
        
        "Придумай простую задачу по программированию на Python. "
        "Напиши решение и объясни каждую строку на русском.",
    ]
    
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    
    client = openai.OpenAI(**client_kwargs)
    samples = []
    
    for i in range(n_samples):
        try:
            prompt = random.choice(prompts)
            response = client.chat.completions.create(
                model="gpt-4o-mini" if provider == "openai" else "qwen-plus",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.9,
            )
            text = response.choices[0].message.content
            samples.append(text)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i+1}/{n_samples} samples...")
        except Exception as e:
            print(f"  [!] Error at sample {i}: {e}")
            continue
    
    return samples


def main():
    p = argparse.ArgumentParser(description="Generate Synthetic STEM Data for TARS")
    p.add_argument('--provider', choices=['offline', 'openai', 'qwen'],
                   default='offline',
                   help="Data generation method")
    p.add_argument('--api_key', type=str, default=None)
    p.add_argument('--base_url', type=str, default=None)
    p.add_argument('--output', type=str, default='data/synthetic_stem.txt')
    p.add_argument('--n_samples', type=int, default=10000)
    args = p.parse_args()
    
    print(f"{'═'*60}")
    print(f"  TARS Synthetic STEM Data Generator")
    print(f"  Provider: {args.provider} | Samples: {args.n_samples}")
    print(f"{'═'*60}\n")
    
    if args.provider == 'offline':
        samples = generate_offline(args.n_samples)
    else:
        samples = generate_with_api(args.provider, args.api_key, args.n_samples,
                                     args.base_url)
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(s.strip() + "\n\n")
    
    total_chars = sum(len(s) for s in samples)
    print(f"\n✓ Generated {len(samples)} samples ({total_chars/1e6:.1f}M chars)")
    print(f"  Saved to: {args.output}")
    print(f"  Math: ~40% | Logic: ~35% | Code: ~25%")


if __name__ == "__main__":
    main()
