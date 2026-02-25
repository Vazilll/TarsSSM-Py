"""
═══════════════════════════════════════════════════════════════
  HuggingFace Dataset Downloader — Датасеты для обучения ТАРС
═══════════════════════════════════════════════════════════════

Скачивает профессиональные датасеты из HuggingFace.

Категории:
  --preset code     → Датасеты для программирования (Rust, C++, ASM, Dart, C)
  --preset chat     → Русские диалоги, шутки, харизма
  --preset agent    → Управление интерфейсом, вызов API, агенты
  --preset instruct → Instruction tuning (Alpaca, ShareGPT)
  --preset all      → Все вышеперечисленное

Требования:
  pip install datasets

Использование:
  python training/download_hf_dataset.py --preset all
  python training/download_hf_dataset.py --dataset IlyaGusev/ru_turbo_alpaca
  python training/download_hf_dataset.py --preset code --count 5000
"""

import os
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════════
# Пресеты датасетов по категориям
# ═══════════════════════════════════════════════════════════════

PRESETS = {
    # ─── Код: Rust, C++, ASM, C, Dart, Python ───
    "code": [
        {
            "name": "ise-uiuc/Magicoder-Evol-Instruct-110K",
            "desc": "110K задач по коду на разных языках (C++, Rust, Python, JS)",
            "count": 20000,
            "format": "instruct",
        },
        {
            "name": "bigcode/the-stack-smol",
            "desc": "Примеры кода из GitHub (asm, c, cpp, dart, rust)",
            "count": 10000,
            "format": "code",
            "subsets": ["default"],
        },
        {
            "name": "sahil2801/CodeAlpaca-20k",
            "desc": "20K задач instruction-tuning для программирования",
            "count": 20000,
            "format": "instruct",
        },
    ],
    
    # ─── Харизма, шутки, русские диалоги ───
    "chat": [
        {
            "name": "Vikhrmodels/ru_turbo_saiga",
            "desc": "50K+ русских инструкций и ответов (Saiga)",
            "count": 50000,
            "format": "chat",
        },
        {
            "name": "ai-forever/ru_sharegpt",
            "desc": "Диалоги людей с ChatGPT (русские)",
            "count": 20000,
            "format": "sharegpt",
        },
        {
            "name": "Den4ikAI/russian_instructions_2",
            "desc": "Русские инструкции для ИИ",
            "count": 30000,
            "format": "instruct",
        },
    ],
    
    # ─── Агенты и управление интерфейсом ───
    "agent": [
        {
            "name": "glaiveai/glaive-function-calling-v2",
            "desc": "Обучение вызову функций и API (function calling)",
            "count": 10000,
            "format": "chat",
        },
    ],
    
    # ─── Instruction tuning (основа) ───
    "instruct": [
        {
            "name": "Vikhrmodels/ru_turbo_saiga",
            "desc": "50K+ русских инструкций (основной датасет)",
            "count": 50000,
            "format": "chat",
        },
        {
            "name": "Den4ikAI/russian_instructions_2",
            "desc": "Русские инструкции для ИИ",
            "count": 20000,
            "format": "instruct",
        },
    ],
}


def format_row(row: dict, fmt: str) -> str:
    """Форматирует одну строку датасета в текст для обучения."""
    
    # Формат Alpaca / Instruct
    if fmt == "instruct":
        inst = row.get("instruction", row.get("prompt", row.get("question", "")))
        if not isinstance(inst, str): inst = str(inst) if inst else ""
        inst = inst.strip()
        
        inp = row.get("input", row.get("context", ""))
        if not isinstance(inp, str): inp = str(inp) if inp else ""
        inp = inp.strip()
        
        out = row.get("output", row.get("response", row.get("completion", row.get("answer", ""))))
        if not isinstance(out, str): out = str(out) if out else ""
        out = out.strip()
        
        if not inst and not out:
            return ""
        if inp:
            return f"Запрос: {inst}\nКонтекст: {inp}\nОтвет: {out}"
        return f"Запрос: {inst}\nОтвет: {out}"
    
    # Формат ShareGPT / Conversations
    if fmt == "sharegpt":
        convs = row.get("conversations", row.get("messages", []))
        if not convs:
            return ""
        dialog = []
        for msg in convs:
            role_raw = msg.get("from", msg.get("role", "user"))
            role = "Вопрос" if role_raw in ("human", "user") else "Ответ"
            text = msg.get("value", msg.get("content", "")).strip()
            if text:
                dialog.append(f"{role}: {text}")
        return "\n".join(dialog) if dialog else ""
    
    # Формат Chat (system/user/assistant messages)
    if fmt == "chat":
        messages = row.get("messages", row.get("conversations", []))
        if not messages:
            # Special parsing for glaive function calling dataset which has system/chat as strings
            if "chat" in row and isinstance(row["chat"], str):
                system_prompt = row.get("system", "").strip()
                chat_str = row["chat"].strip()
                if system_prompt:
                    return f"Система: {system_prompt}\nДиалог:\n{chat_str}"
                return f"Диалог:\n{chat_str}"
                
            # Fallback: text field
            return row.get("text", row.get("content", "")).strip()
        
        dialog = []
        for msg in messages:
            role = msg.get("role", msg.get("from", "user"))
            text = msg.get("content", msg.get("value", "")).strip()
            if text:
                if role == "system":
                    dialog.append(f"Система: {text}")
                elif role in ("user", "human"):
                    dialog.append(f"Запрос: {text}")
                else:
                    dialog.append(f"Ответ: {text}")
        return "\n".join(dialog) if dialog else ""
    
    # Формат Code (сырой код)
    if fmt == "code":
        content = row.get("content", row.get("text", row.get("code", ""))).strip()
        lang = row.get("lang", row.get("language", "unknown"))
        if content and len(content) > 50:
            return f"Язык: {lang}\n```\n{content[:8000]}\n```"
        return ""
    
    # Fallback: text
    text = row.get("text", "").strip()
    if text:
        return text
    return json.dumps(row, ensure_ascii=False)[:2000]


def download_one_dataset(ds_config: dict, output_dir: str) -> str:
    """Скачивает один датасет и возвращает текст."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ pip install datasets")
        return ""
    
    name = ds_config["name"]
    count = ds_config.get("count", 10000)
    fmt = ds_config.get("format", "instruct")
    safe_name = name.replace("/", "_")
    output_file = os.path.join(output_dir, f"hf_{safe_name}.txt")
    
    # Проверяем кеш
    if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ {name}: уже скачан ({size_mb:.1f} MB)")
        with open(output_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    print(f"  ↓ {name}: {ds_config.get('desc', '')}...")
    
    try:
        # Загрузка
        subsets = ds_config.get("subsets", [])
        if subsets:
            ds = load_dataset(name, subsets[0], split="train", streaming=False)
        else:
            ds = load_dataset(name, split="train", streaming=False)
        
        if len(ds) > count:
            ds = ds.select(range(count))
        
        # Форматирование
        texts = []
        for row in ds:
            text = format_row(row, fmt)
            if text and len(text) > 20:
                texts.append(text)
        
        corpus = "\n\n".join(texts)
        
        # Сохранение
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(corpus)
        
        size_mb = len(corpus.encode('utf-8')) / (1024 * 1024)
        print(f"  ✓ {name}: {len(texts)} примеров, {size_mb:.1f} MB")
        return corpus
        
    except Exception as e:
        print(f"  ⚠ {name}: ошибка — {e}")
        return ""


def download_preset(preset: str, output_dir: str = None, count_override: int = None) -> list:
    """Скачивает все датасеты из пресета."""
    if output_dir is None:
        output_dir = str(ROOT / "data")
    os.makedirs(output_dir, exist_ok=True)
    
    if preset == "all":
        datasets = []
        for p in PRESETS.values():
            datasets.extend(p)
        # Дедупликация по имени
        seen = set()
        unique = []
        for d in datasets:
            if d["name"] not in seen:
                seen.add(d["name"])
                unique.append(d)
        datasets = unique
    else:
        datasets = PRESETS.get(preset, [])
    
    if not datasets:
        print(f"❌ Неизвестный пресет: {preset}. Доступные: {', '.join(PRESETS.keys())}, all")
        return []
    
    print(f"[HF] ═══ Скачиваю пресет '{preset}': {len(datasets)} датасетов ═══")
    
    results = []
    for ds_config in datasets:
        if count_override:
            ds_config = {**ds_config, "count": count_override}
        text = download_one_dataset(ds_config, output_dir)
        if text:
            results.append(text)
    
    total_mb = sum(len(t.encode('utf-8')) for t in results) / (1024 * 1024)
    print(f"[HF] ═══ Готово: {len(results)} датасетов, {total_mb:.1f} MB ═══")
    return results


def download_single(dataset_name: str, max_samples: int = 10000, 
                    output_dir: str = None) -> str:
    """Скачивает один конкретный датасет по имени."""
    if output_dir is None:
        output_dir = str(ROOT / "data")
    
    config = {
        "name": dataset_name,
        "count": max_samples,
        "format": "instruct",  # default guess
    }
    return download_one_dataset(config, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка датасетов из HuggingFace для обучения ТАРС",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Пресеты:
  code     — Код: Rust, C++, ASM, Dart, C (Magicoder, The Stack, CodeAlpaca)
  chat     — Русские диалоги, шутки, харизма (Alpaca, ShareGPT)
  agent    — Управление интерфейсом, function calling
  instruct — Instruction tuning
  all      — Все вышеперечисленное

Примеры:
  python training/download_hf_dataset.py --preset all
  python training/download_hf_dataset.py --preset code --count 5000
  python training/download_hf_dataset.py --dataset IlyaGusev/ru_turbo_alpaca
        """
    )
    parser.add_argument("--preset", type=str, default=None,
                        help="Пресет датасетов: code, chat, agent, instruct, all")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Конкретный датасет HuggingFace")
    parser.add_argument("--count", type=int, default=None,
                        help="Количество примеров")
    parser.add_argument("--output", type=str, default=None,
                        help="Директория для сохранения")
    args = parser.parse_args()
    
    if args.preset:
        download_preset(args.preset, args.output, args.count)
    elif args.dataset:
        download_single(args.dataset, args.count or 10000, args.output)
    else:
        # По умолчанию скачиваем всё
        download_preset("all", args.output, args.count)


if __name__ == "__main__":
    main()
