"""
═══════════════════════════════════════════════════════════════
  HuggingFace Dataset Downloader — Датасеты для обучения ТАРС
═══════════════════════════════════════════════════════════════

Скачивает только ПРОВЕРЕННЫЕ быстрые датасеты из HuggingFace.

Категории:
  --preset code     → Код (Magicoder, CodeAlpaca, commitpackft)
  --preset math     → Математика и логика
  --preset thinking → Reasoning / Chain-of-Thought
  --preset chat     → Русские диалоги, харизма
  --preset instruct → Instruction tuning
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

ROOT = Path(__file__).parent.parent

# ═══════════════════════════════════════════════════════════════
# ПРОВЕРЕННЫЕ БЫСТРЫЕ датасеты (только parquet/arrow формат)
# ═══════════════════════════════════════════════════════════════

PRESETS = {
    # ─── Код: C++, Rust, Python, JS ───
    "code": [
        {
            "name": "ise-uiuc/Magicoder-Evol-Instruct-110K",
            "desc": "110K задач по коду",
            "count": 15000,
            "format": "instruct",
        },
        {
            "name": "sahil2801/CodeAlpaca-20k",
            "desc": "20K instruction-tuning для кода",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "codeparrot/self-instruct-starcoder",
            "desc": "StarCoder self-instruct",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "d0rj/conala-mined-ru",
            "desc": "Код + комментарии на русском",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "bigcode/commitpackft",
            "desc": "Коммиты как инструкции",
            "count": 10000,
            "format": "instruct",
        },
    ],

    # ─── Математика и логика ───
    "math": [
        {
            "name": "d0rj/MathInstruct-ru",
            "desc": "Мат. задачи с решениями (русский)",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "d0rj/muInstruct-ru",
            "desc": "Авто-сгенерированные мат. задачи (русский)",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "TIGER-Lab/MathInstruct",
            "desc": "Мат. задачи с CoT решениями",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "meta-math/MetaMathQA",
            "desc": "MetaMath: переформулированные GSM8K+MATH",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "openai/gsm8k",
            "desc": "8.5K задач с пошаговыми решениями",
            "count": 8500,
            "format": "instruct",
        },
    ],

    # ─── Мышление: Chain-of-Thought ───
    "thinking": [
        {
            "name": "open-thoughts/OpenThoughts-114k",
            "desc": "114K задач с цепочками рассуждений",
            "count": 10000,
            "format": "sharegpt",
        },
    ],

    # ─── Чат и диалоги ───
    "chat": [
        {
            "name": "Den4ikAI/russian_instructions_2",
            "desc": "Русские инструкции — основной чат",
            "count": 30000,
            "format": "instruct",
        },
        {
            "name": "IlyaGusev/ru_turbo_alpaca",
            "desc": "GPT-4 русские инструкции",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "IlyaGusev/ru_turbo_alpaca_evol_instruct",
            "desc": "Эволюционные цепочки reasoning",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "Vikhrmodels/GrandMaster-PRO-MAX",
            "desc": "Сложный русский reasoning",
            "count": 10000,
            "format": "chat",
        },
    ],

    # ─── Самосознание ТАРС ───
    "selfaware": [
        {
            "name": "ai-forever/school_notebooks_QA",
            "desc": "Школьные вопросы-ответы (образование)",
            "count": 10000,
            "format": "instruct",
        },
    ],

    # ─── Наука ───
    "science": [
        {
            "name": "OpenAssistant/oasst1",
            "desc": "Мультиязычные диалоги",
            "count": 15000,
            "format": "chat",
        },
        {
            "name": "OpenAssistant/oasst2",
            "desc": "OpenAssistant v2",
            "count": 10000,
            "format": "chat",
        },
    ],

    # ─── Instruction tuning ───
    "instruct": [
        {
            "name": "d0rj/OpenOrca-ru",
            "desc": "Русские инструкции (перевод OpenOrca)",
            "count": 30000,
            "format": "instruct",
        },
        {
            "name": "d0rj/OpenHermes-2.5-ru",
            "desc": "Русские инструкции GPT-4 качества",
            "count": 20000,
            "format": "sharegpt",
        },
    ],

    # ─── Русский reasoning ───
    "russian": [
        {
            "name": "d0rj/R1-Distill-SFT_v1-ru",
            "desc": "DeepSeek R1 reasoning (русский)",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "d0rj/ROMB-1.0",
            "desc": "Олимпиадные задачи (русский)",
            "count": 3000,
            "format": "instruct",
        },
    ],

    # ─── DPO (preference) ───
    "dpo": [
        {
            "name": "Intel/orca_dpo_pairs",
            "desc": "DPO пары (chosen vs rejected)",
            "count": 5000,
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
            if "chat" in row and isinstance(row["chat"], str):
                system_prompt = row.get("system", "").strip()
                chat_str = row["chat"].strip()
                if system_prompt:
                    return f"Система: {system_prompt}\nДиалог:\n{chat_str}"
                return f"Диалог:\n{chat_str}"
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
    subsets = ds_config.get("subsets", [])
    
    output_file = os.path.join(output_dir, f"hf_{safe_name}.txt")
    
    # Проверяем кеш
    if os.path.exists(output_file) and os.path.getsize(output_file) > 10000:
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  ✓ {name}: уже скачан ({size_mb:.1f} MB)")
        with open(output_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    print(f"  ↓ {name}: {ds_config.get('desc', '')}...")
    
    try:
        # Загрузка (берём только count строк — быстрее!)
        split_str = f"train[:{count}]"
        try:
            if subsets:
                ds = load_dataset(name, subsets[0], split=split_str, streaming=False)
            else:
                ds = load_dataset(name, split=split_str, streaming=False)
        except Exception:
            # Fallback — грузим всё и обрезаем
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
        for pname, pdata in PRESETS.items():
            datasets.extend(pdata)
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
    
    # Подсчёт сколько будет
    total_count = sum(d.get("count", 10000) for d in datasets)
    print(f"[HF] ═══ Пресет '{preset}': {len(datasets)} датасетов, ~{total_count:,} примеров ═══")
    
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
        "format": "instruct",
    }
    return download_one_dataset(config, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Загрузка датасетов из HuggingFace для обучения ТАРС",
    )
    parser.add_argument("--preset", type=str, default=None,
                        help="Пресет: code, math, thinking, chat, instruct, all")
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
        download_preset("all", args.output, args.count)


if __name__ == "__main__":
    main()
