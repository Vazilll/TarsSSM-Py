"""
═══════════════════════════════════════════════════════════════
  HuggingFace Dataset Downloader — Датасеты для обучения ТАРС
═══════════════════════════════════════════════════════════════

Скачивает только ПРОВЕРЕННЫЕ быстрые датасеты из HuggingFace.

Категории:
  --preset code     → Код (Magicoder, CodeAlpaca, StarCoder, commitpackft)
  --preset math     → Математика и логика (MathInstruct-ru, orca-math-ru, GSM8K)
  --preset thinking → Reasoning / Chain-of-Thought (OpenThoughts)
  --preset chat     → Русские диалоги (russian_instructions, GrandMaster)
  --preset instruct → Instruction tuning (OpenOrca-ru, OpenHermes-ru, ru-instruct)
  --preset russian  → Русский reasoning (R1-Distill, ROMB)
  --preset science  → Наука (OpenAssistant, школьные QA)
  --preset quality  → Обратная связь (Feedback-Collection-ru)
  --preset dpo      → DPO/RLHF (orca_dpo, full-hh-rlhf-ru)
  --preset all      → Все вышеперечисленное (~580K+ примеров)

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
            "desc": "110K задач по коду (лучший code-instruct)",
            "count": 25000,
            "format": "instruct",
        },
        {
            "name": "sahil2801/CodeAlpaca-20k",
            "desc": "20K instruction-tuning для кода",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "codeparrot/self-instruct-starcoder",
            "desc": "StarCoder self-instruct",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "d0rj/conala-mined-ru",
            "desc": "Код + комментарии на русском",
            "count": 8000,
            "format": "instruct",
        },
        {
            "name": "bigcode/commitpackft",
            "desc": "Коммиты как инструкции",
            "count": 15000,
            "format": "instruct",
        },
    ],

    # ─── Математика и логика ───
    "math": [
        {
            "name": "d0rj/MathInstruct-ru",
            "desc": "Мат. задачи с решениями (русский)",
            "count": 15000,
            "format": "instruct",
        },
        {
            "name": "d0rj/orca-math-word-problems-200k-ru",
            "desc": "200K текстовых мат. задач (русский) ★",
            "count": 30000,
            "format": "instruct",
        },
        {
            "name": "d0rj/muInstruct-ru",
            "desc": "Авто-сгенерированные мат. задачи (русский)",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "TIGER-Lab/MathInstruct",
            "desc": "Мат. задачи с CoT решениями",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "meta-math/MetaMathQA",
            "desc": "MetaMath: переформулированные GSM8K+MATH",
            "count": 10000,
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
            "desc": "114K задач с цепочками рассуждений ★",
            "count": 20000,
            "format": "sharegpt",
        },
    ],

    # ─── Чат и диалоги ───
    "chat": [
        {
            "name": "Den4ikAI/russian_instructions_2",
            "desc": "Русские инструкции — основной чат ★",
            "count": 50000,
            "format": "instruct",
        },
        {
            "name": "IlyaGusev/ru_turbo_alpaca",
            "desc": "GPT-4 русские инструкции",
            "count": 15000,
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
            "desc": "Сложный русский reasoning ★",
            "count": 15000,
            "format": "chat",
        },
    ],

    # ─── Instruction tuning (основа интеллекта) ───
    "instruct": [
        {
            "name": "d0rj/OpenOrca-ru",
            "desc": "Русские инструкции (перевод OpenOrca) ★",
            "count": 50000,
            "format": "instruct",
        },
        {
            "name": "d0rj/OpenHermes-2.5-ru",
            "desc": "Русские инструкции GPT-4 качества ★",
            "count": 40000,
            "format": "sharegpt",
        },
        {
            "name": "d0rj/ru-instruct",
            "desc": "Обширные русские инструкции (2024) ★",
            "count": 30000,
            "format": "instruct",
        },
        {
            "name": "lksy/ru_instruct_gpt4",
            "desc": "GPT-4 инструкции на русском ★",
            "count": 20000,
            "format": "instruct",
        },
    ],

    # ─── Русский reasoning ───
    "russian": [
        {
            "name": "d0rj/R1-Distill-SFT_v1-ru",
            "desc": "DeepSeek R1 reasoning (русский) ★",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "d0rj/ROMB-1.0",
            "desc": "Олимпиадные задачи (русский)",
            "count": 5000,
            "format": "instruct",
        },
    ],

    # ─── Наука и общие знания ───
    "science": [
        {
            "name": "OpenAssistant/oasst1",
            "desc": "Мультиязычные диалоги (качество)",
            "count": 15000,
            "format": "chat",
        },
        {
            "name": "OpenAssistant/oasst2",
            "desc": "OpenAssistant v2 — улучшенные диалоги",
            "count": 15000,
            "format": "chat",
        },
        {
            "name": "ai-forever/school_notebooks_QA",
            "desc": "Школьные вопросы-ответы (образование)",
            "count": 15000,
            "format": "instruct",
        },
    ],

    # ─── Качество ответов и обратная связь ───
    "quality": [
        {
            "name": "d0rj/Feedback-Collection-ru",
            "desc": "Обратная связь + оценки качества (рус) ★",
            "count": 10000,
            "format": "instruct",
        },
    ],

    # ─── DPO / RLHF (preference learning) ───
    "dpo": [
        {
            "name": "Intel/orca_dpo_pairs",
            "desc": "DPO пары (chosen vs rejected)",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "d0rj/full-hh-rlhf-ru",
            "desc": "RLHF пары на русском (harmless+helpful) ★",
            "count": 15000,
            "format": "instruct",
        },
    ],
}


import re as _re

# ═══════════════════════════════════════════════════════════════
# QUALITY PIPELINE — Фильтрация и очистка данных
# ═══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Очистка текста от мусора."""
    if not text:
        return ""
    # HTML теги
    text = _re.sub(r'<[^>]+>', '', text)
    # URLs
    text = _re.sub(r'https?://\S+', '', text)
    # Email
    text = _re.sub(r'\S+@\S+\.\S+', '', text)
    # Повторяющиеся пробелы
    text = _re.sub(r'[ \t]{3,}', '  ', text)
    # Повторяющиеся переносы (>3 → 2)
    text = _re.sub(r'\n{4,}', '\n\n\n', text)
    # Повторяющиеся знаки препинания (... → ..., !!! → !)
    text = _re.sub(r'([!?.])\1{3,}', r'\1\1\1', text)
    # Non-printable (кроме \n\r\t)
    text = _re.sub(r'[^\S\n\r\t]+', ' ', text)
    return text.strip()


def is_quality_text(text: str, min_len: int = 50, max_len: int = 15000) -> bool:
    """Проверка качества текста. Returns True если текст достаточно хороший."""
    if not text:
        return False
    
    length = len(text)
    
    # Длина
    if length < min_len or length > max_len:
        return False
    
    # Соотношение букв к общему количеству символов (>40%)
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count / max(length, 1) < 0.4:
        return False
    
    # Детекция повторений — строки
    lines = text.split('\n')
    if len(lines) > 3:
        unique_lines = set(line.strip().lower() for line in lines if line.strip())
        if len(unique_lines) < len(lines) * 0.5:  # >50% дублей
            return False
    
    # Детекция повторений — триграммы символов
    if length > 200:
        trigrams = [text[i:i+3] for i in range(0, min(length, 1000) - 2)]
        unique_tri = set(trigrams)
        if len(unique_tri) < len(trigrams) * 0.15:  # слишком мало уникальных
            return False
    
    # Слишком много спецсимволов (>30% не-буквы-цифры-пробелы)
    normal_chars = sum(1 for c in text if c.isalnum() or c.isspace() or c in '.,!?;:-()"\'/«»')
    if normal_chars / max(length, 1) < 0.7:
        return False
    
    return True


def deduplicate_texts(texts: list, max_hash_len: int = 200) -> list:
    """Дедупликация по хешу первых N символов (нормализованных)."""
    import hashlib
    seen = set()
    unique = []
    for text in texts:
        # Нормализуем: lowercase, убираем пробелы
        norm = text[:max_hash_len].lower().replace(' ', '').replace('\n', '')
        h = hashlib.md5(norm.encode('utf-8', errors='ignore')).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(text)
    return unique


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
        convs = row.get("conversations", row.get("messages", row.get("conversation", [])))
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
        messages = row.get("messages", row.get("conversations", row.get("conversation", [])))
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
    """Скачивает один датасет с фильтрацией качества."""
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
        # Скачиваем с запасом (+30%) чтобы после фильтрации осталось count
        fetch_count = int(count * 1.3)
        split_str = f"train[:{fetch_count}]"
        try:
            if subsets:
                ds = load_dataset(name, subsets[0], split=split_str, streaming=False)
            else:
                ds = load_dataset(name, split=split_str, streaming=False)
        except Exception:
            if subsets:
                ds = load_dataset(name, subsets[0], split="train", streaming=False)
            else:
                ds = load_dataset(name, split="train", streaming=False)
            if len(ds) > fetch_count:
                ds = ds.select(range(fetch_count))
        
        # ═══ Quality Pipeline ═══
        raw_count = len(ds)
        
        # 1. Format
        texts = []
        for row in ds:
            text = format_row(row, fmt)
            if text:
                texts.append(text)
        
        after_format = len(texts)
        
        # 2. Clean
        texts = [clean_text(t) for t in texts]
        
        # 3. Quality filter
        texts = [t for t in texts if is_quality_text(t)]
        after_quality = len(texts)
        
        # 4. Deduplicate
        texts = deduplicate_texts(texts)
        after_dedup = len(texts)
        
        # 5. Trim to count
        if len(texts) > count:
            texts = texts[:count]
        
        corpus = "\n\n".join(texts)
        
        # Сохранение
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(corpus)
        
        size_mb = len(corpus.encode('utf-8')) / (1024 * 1024)
        dropped = raw_count - len(texts)
        drop_pct = (dropped / max(raw_count, 1)) * 100
        print(f"  ✓ {name}: {len(texts)} примеров ({size_mb:.1f} MB) "
              f"[quality: -{dropped} ({drop_pct:.0f}%) | dedup: {after_quality - after_dedup}]")
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
