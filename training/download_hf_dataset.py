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
    # ─── Код: C++, Rust, Python, JS, коммиты, фидбек ───
    "code": [
        {
            "name": "ise-uiuc/Magicoder-Evol-Instruct-110K",
            "desc": "110K задач по коду (C++, Rust, Python, JS)",
            "count": 15000,
            "format": "instruct",
        },
        {
            "name": "sahil2801/CodeAlpaca-20k",
            "desc": "20K instruction-tuning для программирования",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "codeparrot/self-instruct-starcoder",
            "desc": "StarCoder self-instruct — генерация кода",
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
            "desc": "Коммиты как инструкции (высокое качество)",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "m-a-p/Code-Feedback",
            "desc": "70K разговоров о коде с фидбеком",
            "count": 5000,
            "format": "sharegpt",
        },
        {
            "name": "m-a-p/CodeFeedback-Filtered-Instruction",
            "desc": "157K отфильтрованных инструкций по коду",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "theblackcat102/evol-codealpaca-v1",
            "desc": "110K эволюционных задач по коду",
            "count": 0,
            "format": "instruct",
        },
    ],
    
    # ─── Математика и логика ───
    "math": [
        {
            "name": "d0rj/MathInstruct-ru",
            "desc": "Математические задачи с решениями (русский)",
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
            "name": "d0rj/ROMB-1.0",
            "desc": "Олимпиадные задачи и экзамены (русский)",
            "count": 3000,
            "format": "instruct",
        },
        {
            "name": "TIGER-Lab/MathInstruct",
            "desc": "Математические задачи с CoT решениями",
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
        {
            "name": "d0rj/R1-Distill-SFT_v1-ru",
            "desc": "DeepSeek R1 reasoning traces — русский перевод",
            "count": 5000,
            "format": "instruct",
        },
    ],
    
    # ─── МЫШЛЕНИЕ: Chain-of-Thought, Reasoning Traces ───
    "thinking": [
        {
            "name": "open-thoughts/OpenThoughts-114k",
            "desc": "114K задач с длинными цепочками рассуждений",
            "count": 10000,
            "format": "sharegpt",
        },
        {
            "name": "ServiceNow-AI/R1-Distill-SFT",
            "desc": "R1 distill: reasoning traces для math + code",
            "count": 10000,
            "format": "sharegpt",
            "subsets": ["v0"],
        },
        # camel-ai/* удалены — слишком медленная индексация (7+ часов)
    ],
    
    # ─── Сложные эволюционные инструкции ───
    "evolinstruct": [
        {
            "name": "WizardLMTeam/WizardLM_evol_instruct_V2_196k",
            "desc": "WizardLM: 196K эволюционных инструкций",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "WizardLMTeam/WizardLM_evol_instruct_70k",
            "desc": "WizardLM: 70K базовых эволюционных инструкций",
            "count": 0,
            "format": "instruct",
        },
        {
            "name": "cognitivecomputations/dolphin",
            "desc": "Dolphin: 1M+ uncensored инструкций",
            "count": 10000,
            "format": "sharegpt",
        },
        {
            "name": "lmsys/lmsys-chat-1m",
            "desc": "1M реальных диалогов с LLM (Chatbot Arena)",
            "count": 10000,
            "format": "chat",
        },
    ],
    
    # ─── Харизма, юмор, диалоги, личность, ролеплей ───
    "chat": [
        {
            "name": "Vikhrmodels/ru_turbo_saiga",
            "desc": "50K+ русских инструкций (Saiga)",
            "count": 10000,
            "format": "chat",
        },
        {
            "name": "Den4ikAI/russian_instructions_2",
            "desc": "Русские инструкции для ИИ",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "IlyaGusev/pikabu",
            "desc": "Pikabu — юмор, сарказм, живой язык",
            "count": 20000,
            "format": "text",
        },
        {
            "name": "IlyaGusev/ru_turbo_alpaca",
            "desc": "GPT-4 русские инструкции (высшее качество)",
            "count": 10000,
            "format": "instruct",
        },
        {
            "name": "IlyaGusev/saiga_scored",
            "desc": "Оцененные русские диалоги (по качеству)",
            "count": 10000,
            "format": "chat",
        },
        {
            "name": "inkoziev/Conversations",
            "desc": "9M русских диалогов (форумы, книги, шутки)",
            "count": 20000,
            "format": "text",
        },
        {
            "name": "IlyaGusev/gpt_roleplay_realm",
            "desc": "GPT-4 ролеплей — русские персонажи",
            "count": 5000,
            "format": "chat",
        },
        {
            "name": "d0rj/dialogsum-ru",
            "desc": "Русские диалоги + суммаризация",
            "count": 5000,
            "format": "text",
        },
        {
            "name": "IlyaGusev/ru_sharegpt_cleaned",
            "desc": "Очищенные русские ShareGPT диалоги",
            "count": 5000,
            "format": "sharegpt",
        },
    ],
    
    # ─── Самосознание, рефлексия, personality ───
    "selfaware": [
        {
            "name": "nazlicanto/persona-based-chat",
            "desc": "Диалоги с устойчивой персоной/характером",
            "count": 5000,
            "format": "chat",
        },
        {
            "name": "Anthropic/hh-rlhf",
            "desc": "Helpful/Harmless: самосознание ИИ",
            "count": 10000,
            "format": "chat",
        },
        {
            "name": "HuggingFaceH4/ultrachat_200k",
            "desc": "200K глубоких multi-turn диалогов",
            "count": 10000,
            "format": "sharegpt",
        },
        {
            "name": "stingning/ultrachat",
            "desc": "1.5M диалогов — вопросы о мире, себе, идеях",
            "count": 5000,
            "format": "chat",
        },
    ],
    
    # ─── SQL и работа с базами данных ───
    "sql": [
        {
            "name": "gretelai/synthetic_text_to_sql",
            "desc": "100K+ синтетических text-to-SQL примеров",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "Clinton/Text-to-sql-v1",
            "desc": "Text-to-SQL: вопрос → SQL запрос",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "b-mc2/sql-create-context",
            "desc": "SQL с контекстом создания таблиц",
            "count": 5000,
            "format": "instruct",
        },
    ],
    
    # ─── Креативное письмо, истории, стиль ───
    "creative": [
        {
            "name": "Dampfinchen/Creative_Writing_Multiturn",
            "desc": "Multi-turn креативное письмо",
            "count": 5000,
            "format": "sharegpt",
        },
        {
            "name": "d0rj/librusec",
            "desc": "Русская литература — богатый стиль и язык",
            "count": 10000,
            "format": "text",
        },
        {
            "name": "IlyaGusev/gazeta",
            "desc": "Газета.ру — журналистский стиль",
            "count": 5000,
            "format": "text",
        },
    ],
    
    # ─── Суммаризация и понимание текста ───
    "summarize": [
        {
            "name": "d0rj/curation-corpus-ru",
            "desc": "Русский корпус суммаризации новостей",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "IlyaGusev/gazeta",
            "desc": "Газета.ру — статьи + заголовки как суммари",
            "count": 5000,
            "format": "text",
        },
    ],
    
    # ─── Мировые знания, факты, QA ───
    "trivia": [
        {
            "name": "mandarjoshi/trivia_qa",
            "desc": "95K вопросов с evidence-based ответами",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "ai-forever/school_notebooks_QA",
            "desc": "Русские школьные вопросы-ответы",
            "count": 5000,
            "format": "instruct",
        },
        {
            "name": "NousResearch/nous-hermes-llama2-13b",
            "desc": "Nous: качественные QA (GPT-4 уровень)",
            "count": 0,
            "format": "instruct",
        },
        {
            "name": "allenai/ai2_arc",
            "desc": "ARC: научные вопросы (школьный экзамен)",
            "count": 3000,
            "format": "instruct",
        },
    ],
    
    # ─── Агенты, function calling, интерфейс ───
    "agent": [
        {
            "name": "glaiveai/glaive-function-calling-v2",
            "desc": "Вызов функций и API",
            "count": 5000,
            "format": "chat",
        },
        {
            "name": "NousResearch/hermes-function-calling-v1",
            "desc": "Hermes: продвинутый function calling (JSON)",
            "count": 5000,
            "format": "sharegpt",
        },
        {
            "name": "Salesforce/xlam-function-calling-60k",
            "desc": "xLAM: 60K примеров вызова API",
            "count": 5000,
            "format": "chat",
        },
    ],
    
    # ─── Instruction tuning (русский, основа) ───
    "instruct": [
        {
            "name": "d0rj/OpenOrca-ru",
            "desc": "4.2M русских инструкций (перевод OpenOrca)",
            "count": 50000,
            "format": "instruct",
        },
        {
            "name": "d0rj/OpenHermes-2.5-ru",
            "desc": "1M русских инструкций GPT-4 качества",
            "count": 30000,
            "format": "sharegpt",
        },
        {
            "name": "d0rj/ru-instruct",
            "desc": "754K русских инструкций (сборник)",
            "count": 20000,
            "format": "instruct",
        },
    ],
    
    # ─── Русский reasoning ───
    "russian": [
        {
            "name": "IlyaGusev/ru_turbo_alpaca_evol_instruct",
            "desc": "Эволюционные цепочки (сложный reasoning)",
            "count": 15000,
            "format": "instruct",
        },
        {
            "name": "Vikhrmodels/GrandMaster-PRO-MAX",
            "desc": "Сложный русский reasoning + многошаговые задачи",
            "count": 15000,
            "format": "chat",
        },
        {
            "name": "ai-forever/school_notebooks_QA",
            "desc": "Школьные вопросы-ответы (образование)",
            "count": 10000,
            "format": "instruct",
        },
    ],
    
    # ─── Наука и рассуждения ───
    "science": [
        {
            "name": "OpenAssistant/oasst1",
            "desc": "Мультиязычные диалоги (вкл. русский)",
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
            "name": "Open-Orca/OpenOrca",
            "desc": "Разнообразные инструкции (GPT-4)",
            "count": 20000,
            "format": "chat",
        },
        {
            "name": "teknium/OpenHermes-2.5",
            "desc": "1M+ инструкций (GPT-4, Code, Math)",
            "count": 20000,
            "format": "sharegpt",
        },
        {
            "name": "BAAI/Infinity-Instruct",
            "desc": "Огромный инструкционный датасет",
            "count": 10000,
            "format": "sharegpt",
        },
    ],
    
    # ═══════════════════════════════════════════════════════
    #  MASSIVE DATASETS (для 1B модели, 50-80 GB)
    # ═══════════════════════════════════════════════════════
    
    "massive": [
        {
            "name": "cc100",
            "desc": "CC-100 Russian — 46 GB веб-текста",
            "count": 5_000_000,
            "format": "text",
            "subsets": ["ru"],
            "streaming": True,
            "shard_size_mb": 1024,
        },
        {
            "name": "uonlp/CulturaX",
            "desc": "CulturaX Russian — cleaned mC4 + OSCAR",
            "count": 3_000_000,
            "format": "text",
            "subsets": ["ru"],
            "streaming": True,
            "shard_size_mb": 1024,
        },
        {
            "name": "oscar-corpus/OSCAR-2301",
            "desc": "OSCAR Russian — CommonCrawl",
            "count": 2_000_000,
            "format": "text",
            "subsets": ["ru"],
            "streaming": True,
            "shard_size_mb": 1024,
        },
    ],
    
    "quality": [
        {
            "name": "wikimedia/wikipedia",
            "desc": "Русская Wikipedia (~3 GB)",
            "count": 50_000,
            "format": "text",
            "subsets": ["20231101.ru"],
        },
        {
            "name": "IlyaGusev/gazeta",
            "desc": "Газета.ру — новости (~500 MB)",
            "count": 5_000,
            "format": "text",
        },
        {
            "name": "d0rj/librusec",
            "desc": "Русская литература — богатый стиль",
            "count": 10_000,
            "format": "text",
        },
        {
            "name": "IlyaGusev/ru_turbo_alpaca",
            "desc": "GPT-4 русские инструкции",
            "count": 10_000,
            "format": "instruct",
        },
        {
            "name": "IlyaGusev/ru_turbo_alpaca_evol_instruct",
            "desc": "Эволюционные цепочки reasoning",
            "count": 10_000,
            "format": "instruct",
        },
        {
            "name": "Vikhrmodels/GrandMaster-PRO-MAX",
            "desc": "Сложный русский reasoning",
            "count": 10_000,
            "format": "chat",
        },
    ],
    
    "reasoning": [
        {
            "name": "OpenAssistant/oasst2",
            "desc": "OpenAssistant v2",
            "count": 10_000,
            "format": "chat",
        },
        {
            "name": "TIGER-Lab/MathInstruct",
            "desc": "Математические задачи (CoT)",
            "count": 5_000,
            "format": "instruct",
        },
        {
            "name": "Open-Orca/OpenOrca",
            "desc": "Инструкции GPT-4 качества",
            "count": 20_000,
            "format": "chat",
        },
        {
            "name": "teknium/OpenHermes-2.5",
            "desc": "1M+ инструкций (GPT-4, Code, Math)",
            "count": 20_000,
            "format": "sharegpt",
        },
        {
            "name": "BAAI/Infinity-Instruct",
            "desc": "Огромный мультиязычный датасет",
            "count": 10_000,
            "format": "sharegpt",
        },
    ],
    
    "dpo": [
        {
            "name": "Anthropic/hh-rlhf",
            "desc": "Human preference (chosen/rejected)",
            "count": 10_000,
            "format": "chat",
        },
        {
            "name": "Intel/orca_dpo_pairs",
            "desc": "DPO пары (chosen vs rejected)",
            "count": 5_000,
            "format": "instruct",
        },
        {
            "name": "d0rj/full-hh-rlhf-ru",
            "desc": "Human preference на русском",
            "count": 10_000,
            "format": "chat",
        },
    ],
    
    "max": [],  # = massive + quality + reasoning + dpo + all
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
    """Скачивает один датасет и возвращает текст (или путь к шардам)."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ pip install datasets")
        return ""
    
    name = ds_config["name"]
    count = ds_config.get("count", 10000)
    fmt = ds_config.get("format", "instruct")
    safe_name = name.replace("/", "_")
    is_streaming = ds_config.get("streaming", False)
    
    if is_streaming:
        return download_streaming(ds_config, output_dir)
    
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
        subsets = ds_config.get("subsets", [])
        try:
            if subsets:
                ds = load_dataset(name, subsets[0], split=split_str, streaming=False)
            else:
                ds = load_dataset(name, split=split_str, streaming=False)
        except Exception:
            # Некоторые датасеты не поддерживают slicing — грузим всё
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


def download_streaming(ds_config: dict, output_dir: str) -> str:
    """Скачивает большой датасет в режиме streaming (пишет на диск, не в RAM).
    
    Для датасетов 10+ GB: CC-100, CulturaX, OSCAR.
    Пишет в шарды по shard_size_mb для streaming DataLoader.
    """
    from datasets import load_dataset
    import time as _time
    
    name = ds_config["name"]
    count = ds_config.get("count", 5_000_000)
    fmt = ds_config.get("format", "text")
    subsets = ds_config.get("subsets", [])
    safe_name = name.replace("/", "_")
    shard_mb = ds_config.get("shard_size_mb", 1024)
    
    shard_dir = os.path.join(output_dir, f"shards_{safe_name}")
    marker = os.path.join(shard_dir, "_COMPLETE")
    
    # Проверяем кеш
    if os.path.exists(marker):
        # Считаем размер
        total = sum(
            os.path.getsize(os.path.join(shard_dir, f))
            for f in os.listdir(shard_dir) if f.endswith('.txt')
        )
        total_gb = total / (1024**3)
        n_shards = len([f for f in os.listdir(shard_dir) if f.endswith('.txt')])
        print(f"  ✓ {name}: уже скачан ({n_shards} шардов, {total_gb:.1f} GB)")
        return f"SHARDS:{shard_dir}"
    
    os.makedirs(shard_dir, exist_ok=True)
    
    print(f"  ↓↓ {name}: STREAMING режим ({count:,} примеров)...")
    print(f"     Шарды: {shard_dir}")
    
    try:
        if subsets:
            ds = load_dataset(name, subsets[0], split="train", streaming=True)
        else:
            ds = load_dataset(name, split="train", streaming=True)
        
        shard_idx = 0
        shard_texts = []
        shard_bytes = 0
        total_items = 0
        total_bytes = 0
        t0 = _time.time()
        
        for i, row in enumerate(ds):
            if i >= count:
                break
            
            text = format_row(row, fmt)
            if not text or len(text) < 30:
                continue
            
            text_bytes = len(text.encode('utf-8'))
            shard_texts.append(text)
            shard_bytes += text_bytes
            total_items += 1
            total_bytes += text_bytes
            
            # Записать шард если набрали shard_mb
            if shard_bytes >= shard_mb * 1024 * 1024:
                shard_file = os.path.join(shard_dir, f"shard_{shard_idx:04d}.txt")
                with open(shard_file, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(shard_texts))
                
                shard_idx += 1
                elapsed = _time.time() - t0
                speed = total_items / elapsed if elapsed > 0 else 0
                gb = total_bytes / (1024**3)
                print(f"     Шард {shard_idx}: {len(shard_texts):,} примеров, "
                      f"{shard_bytes/1024/1024:.0f} MB | "
                      f"Всего: {total_items:,} ({gb:.2f} GB) | "
                      f"{speed:.0f} ex/s")
                
                shard_texts = []
                shard_bytes = 0
            
            # Прогресс каждые 100K
            if total_items % 100_000 == 0 and total_items > 0:
                elapsed = _time.time() - t0
                gb = total_bytes / (1024**3)
                pct = 100 * i / count
                print(f"     ... {total_items:,} ({pct:.0f}%, {gb:.2f} GB, "
                      f"{elapsed:.0f}s)")
        
        # Последний шард
        if shard_texts:
            shard_file = os.path.join(shard_dir, f"shard_{shard_idx:04d}.txt")
            with open(shard_file, 'w', encoding='utf-8') as f:
                f.write("\n\n".join(shard_texts))
            shard_idx += 1
        
        # Маркер завершения
        with open(marker, 'w') as f:
            f.write(f"{total_items} items, {total_bytes} bytes, {shard_idx} shards\n")
        
        total_gb = total_bytes / (1024**3)
        elapsed = _time.time() - t0
        print(f"  ✓ {name}: {total_items:,} → {shard_idx} шардов, "
              f"{total_gb:.2f} GB за {elapsed:.0f}s")
        
        return f"SHARDS:{shard_dir}"
        
    except Exception as e:
        print(f"  ⚠ {name}: streaming ошибка — {e}")
        import traceback
        traceback.print_exc()
        return ""


def download_preset(preset: str, output_dir: str = None, count_override: int = None) -> list:
    """Скачивает все датасеты из пресета."""
    if output_dir is None:
        output_dir = str(ROOT / "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 'max' = ВСЕ датасеты (massive + quality + reasoning + dpo + existing)
    if preset in ("all", "max"):
        datasets = []
        include_massive = (preset == "max")
        for pname, pdata in PRESETS.items():
            if pname == "max":
                continue
            if pname == "massive" and not include_massive:
                continue  # 'all' пропускает streaming-датасеты
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
