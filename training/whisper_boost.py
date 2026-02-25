"""
═══════════════════════════════════════════════════════════════
  Whisper Vocabulary Boost — Контекстная настройка STT
═══════════════════════════════════════════════════════════════

Анализирует обучающий корпус и извлекает:
  1. Топ-1000 ключевых слов → hotwords для Whisper
  2. Частотный словарь → initial_prompt для лучшего контекста
  3. Доменные термины → спец. лексика ТАРС

Whisper использует эти данные для улучшения распознавания
специфичных слов (технические термины, имена, команды).

Использование:
  python training/whisper_boost.py
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("Whisper.Boost")

# Стоп-слова (русские), которые не несут информации
STOP_WORDS = {
    "и", "в", "на", "с", "по", "для", "от", "из", "что", "как",
    "это", "не", "но", "он", "она", "они", "его", "её", "их",
    "был", "была", "было", "были", "быть", "есть", "будет",
    "к", "до", "за", "при", "об", "о", "у", "а", "же",
    "то", "ли", "бы", "вот", "ещё", "уже", "все", "весь",
    "так", "тоже", "только", "тот", "та", "те", "этот", "эта",
    "нет", "да", "может", "можно", "нужно", "надо", "там",
    "тут", "очень", "более", "когда", "где", "если", "чем",
    "или", "между", "через", "после", "перед", "над", "под",
    "также", "который", "которая", "которое", "которые",
    "один", "два", "три", "четыре", "пять", "первый", "второй",
    "другой", "каждый", "свой", "этих", "всех", "однако",
    "запрос", "ответ", "вопрос", "контекст", "система",
}


def extract_keywords(text: str, top_k: int = 1000) -> list:
    """Извлекает топ-K значимых слов из текста."""
    # Очистка и токенизация
    words = re.findall(r'[а-яА-ЯёЁa-zA-Z]{3,}', text.lower())
    
    # Фильтрация стоп-слов
    meaningful = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    # Частотный анализ
    freq = Counter(meaningful)
    
    # Топ-K по частоте
    return [word for word, count in freq.most_common(top_k)]


def extract_domain_terms(text: str, min_freq: int = 5) -> list:
    """Извлекает доменные термины (технические слова)."""
    # Ищем слова с заглавной буквы (имена, термины)
    proper_nouns = re.findall(r'\b([А-ЯЁA-Z][а-яёa-z]{2,})\b', text)
    freq = Counter(proper_nouns)
    
    # Ищем технические термины (латиница, аббревиатуры)
    tech_terms = re.findall(r'\b([A-Z]{2,}[a-z]*|[A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
    tech_freq = Counter(tech_terms)
    
    domain = []
    for word, count in freq.most_common(200):
        if count >= min_freq:
            domain.append(word)
    for word, count in tech_freq.most_common(100):
        if count >= min_freq:
            domain.append(word)
    
    return domain


def build_initial_prompt(keywords: list, max_words: int = 50) -> str:
    """Создаёт initial_prompt для Whisper из ключевых слов."""
    # Берём самые частотные слова как контекст
    prompt_words = keywords[:max_words]
    return " ".join(prompt_words)


def create_whisper_config(data_dir: str = None, output_path: str = None):
    """
    Создаёт конфигурацию для Whisper из обучающих данных.
    
    Выходной файл models/voice/whisper_context.json содержит:
    - hotwords: список ключевых слов для бустинга
    - initial_prompt: контекстная фраза
    - domain_terms: технические термины
    """
    if data_dir is None:
        data_dir = str(ROOT / "data")
    if output_path is None:
        output_path = str(ROOT / "models" / "voice" / "whisper_context.json")
    
    logger.info("═" * 60)
    logger.info("  Whisper Vocabulary Boost")
    logger.info("═" * 60)
    
    # Собираем весь текст
    all_text = ""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logger.error(f"❌ Директория не найдена: {data_dir}")
        return False
    
    txt_files = sorted(data_path.glob("*.txt"))
    for f in txt_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                text = fh.read()
            all_text += " " + text
            logger.info(f"  📄 {f.name}: {len(text):,} символов")
        except Exception:
            pass
    
    if not all_text:
        logger.warning("⚠ Нет данных для анализа")
        return False
    
    total_mb = len(all_text.encode('utf-8')) / (1024 * 1024)
    logger.info(f"\n  Итого текста: {total_mb:.1f} MB")
    
    # Извлечение
    logger.info("\n  Извлечение ключевых слов...")
    keywords = extract_keywords(all_text, top_k=1000)
    logger.info(f"  ✓ {len(keywords)} ключевых слов")
    
    logger.info("  Извлечение доменных терминов...")
    domain_terms = extract_domain_terms(all_text)
    logger.info(f"  ✓ {len(domain_terms)} доменных терминов")
    
    initial_prompt = build_initial_prompt(keywords, max_words=50)
    logger.info(f"  ✓ Initial prompt: {len(initial_prompt)} символов")
    
    # TARS-специфичные команды
    tars_commands = [
        "ТАРС", "Тарс", "тарс",
        "привет", "помоги", "расскажи", "объясни", "найди",
        "открой", "закрой", "запусти", "останови", "выключи",
        "покажи", "скажи", "напомни", "запомни", "забудь",
        "переведи", "напиши", "прочитай", "посчитай",
        "какая погода", "который час", "что нового",
    ]
    
    # Объединяем hotwords
    all_hotwords = list(set(keywords[:200] + domain_terms + tars_commands))
    
    # Сохранение
    config = {
        "hotwords": all_hotwords,
        "initial_prompt": initial_prompt,
        "domain_terms": domain_terms[:100],
        "tars_commands": tars_commands,
        "keywords_count": len(keywords),
        "corpus_size_mb": round(total_mb, 1),
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n  ✅ Whisper контекст сохранён: {output_path}")
    logger.info(f"     {len(all_hotwords)} hotwords, prompt: {len(initial_prompt)} символов")
    logger.info(f"     Whisper теперь лучше понимает контекст ТАРС")
    
    return True


if __name__ == "__main__":
    create_whisper_config()
