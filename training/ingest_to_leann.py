"""
═══════════════════════════════════════════════════════════════
  LEANN Ingestion — Загрузка корпуса в векторную память ТАРС
═══════════════════════════════════════════════════════════════

Разбивает тексты (Wikipedia, датасеты) на чанки и загружает
в LEANN индекс для RAG (Retrieval-Augmented Generation).

ТАРС сможет вспоминать знания из памяти при ответах.

Использование:
  python training/ingest_to_leann.py                      # Всё из data/
  python training/ingest_to_leann.py --file data/wiki_ru.txt
  python training/ingest_to_leann.py --chunk-size 500
"""

import os
import sys
import re
import logging
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("LEANN.Ingest")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Разбивает текст на чанки фиксированного размера с перекрытием.
    
    Перекрытие нужно, чтобы LEANN не терял контекст на границах предложений.
    
    Args:
        text: Исходный текст
        chunk_size: Размер чанка в символах
        overlap: Перекрытие между чанками
        
    Returns:
        Список текстовых чанков
    """
    # Разбиваем по абзацам
    paragraphs = re.split(r'\n{2,}', text)
    
    chunks = []
    current = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 30:
            continue
            
        if len(current) + len(para) < chunk_size:
            current += " " + para if current else para
        else:
            if current:
                chunks.append(current.strip())
            # Если параграф сам больше chunk_size — режем
            if len(para) > chunk_size:
                words = para.split()
                sub = ""
                for w in words:
                    if len(sub) + len(w) + 1 > chunk_size:
                        if sub:
                            chunks.append(sub.strip())
                        sub = w
                    else:
                        sub += " " + w if sub else w
                current = sub
            else:
                current = para
    
    if current and len(current) > 30:
        chunks.append(current.strip())
    
    return chunks


def batch_embeddings(texts: list, model, batch_size: int = 64) -> list:
    """
    Батчевое создание эмбеддингов (быстрее чем по одному).
    
    Args:
        texts: Список текстов
        model: SentenceTransformer модель
        batch_size: Размер батча
        
    Returns:
        Список numpy массивов (эмбеддинги)
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, batch_size=batch_size)
        all_embeddings.extend(embs)
    return all_embeddings


def fallback_embedding(text: str) -> np.ndarray:
    """TF-IDF fallback если нет SentenceTransformer."""
    words = text.lower().split()
    vec = np.zeros(384)
    for w in words:
        idx = hash(w) % 384
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def ingest_file(file_path: str, index_path: str = None,
                chunk_size: int = 500, max_chunks: int = None):
    """
    Загружает текстовый файл в LEANN индекс.
    
    Args:
        file_path: Путь к текстовому файлу
        index_path: Путь к LEANN индексу (default: memory/leann.index)
        chunk_size: Размер чанка
        max_chunks: Максимум чанков (None = все)
    """
    if index_path is None:
        index_path = str(ROOT / "memory" / "leann.index")
    
    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не найден: {file_path}")
        return
    
    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    file_mb = len(text.encode('utf-8')) / (1024 * 1024)
    logger.info(f"[LEANN] Файл: {file_path} ({file_mb:.1f} MB)")
    
    # Чанкуем
    chunks = chunk_text(text, chunk_size=chunk_size)
    if max_chunks:
        chunks = chunks[:max_chunks]
    logger.info(f"[LEANN] Чанков: {len(chunks)} (по ~{chunk_size} символов)")
    
    # Загружаем существующий индекс
    existing_texts = []
    existing_embeddings = []
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_texts = data.get("texts", [])
                existing_embeddings = [np.array(e) for e in data.get("embeddings", [])]
            logger.info(f"[LEANN] Существующий индекс: {len(existing_texts)} документов")
        except Exception:
            pass
    
    # Дедупликация: не добавляем уже существующие чанки
    existing_set = set(existing_texts)
    new_chunks = [c for c in chunks if c not in existing_set]
    
    if not new_chunks:
        logger.info("[LEANN] ✓ Все чанки уже в индексе. Пропускаю.")
        return
    
    logger.info(f"[LEANN] Новых чанков: {len(new_chunks)}")
    
    # Эмбеддинги
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model_path = str(ROOT / "models" / "embeddings")
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("[LEANN] Используется SentenceTransformer для эмбеддингов")
    except ImportError:
        logger.warning("[LEANN] ⚠ sentence-transformers не установлен, используется TF-IDF fallback")
        logger.warning("[LEANN]   pip install sentence-transformers")
    
    logger.info(f"[LEANN] Создаю эмбеддинги для {len(new_chunks)} чанков...")
    
    if model:
        new_embeddings = batch_embeddings(new_chunks, model, batch_size=128)
    else:
        new_embeddings = [fallback_embedding(c) for c in new_chunks]
    
    # Объединяем
    all_texts = existing_texts + new_chunks
    all_embeddings = existing_embeddings + new_embeddings
    
    # Сохраняем
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    data = {
        "texts": all_texts,
        "embeddings": [emb.tolist() for emb in all_embeddings],
    }
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    
    index_mb = os.path.getsize(index_path) / (1024 * 1024)
    logger.info(f"[LEANN] ✓ Индекс сохранён: {len(all_texts)} документов, {index_mb:.1f} MB")
    logger.info(f"[LEANN]   Путь: {index_path}")


def ingest_all(data_dir: str = None, index_path: str = None,
               chunk_size: int = 500):
    """
    Автоматически загружает ВСЕ текстовые файлы из data/ в LEANN.
    """
    if data_dir is None:
        data_dir = str(ROOT / "data")
    if index_path is None:
        index_path = str(ROOT / "memory" / "leann.index")
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ Директория не найдена: {data_dir}")
        return
    
    logger.info("═" * 60)
    logger.info("  LEANN — Загрузка ВСЕХ данных в векторную память")
    logger.info("═" * 60)
    print()
    
    txt_files = sorted(Path(data_dir).glob("*.txt"))
    
    if not txt_files:
        logger.info("⚠ Нет .txt файлов в data/. Сначала скачайте данные:")
        logger.info("  python training/download_all.py")
        return
    
    logger.info(f"Найдено {len(txt_files)} файлов для загрузки:")
    for f in txt_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  • {f.name} ({size_mb:.1f} MB)")
    logger.info()
    
    for f in txt_files:
        ingest_file(str(f), index_path=index_path, chunk_size=chunk_size)
        logger.info()
    
    # Проверяем итоговый размер
    if os.path.exists(index_path):
        index_mb = os.path.getsize(index_path) / (1024 * 1024)
        with open(index_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total_docs = len(data.get("texts", []))
        logger.info("═" * 60)
        logger.info(f"  ✅ LEANN готов: {total_docs} документов, {index_mb:.1f} MB")
        logger.info(f"  ТАРС теперь помнит всё из data/")
        logger.info("═" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка данных в LEANN память ТАРС")
    parser.add_argument("--file", type=str, default=None,
                        help="Конкретный .txt файл для загрузки")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Размер чанка (default: 500 символов)")
    parser.add_argument("--index", type=str, default=None,
                        help="Путь к индексу LEANN (default: memory/leann.index)")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Максимум чанков из файла")
    args = parser.parse_args()
    
    if args.file:
        ingest_file(args.file, index_path=args.index,
                    chunk_size=args.chunk_size, max_chunks=args.max_chunks)
    else:
        ingest_all(index_path=args.index, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
