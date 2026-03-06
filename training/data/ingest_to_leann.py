"""
═══════════════════════════════════════════════════════════════
  LEANN Ingestion — Загрузка корпуса в векторную память ТАРС
═══════════════════════════════════════════════════════════════

Разбивает тексты (Wikipedia, датасеты, PDF, Word, Excel) на чанки
и загружает в LEANN индекс для RAG (Retrieval-Augmented Generation).

Поддерживаемые форматы:
  - .txt, .md, .rst — текстовые
  - .pdf — PDF документы (PyPDF2)
  - .docx — Word документы (python-docx)
  - .xlsx — Excel таблицы (openpyxl)
  - .csv — CSV таблицы
  - .json — JSON файлы

Использование:
  python training/ingest_to_leann.py                              # Всё из data/
  python training/ingest_to_leann.py --file data/spec.pdf         # PDF файл
  python training/ingest_to_leann.py --file data/report.docx      # Word файл
  python training/ingest_to_leann.py --chunk-size 500
"""

import os
import sys
import re
import gc
import logging
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LEANN.Ingest")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Разбивает текст на чанки фиксированного размера с перекрытием.
    """
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


def batch_embeddings(texts: list, model, batch_size: int = 256):
    """
    Батчевое создание эмбеддингов. Возвращает (int8 array, float32 scales).
    """
    all_int8 = []
    all_scales = []
    total = len(texts)
    
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, batch_size=batch_size)
        embs_f32 = embs.astype(np.float32)
        
        # Per-vector int8 квантизация
        scales = np.abs(embs_f32).max(axis=1) / 127.0
        scales = np.maximum(scales, 1e-8)
        embs_q = np.clip(np.round(embs_f32 / scales[:, np.newaxis]), -128, 127).astype(np.int8)
        
        all_int8.append(embs_q)
        all_scales.append(scales.astype(np.float32))
        
        if (i // batch_size) % 10 == 0 and i > 0:
            logger.info(f"[LEANN]   Прогресс: {i}/{total} ({100*i//total}%)")
    
    return np.vstack(all_int8), np.concatenate(all_scales)


def fallback_embedding(text: str):
    """TF-IDF fallback. Returns (int8_vec, scale)."""
    words = text.lower().split()
    vec = np.zeros(384, dtype=np.float32)
    for w in words:
        idx = hash(w) % 384
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    scale = np.abs(vec).max() / 127.0
    if scale < 1e-8:
        scale = 1e-8
    vec_q = np.clip(np.round(vec / scale), -128, 127).astype(np.int8)
    return vec_q, np.float32(scale)


def load_index(index_path: str):
    """
    Загрузка индекса. Поддерживает форматы:
    - int8 + scales (.npz)
    - float16 (.npz, старый)
    - JSON (обратная совместимость)
    
    Returns: (texts, embeddings_int8, scales)
    """
    npz_path = index_path.replace(".index", ".npz")
    texts_path = index_path.replace(".index", ".texts.json")
    
    if os.path.exists(npz_path) and os.path.exists(texts_path):
        try:
            data = np.load(npz_path)
            embeddings = data["embeddings"]
            with open(texts_path, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            
            if "scales" in data:
                scales = data["scales"]
            else:
                # Миграция float16 → int8
                embs_f32 = embeddings.astype(np.float32)
                scales = np.abs(embs_f32).max(axis=1) / 127.0
                scales = np.maximum(scales, 1e-8).astype(np.float32)
                embeddings = np.clip(np.round(embs_f32 / scales[:, np.newaxis]), -128, 127).astype(np.int8)
                save_index(index_path, texts, embeddings, scales)
                logger.info(f"[LEANN] Миграция float16 → int8")
            
            logger.info(f"[LEANN] Загружен индекс: {len(texts)} документов, {embeddings.nbytes / 1024 / 1024:.1f} MB int8")
            return texts, embeddings, scales
        except Exception as e:
            logger.warning(f"[LEANN] Ошибка чтения npz: {e}")
    
    if os.path.exists(index_path):
        try:
            file_size = os.path.getsize(index_path)
            if file_size < 100:
                return [], None, None
            logger.info(f"[LEANN] Чтение JSON индекса ({file_size / 1024 / 1024:.1f} MB)...")
            with open(index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            texts = data.get("texts", [])
            raw_embs = data.get("embeddings", [])
            if raw_embs:
                embs_f32 = np.array(raw_embs, dtype=np.float32)
                scales = np.abs(embs_f32).max(axis=1) / 127.0
                scales = np.maximum(scales, 1e-8).astype(np.float32)
                embeddings = np.clip(np.round(embs_f32 / scales[:, np.newaxis]), -128, 127).astype(np.int8)
                save_index(index_path, texts, embeddings, scales)
                os.remove(index_path)
                logger.info(f"[LEANN] Мигрировано JSON → int8")
                return texts, embeddings, scales
            return texts, None, None
        except Exception as e:
            logger.warning(f"[LEANN] Ошибка JSON: {e}")
    
    return [], None, None


def save_index(index_path: str, texts: list, embeddings: np.ndarray, scales: np.ndarray = None):
    """
    Сохранение индекса в int8 формате:
    - .npz: int8 эмбеддинги + float32 scales
    - .texts.json: тексты
    
    RAM: 1M docs × 384 dim × int8 = ~375 MB (vs 8.9 GB JSON)
    """
    npz_path = index_path.replace(".index", ".npz")
    texts_path = index_path.replace(".index", ".texts.json")
    
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    
    if scales is not None:
        np.savez_compressed(npz_path, embeddings=embeddings, scales=scales)
    else:
        np.savez_compressed(npz_path, embeddings=embeddings)
    
    with open(texts_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False)
    
    npz_mb = os.path.getsize(npz_path) / (1024 * 1024)
    texts_mb = os.path.getsize(texts_path) / (1024 * 1024)
    logger.info(f"[LEANN] ✓ Сохранено: {len(texts)} docs, "
                f"embeddings={npz_mb:.1f} MB (int8), texts={texts_mb:.1f} MB")


def ingest_file(file_path: str, index_path: str = None,
                chunk_size: int = 500, max_chunks: int = None,
                existing_texts: list = None, existing_embeddings=None,
                model=None):
    """
    Загружает текстовый файл в LEANN индекс.
    
    Принимает существующие тексты/эмбеддинги чтобы не перечитывать индекс
    для каждого файла.
    
    Returns: (texts, embeddings) — обновлённые
    """
    if index_path is None:
        # На Colab → сохраняем прямо на Drive
        drive_idx = ROOT / ".." / "drive" / "MyDrive" / "TarsMemory" / "leann.index"
        if drive_idx.parent.exists():
            index_path = str(drive_idx.resolve())
        else:
            index_path = str(ROOT / "memory" / "leann.index")
    
    if not os.path.exists(file_path):
        logger.error(f"❌ Файл не найден: {file_path}")
        return existing_texts or [], existing_embeddings
    
    # ═══ Определяем формат и читаем файл ═══
    ext = Path(file_path).suffix.lower()
    SUPPORTED_DOC_FORMATS = {'.pdf', '.docx', '.xlsx', '.csv', '.json', '.xml'}
    
    if ext in SUPPORTED_DOC_FORMATS:
        # Используем document_tools для структурированных форматов
        try:
            from tools.document_tools import read_document
            text = read_document(file_path, max_pages=100, max_rows=2000)
            logger.info(f"[LEANN] Прочитан {ext} через document_tools")
        except ImportError as e:
            logger.error(f"❌ Для {ext} нужно: pip install PyPDF2 python-docx openpyxl")
            return existing_texts or [], existing_embeddings, None
        except Exception as e:
            logger.error(f"❌ Ошибка чтения {ext}: {e}")
            return existing_texts or [], existing_embeddings, None
    else:
        # Текстовые форматы (.txt, .md, .rst, etc.)
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
    
    file_mb = len(text.encode('utf-8')) / (1024 * 1024)
    logger.info(f"[LEANN] Файл: {file_path} ({file_mb:.1f} MB, {ext})")
    
    # Чанкуем
    chunks = chunk_text(text, chunk_size=chunk_size)
    if max_chunks:
        chunks = chunks[:max_chunks]
    logger.info(f"[LEANN] Чанков: {len(chunks)} (по ~{chunk_size} символов)")
    
    # Освобождаем текст из RAM
    del text
    gc.collect()
    
    # Загружаем существующий индекс если не передан
    if existing_texts is None:
        existing_texts, existing_embeddings, existing_scales = load_index(index_path)
    else:
        existing_scales = None  # будет передано извне
    
    # Дедупликация
    existing_set = set(existing_texts)
    new_chunks = [c for c in chunks if c not in existing_set]
    del chunks
    gc.collect()
    
    if not new_chunks:
        logger.info("[LEANN] ✓ Все чанки уже в индексе. Пропуск.")
        return existing_texts, existing_embeddings, existing_scales
    
    logger.info(f"[LEANN] Новых чанков: {len(new_chunks)}")
    
    # Создаём эмбеддинги
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model_path = str(ROOT / "models" / "embeddings")
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                model = SentenceTransformer(model_path)
            else:
                model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("[LEANN] Используется SentenceTransformer")
        except ImportError:
            logger.warning("[LEANN] ⚠ TF-IDF fallback")
    
    logger.info(f"[LEANN] Создаю эмбеддинги для {len(new_chunks)} чанков (int8)...")
    
    if model:
        new_embs, new_scales = batch_embeddings(new_chunks, model, batch_size=256)
    else:
        results = [fallback_embedding(c) for c in new_chunks]
        new_embs = np.array([r[0] for r in results], dtype=np.int8)
        new_scales = np.array([r[1] for r in results], dtype=np.float32)
    
    # Объединяем
    all_texts = existing_texts + new_chunks
    if existing_embeddings is not None and len(existing_embeddings) > 0:
        all_embeddings = np.vstack([existing_embeddings, new_embs])
        all_scales = np.concatenate([existing_scales, new_scales]) if existing_scales is not None else new_scales
    else:
        all_embeddings = new_embs
        all_scales = new_scales
    
    del new_chunks, new_embs, new_scales, existing_texts, existing_embeddings
    gc.collect()
    
    return all_texts, all_embeddings, all_scales


def ingest_all(data_dir: str = None, index_path: str = None,
               chunk_size: int = 500):
    """
    Автоматически загружает ВСЕ текстовые файлы из data/ в LEANN.
    Оптимизировано: один проход, общий индекс.
    """
    if data_dir is None:
        data_dir = str(ROOT / "data")
    if index_path is None:
        # На Colab → сохраняем прямо на Drive
        drive_idx = ROOT / ".." / "drive" / "MyDrive" / "TarsMemory" / "leann.index"
        if drive_idx.parent.exists():
            index_path = str(drive_idx.resolve())
        else:
            index_path = str(ROOT / "memory" / "leann.index")
    
    if not os.path.exists(data_dir):
        logger.error(f"❌ Директория не найдена: {data_dir}")
        return
    
    logger.info("═" * 60)
    logger.info("  LEANN — Загрузка данных в векторную память")
    logger.info("═" * 60)
    
    # Ищем ВСЕ поддерживаемые форматы
    supported_exts = ['*.txt', '*.md', '*.rst', '*.pdf', '*.docx', '*.xlsx', '*.csv', '*.json']
    all_files = []
    for ext in supported_exts:
        all_files.extend(sorted(Path(data_dir).glob(ext)))
    # Dedup (в случае одного файла, подходящего под несколько паттернов)
    seen = set()
    txt_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            txt_files.append(f)
    
    if not txt_files:
        logger.info("⚠ Нет файлов поддерживаемых форматов в data/")
        return
    
    total_mb = sum(f.stat().st_size for f in txt_files) / (1024 * 1024)
    exts_found = set(f.suffix for f in txt_files)
    logger.info(f"Найдено {len(txt_files)} файлов ({total_mb:.0f} MB)")
    logger.info(f"Форматы: {', '.join(sorted(exts_found))}")
    
    # Загружаем существующий индекс ОДИН раз
    all_texts, all_embeddings, all_scales = load_index(index_path)
    
    # Загружаем модель ОДИН раз
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        model_path = str(ROOT / "models" / "embeddings")
        if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("[LEANN] Используется SentenceTransformer")
    except ImportError:
        logger.warning("[LEANN] ⚠ TF-IDF fallback")
    
    # Обрабатываем все файлы с общим индексом
    for i, f in enumerate(txt_files):
        logger.info(f"\n[{i+1}/{len(txt_files)}] {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        all_texts, all_embeddings, all_scales = ingest_file(
            str(f), index_path=index_path,
            chunk_size=chunk_size,
            existing_texts=all_texts,
            existing_embeddings=all_embeddings,
            model=model
        )
        
        # Промежуточное сохранение каждые 5 файлов
        if (i + 1) % 5 == 0 and all_embeddings is not None:
            save_index(index_path, all_texts, all_embeddings, all_scales)
            logger.info(f"[LEANN] Промежуточное сохранение ({len(all_texts)} документов)")
    
    # Финальное сохранение
    if all_embeddings is not None and len(all_texts) > 0:
        save_index(index_path, all_texts, all_embeddings, all_scales)
    
    # RAM отчёт
    if all_embeddings is not None:
        ram_mb = all_embeddings.nbytes / (1024 * 1024)
        logger.info(f"\n{'═' * 60}")
        logger.info(f"  ✅ LEANN готов: {len(all_texts)} документов")
        logger.info(f"  💾 RAM эмбеддингов: {ram_mb:.0f} MB (int8)")
        logger.info(f"  ТАРС теперь помнит всё из data/")
        logger.info(f"{'═' * 60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Загрузка данных в LEANN память ТАРС")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()
    
    if args.file:
        texts, embs, scales = ingest_file(args.file, index_path=args.index,
                    chunk_size=args.chunk_size, max_chunks=args.max_chunks)
        if embs is not None:
            save_index(args.index or str(ROOT / "memory" / "leann.index"), texts, embs, scales)
    else:
        ingest_all(index_path=args.index, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
