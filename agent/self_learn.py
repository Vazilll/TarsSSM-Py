"""
═══════════════════════════════════════════════════════════════
  ТАРС Self-Learning — Автономное самообучение
═══════════════════════════════════════════════════════════════

Цикл самообучения:
  1. Получить задачу/тему
  2. Поиск в интернете (DuckDuckGo)
  3. Загрузка и очистка контента
  4. Сохранение в LEANN (мгновенная память)
  5. Сохранение в корпус (для дообучения модели)
  6. Когда корпус достаточно большой → triggers retrain

Запуск:
  python self_learn.py "квантовые вычисления"
  python self_learn.py --auto         # автоматический режим
  python self_learn.py --telegram     # с Telegram ботом
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

_ROOT = Path(__file__).resolve().parent.parent  # agent/ → project root
sys.path.insert(0, str(_ROOT))

from memory.leann import LeannIndex
from tools.web_search import search_duckduckgo, fetch_page_text, WebResult

logger = logging.getLogger("Tars.SelfLearn")

# Пути
DATA_DIR = _ROOT / "data"
CORPUS_DIR = DATA_DIR / "self_learned"
RETRAIN_THRESHOLD_MB = 5  # Начать дообучение при 5 MB нового корпуса


class SelfLearner:
    """
    Автономная система самообучения ТАРС.
    
    Пример:
        learner = SelfLearner()
        await learner.learn_topic("нейронные сети Mamba")
        await learner.learn_from_urls(["https://arxiv.org/abs/..."])
    """
    
    def __init__(self, memory: Optional[LeannIndex] = None):
        if memory is None:
            memory = LeannIndex()
        self.memory = memory
        
        # Конфиг
        self.max_pages_per_topic = 5
        self.chunk_size = 500          # символов на чанк
        self.min_content_length = 100  # минимум для полезного контента
        
        # Content validation thresholds
        self._min_word_count = 20      # minimum words for quality content
        self._max_boilerplate_ratio = 0.5  # max ratio of boilerplate markers
        
        # ═══ Rate limiting ═══
        self._min_request_interval = 2.0  # seconds between HTTP requests
        self._last_request_time = 0.0
        
        # ═══ Domain blocklist (security) ═══
        self._blocked_domains = {
            'bit.ly', 'tinyurl.com', 'goo.gl',  # URL shorteners (hide real dest)
            'pastebin.com', 'paste.ee',          # potential malicious content
            'file-upload.com', 'uploadfiles.io',  # download sites
            'localhost', '127.0.0.1', '0.0.0.0', # local network
        }
        
        # Корпусный файл для дообучения
        self.corpus_file = CORPUS_DIR / "auto_learned.txt"
        self._seen_urls: set = set()  # ═══ URL dedup across calls ═══
        
        # Статистика
        self.pages_fetched = 0
        self.docs_added = 0
        self.bytes_collected = 0
        
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"SelfLearner initialized (corpus: {self.corpus_file})")
    
    def _rate_limit(self):
        """Enforce minimum interval between HTTP requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _is_url_safe(self, url: str) -> bool:
        """Check URL against blocklist and scheme validation."""
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return False
            domain = parsed.hostname or ''
            # Check exact match and parent domain
            for blocked in self._blocked_domains:
                if domain == blocked or domain.endswith('.' + blocked):
                    logger.warning(f"SelfLearner: blocked domain {domain}")
                    return False
            return True
        except Exception:
            return False
    
    # ═══════ Основные методы ═══════
    
    async def learn_topic(self, topic: str, depth: int = 1) -> dict:
        """
        Изучить тему из интернета.
        
        Args:
            topic: Тема для изучения
            depth: 1 = только поиск, 2 = + связжанные темы
        
        Returns:
            Статистика обучения
        """
        print(f"\n{'═'*60}")
        print(f"  🧠 ТАРС Self-Learning: {topic}")
        print(f"{'═'*60}\n")
        
        stats = {"topic": topic, "pages": 0, "chunks": 0, "bytes": 0}
        
        # Шаг 1: Поиск
        print(f"  🔍 Поиск: {topic}")
        results = await search_duckduckgo(topic, max_results=self.max_pages_per_topic)
        
        if not results:
            print(f"  ❌ Ничего не найдено")
            return stats
        
        print(f"  📄 Найдено: {len(results)} результатов")
        
        # Шаг 2: Загрузка и обработка каждой страницы
        for i, result in enumerate(results, 1):
            print(f"\n  [{i}/{len(results)}] {result.title[:60]}")
            print(f"       {result.url[:80]}")
            
            # ═══ URL safety: check scheme + domain blocklist ═══
            if not self._is_url_safe(result.url):
                print(f"       ⛔ Blocked domain, skip")
                continue
            
            # ═══ URL dedup: skip already fetched ═══
            if result.url in self._seen_urls:
                print(f"       ⏭ Already fetched, skip")
                continue
            self._seen_urls.add(result.url)
            
            content = await fetch_page_text(result.url, max_chars=5000)
            self._rate_limit()  # enforce minimum interval
            
            if not content or len(content) < self.min_content_length:
                print(f"       ⚠ Мало контента ({len(content) if content else 0} символов), пропуск")
                continue
            
            # Content quality validation
            quality_ok, reason = self._validate_content(content)
            if not quality_ok:
                print(f"       ⚠ Низкое качество: {reason}, пропуск")
                continue
            
            # Разбить на чанки
            chunks = self._make_chunks(content, result.title, result.url)
            
            # Добавить в LEANN
            for chunk in chunks:
                self.memory.add_document(chunk)
                self.docs_added += 1
            
            # Сохранить в корпус для дообучения
            self._save_to_corpus(content, source=result.url, title=result.title)
            
            self.pages_fetched += 1
            stats["pages"] += 1
            stats["chunks"] += len(chunks)
            stats["bytes"] += len(content.encode('utf-8'))
            
            print(f"       ✅ {len(content)} символов → {len(chunks)} чанков → LEANN")
        
        # Шаг 3: Сохранить LEANN
        print(f"\n  💾 Сохранение LEANN ({len(self.memory.texts)} документов)...")
        self.memory.save()
        
        # Шаг 4: Проверить нужно ли дообучение
        corpus_mb = self._corpus_size_mb()
        if corpus_mb >= RETRAIN_THRESHOLD_MB:
            print(f"\n  🔥 Корпус: {corpus_mb:.1f} MB (порог {RETRAIN_THRESHOLD_MB} MB)")
            print(f"  🔥 Готов к дообучению! Запусти:")
            print(f"     python mega_train.py --extra-data {self.corpus_file}")
        else:
            print(f"\n  📊 Корпус: {corpus_mb:.1f}/{RETRAIN_THRESHOLD_MB} MB до дообучения")
        
        self.bytes_collected += stats["bytes"]
        
        print(f"\n{'═'*60}")
        print(f"  ✅ Изучено: {stats['pages']} страниц, {stats['chunks']} чанков")
        print(f"  🧠 LEANN: {len(self.memory.texts)} документов")
        print(f"{'═'*60}\n")
        
        return stats
    
    async def learn_from_urls(self, urls: List[str]) -> dict:
        """Изучить список конкретных URL."""
        stats = {"pages": 0, "chunks": 0, "bytes": 0}
        
        for url in urls:
            # ═══ URL safety check ═══
            if not self._is_url_safe(url):
                print(f"  ⛔ Заблокированный URL: {url[:80]}")
                continue
            
            print(f"\n  📚 Загружаю: {url[:80]}")
            self._rate_limit()
            content = await fetch_page_text(url, max_chars=10000)
            
            if not content or len(content) < self.min_content_length:
                print(f"  ⚠ Пропуск ({len(content)} символов)")
                continue
            
            chunks = self._make_chunks(content, url=url)
            for chunk in chunks:
                self.memory.add_document(chunk)
                self.docs_added += 1
            
            self._save_to_corpus(content, source=url)
            stats["pages"] += 1
            stats["chunks"] += len(chunks)
            stats["bytes"] += len(content.encode('utf-8'))
            
            print(f"  ✅ {len(content)} символов → {len(chunks)} чанков")
        
        self.memory.save()
        return stats
    
    async def auto_learn(self, topics: List[str], interval_min: int = 30,
                         max_cycles: int = 100, max_corpus_mb: float = 100.0):
        """
        Автоматическое обучение: циклически изучает темы.
        
        Args:
            topics: Список тем для изучения
            interval_min: Интервал между циклами (минуты)
            max_cycles: Максимум циклов (защита от бесконечного лупа)
            max_corpus_mb: Максимум размера корпуса (MB)
        """
        print(f"\n  🔄 Автоматическое обучение: {len(topics)} тем")
        print(f"  ⏰ Интервал: {interval_min} мин | Max cycles: {max_cycles}")
        print(f"  (Ctrl+C для остановки)\n")
        
        for cycle in range(1, max_cycles + 1):
            # ═══ Corpus size guard ═══
            if self._corpus_size_mb() >= max_corpus_mb:
                print(f"\n  🛑 Корпус достиг {max_corpus_mb} MB. Остановка.")
                break
            
            print(f"\n{'━'*60}")
            print(f"  Цикл #{cycle}/{max_cycles}")
            print(f"{'━'*60}")
            
            for topic in topics:
                try:
                    await self.learn_topic(topic)
                except Exception as e:
                    print(f"  ❌ Ошибка '{topic}': {e}")
                
                # ═══ Polite pause (5s between requests) ═══
                await asyncio.sleep(5)
            
            if cycle < max_cycles:
                print(f"\n  💤 Следующий цикл через {interval_min} мин...")
                await asyncio.sleep(interval_min * 60)
        
        print(f"\n  ✅ auto_learn завершён ({cycle} циклов)")
    
    # ═══════ Утилиты ═══════
    
    def _make_chunks(self, text: str, title: str = "", url: str = "") -> List[str]:
        """Разбить текст на чанки для LEANN."""
        chunks = []
        # Добавить контекст к первому чанку
        prefix = ""
        if title:
            prefix += f"[{title}] "
        
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size].strip()
            if len(chunk) < 50:
                continue
            if i == 0 and prefix:
                chunk = prefix + chunk
            chunks.append(chunk)
        
        return chunks[:50]  # Макс 50 чанков с одной страницы
    
    def _validate_content(self, text: str) -> tuple:
        """
        Validate content quality before storing in LEANN.
        
        Returns: (is_valid: bool, reason: str)
        """
        import re as _re
        
        # 1. Minimum word count
        words = text.split()
        if len(words) < self._min_word_count:
            return False, f"слишком мало слов ({len(words)})"
        
        # 2. Detect script/CSS/HTML junk
        junk_markers = ['<script', '<style', 'function()', 'var ', 'document.', 
                        'window.', 'addEventListener', '{display:', 'margin:',
                        'padding:', '@media']
        junk_count = sum(1 for m in junk_markers if m in text.lower())
        if junk_count > 3:
            return False, f"обнаружен код/CSS ({junk_count} маркеров)"
        
        # 3. Detect boilerplate / navigation text  
        boilerplate_markers = ['cookie', 'privacy policy', 'terms of service',
                              'subscribe', 'newsletter', 'sign up', 'log in',
                              'политика конфиденциальности', 'подписаться', 'войти']
        bp_count = sum(1 for m in boilerplate_markers if m in text.lower())
        bp_ratio = bp_count / max(1, len(words) / 50)
        if bp_ratio > self._max_boilerplate_ratio:
            return False, f"слишком много boilerplate ({bp_ratio:.1f})"
        
        # 4. Detect repeated characters / gibberish
        if _re.search(r'(.{3,})\1{5,}', text):
            return False, "повторяющийся текст (gibberish)"
        
        return True, "ok"
    
    def _save_to_corpus(self, text: str, source: str = "", title: str = ""):
        """Сохранить в корпус для дообучения."""
        with open(self.corpus_file, 'a', encoding='utf-8') as f:
            header = f"\n\n=== SOURCE: {source} ==="
            if title:
                header += f"\n=== TITLE: {title} ==="
            f.write(header + "\n")
            f.write(text)
            f.write("\n")
    
    def _corpus_size_mb(self) -> float:
        """Размер корпуса в MB."""
        if self.corpus_file.exists():
            return self.corpus_file.stat().st_size / (1024 * 1024)
        return 0.0
    
    def corpus_stats(self) -> dict:
        """Статистика корпуса."""
        size = self._corpus_size_mb()
        lines = 0
        if self.corpus_file.exists():
            with open(self.corpus_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = sum(1 for _ in f)
        return {
            "size_mb": round(size, 2),
            "lines": lines,
            "ready_for_retrain": size >= RETRAIN_THRESHOLD_MB,
            "path": str(self.corpus_file),
        }


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ТАРС Self-Learning")
    parser.add_argument("topic", nargs="*", help="Тема для изучения")
    parser.add_argument("--auto", action="store_true", help="Автоматический режим")
    parser.add_argument("--telegram", action="store_true", help="С Telegram ботом")
    parser.add_argument("--urls", nargs="*", help="Конкретные URL для изучения")
    parser.add_argument("--interval", type=int, default=30, help="Интервал авто-обучения (мин)")
    parser.add_argument("--stats", action="store_true", help="Показать статистику")
    
    args = parser.parse_args()
    
    learner = SelfLearner()
    
    if args.stats:
        stats = learner.corpus_stats()
        print(f"\n📊 Корпус самообучения:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"  LEANN: {len(learner.memory.texts)} документов\n")
        return
    
    if args.telegram:
        # Запуск с Telegram
        from tools.telegram_bot import TarsTelegram
        token = os.environ.get("TARS_TELEGRAM_TOKEN", "")
        if not token:
            print("❌ Установи TARS_TELEGRAM_TOKEN")
            return
        
        bot = TarsTelegram(token=token, memory=learner.memory)
        
        # Запустить бота и обучение параллельно
        tasks = [bot.start()]
        if args.topic:
            tasks.append(learner.auto_learn(args.topic, args.interval))
        
        await asyncio.gather(*tasks)
    
    elif args.auto and args.topic:
        # Автоматический режим
        await learner.auto_learn(args.topic, args.interval)
    
    elif args.urls:
        # Изучить конкретные URL
        await learner.learn_from_urls(args.urls)
    
    elif args.topic:
        # Изучить одну тему
        for topic in args.topic:
            await learner.learn_topic(topic)
    
    else:
        # Интерактивный режим
        print("\n🧠 ТАРС Self-Learning — Интерактивный режим")
        print("Введи тему для изучения (или 'quit')\n")
        
        while True:
            try:
                topic = input("  Тема → ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if topic in ('quit', 'exit', 'q'):
                break
            if topic == 'stats':
                stats = learner.corpus_stats()
                for k, v in stats.items():
                    print(f"  {k}: {v}")
                continue
            if topic:
                await learner.learn_topic(topic)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
