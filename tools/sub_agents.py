"""
═══════════════════════════════════════════════════════════════
  ТАРС Sub-Agent System v2 — Production-Grade
═══════════════════════════════════════════════════════════════

Улучшения над v1:
  1. Retry с exponential backoff
  2. Кэш контента (не скачивать одно дважды)
  3. Rate limiter (вежливость к сайтам)
  4. Quality filter (отсеять мусор)
  5. Deduplication (simhash для текстов)
  6. Agent chaining (агент→подзадача→агент)
  7. Priority queue (важные задачи первыми)
  8. Health monitoring (самодиагностика)
"""

import asyncio
import hashlib
import time
import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

logger = logging.getLogger("Tars.SubAgents")


# ═══════════════════════════════════════════
# Infrastructure
# ═══════════════════════════════════════════

class RateLimiter:
    """Per-domain rate limiter — вежливость к сайтам."""
    
    def __init__(self, requests_per_second: float = 2.0):
        self.min_interval = 1.0 / requests_per_second
        self._last_request: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _get_domain(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except Exception:
            return url
    
    async def acquire(self, url: str):
        domain = self._get_domain(url)
        async with self._lock:
            now = time.time()
            last = self._last_request.get(domain, 0)
            wait = self.min_interval - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request[domain] = time.time()


class ContentCache:
    """LRU кэш загруженных страниц — не скачивать одно дважды."""
    
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self._cache: Dict[str, str] = {}
        self._access_order: List[str] = []
        self.hits = 0
        self.misses = 0
    
    def get(self, url: str) -> Optional[str]:
        if url in self._cache:
            self.hits += 1
            self._access_order.remove(url)
            self._access_order.append(url)
            return self._cache[url]
        self.misses += 1
        return None
    
    def put(self, url: str, content: str):
        if url in self._cache:
            self._access_order.remove(url)
        elif len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[url] = content
        self._access_order.append(url)
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def _text_fingerprint(text: str) -> str:
    """Быстрый fingerprint для дедупликации."""
    words = sorted(set(re.findall(r'[a-zа-яё]{4,}', text.lower())))[:50]
    return hashlib.md5(" ".join(words).encode()).hexdigest()


def _content_quality_score(text: str) -> float:
    """
    Оценка качества контента [0, 1].
    Фильтрует: пустые, мусорные, cookie-баннеры, навигацию.
    """
    if not text or len(text) < 100:
        return 0.0
    
    score = 0.5  # Базовый
    
    # Длина
    if len(text) > 500: score += 0.1
    if len(text) > 2000: score += 0.1
    
    # Предложения (хороший контент имеет их)
    sentences = len(re.findall(r'[.!?]\s', text))
    if sentences > 3: score += 0.1
    if sentences > 10: score += 0.1
    
    # Мусорные паттерны (штрафуем)
    junk = len(re.findall(
        r'cookie|privacy policy|terms of service|sign up|subscribe|advertisement|404|not found',
        text.lower()
    ))
    score -= junk * 0.05
    
    # Информативные слова (бонус)
    info_words = len(re.findall(
        r'tutorial|guide|example|step|install|create|build|use|how to|'
        r'руководство|пример|шаг|создать|использовать|установить',
        text.lower()
    ))
    score += min(info_words * 0.02, 0.2)
    
    return max(0.0, min(1.0, score))


class MsgType(str, Enum):
    TASK = "task"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"


@dataclass
class AgentMessage:
    sender: str
    msg_type: MsgType
    payload: Any
    timestamp: float = field(default_factory=time.time)


class MessageBus:
    """Шина сообщений между агентами."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._log: List[AgentMessage] = []
    
    def create_channel(self, name: str, maxsize: int = 100):
        if name not in self._queues:
            self._queues[name] = asyncio.Queue(maxsize=maxsize)
    
    async def send(self, channel: str, msg: AgentMessage):
        if channel in self._queues:
            await self._queues[channel].put(msg)
            self._log.append(msg)
    
    async def receive(self, channel: str, timeout: float = 30) -> Optional[AgentMessage]:
        if channel not in self._queues:
            return None
        try:
            return await asyncio.wait_for(self._queues[channel].get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


# ═══════════════════════════════════════════
# Sub-Agent Base
# ═══════════════════════════════════════════

@dataclass
class SubAgentResult:
    agent_name: str
    success: bool
    data: Any = None
    error: str = ""
    duration: float = 0.0
    retries: int = 0


class SubAgent:
    """Базовый суб-агент с retry и health monitoring."""
    
    def __init__(self, name: str, max_retries: int = 3):
        self.name = name
        self.max_retries = max_retries
        self.busy = False
        self.tasks_done = 0
        self.tasks_failed = 0
        self.total_time = 0.0
    
    async def run(self, task: dict) -> SubAgentResult:
        raise NotImplementedError
    
    async def run_with_retry(self, task: dict) -> SubAgentResult:
        """Выполнить с retry и exponential backoff."""
        self.busy = True
        last_error = ""
        
        for attempt in range(self.max_retries):
            try:
                result = await self.run(task)
                self.busy = False
                self.tasks_done += 1
                self.total_time += result.duration
                result.retries = attempt
                return result
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    wait = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    await asyncio.sleep(wait)
        
        self.busy = False
        self.tasks_failed += 1
        return SubAgentResult(self.name, False, error=last_error, retries=self.max_retries)
    
    @property
    def health(self) -> float:
        """Здоровье агента [0, 1]."""
        total = self.tasks_done + self.tasks_failed
        if total == 0:
            return 1.0
        return self.tasks_done / total
    
    @property
    def avg_time(self) -> float:
        return self.total_time / max(self.tasks_done, 1)


# ═══════════════════════════════════════════
# Specialized Sub-Agents
# ═══════════════════════════════════════════

class SearchAgent(SubAgent):
    """Поиск с дедупликацией результатов."""
    
    def __init__(self):
        super().__init__("search", max_retries=2)
        self._seen_urls: Set[str] = set()
    
    async def run(self, task: dict) -> SubAgentResult:
        start = time.time()
        query = task.get("query", "")
        max_results = task.get("max_results", 5)
        
        from tools.web_search import search_duckduckgo
        results = await search_duckduckgo(query, max_results=max_results)
        
        # Дедупликация URL
        unique = []
        for r in results:
            if r.url not in self._seen_urls:
                self._seen_urls.add(r.url)
                unique.append({"title": r.title, "url": r.url, "snippet": r.snippet})
        
        return SubAgentResult(
            self.name, True,
            data=unique,
            duration=time.time() - start,
        )


class FetchAgent(SubAgent):
    """Параллельная загрузка с кэшом, rate limiting, quality filter."""
    
    def __init__(self, max_concurrent: int = 5):
        super().__init__("fetch", max_retries=2)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.cache = ContentCache(max_size=200)
        self.rate_limiter = RateLimiter(requests_per_second=3.0)
        self.min_quality = 0.3
    
    async def _fetch_one(self, url: str, max_chars: int = 5000) -> dict:
        async with self.semaphore:
            # Проверить кэш
            cached = self.cache.get(url)
            if cached is not None:
                return {"url": url, "content": cached, "ok": True, "cached": True}
            
            # Rate limit
            await self.rate_limiter.acquire(url)
            
            try:
                from tools.web_search import fetch_page_text
                text = await fetch_page_text(url, max_chars=max_chars)
                
                if not text:
                    return {"url": url, "content": "", "ok": False}
                
                # Quality filter
                quality = _content_quality_score(text)
                if quality < self.min_quality:
                    return {"url": url, "content": "", "ok": False, 
                            "reason": f"low quality ({quality:.2f})"}
                
                # Кэшируем
                self.cache.put(url, text)
                
                return {"url": url, "content": text, "ok": True, 
                        "quality": quality, "cached": False}
            except Exception as e:
                return {"url": url, "content": "", "ok": False, "error": str(e)}
    
    async def run(self, task: dict) -> SubAgentResult:
        start = time.time()
        urls = task.get("urls", [])
        max_chars = task.get("max_chars", 5000)
        
        results = await asyncio.gather(
            *[self._fetch_one(url, max_chars) for url in urls],
            return_exceptions=True,
        )
        
        fetched = []
        for r in results:
            if isinstance(r, dict):
                fetched.append(r)
            else:
                fetched.append({"url": "?", "content": "", "ok": False, "error": str(r)})
        
        # Сортировать по quality (лучшие первыми)
        fetched.sort(key=lambda x: x.get("quality", 0), reverse=True)
        
        return SubAgentResult(self.name, True, data=fetched, duration=time.time() - start)


class AnalyzeAgent(SubAgent):
    """NLP с дедупликацией и quality scoring."""
    
    def __init__(self):
        super().__init__("analyze", max_retries=1)
        self._seen_fingerprints: Set[str] = set()
    
    async def run(self, task: dict) -> SubAgentResult:
        start = time.time()
        texts = task.get("texts", [])
        topic = task.get("topic", "")
        
        # Дедупликация текстов
        unique_texts = []
        for text in texts:
            if not text or len(text) < 50:
                continue
            fp = _text_fingerprint(text)
            if fp not in self._seen_fingerprints:
                self._seen_fingerprints.add(fp)
                unique_texts.append(text)
        
        all_instructions = []
        all_commands = []
        all_facts = []
        
        for text in unique_texts:
            # Инструкции
            for pattern in [
                r'(?:шаг|step)\s*\d+[.:]\s*(.+)',
                r'\d+[.)]\s+(.{20,})',
                r'(?:сначала|далее|затем|потом|нужно|необходимо)\s+(.+)',
                r'(?:first|then|next|finally|you need to)\s+(.+)',
            ]:
                for m in re.findall(pattern, text, re.IGNORECASE | re.MULTILINE):
                    m = m.strip()
                    if 20 < len(m) < 300 and m not in all_instructions:
                        all_instructions.append(m)
            
            # Команды
            for pattern in [
                r'`([^`]{3,100})`',
                r'\$\s+(.{3,100})',
                r'(?:pip|npm|docker|git|python|cargo|apt|brew|yarn)\s+\S+(?:\s+\S+){0,5}',
            ]:
                for m in re.findall(pattern, text, re.MULTILINE):
                    m = m.strip()
                    if 3 < len(m) < 200 and m not in all_commands:
                        all_commands.append(m)
            
            # Факты
            if topic:
                topic_words = set(re.findall(r'[a-zа-яё]{3,}', topic.lower()))
                for sent in re.split(r'[.!?\n]', text):
                    sent = sent.strip()
                    if 30 < len(sent) < 300:
                        words = set(re.findall(r'[a-zа-яё]{3,}', sent.lower()))
                        if len(topic_words & words) >= 1 and sent not in all_facts:
                            all_facts.append(sent)
        
        return SubAgentResult(
            self.name, True,
            data={
                "instructions": all_instructions[:25],
                "commands": all_commands[:20],
                "facts": all_facts[:40],
                "unique_sources": len(unique_texts),
                "deduped": len(texts) - len(unique_texts),
            },
            duration=time.time() - start,
        )


class MemoryAgent(SubAgent):
    """Сохранение в LEANN с дедупликацией."""
    
    def __init__(self, memory=None):
        super().__init__("memory", max_retries=1)
        self.memory = memory
        self._saved_fingerprints: Set[str] = set()
    
    async def run(self, task: dict) -> SubAgentResult:
        start = time.time()
        texts = task.get("texts", [])
        chunk_size = task.get("chunk_size", 500)
        tag = task.get("tag", "")
        
        if not self.memory:
            return SubAgentResult(self.name, False, error="No memory")
        
        added = 0
        skipped = 0
        for text in texts:
            if not text or len(text) < 50:
                continue
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i+chunk_size].strip()
                if len(chunk) < 50:
                    continue
                
                # Дедупликация
                fp = _text_fingerprint(chunk)
                if fp in self._saved_fingerprints:
                    skipped += 1
                    continue
                self._saved_fingerprints.add(fp)
                
                if tag:
                    chunk = f"[{tag}] {chunk}"
                self.memory.add_document(chunk)
                added += 1
        
        return SubAgentResult(
            self.name, True,
            data={"added": added, "skipped": skipped, "total": len(self.memory.texts)},
            duration=time.time() - start,
        )


# ═══════════════════════════════════════════
# Agent Pool v2 — Оркестратор
# ═══════════════════════════════════════════

class AgentPool:
    """
    Пул суб-агентов v2 — production-grade.
    
    Фичи:
      - Retry с backoff
      - Кэш контента
      - Rate limiting
      - Quality filter
      - Дедупликация
      - Health monitoring
      - Agent chaining
    """
    
    def __init__(self, memory=None):
        self.memory = memory
        self.bus = MessageBus()
        
        self.search_agent = SearchAgent()
        self.fetch_agent = FetchAgent(max_concurrent=5)
        self.analyze_agent = AnalyzeAgent()
        self.memory_agent = MemoryAgent(memory)
        
        self.total_tasks = 0
    
    async def research(self, topic: str, depth: int = 5) -> Dict[str, Any]:
        """
        Полное исследование темы — параллельно!
        
        v2 улучшения:
          ✅ Retry при ошибках
          ✅ Кэш (повторный запрос мгновенный)
          ✅ Rate limit (2 req/s per domain)
          ✅ Quality filter (мусор отсеивается)
          ✅ Dedup (нет дубликатов)
        """
        start = time.time()
        print(f"  ⚡ SubAgents v2: '{topic}'")
        
        # ── Фаза 1: Параллельный поиск ──
        search_queries = [
            f"{topic} tutorial руководство",
            f"{topic} examples примеры команды",
            f"{topic} best practices tips",
        ]
        
        search_tasks = [
            self.search_agent.run_with_retry({"query": q, "max_results": depth})
            for q in search_queries
        ]
        search_results = await asyncio.gather(*search_tasks)
        
        all_urls = []
        all_titles = {}
        for sr in search_results:
            if sr.success and sr.data:
                for item in sr.data:
                    url = item["url"]
                    if url not in all_titles:
                        all_urls.append(url)
                        all_titles[url] = item["title"]
        
        search_time = time.time() - start
        cached = self.fetch_agent.cache.hits
        print(f"    🔍 Поиск: {len(all_urls)} URL ({search_time:.1f}s)")
        
        if not all_urls:
            return {"topic": topic, "error": "nothing found", "time": search_time}
        
        # ── Фаза 2: Параллельная загрузка (с кэшом!) ──
        fetch_start = time.time()
        fetch_result = await self.fetch_agent.run_with_retry({
            "urls": all_urls[:12],
            "max_chars": 5000,
        })
        
        fetched_texts = []
        fetched_sources = []
        quality_scores = []
        if fetch_result.success:
            for item in fetch_result.data:
                if item.get("ok") and item.get("content"):
                    fetched_texts.append(item["content"])
                    fetched_sources.append(item["url"])
                    quality_scores.append(item.get("quality", 0))
        
        new_cached = self.fetch_agent.cache.hits - cached
        fetch_time = time.time() - fetch_start
        print(f"    📥 Загрузка: {len(fetched_texts)}/{len(all_urls)} "
              f"({fetch_time:.1f}s, cache: {new_cached})")
        
        # ── Фаза 3+4: Анализ + Сохранение (параллельно) ──
        par_start = time.time()
        analyze_task = self.analyze_agent.run_with_retry({
            "texts": fetched_texts,
            "topic": topic,
        })
        memory_task = self.memory_agent.run_with_retry({
            "texts": fetched_texts,
            "chunk_size": 500,
            "tag": f"Skill:{topic}",
        })
        
        analyze_result, memory_result = await asyncio.gather(analyze_task, memory_task)
        par_time = time.time() - par_start
        
        analysis = analyze_result.data if analyze_result.success else {}
        mem_info = memory_result.data if memory_result.success else {}
        
        total_time = time.time() - start
        self.total_tasks += 1
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        print(f"    🧠 Анализ+Память: ({par_time:.1f}s) "
              f"dedup: -{analysis.get('deduped', 0)}, "
              f"mem_skip: {mem_info.get('skipped', 0)}")
        print(f"    ✅ {total_time:.1f}s | "
              f"{len(analysis.get('instructions', []))} instr, "
              f"{len(analysis.get('commands', []))} cmd, "
              f"{len(analysis.get('facts', []))} facts | "
              f"quality: {avg_quality:.0%} | "
              f"+{mem_info.get('added', 0)} LEANN")
        
        return {
            "topic": topic,
            "sources": fetched_sources,
            "titles": [all_titles.get(u, "") for u in fetched_sources],
            "instructions": analysis.get("instructions", []),
            "commands": analysis.get("commands", []),
            "facts": analysis.get("facts", []),
            "memory_added": mem_info.get("added", 0),
            "memory_total": mem_info.get("total", 0),
            "pages_fetched": len(fetched_texts),
            "avg_quality": avg_quality,
            "total_time": total_time,
            "phase_times": {
                "search": search_time,
                "fetch": fetch_time,
                "analyze_and_memory": par_time,
            },
        }
    
    async def multi_research(self, topics: List[str]) -> List[Dict]:
        """Исследовать НЕСКОЛЬКО тем параллельно."""
        print(f"\n  ⚡⚡ Параллельно: {len(topics)} тем")
        start = time.time()
        
        results = await asyncio.gather(
            *[self.research(topic) for topic in topics],
            return_exceptions=True,
        )
        
        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                output.append({"topic": topics[i], "error": str(r)})
            else:
                output.append(r)
        
        total = time.time() - start
        print(f"\n  ⚡⚡ Все {len(topics)} тем за {total:.1f}s")
        return output
    
    async def deep_research(self, topic: str, depth: int = 5, 
                             follow_links: int = 3) -> Dict[str, Any]:
        """
        Глубокое исследование: research() + follow related links.
        
        Агент находит информацию → извлекает связанные темы → 
        исследует их тоже (agent chaining).
        """
        print(f"\n  🔬 Глубокое исследование: {topic}")
        
        # Первый проход
        result = await self.research(topic, depth=depth)
        
        # Извлечь связанные темы из фактов
        related = self._extract_related_topics(topic, result.get("facts", []))
        
        if related and follow_links > 0:
            print(f"  🔗 Связанные темы: {related[:follow_links]}")
            extra_results = await self.multi_research(related[:follow_links])
            
            # Объединить
            for extra in extra_results:
                if isinstance(extra, dict) and not extra.get("error"):
                    result["instructions"].extend(extra.get("instructions", []))
                    result["commands"].extend(extra.get("commands", []))
                    result["facts"].extend(extra.get("facts", []))
                    result["sources"].extend(extra.get("sources", []))
            
            # Дедупликация
            result["instructions"] = list(dict.fromkeys(result["instructions"]))[:30]
            result["commands"] = list(dict.fromkeys(result["commands"]))[:25]
            result["facts"] = list(dict.fromkeys(result["facts"]))[:50]
        
        return result
    
    def _extract_related_topics(self, topic: str, facts: List[str]) -> List[str]:
        """Извлечь связанные темы из фактов."""
        # Ищем упоминания технологий/инструментов в фактах
        tech_patterns = re.findall(
            r'\b(?:Docker|Kubernetes|Git|Python|npm|pip|API|REST|SQL|Redis|'
            r'Nginx|Apache|SSH|Linux|Windows|CI/CD|AWS|Azure|GCP)\b',
            " ".join(facts), re.IGNORECASE
        )
        
        topic_lower = topic.lower()
        related = []
        for t in set(tech_patterns):
            if t.lower() != topic_lower and t.lower() not in topic_lower:
                related.append(f"{topic} {t}")
        
        return related[:5]
    
    def health_report(self) -> Dict[str, Any]:
        """Полный отчёт здоровья всех агентов."""
        agents = [self.search_agent, self.fetch_agent, 
                  self.analyze_agent, self.memory_agent]
        return {
            "agents": {
                a.name: {
                    "health": f"{a.health:.0%}",
                    "done": a.tasks_done,
                    "failed": a.tasks_failed,
                    "avg_time": f"{a.avg_time:.2f}s",
                } for a in agents
            },
            "cache": {
                "hit_rate": f"{self.fetch_agent.cache.hit_rate:.0%}",
                "hits": self.fetch_agent.cache.hits,
                "misses": self.fetch_agent.cache.misses,
            },
            "total_research": self.total_tasks,
        }
