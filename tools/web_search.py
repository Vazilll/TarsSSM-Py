"""
═══════════════════════════════════════════════════════════════
  ТАРС Web Search — Поиск информации в интернете
═══════════════════════════════════════════════════════════════

Без API ключей! Использует:
  - DuckDuckGo Lite (HTML scraping, без API)
  - Прямой HTTP для страниц
  - Очистка HTML → текст
"""

import re
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from urllib.parse import quote_plus, urljoin
from dataclasses import dataclass

logger = logging.getLogger("Tars.WebSearch")


@dataclass
class WebResult:
    """Результат веб-поиска."""
    title: str
    url: str
    snippet: str
    content: str = ""  # Полный текст страницы (если загружен)


def _clean_html(html: str) -> str:
    """Убрать HTML теги, оставить текст."""
    # Удалить скрипты и стили
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Удалить теги
    text = re.sub(r'<[^>]+>', ' ', text)
    # HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('&quot;', '"').replace('&#39;', "'").replace('&nbsp;', ' ')
    # Очистить пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


async def _http_get(url: str, timeout: int = 10) -> str:
    """HTTP GET запрос (без внешних зависимостей)."""
    try:
        import urllib.request
        import ssl
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml',
            'Accept-Language': 'ru,en;q=0.9',
        })
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=timeout, context=ctx)
        )
        
        data = response.read()
        # Попробовать разные кодировки
        for enc in ['utf-8', 'cp1251', 'latin-1']:
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                continue
        return data.decode('utf-8', errors='replace')
        
    except Exception as e:
        logger.warning(f"HTTP GET failed for {url}: {e}")
        return ""


async def search_duckduckgo(query: str, max_results: int = 5) -> List[WebResult]:
    """
    Поиск через DuckDuckGo Lite (HTML).
    Не требует API ключ!
    """
    encoded = quote_plus(query)
    url = f"https://lite.duckduckgo.com/lite/?q={encoded}"
    
    html = await _http_get(url)
    if not html:
        return []
    
    results = []
    
    # Парсим результаты из DuckDuckGo Lite HTML
    # Формат: <a rel="nofollow" href="URL" class="result-link">Title</a>
    # Потом: <td class="result-snippet">Snippet</td>
    
    links = re.findall(
        r'<a[^>]*rel="nofollow"[^>]*href="([^"]+)"[^>]*class="result-link"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )
    snippets = re.findall(
        r'<td[^>]*class="result-snippet"[^>]*>(.*?)</td>',
        html, re.DOTALL
    )
    
    for i, (link_url, title) in enumerate(links[:max_results]):
        snippet = _clean_html(snippets[i]) if i < len(snippets) else ""
        title_clean = _clean_html(title)
        
        if link_url and title_clean:
            results.append(WebResult(
                title=title_clean,
                url=link_url,
                snippet=snippet[:300],
            ))
    
    # Fallback: если DuckDuckGo Lite не распарсился, 
    # пробуем альтернативный паттерн
    if not results:
        # Ищем любые ссылки с текстом
        all_links = re.findall(
            r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        for link_url, title in all_links[:max_results]:
            title_clean = _clean_html(title).strip()
            if title_clean and len(title_clean) > 5 and 'duckduckgo' not in link_url:
                results.append(WebResult(
                    title=title_clean,
                    url=link_url,
                    snippet="",
                ))
    
    logger.info(f"DuckDuckGo: '{query}' → {len(results)} results")
    return results


async def fetch_page_text(url: str, max_chars: int = 5000) -> str:
    """Скачать страницу и извлечь текст."""
    html = await _http_get(url, timeout=15)
    if not html:
        return ""
    
    text = _clean_html(html)
    
    # Удалить навигацию и мусор (частые паттерны)
    text = re.sub(r'(Cookie|Privacy|Terms of Service|Sign in|Log in).*?\n', '', text)
    
    return text[:max_chars]


async def search_and_learn(query: str, max_results: int = 3) -> List[WebResult]:
    """
    Поиск + загрузка контента страниц.
    Готово для добавления в LEANN.
    """
    results = await search_duckduckgo(query, max_results=max_results)
    
    # Загрузить контент каждой страницы
    for r in results:
        try:
            r.content = await fetch_page_text(r.url, max_chars=3000)
        except Exception as e:
            logger.warning(f"Failed to fetch {r.url}: {e}")
            r.content = r.snippet
    
    return results
