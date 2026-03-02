"""
═══════════════════════════════════════════════════════════════
  MinGRU Action Executor — Runtime Memory Integration
═══════════════════════════════════════════════════════════════

Парсит вывод MinGRU на наличие action-токенов и выполняет их:
  [FILE_SEARCH]    → os.walk / glob поиск файлов
  [FILE_LIST]      → os.listdir
  [FILE_READ]      → open/read
  [FILE_INFO]      → os.stat
  [MEMORY_SAVE]    → leann.add_document
  [MEMORY_SEARCH]  → leann.search
  [MEMORY_LIST]    → список документов в LEANN
  [NOTE_CREATE]    → сохранить заметку в JSON
  [NOTE_DELETE]    → удалить заметку
  [LEANN_SEARCH]   → семантический поиск
  [SYSTEM_INFO]    → GPU/RAM/disk info
  [SYSTEM_TIME]    → datetime.now
  [SYSTEM_STATUS]  → проверка модулей
  [SYSTEM_CMD]     → torch.cuda.empty_cache и т.п.
  [ROUTE_DEEP]     → маркер "передать в Mamba-2"
  [ROUTE_FAST]     → маркер "ответ от MinGRU"
  [EXEC]           → запуск команды

Использование:
  from brain.min_gru.mingru_actions import MinGRUActionExecutor
  executor = MinGRUActionExecutor()
  result = await executor.execute("[FILE_SEARCH] config.json → Ищу...")
"""

import os
import sys
import json
import re
import asyncio
import logging
import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger("mingru_actions")

ROOT = Path(__file__).resolve().parent.parent.parent

# ═══════════════════════════════════════════
# Precompiled regex (вызывается тысячи раз)
# ═══════════════════════════════════════════
_ACTION_RE = re.compile(r'\[(\w+(?:_\w+)*)\]\s*(.*)')
_PAYLOAD_CLEAN_RE = re.compile(r'^[→\-:]+\s*')

# Директории которые пропускаем при поиске файлов
_SKIP_DIRS = frozenset({
    '.git', '__pycache__', '.venv', 'venv', 'node_modules',
    '.mypy_cache', '.pytest_cache', '.eggs', '*.egg-info',
})


class MinGRUActionExecutor:
    """
    Исполнитель действий MinGRU — превращает action-токены в реальные операции.
    
    Оптимизации:
      - Precompiled regex (не пересоздаётся на каждый вызов)
      - Handler dict создаётся один раз в __init__
      - File index кэшируется (обновляется не чаще 30 сек)
      - LEANN lazy-load + совместимость async/sync
      - Notes кэшируются в памяти, записываются лениво
    """

    __slots__ = (
        '_leann', 'notes_path', '_notes', '_handlers',
        '_file_cache', '_file_cache_time', '_sys_info_cache', '_sys_info_time',
    )

    def __init__(self, leann=None, notes_path: Optional[Path] = None):
        self._leann = leann
        self.notes_path = notes_path or ROOT / "data" / "notes.json"
        self._notes = self._load_notes()

        # File index cache (обновляется не чаще 30 сек)
        self._file_cache: Optional[List[str]] = None
        self._file_cache_time: float = 0.0

        # System info cache (обновляется не чаще 10 сек)
        self._sys_info_cache: Optional[str] = None
        self._sys_info_time: float = 0.0

        # Handler dict — ONE создание вместо пересоздания каждый вызов
        self._handlers: Dict[str, callable] = {
            "FILE_SEARCH": self._file_search,
            "FILE_LIST": self._file_list,
            "FILE_READ": self._file_read,
            "FILE_INFO": self._file_info,
            "MEMORY_SAVE": self._memory_save,
            "MEMORY_SEARCH": self._memory_search,
            "MEMORY_LIST": self._memory_list,
            "NOTE_CREATE": self._note_create,
            "NOTE_DELETE": self._note_delete,
            "LEANN_SEARCH": self._leann_search,
            "SYSTEM_INFO": self._system_info,
            "SYSTEM_TIME": self._system_time,
            "SYSTEM_STATUS": self._system_status,
            "SYSTEM_CMD": self._system_cmd,
            "ROUTE_DEEP": self._route_deep,
            "ROUTE_FAST": self._route_fast,
            "EXEC": self._exec_cmd,
        }

    # ═══════════════════════════════════════════
    # Main API
    # ═══════════════════════════════════════════

    async def execute(self, mingru_output: str) -> Optional[str]:
        """
        Парсит вывод MinGRU, находит action-токен и выполняет действие.
        
        Returns:
            Строка результата или None если нет action-токенов.
        """
        action, payload = self._parse_action(mingru_output)
        if not action:
            return None

        handler = self._handlers.get(action)
        if handler:
            try:
                result = await handler(payload)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[{action}] {payload} → {result[:100] if result else 'None'}")
                return result
            except Exception as e:
                logger.error(f"[{action}] Ошибка: {e}")
                return f"Ошибка [{action}]: {e}"

        return None

    def parse_route(self, mingru_output: str) -> str:
        """Определяет маршрут: 'fast', 'deep', или 'unknown'."""
        action, _ = self._parse_action(mingru_output)
        if action == "ROUTE_DEEP":
            return "deep"
        elif action == "ROUTE_FAST":
            return "fast"
        return "unknown"

    # ═══════════════════════════════════════════
    # Action Parser (precompiled regex)
    # ═══════════════════════════════════════════

    @staticmethod
    def _parse_action(text: str) -> Tuple[Optional[str], str]:
        """Ищет [ACTION_TOKEN] в тексте. Precompiled regex — O(1) аллокаций."""
        match = _ACTION_RE.search(text)
        if match:
            action = match.group(1).upper()
            payload = _PAYLOAD_CLEAN_RE.sub('', match.group(2).strip())
            return action, payload
        return None, text

    # ═══════════════════════════════════════════
    # File Operations (кэшированный индекс)
    # ═══════════════════════════════════════════

    def _build_file_index(self) -> List[str]:
        """Построить индекс файлов (кэш 30 сек). Не блокирует event loop."""
        import time
        now = time.monotonic()
        if self._file_cache is not None and (now - self._file_cache_time) < 30.0:
            return self._file_cache

        files = []
        for root_dir, dirs, filenames in os.walk(ROOT):
            # Пропуск скрытых и cache
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith('.')]
            for f in filenames:
                rel = os.path.relpath(os.path.join(root_dir, f), ROOT)
                files.append(rel)
            if len(files) >= 5000:  # Жёсткий лимит для скорости
                break

        self._file_cache = files
        self._file_cache_time = now
        return files

    async def _file_search(self, query: str) -> str:
        """Поиск файлов. Использует кэшированный индекс."""
        pattern = query.strip()
        if not pattern:
            return "Укажите имя или паттерн файла."

        # Запускаем в thread pool чтобы не блокировать event loop
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, self._build_file_index)

        pattern_lower = pattern.lower()
        results = []
        for f in files:
            basename = os.path.basename(f)
            if fnmatch(basename, pattern) or pattern_lower in basename.lower():
                results.append(f)
                if len(results) >= 20:
                    break

        if results:
            return f"Найдено {len(results)}: " + ", ".join(results[:10])
        return f"Файл '{pattern}' не найден."

    async def _file_list(self, path: str) -> str:
        """Содержимое директории."""
        target = ROOT / (path.strip().rstrip('/') if path.strip() else '.')
        if not target.exists():
            return f"Директория {path} не найдена."

        items = []
        try:
            for item in sorted(target.iterdir()):
                if item.name.startswith('.'):
                    continue
                prefix = "📁" if item.is_dir() else "📄"
                items.append(f"{prefix} {item.name}")
                if len(items) >= 30:
                    break
        except PermissionError:
            return "Нет доступа к директории."

        return "\n".join(items) if items else "Пусто."

    async def _file_read(self, path: str) -> str:
        """Чтение файла (первые 500 символов)."""
        target = ROOT / path.strip()
        if not target.exists():
            return f"Файл {path} не найден."
        try:
            text = target.read_text(encoding='utf-8', errors='replace')
            return text[:500] + ("..." if len(text) > 500 else "")
        except Exception as e:
            return f"Ошибка чтения: {e}"

    async def _file_info(self, path: str) -> str:
        """Информация о файле."""
        target = ROOT / path.strip()
        if not target.exists():
            return f"Файл {path} не найден."
        stat = target.stat()
        size = stat.st_size
        if size > 1048576:
            size_str = f"{size / 1048576:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.0f} KB"
        else:
            size_str = f"{size} B"
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        return f"{path}: {size_str}, изменён {mtime:%Y-%m-%d %H:%M}"

    # ═══════════════════════════════════════════
    # Memory Operations (LEANN, async/sync safe)
    # ═══════════════════════════════════════════

    @property
    def leann(self):
        """Lazy-load LEANN (один раз, потом кэш)."""
        if self._leann is None:
            try:
                from memory.leann import LeannIndex
                self._leann = LeannIndex()
                self._leann.load()
                logger.info(f"LEANN загружен: {len(self._leann.texts)} документов")
            except Exception as e:
                logger.warning(f"LEANN не загружен: {e}")
        return self._leann

    async def _call_leann(self, method_name: str, *args, **kwargs):
        """Вызов метода LEANN — автоопределяет async/sync."""
        if not self.leann:
            return None
        method = getattr(self.leann, method_name)
        result = method(*args, **kwargs)
        # Если coroutine — await, иначе вернуть как есть
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _memory_save(self, text: str) -> str:
        """Сохранить в LEANN."""
        if not text:
            return "Что сохранить? Укажите текст."
        result = await self._call_leann("add_document", text)
        if result is not None or self.leann:
            return f"Сохранено в память: {text[:80]}"
        return "LEANN недоступен."

    async def _memory_search(self, query: str) -> str:
        """Поиск в LEANN."""
        if not query:
            return "Что искать?"
        results = await self._call_leann("search", query, top_k=3)
        if results:
            return "Найдено:\n" + "\n".join(f"  • {r[:100]}" for r in results)
        if self.leann:
            return "Ничего не найдено в памяти."
        return "LEANN недоступен."

    async def _memory_list(self, _: str) -> str:
        """Список документов в LEANN."""
        if self.leann and hasattr(self.leann, 'texts'):
            n = len(self.leann.texts)
            if n == 0:
                return "Память пуста."
            last = self.leann.texts[-3:] if n > 3 else self.leann.texts
            return f"В памяти {n} документов. Последние:\n" + "\n".join(f"  • {t[:80]}" for t in last)
        return "LEANN недоступен."

    async def _leann_search(self, query: str) -> str:
        """Семантический поиск (алиас MEMORY_SEARCH)."""
        return await self._memory_search(query)

    # ═══════════════════════════════════════════
    # Notes (JSON, кэш в RAM)
    # ═══════════════════════════════════════════

    def _load_notes(self) -> List[Dict]:
        if self.notes_path.exists():
            try:
                return json.loads(self.notes_path.read_text(encoding='utf-8'))
            except Exception:
                pass
        return []

    def _save_notes(self):
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        self.notes_path.write_text(
            json.dumps(self._notes, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )

    async def _note_create(self, text: str) -> str:
        if not text:
            return "Что записать?"
        self._notes.append({
            "text": text,
            "created": datetime.datetime.now().isoformat(),
        })
        self._save_notes()
        return f"Заметка создана: {text[:80]}"

    async def _note_delete(self, query: str) -> str:
        if not query:
            return "Что удалить?"
        before = len(self._notes)
        q_lower = query.lower()
        self._notes = [n for n in self._notes if q_lower not in n.get("text", "").lower()]
        after = len(self._notes)
        if before != after:
            self._save_notes()
            return f"Удалено {before - after} заметок."
        return f"Заметка '{query}' не найдена."

    # ═══════════════════════════════════════════
    # System (кэшированный info)
    # ═══════════════════════════════════════════

    async def _system_info(self, _: str) -> str:
        """GPU/RAM info. Кэш 10 сек — не дёргать GPU каждый вызов."""
        import time
        now = time.monotonic()
        if self._sys_info_cache and (now - self._sys_info_time) < 10.0:
            return self._sys_info_cache

        info = []
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / 1073741824
                used = torch.cuda.memory_allocated() / 1073741824
                info.append(f"GPU: {name} ({used:.1f}/{total:.1f} GB)")
            else:
                info.append("GPU: нет")
        except ImportError:
            info.append("GPU: torch не установлен")

        try:
            import psutil
            ram = psutil.virtual_memory()
            info.append(f"RAM: {ram.used/1073741824:.1f}/{ram.total/1073741824:.1f} GB ({ram.percent}%)")
        except ImportError:
            pass

        result = " | ".join(info) if info else "Информация недоступна."
        self._sys_info_cache = result
        self._sys_info_time = now
        return result

    async def _system_time(self, _: str) -> str:
        now = datetime.datetime.now()
        return f"{now:%Y-%m-%d %H:%M:%S} ({now:%A})"

    async def _system_status(self, _: str) -> str:
        status = []
        try:
            from brain.min_gru.mingru_lm import MinGRU_LM
            status.append("MinGRU: ✓")
        except Exception:
            status.append("MinGRU: ✗")

        if self.leann and hasattr(self.leann, 'texts'):
            status.append(f"LEANN: ✓ ({len(self.leann.texts)} docs)")
        else:
            status.append("LEANN: ✗")

        status.append(f"Notes: {len(self._notes)}")
        return " | ".join(status)

    async def _system_cmd(self, cmd: str) -> str:
        """Безопасные системные команды."""
        cmd_lower = cmd.lower()
        if "cache" in cmd_lower or "empty" in cmd_lower:
            try:
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "Кэш очищен (gc + cuda)."
            except Exception:
                import gc
                gc.collect()
                return "Кэш очищен (gc)."
        return "Неизвестная команда."

    async def _route_deep(self, payload: str) -> str:
        return f"[ROUTE_DEEP] {payload}"

    async def _route_fast(self, payload: str) -> str:
        return payload

    async def _exec_cmd(self, cmd: str) -> str:
        """Безопасный запуск (только python/pip внутри проекта)."""
        if not cmd.startswith("python ") and not cmd.startswith("pip "):
            return "Допустимы только python и pip команды."
        return f"[EXEC] Запрос: {cmd}. Требуется подтверждение."
