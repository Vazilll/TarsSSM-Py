"""
═══════════════════════════════════════════════════════════════
  TARS Tool Registry — Plugin System
═══════════════════════════════════════════════════════════════

Портировано из PicoClaw (Go → Python):
  - tools/registry.go → ToolRegistry 
  - tools/types.go → Tool interface
  - tools/shell.go → ShellTool (safe command execution)

Позволяет ТАРС динамически регистрировать и вызывать инструменты.
Каждый инструмент: name, description, parameters schema, execute().
"""

import os
import sys
import json
import time
import asyncio
import logging
import subprocess
import platform
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod


logger = logging.getLogger("Tars.Tools")


# ═══════════════════════════════════════════
# Tool Interface (из PicoClaw types.go)
# ═══════════════════════════════════════════

@dataclass
class ToolResult:
    """Результат выполнения инструмента."""
    output: str         # Результат для пользователя
    llm_output: str     # Результат для LLM (может быть подробнее)
    is_error: bool = False
    silent: bool = False    # Не показывать пользователю
    execution_time: float = 0.0

    @staticmethod
    def success(output: str, llm_output: str = "") -> 'ToolResult':
        return ToolResult(output=output, llm_output=llm_output or output)
    
    @staticmethod
    def error(message: str) -> 'ToolResult':
        return ToolResult(output=message, llm_output=message, is_error=True)


class Tool(ABC):
    """Базовый класс для инструментов ТАРС."""
    
    @abstractmethod
    def name(self) -> str:
        """Уникальное имя инструмента."""
        ...
    
    @abstractmethod
    def description(self) -> str:
        """Описание для LLM."""
        ...
    
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema параметров."""
        ...
    
    @abstractmethod
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        """Выполнить инструмент."""
        ...


# ═══════════════════════════════════════════
# Tool Registry (из PicoClaw registry.go)
# ═══════════════════════════════════════════

class ToolRegistry:
    """
    Реестр инструментов ТАРС.
    
    Позволяет:
      - Регистрировать инструменты динамически
      - Вызывать по имени
      - Получать JSON-схему всех инструментов (для LLM)
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self.logger = logging.getLogger("Tars.ToolRegistry")
    
    def register(self, tool: Tool):
        """Зарегистрировать инструмент."""
        name = tool.name()
        if name in self._tools:
            self.logger.warning(f"Tool '{name}' already registered, overwriting")
        self._tools[name] = tool
        self.logger.info(f"Tool registered: {name}")
    
    def unregister(self, name: str):
        """Удалить инструмент."""
        if name in self._tools:
            del self._tools[name]
    
    def get(self, name: str) -> Optional[Tool]:
        """Получить инструмент по имени."""
        return self._tools.get(name)
    
    def list_tools(self) -> List[str]:
        """Список имён инструментов."""
        return list(self._tools.keys())
    
    async def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Вызвать инструмент по имени."""
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult.error(f"Unknown tool: {tool_name}")
        
        start = time.time()
        try:
            result = await tool.execute(args)
            result.execution_time = time.time() - start
            return result
        except Exception as e:
            self.logger.error(f"Tool '{tool_name}' error: {e}")
            return ToolResult.error(f"Tool error: {e}")
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """
        JSON-схемы всех инструментов (для LLM function calling).
        
        Формат совместим с OpenAI tools API.
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters(),
                }
            })
        return schemas


# ═══════════════════════════════════════════
# Shell Tool (из PicoClaw shell.go)
# ═══════════════════════════════════════════

class ShellTool(Tool):
    """
    Безопасное выполнение shell-команд.
    
    Из PicoClaw:
      - Таймаут (default 30s)
      - Whitelisted директории
      - Блокировка опасных команд
    """
    
    BLOCKED_COMMANDS = {
        'rm -rf /', 'format', 'del /s /q',
        'mkfs', 'dd if=', ':(){:|:&};:',
        'shutdown', 'reboot', 'halt',
    }
    
    def __init__(self, workspace: str = ".", timeout: int = 30,
                 allowed_dirs: Optional[List[str]] = None):
        self.workspace = os.path.abspath(workspace)
        self.timeout = timeout
        self.allowed_dirs = allowed_dirs or [self.workspace]
    
    def name(self) -> str:
        return "shell"
    
    def description(self) -> str:
        return (
            "Execute a shell command. Commands run in the workspace directory. "
            "Dangerous commands (rm -rf /, format, etc.) are blocked. "
            f"Timeout: {self.timeout}s."
        )
    
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": f"Working directory (default: {self.workspace})",
                },
            },
            "required": ["command"],
        }
    
    def _is_blocked(self, command: str) -> bool:
        """Проверка на опасные команды."""
        cmd_lower = command.lower().strip()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return True
        return False
    
    def _validate_cwd(self, cwd: str) -> bool:
        """Проверка что директория в разрешённом списке."""
        abs_cwd = os.path.abspath(cwd)
        return any(abs_cwd.startswith(os.path.abspath(d)) for d in self.allowed_dirs)
    
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        cwd = args.get("cwd", self.workspace)
        
        if not command:
            return ToolResult.error("command is required")
        
        if self._is_blocked(command):
            return ToolResult.error(f"Blocked dangerous command: {command}")
        
        if not self._validate_cwd(cwd):
            return ToolResult.error(f"Directory '{cwd}' is outside allowed paths")
        
        try:
            is_windows = platform.system() == "Windows"
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                shell=True,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult.error(f"Command timed out after {self.timeout}s")
            
            output = stdout.decode('utf-8', errors='replace')
            error = stderr.decode('utf-8', errors='replace')
            
            if proc.returncode != 0:
                return ToolResult(
                    output=f"Exit code {proc.returncode}\n{error}\n{output}".strip(),
                    llm_output=f"Command failed (exit {proc.returncode}): {error}",
                    is_error=True,
                )
            
            combined = output
            if error:
                combined += f"\n[stderr] {error}"
            
            return ToolResult.success(combined[:2000])  # Limit output
            
        except Exception as e:
            return ToolResult.error(f"Shell error: {e}")


# ═══════════════════════════════════════════
# File Search Tool (из OpenClaw)
# ═══════════════════════════════════════════

class FileSearchTool(Tool):
    """Поиск файлов по имени и содержимому."""
    
    def __init__(self, workspace: str = "."):
        self.workspace = os.path.abspath(workspace)
    
    def name(self) -> str:
        return "file_search"
    
    def description(self) -> str:
        return "Search for files by name pattern or content in the workspace."
    
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Filename pattern (glob, e.g. '*.py')",
                },
                "content": {
                    "type": "string",
                    "description": "Search for this text inside files",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                },
            },
        }
    
    async def execute(self, args: Dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern", "")
        content = args.get("content", "")
        max_results = args.get("max_results", 10)
        
        if not pattern and not content:
            return ToolResult.error("Either 'pattern' or 'content' is required")
        
        results = []
        workspace = Path(self.workspace)
        
        try:
            if pattern:
                # Поиск по имени
                for f in workspace.rglob(pattern):
                    if f.is_file() and not any(p.startswith('.') for p in f.parts):
                        rel = f.relative_to(workspace)
                        size_kb = f.stat().st_size / 1024
                        results.append(f"{rel} ({size_kb:.1f} KB)")
                        if len(results) >= max_results:
                            break
            
            if content:
                # Поиск по содержимому
                for f in workspace.rglob("*"):
                    if not f.is_file() or f.stat().st_size > 1_000_000:
                        continue
                    if any(p.startswith('.') for p in f.parts):
                        continue
                    try:
                        text = f.read_text(encoding='utf-8', errors='ignore')
                        if content.lower() in text.lower():
                            rel = f.relative_to(workspace)
                            # Найти строку с совпадением
                            for i, line in enumerate(text.split('\n'), 1):
                                if content.lower() in line.lower():
                                    results.append(f"{rel}:{i}: {line.strip()[:100]}")
                                    break
                            if len(results) >= max_results:
                                break
                    except Exception:
                        pass
            
            if not results:
                return ToolResult.success("No results found")
            
            return ToolResult.success("\n".join(results))
            
        except Exception as e:
            return ToolResult.error(f"Search error: {e}")


# ═══════════════════════════════════════════
# Cron Scheduler (из PicoClaw cron.go)
# ═══════════════════════════════════════════

class CronTask:
    """Периодическая задача."""
    def __init__(self, name: str, interval_seconds: int, 
                 callback: Callable, enabled: bool = True):
        self.name = name
        self.interval = interval_seconds
        self.callback = callback
        self.enabled = enabled
        self.last_run: float = 0.0
        self.run_count: int = 0
        self.last_error: Optional[str] = None


class CronScheduler:
    """
    Планировщик периодических задач (из PicoClaw cron.go).
    
    Задачи:
      - Авто-бэкап моделей на Drive
      - Синхронизация LEANN памяти
      - Очистка старых логов
      - Health check GPU
    """
    
    def __init__(self):
        self.tasks: Dict[str, CronTask] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger("Tars.Cron")
    
    def register(self, name: str, interval_seconds: int, 
                 callback: Callable, enabled: bool = True):
        """Зарегистрировать периодическую задачу."""
        self.tasks[name] = CronTask(name, interval_seconds, callback, enabled)
        self.logger.info(f"Cron: registered '{name}' (every {interval_seconds}s)")
    
    def unregister(self, name: str):
        if name in self.tasks:
            del self.tasks[name]
    
    async def _run_loop(self):
        """Основной цикл проверки задач."""
        while self._running:
            now = time.time()
            for task in self.tasks.values():
                if not task.enabled:
                    continue
                if now - task.last_run >= task.interval:
                    try:
                        if asyncio.iscoroutinefunction(task.callback):
                            await task.callback()
                        else:
                            task.callback()
                        task.last_run = now
                        task.run_count += 1
                        task.last_error = None
                    except Exception as e:
                        task.last_error = str(e)
                        self.logger.error(f"Cron '{task.name}' error: {e}")
            await asyncio.sleep(1)
    
    def start(self):
        """Запустить планировщик."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run_loop())
        self.logger.info("Cron scheduler started")
    
    def stop(self):
        """Остановить планировщик."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        self.logger.info("Cron scheduler stopped")
    
    def status(self) -> List[Dict[str, Any]]:
        """Статус всех задач."""
        return [{
            "name": t.name,
            "interval": t.interval,
            "enabled": t.enabled,
            "run_count": t.run_count,
            "last_error": t.last_error,
            "last_run": time.strftime("%H:%M:%S", time.localtime(t.last_run)) if t.last_run else "never",
        } for t in self.tasks.values()]


# ═══════════════════════════════════════════
# Query Expansion (из OpenClaw query-expansion.ts)
# ═══════════════════════════════════════════

# Синонимы для расширения запросов (русский + английский)
_SYNONYM_MAP = {
    # Русский
    "привет": ["здравствуйте", "добрый день", "хай"],
    "модель": ["нейросеть", "сеть", "model"],
    "обучение": ["тренировка", "training", "train"],
    "помощь": ["помоги", "подскажи", "help"],
    "код": ["программа", "скрипт", "code"],
    "ошибка": ["баг", "bug", "error", "проблема"],
    "память": ["memory", "ram", "воспоминания"],
    "поиск": ["найти", "search", "искать"],
    "файл": ["file", "документ"],
    "python": ["питон", "py"],
    # English
    "help": ["assist", "support"],
    "error": ["bug", "issue", "problem"],
    "code": ["script", "program", "source"],
    "model": ["neural network", "weights"],
    "search": ["find", "query", "lookup"],
    "memory": ["ram", "storage", "cache"],
    "train": ["training", "learn", "fine-tune"],
}


def expand_query(query: str, max_expansions: int = 3) -> str:
    """
    Расширение поискового запроса синонимами.
    
    Из OpenClaw query-expansion.ts:
    Добавляет связанные термины для лучшего recall.
    
    Пример: "ошибка в модели" → "ошибка баг bug error в модели нейросеть"
    """
    import re
    tokens = re.findall(r'[a-zа-яё0-9_]+', query.lower())
    
    expanded = list(tokens)
    added = 0
    
    for token in tokens:
        if token in _SYNONYM_MAP and added < max_expansions:
            # Добавляем 1-2 синонима
            syns = _SYNONYM_MAP[token][:2]
            expanded.extend(syns)
            added += len(syns)
    
    return " ".join(expanded)


# ═══════════════════════════════════════════
# Convenience: создать реестр с базовыми инструментами
# ═══════════════════════════════════════════

def create_default_registry(workspace: str = ".") -> ToolRegistry:
    """Создать реестр с базовыми инструментами ТАРС."""
    registry = ToolRegistry()
    registry.register(ShellTool(workspace))
    registry.register(FileSearchTool(workspace))
    
    # Document tools (PDF, Word, Excel)
    try:
        from tools.document_tools import (
            DocumentReadTool, DocumentWriteTool, SpreadsheetTool
        )
        registry.register(DocumentReadTool())
        registry.register(DocumentWriteTool())
        registry.register(SpreadsheetTool())
    except ImportError as e:
        logger.debug(f"Document tools not available: {e}")
    
    return registry
