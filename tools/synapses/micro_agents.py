"""
═══════════════════════════════════════════════════════════════
  ТАРС Micro-Agents (Синапсы) — Параллельные мини-исполнители
═══════════════════════════════════════════════════════════════

Синапсы = тупые и быстрые исполнители. Без LoRA, без рассуждений.
Каждый синапс делает ОДНО действие и возвращает результат.

5 типов:
  SearchSynapse   — поиск в LEANN + Web
  FileSynapse     — чтение/поиск файлов
  ExecSynapse     — выполнение команд
  MemorySynapse   — запись в LEANN
  AnalyzeSynapse  — NLP-анализ текста

Оркестратор SynapsePool запускает нужные синапсы параллельно
и возвращает объединённые результаты.
"""

import asyncio
import time
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("Tars.Synapses")

ROOT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════
# Result
# ═══════════════════════════════════════════

@dataclass
class SynapseResult:
    """Результат работы одного синапса."""
    synapse: str
    success: bool
    data: Any = None
    error: str = ""
    duration: float = 0.0

    @property
    def summary(self) -> str:
        if not self.success:
            return f"[{self.synapse}] ❌ {self.error}"
        if isinstance(self.data, str):
            return f"[{self.synapse}] {self.data[:200]}"
        if isinstance(self.data, list):
            return f"[{self.synapse}] {len(self.data)} items"
        if isinstance(self.data, dict):
            return f"[{self.synapse}] {list(self.data.keys())}"
        return f"[{self.synapse}] ✅"


# ═══════════════════════════════════════════
# Base Synapse
# ═══════════════════════════════════════════

class Synapse:
    """Базовый синапс — один тип действия, без retry (быстрый)."""
    
    def __init__(self, name: str):
        self.name = name
        self.calls = 0
        self.total_time = 0.0
    
    async def fire(self, task: dict) -> SynapseResult:
        """Выстрелить синапс — выполнить задачу."""
        t0 = time.time()
        try:
            result = await self._execute(task)
            dur = time.time() - t0
            self.calls += 1
            self.total_time += dur
            return SynapseResult(self.name, True, data=result, duration=dur)
        except Exception as e:
            dur = time.time() - t0
            logger.warning(f"[{self.name}] Error: {e}")
            return SynapseResult(self.name, False, error=str(e), duration=dur)
    
    async def _execute(self, task: dict) -> Any:
        raise NotImplementedError


# ═══════════════════════════════════════════
# 5 Synapse Types
# ═══════════════════════════════════════════

class SearchSynapse(Synapse):
    """🔍 Поиск в LEANN + Web."""
    
    def __init__(self, leann=None):
        super().__init__("search")
        self._leann = leann
    
    async def _execute(self, task: dict) -> Any:
        query = task.get("query", "")
        source = task.get("source", "all")  # "leann", "web", "all"
        results = []
        
        # LEANN search
        if source in ("leann", "all") and self._leann:
            try:
                docs = self._leann.search(query, top_k=5)
                if docs:
                    results.extend([{"source": "leann", "text": d} for d in docs])
            except Exception as e:
                logger.debug(f"LEANN search error: {e}")
        
        # Web search
        if source in ("web", "all"):
            try:
                from tools.web_search import search_duckduckgo, fetch_page_text
                web_results = await search_duckduckgo(query, max_results=3)
                for r in web_results[:3]:
                    text = await fetch_page_text(r.url, max_chars=2000)
                    if text and len(text) > 100:
                        results.append({
                            "source": "web",
                            "url": r.url,
                            "title": r.title,
                            "text": text[:2000],
                        })
            except Exception as e:
                logger.debug(f"Web search error: {e}")
        
        return results


class FileSynapse(Synapse):
    """📂 Чтение и поиск файлов."""
    
    def __init__(self):
        super().__init__("file")
    
    async def _execute(self, task: dict) -> Any:
        action = task.get("action", "search")  # "search", "read", "list"
        target = task.get("target", "")
        
        if action == "read":
            path = Path(target)
            # ═══ Path traversal guard: only allow files under ROOT ═══
            try:
                path.resolve().relative_to(ROOT)
            except ValueError:
                return f"Blocked: {target} (only project files allowed)"
            if path.exists() and path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                return text[:5000]
            return f"File not found: {target}"
        
        elif action == "list":
            path = Path(target) if target else ROOT
            if path.exists() and path.is_dir():
                items = []
                for f in sorted(path.iterdir())[:50]:
                    if f.name.startswith(".") or f.name == "__pycache__":
                        continue
                    kind = "dir" if f.is_dir() else "file"
                    items.append(f"{kind}: {f.name}")
                return items
            return f"Directory not found: {target}"
        
        else:  # search
            query = target.lower()
            found = []
            for f in ROOT.rglob("*"):
                if f.is_file() and query in f.name.lower():
                    if "__pycache__" not in str(f) and ".git" not in str(f) and "venv" not in str(f):
                        found.append(str(f.relative_to(ROOT)))
                        if len(found) >= 10:
                            break
            return found


class ExecSynapse(Synapse):
    """🛠 Выполнение безопасных команд."""
    
    ALLOWED = {"python", "pip", "git", "echo", "dir", "ls", "cat", "type", "where", "which"}
    
    def __init__(self):
        super().__init__("exec")
    
    async def _execute(self, task: dict) -> Any:
        cmd = task.get("command", "")
        if not cmd:
            return "Empty command"
        
        # Security: parse into argv (no shell interpretation)
        import shlex
        try:
            argv = shlex.split(cmd)
        except ValueError as e:
            return f"Invalid command syntax: {e}"
        
        if not argv:
            return "Empty command"
        
        first_word = argv[0].lower()
        if first_word not in self.ALLOWED:
            return f"Blocked: '{first_word}' not in allowed commands"
        
        try:
            # ═══ Use exec (not shell) to prevent injection via &&/;/| ═══
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(ROOT),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode("utf-8", errors="replace")
            if stderr:
                output += "\n" + stderr.decode("utf-8", errors="replace")
            return output[:3000]
        except asyncio.TimeoutError:
            return "Command timed out (30s)"
        except Exception as e:
            return f"Exec error: {e}"


class MemorySynapse(Synapse):
    """🧠 Запись/чтение LEANN памяти."""
    
    def __init__(self, leann=None):
        super().__init__("memory")
        self._leann = leann
    
    async def _execute(self, task: dict) -> Any:
        action = task.get("action", "save")  # "save", "search"
        text = task.get("text", "")
        
        if not self._leann:
            return "LEANN not available"
        
        if action == "save" and text:
            self._leann.add_document(text)
            return f"Saved to LEANN ({len(text)} chars)"
        
        elif action == "search" and text:
            results = self._leann.search(text, top_k=5)
            return results or []
        
        return "No action taken"


class AnalyzeSynapse(Synapse):
    """📊 NLP-анализ текста."""
    
    def __init__(self):
        super().__init__("analyze")
    
    async def _execute(self, task: dict) -> Any:
        texts = task.get("texts", [])
        topic = task.get("topic", "")
        
        instructions = []
        commands = []
        facts = []
        
        for text in texts:
            if not text or len(text) < 50:
                continue
            
            # Инструкции
            for m in re.findall(r'(?:шаг|step)\s*\d+[.:]\s*(.+)', text, re.I | re.M):
                m = m.strip()
                if 20 < len(m) < 300 and m not in instructions:
                    instructions.append(m)
            
            # Команды
            for m in re.findall(r'`([^`]{3,100})`', text):
                m = m.strip()
                if m not in commands:
                    commands.append(m)
            
            # Факты по теме
            if topic:
                topic_words = set(re.findall(r'[a-zа-яё]{3,}', topic.lower()))
                for sent in re.split(r'[.!?\n]', text):
                    sent = sent.strip()
                    if 30 < len(sent) < 300:
                        words = set(re.findall(r'[a-zа-яё]{3,}', sent.lower()))
                        if len(topic_words & words) >= 1 and sent not in facts:
                            facts.append(sent)
        
        return {
            "instructions": instructions[:20],
            "commands": commands[:15],
            "facts": facts[:30],
        }


# ═══════════════════════════════════════════
# SynapsePool — Parallel Dispatch
# ═══════════════════════════════════════════

class SynapsePool:
    """
    Пул синапсов — параллельный запуск мини-исполнителей.
    
    Использование:
        pool = SynapsePool(leann=memory)
        results = await pool.fire_many([
            ("search", {"query": "quantum physics"}),
            ("file",   {"action": "search", "target": "config"}),
            ("memory", {"action": "search", "text": "квантовые"}),
        ])
    """
    
    def __init__(self, leann=None):
        self.leann = leann
        self.synapses: Dict[str, Synapse] = {
            "search":  SearchSynapse(leann),
            "file":    FileSynapse(),
            "exec":    ExecSynapse(),
            "memory":  MemorySynapse(leann),
            "analyze": AnalyzeSynapse(),
        }
    
    async def fire(self, synapse_name: str, task: dict) -> SynapseResult:
        """Запустить один синапс."""
        syn = self.synapses.get(synapse_name)
        if not syn:
            return SynapseResult(synapse_name, False, error=f"Unknown synapse: {synapse_name}")
        return await syn.fire(task)
    
    async def fire_many(self, tasks: List[tuple]) -> List[SynapseResult]:
        """
        Запустить несколько синапсов параллельно.
        
        Args:
            tasks: list of (synapse_name, task_dict)
        Returns:
            list of SynapseResult
        """
        coros = [self.fire(name, task) for name, task in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)
        
        output = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                name = tasks[i][0] if i < len(tasks) else "?"
                output.append(SynapseResult(name, False, error=str(r)))
            else:
                output.append(r)
        return output
    
    async def fire_for_query(self, query: str) -> List[SynapseResult]:
        """
        Автоматически определить какие синапсы нужны и запустить.
        Используется Spine для автоматического dispatch.
        """
        tasks = []
        q_lower = query.lower()
        
        # Всегда ищем в LEANN
        if self.leann:
            tasks.append(("memory", {"action": "search", "text": query}))
        
        # Поиск в интернете для информационных запросов
        info_triggers = ["что", "как", "почему", "зачем", "объясни", "расскажи",
                         "what", "how", "why", "explain", "tell"]
        if any(t in q_lower for t in info_triggers):
            tasks.append(("search", {"query": query, "source": "web"}))
        
        # Файловые операции
        file_triggers = ["файл", "найди", "открой", "прочитай", "file", "find", "read", "open"]
        if any(t in q_lower for t in file_triggers):
            # Extract filename from query
            words = query.split()
            target = words[-1] if len(words) > 1 else ""
            tasks.append(("file", {"action": "search", "target": target}))
        
        # Выполнение команд
        exec_triggers = ["выполни", "запусти", "run", "execute", "pip", "python"]
        if any(t in q_lower for t in exec_triggers):
            # Extract command (after trigger word)
            for trigger in exec_triggers:
                if trigger in q_lower:
                    idx = q_lower.index(trigger)
                    cmd = query[idx:].strip()
                    tasks.append(("exec", {"command": cmd}))
                    break
        
        if not tasks:
            # Fallback — хотя бы LEANN search
            tasks.append(("memory", {"action": "search", "text": query}))
        
        return await self.fire_many(tasks)
    
    def summarize_results(self, results: List[SynapseResult]) -> str:
        """Сжать результаты синапсов в текстовый контекст для Brain."""
        parts = []
        for r in results:
            if not r.success:
                continue
            if isinstance(r.data, str) and r.data:
                parts.append(f"[{r.synapse}] {r.data[:500]}")
            elif isinstance(r.data, list):
                for item in r.data[:5]:
                    if isinstance(item, dict):
                        text = item.get("text", str(item))[:300]
                        parts.append(f"[{r.synapse}] {text}")
                    elif isinstance(item, str):
                        parts.append(f"[{r.synapse}] {item[:300]}")
            elif isinstance(r.data, dict):
                for k, v in r.data.items():
                    if isinstance(v, list) and v:
                        parts.append(f"[{r.synapse}/{k}] {'; '.join(str(x)[:100] for x in v[:3])}")
        
        return "\n".join(parts[:20]) if parts else ""
    
    def health_report(self) -> Dict[str, Any]:
        """Статус всех синапсов."""
        return {
            name: {
                "calls": syn.calls,
                "avg_ms": f"{(syn.total_time / max(syn.calls, 1)) * 1000:.0f}ms",
            }
            for name, syn in self.synapses.items()
        }

