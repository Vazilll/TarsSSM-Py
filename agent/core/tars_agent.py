"""
═══════════════════════════════════════════════════════════════════
  ТАРС v3 — Автономный ИИ-агент
═══════════════════════════════════════════════════════════════════

Полный цикл:
  Задача → Понимание → Поиск (LEANN + Web) → Планирование →
  → Выполнение (SubAgents) → Обучение → Ответ

Компоненты:

  ┌─────────────────────────────────────────────────────────┐
  │                    TarsAgent                            │
  │                                                         │
  │  ┌──────────┐  ┌──────────┐  ┌────────────┐           │
  │  │ Classifier│  │ Planner  │  │ Executor   │           │
  │  │ (intent) │→ │ (what?)  │→ │ (do it!)   │           │
  │  └──────────┘  └──────────┘  └─────┬──────┘           │
  │                                     │                   │
  │  ┌──────────────────────────────────┼──────────────┐   │
  │  │         SubAgent Pool (parallel)  ↓              │   │
  │  │  ┌─────────┐ ┌─────────┐ ┌──────────┐          │   │
  │  │  │ Search  │ │ Fetch   │ │ Analyze  │          │   │
  │  │  │ Agent   │ │ Agent   │ │ Agent    │          │   │
  │  │  └─────────┘ └─────────┘ └──────────┘          │   │
  │  │  ┌─────────┐ ┌─────────┐ ┌──────────┐          │   │
  │  │  │ Memory  │ │ Shell   │ │ Skill    │          │   │
  │  │  │ Agent   │ │ Agent   │ │ Agent    │          │   │
  │  │  └─────────┘ └─────────┘ └──────────┘          │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  ┌──────────┐  ┌──────────┐  ┌────────────┐           │
  │  │  LEANN   │  │  Skills  │  │  Telegram  │           │
  │  │  Memory  │  │  Store   │  │  Bot       │           │
  │  └──────────┘  └──────────┘  └────────────┘           │
  └─────────────────────────────────────────────────────────┘

Запуск:
  python tars_agent.py                    # CLI
  python tars_agent.py --telegram         # + Telegram бот
  python tars_agent.py --auto-learn       # + автообучение
"""

import os
import re
import sys
import time
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # project root (agent/ → TarsSSM-Py/)
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "agent"))

from memory.leann import LeannIndex
from tools import (
    ToolRegistry, ToolResult, ShellTool, FileSearchTool,
    CronScheduler, create_default_registry, expand_query,
)
from tools.sub_agents import AgentPool
from tools.web_search import search_duckduckgo, fetch_page_text
from agent.skill_learn import SkillLearner, SkillRegistry, Skill
from agent.document_sense import DocumentSense
from brain.doubt_engine import SafetyGate, DoubtVerdict

logger = logging.getLogger("Tars.Agent")


# ═══════════════════════════════════════════════════════
# 1. Intent Classification — Понимание задачи
# ═══════════════════════════════════════════════════════

class Intent:
    LEARN      = "learn"        # Обучение навыку
    WEB_SEARCH = "web_search"   # Поиск в интернете
    SEARCH     = "search"       # Поиск в памяти
    EXECUTE    = "execute"      # Выполнить команду
    FILE_OP    = "file_op"      # Файловые операции
    DOC_WORK   = "doc_work"     # Работа с документами (PDF/Word/Excel)
    REMEMBER   = "remember"     # Запомнить
    ANALYZE    = "analyze"      # Анализ
    CODE       = "code"         # Код
    SKILL_USE  = "skill_use"    # Применить навык
    STATUS     = "status"       # Статус системы
    CHAT       = "chat"         # Общение

_PATTERNS = {
    Intent.LEARN: [
        r'научись|learn|изучи|обучись|освой|выучи|научи\w+|обучи\w+|study|master',
    ],
    Intent.WEB_SEARCH: [
        r'в интернет|в сети|online|web|загугли|google|найди в инет|поищи в инет|из интернета',
    ],
    Intent.SEARCH: [
        r'найди|поиск|search|find|где|what is|что такое|расскаж|покажи|show|сколько',
    ],
    Intent.EXECUTE: [
        r'выполни|запусти|run|execute|создай|сделай|install|устано|скачай|download|удали|delete',
    ],
    Intent.FILE_OP: [
        r'файл|file|папк|folder|директори|directory|открой|скопируй|copy',
    ],
    Intent.DOC_WORK: [
        r'pdf|docx?|xlsx?|word|excel|ворд|эксел',
        r'прочитай|прочти|read|прочесть',
        r'спецификац|specification|spec\b',
        r'документ[аеыи]|document|отчёт|отчет|report',
        r'таблиц[уае]|table|spreadsheet',
        r'ТЗ|техническое\s*задание|контракт|договор',
        r'заполни|дополни|append|напиши\s+в\s+\w+\.docx',
        r'создай\s+(?:таблиц|excel|xlsx|word|docx|документ)',
    ],
    Intent.REMEMBER: [
        r'запомни|remember|не забудь|заметка|note|сохрани в памят',
    ],
    Intent.ANALYZE: [
        r'проанализируй|analyze|сравни|compare|оцени|evaluate|почему|why|как работает',
    ],
    Intent.CODE: [
        r'код|code|скрипт|script|функци|function|класс|class|python|программ',
    ],
    Intent.SKILL_USE: [
        r'используй навык|use skill|примени|apply|как.*навык',
    ],
    Intent.STATUS: [
        r'статус|status|здоровье|health|сколько.*памят|диагностик',
    ],
}


def classify_intent(query: str) -> str:
    """
    Классификация намерения с weighted scoring и tiebreaking.
    
    Улучшения:
      - Приоритеты: DOC_WORK > EXECUTE > CODE > FILE_OP > WEB_SEARCH > ...
      - Position bonus: ранние совпадения важнее
      - Fallback: CHAT если нет совпадений
    """
    q = query.lower()
    
    # Приоритеты для tiebreaking (выше = приоритетнее при равном score)
    PRIORITY = {
        Intent.DOC_WORK:    12,
        Intent.EXECUTE:     10,
        Intent.CODE:         9,
        Intent.FILE_OP:      8,
        Intent.WEB_SEARCH:   7,
        Intent.ANALYZE:      6,
        Intent.LEARN:        5,
        Intent.SEARCH:       4,
        Intent.REMEMBER:     3,
        Intent.SKILL_USE:    2,
        Intent.STATUS:       1,
        Intent.CHAT:         0,
    }
    
    scores = {}
    for intent, patterns in _PATTERNS.items():
        score = 0.0
        for p in patterns:
            for m in re.finditer(p, q):
                # Position bonus: совпадение в первых 30 символах = ×1.5
                pos_bonus = 1.5 if m.start() < 30 else 1.0
                score += pos_bonus
        scores[intent] = score
    
    # Tiebreaking: при равном score выбираем по приоритету
    best = max(scores, key=lambda k: (scores[k], PRIORITY.get(k, 0)))
    return best if scores[best] > 0 else Intent.CHAT


# ═══════════════════════════════════════════════════════
# 2. Execution Steps — Отслеживание шагов
# ═══════════════════════════════════════════════════════

@dataclass
class Step:
    """Один шаг работы агента."""
    action: str         # classify, search, web, learn, tool, think, respond, memory, skill
    detail: str
    result: str = ""
    duration: float = 0.0

    def __str__(self):
        icons = {"classify": "🏷", "search": "🔍", "web": "🌐", "learn": "🎓",
                 "tool": "🔧", "think": "💭", "respond": "💬", "memory": "🧠",
                 "skill": "🎯", "fetch": "📥", "analyze": "📊"}
        icon = icons.get(self.action, "▶")
        t = f" [{self.duration:.1f}s]" if self.duration > 0.01 else ""
        return f"  {icon} {self.detail}{t}" + (f"\n     → {self.result[:120]}" if self.result else "")


@dataclass
class AgentResult:
    query: str
    intent: str
    answer: str
    steps: List[Step] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    total_time: float = 0.0
    task_spec: Any = None  # TaskSpec если было глубокое мышление


def _auto_generate_tz(query: str) -> list:
    """
    Авто-генерация пунктов ТЗ из текста запроса.
    
    Улучшения v2:
      - Извлекает сущности (файлы, URL, код)
      - Генерирует измеримые пункты на основе intent + entities
      - Контекстно-зависимые проверки
    """
    import re
    points = []
    q_lower = query.lower()
    
    # ═══ Извлечение сущностей ═══
    files = re.findall(r'[\w/\\.-]+\.(?:py|txt|pdf|docx?|xlsx?|json|csv|md|html)', query)
    urls = re.findall(r'https?://\S+', query)
    code_keywords = re.findall(r'\b(?:функци\w*|class|def|return|import|модуль|метод)\b', q_lower)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
    quoted = re.findall(r'"([^"]+)"|«([^»]+)»', query)
    quoted_flat = [q[0] or q[1] for q in quoted]
    
    # ═══ Базовые пункты ═══
    points.append("Ответ непосредственно отвечает на запрос")
    
    if len(query) > 100:
        points.append("Все части многосоставного запроса обработаны")
    
    # ═══ Контекстные пункты ═══
    if files:
        for f in files[:3]:
            points.append(f"Файл '{f}' обработан корректно")
        if any(f.endswith(('.py', '.js', '.ts')) for f in files):
            points.append("Код из файлов синтаксически верен")
    
    if urls:
        points.append("Информация из URL валидна и актуальна")
    
    if code_keywords:
        points.append("Код синтаксически корректен")
        points.append("Обработаны граничные случаи (пустой вход, None, ошибки)")
        points.append("Код решает поставленную задачу")
    
    if any(w in q_lower for w in ["объясни", "расскажи", "как работает", "explain", "what is"]):
        points.append("Объяснение структурировано (вступление → суть → пример)")
    
    if any(w in q_lower for w in ["анализ", "сравни", "analyze", "compare"]):
        points.append("Анализ подкреплён конкретными данными")
        points.append("Есть финальное заключение")
    
    if any(w in q_lower for w in ["сделай", "создай", "удали", "запусти", "create", "make", "run"]):
        points.append("Действие выполнено и результат проверен")
    
    if numbers:
        points.append("Числовые данные в ответе точны")
    
    if quoted_flat:
        for q_text in quoted_flat[:2]:
            points.append(f"Учтено упоминание: «{q_text[:50]}»")
    
    # Гарантируем минимум 3 пункта
    if len(points) < 3:
        points.append("Ответ полный и без пропусков")
    
    return points


# ═══════════════════════════════════════════════════════
# 3. TarsAgent — Ядро системы
# ═══════════════════════════════════════════════════════

class TarsAgent:
    """
    ТАРС — автономный ИИ-агент.
    
    Цикл:
      1. Классификация намерения
      2. Поиск контекста (LEANN + навыки)
      3. Планирование (какие агенты нужны?)
      4. Параллельное выполнение (SubAgents)
      5. Формирование ответа
      6. Обучение (сохранение в память)
    """
    
    def __init__(self, workspace: str = ".",
                 model_path: str = "models/embeddings",
                 verbose: bool = True):
        self.workspace = os.path.abspath(workspace)
        self.verbose = verbose
        
        # ── Подсистемы ──
        self.memory = LeannIndex(model_path=model_path)
        self.tools = create_default_registry(workspace)
        self.skill_registry = SkillRegistry()
        self.skill_learner = SkillLearner(self.skill_registry, self.memory)
        self.pool = AgentPool(memory=self.memory)
        self.cron = CronScheduler()
        
        # ── DocumentSense — автономный обработчик документов ──
        self.doc_sense = DocumentSense(
            workspace=workspace,
            memory=self.memory,
        )
        self._log(f"DocumentSense: готов (workspace={workspace})")
        
        # ── Spine (Спинной мозг) ──
        self.spine = None
        try:
            from brain.rrn import RrnCore
            self.spine = RrnCore()
            self._log("Spine (RRN + MinGRU + Synapses) активирован")
        except Exception as e:
            self._log(f"Spine не доступен: {e}")
        
        # ── Состояние ──
        self.history: List[Dict[str, str]] = []
        self.max_history = 20
        self.total_tasks = 0
        
        # ── SafetyGate (DoubtEngine v4) ──
        self.safety_gate = SafetyGate()
        self._log("SafetyGate активирован")
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"  🤖 {msg}")
    
    # ─────────────────────────────────────────
    # Главный цикл
    # ─────────────────────────────────────────
    
    async def run(self, query: str) -> AgentResult:
        """Полный цикл обработки задачи (Нервная система ТАРС)."""
        # Input validation: cap query length to prevent abuse
        MAX_QUERY_LEN = 4096
        if len(query) > MAX_QUERY_LEN:
            query = query[:MAX_QUERY_LEN]
            logger.warning(f"Query truncated to {MAX_QUERY_LEN} chars")
        
        start = time.time()
        steps: List[Step] = []
        
        # 1. CLASSIFY
        intent = classify_intent(query)
        steps.append(Step("classify", f"Намерение: {intent}"))
        self._log(f"Задача: {query}")
        self._log(f"Intent: {intent}")
        
        # 1.5 SPINE — Спинной мозг (3 режима)
        spine_ctx = None
        if self.spine:
            try:
                spine_result = await self.spine.process(query)
                mode = spine_result["mode"]
                steps.append(Step("spine", f"Mode {mode} (conf={spine_result['confidence']:.0%})"))
                
                # Mode 1: Рефлекс — мгновенный ответ
                if mode == 1 and spine_result["response"]:
                    self._log(f"⚡ Рефлекс (mode 1, {spine_result['confidence']:.0%})")
                    self._update_history(query, spine_result["response"])
                    self.total_tasks += 1
                    return AgentResult(
                        query=query, intent=intent,
                        answer=spine_result["response"],
                        steps=steps, total_time=time.time() - start,
                    )
                
                # Mode 2: Действие — синапсы ответили
                if mode == 2 and spine_result["response"]:
                    self._log(f"🔧 Действие (mode 2, синапсы)")
                    self._update_history(query, spine_result["response"])
                    self.total_tasks += 1
                    return AgentResult(
                        query=query, intent=intent,
                        answer=spine_result["response"],
                        steps=steps, total_time=time.time() - start,
                    )
                
                # Mode 3: Глубокое мышление — данные для Brain
                if mode == 3 and spine_result["context"]:
                    self._log(f"🧠 Глубокий анализ (mode 3)")
                    spine_ctx = spine_result["context"]
                    steps.append(Step("synapse_data", f"Контекст: {len(spine_ctx)} chars"))
                    
                    # Auto-ТЗ: система генерирует ТЗ для самопроверки
                    try:
                        from brain.mamba2.critic import TaskSpec
                        tz_points = _auto_generate_tz(query)
                        task_spec = TaskSpec(query=query, points=tz_points, approved=True)
                        self._task_spec = task_spec  # Сохранить для передачи в Brain
                        steps.append(Step("task_spec", 
                            f"ТЗ: {len(tz_points)} пунктов — {', '.join(tz_points[:3])}..."))
                        self._log(f"📋 ТЗ ({len(tz_points)} пунктов): {tz_points[:2]}")
                    except Exception as e:
                        self._log(f"Auto-TZ error: {e}")
                        self._task_spec = None
            except Exception as e:
                self._log(f"Spine error: {e}")
        
        # 2. CONTEXT — поиск в LEANN + навыках
        context = await self._gather_context(query, intent, steps)
        
        # Добавить Spine context если есть (Mode 3)
        if spine_ctx:
            context.insert(0, spine_ctx)
        
        # 3. EXECUTE — диспетчер по intent
        handler = self._get_handler(intent)
        answer = await handler(query, context, steps)
        
        # 4. LEARN — сохранить в историю
        self._update_history(query, answer)
        self.total_tasks += 1
        
        result = AgentResult(
            query=query, intent=intent, answer=answer,
            steps=steps, context=context,
            total_time=time.time() - start,
        )
        
        self._log(f"Готово: {result.total_time:.1f}s, {len(steps)} шагов")
        return result
    
    async def _gather_context(self, query: str, intent: str,
                               steps: List[Step]) -> List[str]:
        """Собрать контекст из всех источников ПАРАЛЛЕЛЬНО."""
        t = time.time()
        context = []
        
        # ═══ Parallel context gathering (asyncio) ═══
        async def _search_leann():
            try:
                return self.memory.search(query, top_k=5) or []
            except Exception as e:
                self._log(f"LEANN search error: {e}")
                return []
        
        async def _search_skills():
            skill = self.skill_registry.get(query)
            if skill:
                return [f"[Skill:{skill.name}] {skill.description}"]
            return []
        
        async def _search_history():
            results = []
            for h in self.history[-4:]:
                if any(w in h.get("content", "").lower() for w in query.lower().split() if len(w) > 3):
                    results.append(f"[History] {h['content'][:200]}")
                    break
            return results
        
        async def _scan_documents():
            """Автоматическое обнаружение и чтение документов из запроса."""
            try:
                docs = await self.doc_sense.detect_and_read(query)
                if docs:
                    return self.doc_sense.format_context(docs, max_chars=3000)
            except Exception as e:
                logger.debug(f"DocumentSense error: {e}")
            return []
        
        # Запускаем ВСЁ параллельно (включая DocumentSense)
        leann_res, skill_res, hist_res, doc_res = await asyncio.gather(
            _search_leann(),
            _search_skills(),
            _search_history(),
            _scan_documents(),
        )
        
        # Документы идут первыми (наиболее релевантный контекст)
        context.extend(doc_res)
        context.extend(leann_res)
        context.extend(skill_res)
        context.extend(hist_res)
        
        dur = time.time() - t
        sources = []
        if doc_res: sources.append(f"{len(doc_res)} doc")
        if leann_res: sources.append(f"{len(leann_res)} leann")
        if skill_res: sources.append(f"{len(skill_res)} skill")
        if hist_res: sources.append(f"{len(hist_res)} hist")
        steps.append(Step("search",
            f"Контекст: {len(context)} источников ({', '.join(sources) or 'empty'})",
            duration=dur))
        return context
    
    def _get_handler(self, intent: str):
        """Получить обработчик для intent."""
        return {
            Intent.LEARN:      self._do_learn,
            Intent.WEB_SEARCH: self._do_web_search,
            Intent.SEARCH:     self._do_search,
            Intent.EXECUTE:    self._do_execute,
            Intent.FILE_OP:    self._do_file_op,
            Intent.DOC_WORK:   self._do_doc_work,
            Intent.REMEMBER:   self._do_remember,
            Intent.ANALYZE:    self._do_analyze,
            Intent.CODE:       self._do_code,
            Intent.SKILL_USE:  self._do_skill_use,
            Intent.STATUS:     self._do_status,
            Intent.CHAT:       self._do_chat,
        }.get(intent, self._do_chat)
    
    # ─────────────────────────────────────────
    # Обработчики
    # ─────────────────────────────────────────
    
    async def _do_learn(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Обучение навыку через суб-агенты."""
        topic = re.sub(r'^(научись|learn|изучи|обучись|освой|выучи|научи\w+)\s*', '',
                       query, flags=re.IGNORECASE).strip()
        if not topic:
            return "🤔 Чему научиться? Пример: 'научись Docker'"
        
        existing = self.skill_registry.get(topic)
        if existing and existing.confidence > 0.7:
            steps.append(Step("skill", f"Уже знаю: {topic} ({existing.confidence:.0%})"))
            return f"✅ Я уже знаю **{topic}**!\n\n{existing.summary()}"
        
        steps.append(Step("learn", f"Изучаю: {topic}"))
        t = time.time()
        
        # Глубокое исследование через суб-агенты
        research = await self.pool.deep_research(topic, depth=5, follow_links=2)
        
        # Собрать skill
        skill = Skill(
            name=topic,
            description=research["facts"][0][:200] if research.get("facts") else f"Навык: {topic}",
            category=self.skill_learner._categorize(topic),
            instructions=research.get("instructions", []),
            commands=[{"cmd": c, "desc": ""} for c in research.get("commands", [])],
            knowledge=research.get("facts", []),
            sources=research.get("sources", []),
            created=time.time(),
            updated=time.time(),
        )
        skill.confidence = min(1.0, (
            len(skill.instructions) * 0.08 +
            len(skill.commands) * 0.12 +
            len(skill.knowledge) * 0.04 +
            len(skill.sources) * 0.04
        ))
        skill.examples = self.skill_learner._generate_examples(topic, skill)
        self.skill_registry.add(skill)
        
        dur = time.time() - t
        steps.append(Step("learn", f"Навык создан: {skill.confidence:.0%}",
                         result=f"{len(skill.instructions)} instr, {len(skill.commands)} cmd",
                         duration=dur))
        
        answer = f"🎓 **Научился: {topic}!** ({dur:.1f}s)\n\n{skill.summary()}\n"
        if skill.instructions[:3]:
            answer += "\n📋 **Ключевые шаги:**\n"
            for i, inst in enumerate(skill.instructions[:3], 1):
                answer += f"  {i}. {inst}\n"
        if skill.commands[:3]:
            answer += "\n🔧 **Команды:**\n"
            for cmd in skill.commands[:3]:
                answer += f"  `{cmd['cmd']}`\n"
        
        return answer
    
    async def _do_web_search(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Поиск в интернете через суб-агенты."""
        q = re.sub(r'^(найди в интернет\w*|загугли|поищи в инет\w*|найди в сети)\s*',
                   '', query, flags=re.IGNORECASE).strip() or query
        
        t = time.time()
        steps.append(Step("web", f"Поиск: {q}"))
        
        research = await self.pool.research(q, depth=5)
        dur = time.time() - t
        
        if not research.get("sources"):
            steps.append(Step("web", "Ничего не найдено", duration=dur))
            return "🤷 Ничего не найдено в интернете"
        
        steps.append(Step("web", f"{len(research['sources'])} источников",
                         result=f"quality: {research.get('avg_quality', 0):.0%}",
                         duration=dur))
        
        parts = [f"🌐 **Результаты: {q}**\n"]
        for i, (src, title) in enumerate(
            zip(research["sources"][:5], research.get("titles", [])[:5]), 1
        ):
            parts.append(f"  {i}. **{title}**")
            parts.append(f"     {src}")
        
        if research.get("facts"):
            parts.append(f"\n📚 **Ключевые факты:**")
            for f in research["facts"][:5]:
                parts.append(f"  • {f[:150]}")
        
        parts.append(f"\n💾 +{research.get('memory_added', 0)} в LEANN")
        return "\n".join(parts)
    
    async def _do_search(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Поиск в LEANN памяти. Если нет — fallback в интернет."""
        if ctx:
            parts = ["📚 **Из памяти ТАРС:**\n"]
            for i, doc in enumerate(ctx[:5], 1):
                snippet = doc[:200] + "..." if len(doc) > 200 else doc
                parts.append(f"  {i}. {snippet}")
            return "\n".join(parts)
        
        # Fallback: поиск в интернете
        steps.append(Step("think", "В памяти пусто → ищу в интернете"))
        return await self._do_web_search(query, ctx, steps)
    
    async def _do_execute(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Выполнение команды."""
        cmd = self._extract_command(query)
        if not cmd:
            return "🤔 Не понял команду. Пример: `запусти 'pip list'`"
        
        # ═══ SafetyGate: pre-execute check (fail-closed) ═══
        try:
            verdict = self.safety_gate.check("shell", {"command": cmd})
            if verdict.is_blocked:
                steps.append(Step("safety", f"⛔ Заблокировано: {verdict.reason}"))
                self._log(f"SafetyGate BLOCK: {cmd} → {verdict.reason}")
                return f"⛔ **Действие заблокировано SafetyGate**: {verdict.reason}"
            if verdict.is_flagged:
                steps.append(Step("safety", f"⚠️ Подозрительно: {verdict.reason}"))
                logger.warning(f"SafetyGate FLAG: {cmd} → {verdict.reason}")
        except Exception as e:
            # Fail-closed: если SafetyGate упал — блокируем (для действий)
            steps.append(Step("safety", f"⛔ SafetyGate error (fail-closed): {e}"))
            return f"⛔ **Действие заблокировано** (ошибка проверки): {e}"
        
        # Проверить: есть ли навык для этой операции?
        related_skill = self._find_relevant_skill(query)
        if related_skill:
            steps.append(Step("skill", f"Навык: {related_skill.name}",
                             result=f"confidence: {related_skill.confidence:.0%}"))
        
        steps.append(Step("tool", f"Shell: {cmd}"))
        t = time.time()
        
        result = await self.tools.execute("shell", {
            "command": cmd, "cwd": self.workspace
        })
        dur = time.time() - t
        steps.append(Step("tool", f"Завершено ({dur:.1f}s)",
                         result=result.output[:100]))
        
        # Сохранить в память
        self.memory.add_document(f"[Exec] {cmd} → {result.output[:200]}")
        
        if result.is_error:
            return f"❌ **Ошибка:**\n```\n{result.output[:1000]}\n```"
        return f"✅ **Выполнено:**\n```\n{result.output[:1500]}\n```"
    
    async def _do_file_op(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Файловые операции."""
        # Если упоминаются документы → перенаправить в doc_work
        if self.doc_sense.detector.has_document_intent(query):
            return await self._do_doc_work(query, ctx, steps)
        
        pattern = self._extract_file_pattern(query)
        content_q = self._extract_quoted(query)
        
        args = {}
        if pattern:
            args["pattern"] = pattern
        if content_q:
            args["content"] = content_q
        if not args:
            args["pattern"] = "*"
        
        steps.append(Step("tool", f"FileSearch: {args}"))
        result = await self.tools.execute("file_search", args)
        
        return f"📁 **Файлы:**\n```\n{result.output[:2000]}\n```"
    
    async def _do_doc_work(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Автономная работа с документами (PDF/Word/Excel)."""
        t = time.time()
        
        # 1. Найти и прочитать документы
        docs = await self.doc_sense.detect_and_read(query)
        
        if docs:
            parts = []
            for doc in docs:
                if doc.error:
                    parts.append(f"⚠ **{doc.name}**: {doc.error}")
                else:
                    parts.append(f"📄 **{doc.name}** ({doc.format}, {doc.size_bytes/1024:.0f} KB):")
                    text_preview = doc.text[:2000]
                    if len(doc.text) > 2000:
                        text_preview += f"\n... [{len(doc.text)} символов всего]"
                    parts.append(text_preview)
                    if doc.chunks_ingested > 0:
                        parts.append(f"💾 +{doc.chunks_ingested} чанков → LEANN")
            
            dur = time.time() - t
            steps.append(Step("tool", f"DocSense: {len(docs)} файлов", duration=dur))
            return "\n\n".join(parts)
        
        # 2. Если нет конкретных файлов — показать что есть в workspace
        steps.append(Step("tool", "DocSense: поиск документов в workspace"))
        self.doc_sense.detector._build_file_index()
        available = list(self.doc_sense.detector._file_index.values())[:20]
        
        if available:
            parts = ["📁 **Документы в workspace:**\n"]
            for p in available:
                path = Path(p)
                size_kb = path.stat().st_size / 1024
                parts.append(f"  • {path.name} ({path.suffix}, {size_kb:.0f} KB)")
            parts.append(f"\nСкажи: 'прочитай {Path(available[0]).name}'")
            return "\n".join(parts)
        
        return "📁 Документов (PDF/Word/Excel) в workspace не найдено."
    
    async def _do_remember(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Сохранение в память."""
        info = re.sub(r'^(запомни|remember|не забудь|заметка|сохрани)[:\s]*',
                      '', query, flags=re.IGNORECASE).strip()
        if not info:
            return "🤔 Что запомнить?"
        
        self.memory.add_document(info)
        steps.append(Step("memory", f"Сохранено: {info[:80]}"))
        return f"✅ Запомнил: {info}"
    
    async def _do_analyze(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Анализ: контекст из памяти + интернет."""
        t = time.time()
        
        # Собрать больше контекста
        if len(ctx) < 3:
            steps.append(Step("think", "Мало контекста → ищу в интернете"))
            research = await self.pool.research(query, depth=3)
            if research.get("facts"):
                ctx.extend(research["facts"][:5])
        
        dur = time.time() - t
        
        parts = ["🔍 **Анализ:**\n"]
        if ctx:
            for i, doc in enumerate(ctx[:5], 1):
                snippet = doc[:250] + "..." if len(doc) > 250 else doc
                parts.append(f"  📄 [{i}] {snippet}\n")
            parts.append(f"\n💡 На основе {len(ctx)} источников")
        else:
            parts.append("Нет данных для анализа")
        
        steps.append(Step("analyze", f"{len(ctx)} источников", duration=dur))
        return "\n".join(parts)
    
    async def _do_code(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Выполнение кода."""
        code = self._extract_code(query)
        if not code:
            return await self._do_search(query, ctx, steps)
        
        # ═══ SafetyGate: check code before execution (fail-closed) ═══
        try:
            verdict = self.safety_gate.check("execute_script", {"code": code})
            if verdict.is_blocked:
                steps.append(Step("safety", f"⛔ Код заблокирован: {verdict.reason}"))
                return f"⛔ **Код заблокирован SafetyGate**: {verdict.reason}"
        except Exception as e:
            steps.append(Step("safety", f"⛔ SafetyGate error: {e}"))
            return f"⛔ **Код заблокирован** (ошибка проверки): {e}"
        
        steps.append(Step("tool", f"Python: {code[:60]}"))
        
        # Security: write code to temp file instead of embedding in shell string
        # This prevents injection via quote escaping
        import tempfile
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', dir=self.workspace,
                delete=False, encoding='utf-8'
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            
            result = await self.tools.execute("shell", {
                "command": f'python "{tmp_path}"', "cwd": self.workspace
            })
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        
        if result.is_error:
            return f"❌ Ошибка:\n```\n{result.output}\n```"
        return f"✅ Результат:\n```\n{result.output[:1000]}\n```"
    
    async def _do_skill_use(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Применить навык."""
        skill = self._find_relevant_skill(query)
        if not skill:
            skills = self.skill_registry.list_all()
            if skills:
                names = ", ".join(s.name for s in skills[:10])
                return f"🤷 Не нашёл навык. Доступные: {names}"
            return "🤷 У меня пока нет навыков. Скажи: 'научись Docker'"
        
        skill.use_count += 1
        self.skill_registry.add(skill)
        steps.append(Step("skill", f"Применяю: {skill.name}"))
        
        parts = [f"🎯 **{skill.name}** (use #{skill.use_count})\n"]
        if skill.instructions:
            parts.append("📋 **Инструкции:**")
            for i, inst in enumerate(skill.instructions[:5], 1):
                parts.append(f"  {i}. {inst}")
        if skill.commands:
            parts.append("\n🔧 **Команды:**")
            for cmd in skill.commands[:5]:
                parts.append(f"  `{cmd['cmd']}`")
        
        return "\n".join(parts)
    
    async def _do_status(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Статус системы."""
        health = self.pool.health_report()
        skills = self.skill_registry.list_all()
        doc_status = self.doc_sense.get_status()
        
        parts = [
            "📊 **ТАРС Status**\n",
            f"  🧠 LEANN: {len(self.memory.texts)} документов",
            f"  📄 DocSense: {doc_status['indexed_files']} файлов, {doc_status['cached_docs']} кеш",
            f"  🎯 Skills: {len(skills)}",
            f"  📝 History: {len(self.history)} сообщений",
            f"  🔧 Tools: {', '.join(self.tools.list_tools())}",
            f"  📋 Tasks done: {self.total_tasks}",
            "\n🏥 **SubAgents Health:**",
        ]
        for name, info in health["agents"].items():
            parts.append(f"  {name}: {info['health']} ({info['done']} done, avg {info['avg_time']})")
        parts.append(f"\n📦 **Cache:** hit-rate {health['cache']['hit_rate']}")
        
        if skills:
            parts.append("\n🎯 **Навыки:**")
            for s in skills[:5]:
                parts.append(f"  • {s.name} ({s.category}) — {s.confidence:.0%}")
        
        return "\n".join(parts)
    
    async def _do_chat(self, query: str, ctx: List[str], steps: List[Step]) -> str:
        """Общение с контекстом."""
        if ctx:
            parts = [f"На основе {len(ctx)} документов:\n"]
            for doc in ctx[:2]:
                snippet = doc[:200] + "..." if len(doc) > 200 else doc
                parts.append(f"📄 {snippet}")
            return "\n".join(parts)
        return "Привет! Я ТАРС. Что мне сделать?\n  • научись Docker\n  • найди в интернете...\n  • запусти 'pip list'\n  • /status"
    
    # ─────────────────────────────────────────
    # Утилиты
    # ─────────────────────────────────────────
    
    def _extract_command(self, query: str) -> Optional[str]:
        m = re.search(r'[`"\'](.*?)[`"\']', query)
        if m: return m.group(1)
        m = re.search(r'(?:выполни|запусти|run|execute)\s+(.+)', query, re.I)
        if m: return m.group(1).strip()
        return None
    
    def _extract_file_pattern(self, query: str) -> Optional[str]:
        m = re.search(r'(\*\.\w+)', query)
        if m: return m.group(1)
        if re.search(r'python\s*(файл|file)', query, re.I): return "*.py"
        if re.search(r'текстов\w*\s*(файл|file)', query, re.I): return "*.txt"
        return None
    
    def _extract_quoted(self, query: str) -> Optional[str]:
        m = re.search(r'[\'"]([^"\']+)[\'"]', query)
        return m.group(1) if m else None
    
    def _extract_code(self, query: str) -> Optional[str]:
        m = re.search(r'```(?:python)?\s*\n?(.*?)```', query, re.DOTALL)
        if m: return m.group(1).strip()
        m = re.search(r'`([^`]+)`', query)
        if m and ('print' in m.group(1) or 'import' in m.group(1)):
            return m.group(1)
        return None
    
    def _find_relevant_skill(self, query: str) -> Optional[Skill]:
        results = self.skill_registry.search(query)
        return results[0] if results else None
    
    def _update_history(self, query: str, answer: str):
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]


# ═══════════════════════════════════════════════════════
# 4. Interactive CLI — Интерфейс
# ═══════════════════════════════════════════════════════

async def interactive_cli(auto_learn_topics: List[str] = None,
                          telegram: bool = False):
    """Запуск ТАРС Agent в интерактивном режиме."""
    print()
    print("═" * 62)
    print("  🤖 ТАРС v3 — Автономный ИИ-агент")
    print("═" * 62)
    print()
    
    agent = TarsAgent(workspace=".")
    
    # ═══ Startup document scan (фоновая индексация) ═══
    try:
        scan_result = await agent.doc_sense.scan_workspace(max_files=50)
        if scan_result.files_ingested > 0:
            print(f"  📄 DocumentSense: +{scan_result.files_ingested} файлов "
                  f"({scan_result.total_chunks} чанков) → LEANN")
    except Exception as e:
        print(f"  ⚠ DocumentSense scan: {e}")
    
    doc_status = agent.doc_sense.get_status()
    print(f"  📂 Workspace : {agent.workspace}")
    print(f"  🧠 LEANN    : {len(agent.memory.texts)} документов")
    print(f"  📄 DocSense : {doc_status['indexed_files']} файлов проиндексировано")
    print(f"  🔧 Tools    : {', '.join(agent.tools.list_tools())}")
    print(f"  🎯 Skills   : {len(agent.skill_registry.list_all())}")
    print(f"  ⚡ SubAgents: search, fetch, analyze, memory")
    print()
    print("  Команды: /status /skills /health /history /quit")
    print("  Примеры:")
    print("    научись Docker")
    print("    найди в интернете Mamba SSM")
    print("    прочитай spec.pdf")
    print("    запомни мой сервер: 192.168.1.100")
    print()
    
    # Автообучение (фоновая задача)
    background_tasks = []
    if auto_learn_topics:
        async def auto_learn():
            from agent.self_learn import SelfLearner
            sl = SelfLearner(memory=agent.memory)
            await sl.auto_learn(auto_learn_topics, interval_min=60)
        background_tasks.append(asyncio.create_task(auto_learn()))
    
    # Telegram (фоновая задача)
    if telegram:
        token = os.environ.get("TARS_TELEGRAM_TOKEN", "")
        if token:
            from tools.telegram_bot import TarsTelegram
            bot = TarsTelegram(token=token, memory=agent.memory)
            background_tasks.append(asyncio.create_task(bot.start()))
            print("  📱 Telegram бот запущен!\n")
    
    while True:
        try:
            query = input("  Ты → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Пока!")
            break
        
        if not query:
            continue
        
        # Системные команды
        if query == "/quit":
            print("  👋 Пока!")
            break
        elif query == "/status":
            r = await agent.run("статус системы")
            print(f"\n{r.answer}\n")
            continue
        elif query == "/skills":
            skills = agent.skill_registry.list_all()
            if skills:
                print(f"\n  🎯 Навыки ({len(skills)}):")
                for s in skills:
                    print(f"    • {s.name} ({s.category}) — {s.confidence:.0%}, использован {s.use_count}x")
            else:
                print("  🤷 Нет навыков. Скажи: 'научись Docker'")
            print()
            continue
        elif query == "/health":
            health = agent.pool.health_report()
            print(f"\n  🏥 SubAgents Health:")
            for name, info in health["agents"].items():
                bar = "█" * int(float(info["health"].rstrip("%")) / 10)
                print(f"    {name:8s} {info['health']:>4s} {bar} ({info['done']} done, avg {info['avg_time']})")
            print(f"    {'cache':8s} hit-rate {health['cache']['hit_rate']}")
            print()
            continue
        elif query == "/history":
            print(f"\n  📜 История ({len(agent.history)} записей):")
            for h in agent.history[-10:]:
                role = "👤" if h["role"] == "user" else "🤖"
                print(f"    {role} {h['content'][:80]}")
            print()
            continue
        
        # Выполнить задачу
        print()
        result = await agent.run(query)
        print()
        print(f"  ТАРС → {result.answer}")
        print()
        
        # Показать шаги
        if agent.verbose and result.steps:
            print(f"  ─── Pipeline ({result.total_time:.1f}s) ───")
            for s in result.steps:
                print(str(s))
            print()
    
    # Остановить фоновые задачи
    for t in background_tasks:
        t.cancel()


# ═══════════════════════════════════════════════════════
# 5. Entry Point
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ТАРС Agent")
    parser.add_argument("--telegram", action="store_true", help="Запустить Telegram бота")
    parser.add_argument("--auto-learn", nargs="*", help="Темы для автообучения")
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    asyncio.run(interactive_cli(
        auto_learn_topics=args.auto_learn,
        telegram=args.telegram,
    ))
