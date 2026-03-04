"""
═══════════════════════════════════════════════════════════════
  ТАРС Skill System — Автоматическое обучение навыкам
═══════════════════════════════════════════════════════════════

ТАРС может научиться ЛЮБОМУ навыку:
  - "научись работать с Excel" → ищет инфо, создаёт skill
  - "научись управлять Arduino" → изучает, сохраняет инструкции
  - "научись API Telegram" → документация → skill

Вдохновлено PicoClaw skills/ и OpenClaw skills/.

Каждый skill = JSON-файл с:
  - name, description
  - instructions (пошаговые инструкции)
  - commands (готовые команды)
  - examples (примеры использования)
  - sources (откуда обучился)

Использование:
  learner = SkillLearner()
  skill = await learner.learn("работа с Docker")
  print(skill.instructions)
"""

import os
import json
import time
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("Tars.Skills")

_ROOT = Path(__file__).resolve().parent.parent  # agent/ → project root
sys.path.insert(0, str(_ROOT))
SKILLS_DIR = _ROOT / "skills"
SKILLS_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════
# Skill Data Model
# ═══════════════════════════════════════════

@dataclass
class Skill:
    """Навык ТАРС."""
    name: str                       # "Docker management"
    description: str                # "Управление Docker контейнерами"
    category: str = "general"       # app, code, hardware, api, system, general
    
    instructions: List[str] = field(default_factory=list)  # Пошаговые инструкции
    commands: List[Dict[str, str]] = field(default_factory=list)  # {cmd, desc}
    examples: List[Dict[str, str]] = field(default_factory=list)  # {input, output}
    knowledge: List[str] = field(default_factory=list)     # Факты и правила
    
    sources: List[str] = field(default_factory=list)       # Откуда обучился
    created: float = 0.0
    updated: float = 0.0
    confidence: float = 0.0        # 0-1, насколько уверен в навыке
    use_count: int = 0             # Сколько раз использовал
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Skill':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: Optional[str] = None):
        """Сохранить skill на диск."""
        if not path:
            safe_name = re.sub(r'[^\w\-]', '_', self.name.lower())
            path = str(SKILLS_DIR / f"{safe_name}.json")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Skill saved: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'Skill':
        """Загрузить skill с диска."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))
    
    def summary(self) -> str:
        """Краткое описание навыка."""
        return (
            f"🎯 {self.name}\n"
            f"   {self.description}\n"
            f"   📋 {len(self.instructions)} инструкций, "
            f"🔧 {len(self.commands)} команд, "
            f"📚 {len(self.knowledge)} фактов\n"
            f"   🎯 Уверенность: {self.confidence:.0%}"
        )


# ═══════════════════════════════════════════
# Skill Registry — Реестр навыков
# ═══════════════════════════════════════════

class SkillRegistry:
    """Реестр всех навыков ТАРС."""
    
    def __init__(self, skills_dir: str = ""):
        self.skills_dir = Path(skills_dir) if skills_dir else SKILLS_DIR
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._skills: Dict[str, Skill] = {}
        self._load_all()
    
    def _load_all(self):
        """Загрузить все skill-файлы."""
        for f in self.skills_dir.glob("*.json"):
            try:
                skill = Skill.load(str(f))
                self._skills[skill.name.lower()] = skill
            except Exception as e:
                logger.warning(f"Failed to load skill {f}: {e}")
        
        if self._skills:
            logger.info(f"Skills loaded: {len(self._skills)}")
    
    def get(self, name: str) -> Optional[Skill]:
        """Найти навык по имени (нечёткий поиск)."""
        name_lower = name.lower()
        
        # Точное совпадение
        if name_lower in self._skills:
            return self._skills[name_lower]
        
        # Частичное совпадение
        for key, skill in self._skills.items():
            if name_lower in key or key in name_lower:
                return skill
        
        return None
    
    def add(self, skill: Skill):
        """Добавить/обновить навык."""
        self._skills[skill.name.lower()] = skill
        skill.save(str(self.skills_dir / f"{re.sub(r'[^a-zа-яё0-9]', '_', skill.name.lower())}.json"))
    
    def list_all(self) -> List[Skill]:
        """Все навыки."""
        return list(self._skills.values())
    
    def search(self, query: str) -> List[Skill]:
        """Поиск навыков по ключевым словам."""
        q = query.lower()
        results = []
        for skill in self._skills.values():
            score = 0
            if q in skill.name.lower():
                score += 3
            if q in skill.description.lower():
                score += 2
            if q in skill.category:
                score += 1
            for kw in q.split():
                if any(kw in k.lower() for k in skill.knowledge):
                    score += 1
            if score > 0:
                results.append((score, skill))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in results]


# ═══════════════════════════════════════════
# Skill Learner — Автоматическое обучение
# ═══════════════════════════════════════════

# Паттерны для извлечения инструкций из текста
_INSTRUCTION_PATTERNS = [
    r'(?:шаг|step)\s*\d+[.:]\s*(.+)',
    r'\d+[.)]\s+(.{20,})',
    r'(?:сначала|первым делом|далее|затем|потом)\s+(.+)',
    r'(?:нужно|необходимо|следует|можно)\s+(.+)',
]

_COMMAND_PATTERNS = [
    r'(?:```|`)\s*(?:bash|shell|cmd|powershell)?\s*\n?\s*(.+?)(?:```|`)',
    r'\$\s+(.+)',
    r'(?:команда|command)[:\s]+(.+)',
    r'(?:pip install|npm install|apt install|brew install)\s+\S+',
    r'(?:docker|git|python|node|cargo)\s+\S+(?:\s+\S+)*',
]


class SkillLearner:
    """
    Автоматическое обучение навыкам из интернета.
    
    Пример:
        learner = SkillLearner()
        skill = await learner.learn("Docker")
        print(skill.summary())
    """
    
    def __init__(self, registry: Optional[SkillRegistry] = None,
                 memory=None):
        self.registry = registry or SkillRegistry()
        self.memory = memory  # LeannIndex
        
        # Параллельные суб-агенты
        try:
            from tools.sub_agents import AgentPool
            self._pool = AgentPool(memory=memory)
        except ImportError:
            self._pool = None
    
    async def learn(self, topic: str, depth: int = 5) -> Skill:
        """
        Научиться навыку — через параллельные суб-агенты!
        
        Было: ~15-30 сек (последовательно)
        Стало: ~3-7 сек (параллельно через AgentPool)
        """
        print(f"\n{'═'*60}")
        print(f"  🎓 ТАРС обучается: {topic}")
        print(f"{'═'*60}\n")
        
        # Проверить существующий
        existing = self.registry.get(topic)
        if existing:
            print(f"  ℹ Навык уже есть! Обновляю...")
            skill = existing
        else:
            skill = Skill(
                name=topic,
                description=f"Навык: {topic}",
                created=time.time(),
            )
        
        skill.category = self._categorize(topic)
        print(f"  📂 Категория: {skill.category}")
        
        # ⚡ Параллельное исследование через AgentPool
        if self._pool:
            print(f"  ⚡ Режим: параллельные суб-агенты")
            research = await self._pool.research(topic, depth=depth)
            
            skill.sources = research.get("sources", [])
            skill.instructions = research.get("instructions", [])
            skill.commands = [{"cmd": c, "desc": ""} for c in research.get("commands", [])]
            skill.knowledge = research.get("facts", [])
        else:
            # Fallback: последовательный режим
            print(f"  ℹ Режим: последовательный")
            from tools.web_search import search_duckduckgo, fetch_page_text
            
            all_content = []
            for query in self._make_search_queries(topic):
                results = await search_duckduckgo(query, max_results=depth)
                for r in results:
                    content = await fetch_page_text(r.url, max_chars=5000)
                    if content and len(content) > 100:
                        all_content.append(content)
                        skill.sources.append(r.url)
            
            for content in all_content:
                for inst in self._extract_instructions(content):
                    if inst not in skill.instructions:
                        skill.instructions.append(inst)
                for cmd in self._extract_commands(content):
                    if cmd not in [c.get("cmd") for c in skill.commands]:
                        skill.commands.append({"cmd": cmd, "desc": ""})
                for fact in self._extract_facts(content, topic):
                    if fact not in skill.knowledge:
                        skill.knowledge.append(fact)
        
        # Примеры
        skill.examples = self._generate_examples(topic, skill)
        
        # Описание
        if skill.knowledge:
            skill.description = skill.knowledge[0][:200]
        
        # Уверенность
        skill.confidence = min(1.0, (
            len(skill.instructions) * 0.1 +
            len(skill.commands) * 0.15 +
            len(skill.knowledge) * 0.05 +
            len(skill.sources) * 0.05
        ))
        
        skill.updated = time.time()
        
        # Сохранить
        self.registry.add(skill)
        if self.memory:
            self.memory.save()
        
        print(f"\n{skill.summary()}")
        print(f"  💾 Сохранено в skills/")
        print(f"{'═'*60}\n")
        
        return skill
    
    async def learn_from_text(self, topic: str, text: str) -> Skill:
        """Научиться из предоставленного текста."""
        skill = Skill(name=topic, description=f"Навык: {topic}", created=time.time())
        skill.category = self._categorize(topic)
        
        skill.instructions = self._extract_instructions(text)
        skill.commands = [{"cmd": c, "desc": ""} for c in self._extract_commands(text)]
        skill.knowledge = self._extract_facts(text, topic)
        skill.confidence = min(1.0, len(skill.instructions) * 0.1 + len(skill.knowledge) * 0.05)
        skill.updated = time.time()
        
        self.registry.add(skill)
        return skill
    
    # ═══════ Извлечение знаний ═══════
    
    def _categorize(self, topic: str) -> str:
        """Определить категорию навыка."""
        t = topic.lower()
        categories = {
            "app": ["excel", "word", "photoshop", "blender", "vscode", "chrome", "firefox",
                     "приложени", "программ", "app", "software"],
            "code": ["python", "javascript", "rust", "code", "программировани", "api",
                      "git", "github", "npm", "pip", "код", "скрипт", "framework"],
            "hardware": ["arduino", "raspberry", "i2c", "spi", "gpio", "датчик", "sensor",
                          "hardware", "железо", "плат", "микроконтроллер", "esp32"],
            "system": ["linux", "windows", "docker", "bash", "terminal", "сервер", "server",
                        "сеть", "network", "ssh", "nginx", "apache", "kubernetes"],
            "data": ["excel", "sql", "database", "данны", "анализ", "pandas", "numpy",
                      "machine learning", "ml", "data", "csv", "json"],
            "ai": ["нейросет", "neural", "gpt", "llm", "обучен", "training", "model",
                    "transformer", "diffusion", "stable", "whisper", "tts"],
        }
        
        for cat, keywords in categories.items():
            if any(kw in t for kw in keywords):
                return cat
        return "general"
    
    def _make_search_queries(self, topic: str) -> List[str]:
        """Создать поисковые запросы для изучения темы."""
        return [
            f"{topic} tutorial руководство",
            f"{topic} commands основные команды",
            f"{topic} examples примеры",
        ]
    
    def _extract_instructions(self, text: str) -> List[str]:
        """Извлечь пошаговые инструкции."""
        instructions = []
        
        for pattern in _INSTRUCTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for m in matches:
                inst = m.strip()
                if 20 < len(inst) < 300:
                    instructions.append(inst)
        
        return instructions[:20]  # Макс 20
    
    def _extract_commands(self, text: str) -> List[str]:
        """Извлечь команды."""
        commands = []
        
        for pattern in _COMMAND_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for m in matches:
                cmd = m.strip()
                if 3 < len(cmd) < 200:
                    commands.append(cmd)
        
        return commands[:15]  # Макс 15
    
    def _extract_facts(self, text: str, topic: str) -> List[str]:
        """Извлечь ключевые факты о теме."""
        facts = []
        
        # Предложения содержащие ключевое слово
        sentences = re.split(r'[.!?\n]', text)
        topic_words = set(re.findall(r'[a-zа-яё]+', topic.lower()))
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30 or len(sent) > 300:
                continue
            
            sent_words = set(re.findall(r'[a-zа-яё]+', sent.lower()))
            overlap = topic_words & sent_words
            
            if overlap and len(overlap) >= 1:
                # Проверить что предложение информативно
                if any(w in sent.lower() for w in [
                    'это', 'is', 'являет', 'позволяет', 'используется',
                    'служит', 'нужен', 'means', 'provides', 'allows',
                    'предназначен', 'поддерживает', 'includes',
                ]):
                    facts.append(sent)
        
        return facts[:30]  # Макс 30 фактов
    
    def _generate_examples(self, topic: str, skill: Skill) -> List[Dict[str, str]]:
        """Сгенерировать примеры использования."""
        examples = []
        
        if skill.commands:
            for cmd_info in skill.commands[:3]:
                examples.append({
                    "input": f"Как использовать {topic}?",
                    "output": f"Выполни: {cmd_info['cmd']}",
                })
        
        if skill.instructions:
            examples.append({
                "input": f"Объясни как начать с {topic}",
                "output": " → ".join(skill.instructions[:3]),
            })
        
        return examples


# ═══════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="ТАРС Skill Learning")
    parser.add_argument("skill", nargs="*", help="Навык для изучения")
    parser.add_argument("--list", action="store_true", help="Список навыков")
    parser.add_argument("--info", type=str, help="Инфо о навыке")
    parser.add_argument("--depth", type=int, default=3, help="Глубина поиска")
    
    args = parser.parse_args()
    
    registry = SkillRegistry()
    learner = SkillLearner(registry)
    
    if args.list:
        skills = registry.list_all()
        if skills:
            print(f"\n🎯 Навыки ТАРС ({len(skills)}):\n")
            for s in skills:
                print(s.summary())
                print()
        else:
            print("\n🤷 Навыков пока нет. Используй: python skill_learn.py 'Docker'\n")
        return
    
    if args.info:
        skill = registry.get(args.info)
        if skill:
            print(f"\n{skill.summary()}")
            if skill.instructions:
                print(f"\n📋 Инструкции:")
                for i, inst in enumerate(skill.instructions[:10], 1):
                    print(f"  {i}. {inst}")
            if skill.commands:
                print(f"\n🔧 Команды:")
                for cmd in skill.commands[:10]:
                    print(f"  $ {cmd['cmd']}")
            if skill.knowledge:
                print(f"\n📚 Знания:")
                for k in skill.knowledge[:5]:
                    print(f"  • {k}")
            print()
        else:
            print(f"\n❌ Навык '{args.info}' не найден\n")
        return
    
    if args.skill:
        for topic in args.skill:
            await learner.learn(topic, depth=args.depth)
    else:
        # Интерактивный
        print("\n🎓 ТАРС Skill Learner")
        print("Введи навык для изучения (или 'list', 'quit')\n")
        
        while True:
            try:
                topic = input("  Навык → ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            
            if topic in ('quit', 'exit', 'q'):
                break
            if topic == 'list':
                for s in registry.list_all():
                    print(f"  {s.summary()}")
                continue
            if topic:
                await learner.learn(topic, depth=args.depth)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
