"""
═══════════════════════════════════════════════════════════════
  knowledge_graph.py — Граф знаний TARS v3 («Obsidian внутри»)
═══════════════════════════════════════════════════════════════

Превращает ВСЮ информацию ТАРС в связанный граф знаний:

  Узлы (Nodes):
    - Заметки пользователя
    - Факты из диалогов (авто)
    - Конспекты лекций (авто)
    - Протоколы встреч (авто)
    - Flashcards (авто)

  Рёбра (Edges):
    - Ключевые слова (авто)
    - Семантическая близость (авто)
    - Временная связь (авто)
    - Ручные связи пользователя

Команды:
  "Запомни: <текст>"
  "Что я знаю про <тему>?"
  "Покажи граф знаний"
  "Свяжи <тема1> с <тема2>"
  "Все заметки за сегодня"
"""

import json
import os
import re
import logging
import time
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger("Tars.KnowledgeGraph")

_ROOT = Path(__file__).parent.parent
_KG_DIR = _ROOT / "data" / "knowledge"
_KG_DIR.mkdir(parents=True, exist_ok=True)
_KG_DB = _KG_DIR / "graph.db"


# ═══════════════════════════════════════════════════════════════
#  Node — Атомарная единица знания
# ═══════════════════════════════════════════════════════════════

class KnowledgeNode:
    """
    Один узел графа знаний.
    Аналог одной заметки в Obsidian.
    """
    
    def __init__(self, title: str, content: str, 
                 node_type: str = "note", tags: List[str] = None):
        self.id = f"kn_{int(time.time()*1000) % 10**10}"
        self.title = title
        self.content = content
        self.node_type = node_type  # note, fact, lecture, meeting, flashcard, idea
        self.tags = tags or []
        self.created = datetime.now().isoformat()
        self.modified = datetime.now().isoformat()
        self.links: List[str] = []      # ID других узлов (ручные)
        self.auto_links: List[str] = [] # ID авто-обнаруженных связей
        self.source = ""                # Откуда пришёл: "dialog", "lecture", "manual"
        self.keywords: List[str] = []   # Авто-извлечённые ключевые слова
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id, "title": self.title, "content": self.content,
            "node_type": self.node_type, "tags": self.tags,
            "created": self.created, "modified": self.modified,
            "links": self.links, "auto_links": self.auto_links,
            "source": self.source, "keywords": self.keywords,
        }
    
    @staticmethod
    def from_dict(d: Dict) -> 'KnowledgeNode':
        n = KnowledgeNode.__new__(KnowledgeNode)
        n.id = d["id"]; n.title = d["title"]; n.content = d["content"]
        n.node_type = d.get("node_type", "note"); n.tags = d.get("tags", [])
        n.created = d.get("created", ""); n.modified = d.get("modified", "")
        n.links = d.get("links", []); n.auto_links = d.get("auto_links", [])
        n.source = d.get("source", ""); n.keywords = d.get("keywords", [])
        return n


# ═══════════════════════════════════════════════════════════════
#  KnowledgeGraph — Основной движок
# ═══════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Граф знаний ТАРС — Obsidian-подобная система, 
    полностью интегрированная с памятью и контекстом.
    
    Преимущества перед Obsidian:
      1. Авто-создание узлов из диалогов, лекций, встреч
      2. Авто-связи по ключевым словам и семантике
      3. Поиск на естественном языке
      4. Интеграция с flashcards (спросил → запомнил → повторил)
    """
    
    def __init__(self):
        self._init_db()
        self._keyword_index: Dict[str, Set[str]] = {}  # keyword → set(node_ids)
        self._rebuild_index()
    
    # ═══ CRUD операции ═══
    
    def add_note(self, title: str, content: str, 
                 tags: List[str] = None, node_type: str = "note",
                 source: str = "manual") -> str:
        """Добавить заметку / узел знаний."""
        node = KnowledgeNode(title, content, node_type, tags)
        node.source = source
        node.keywords = self._extract_keywords(f"{title} {content}")
        
        self._save_node(node)
        
        # Авто-связи — найти похожие узлы
        auto_linked = self._find_related(node)
        if auto_linked:
            node.auto_links = [n_id for n_id, _ in auto_linked[:10]]
            self._update_node(node)
            # Двусторонние связи
            for related_id, _ in auto_linked[:10]:
                self._add_auto_link(related_id, node.id)
        
        # Обновить индекс
        for kw in node.keywords:
            self._keyword_index.setdefault(kw, set()).add(node.id)
        
        n_links = len(auto_linked)
        tag_str = f" #{' #'.join(tags)}" if tags else ""
        return (
            f"📝 Записано: «{title}»{tag_str}\n"
            f"🔗 Автоматически связано с {n_links} узлами"
        )
    
    def add_from_dialog(self, user_msg: str, tars_response: str):
        """
        Авто-создание узла из диалога.
        Вызывается из GIE при каждом взаимодействии.
        """
        # Извлекаем ключевые факты
        facts = self._extract_facts(user_msg, tars_response)
        
        for fact_title, fact_content in facts:
            self.add_note(
                fact_title, fact_content,
                tags=["dialog"], node_type="fact", source="dialog"
            )
    
    def add_from_lecture(self, title: str, summary: str, 
                        key_points: List[str], keywords: List[str]):
        """Авто-создание узла из конспекта лекции."""
        content = f"{summary}\n\nКлючевые тезисы:\n"
        content += "\n".join(f"• {p}" for p in key_points)
        
        self.add_note(
            f"📖 {title}", content,
            tags=["lecture"] + keywords[:5],
            node_type="lecture", source="lecture"
        )
    
    def add_from_meeting(self, title: str, summary: str,
                        action_items: List[str], speakers: List[str]):
        """Авто-создание узла из встречи."""
        content = f"{summary}\n\nУчастники: {', '.join(speakers)}"
        if action_items:
            content += "\n\nЗадачи:\n" + "\n".join(f"□ {a}" for a in action_items)
        
        self.add_note(
            f"🤝 {title}", content,
            tags=["meeting"] + [s.lower() for s in speakers[:3]],
            node_type="meeting", source="meeting"
        )
    
    # ═══ Поиск ═══
    
    def search(self, query: str, limit: int = 10) -> str:
        """
        Поиск по графу знаний.
        «Что я знаю про нейросети?»
        """
        query_keywords = self._extract_keywords(query)
        results: Dict[str, float] = {}  # node_id → relevance
        
        # 1. Поиск по ключевым словам (индекс)
        for kw in query_keywords:
            if kw in self._keyword_index:
                for node_id in self._keyword_index[kw]:
                    results[node_id] = results.get(node_id, 0) + 2.0
        
        # 2. Full-text search в SQLite
        try:
            conn = sqlite3.connect(str(_KG_DB))
            c = conn.cursor()
            for kw in query_keywords:
                c.execute(
                    "SELECT id FROM nodes WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?",
                    (f"%{kw}%", f"%{kw}%", f"%{kw}%")
                )
                for (node_id,) in c.fetchall():
                    results[node_id] = results.get(node_id, 0) + 1.0
            conn.close()
        except Exception:
            pass
        
        if not results:
            return f"🔍 Ничего не найдено по запросу «{query}». Попробуй другие слова."
        
        # Сортировка по релевантности
        sorted_ids = sorted(results.keys(), key=lambda x: -results[x])[:limit]
        
        lines = [f"🔍 Результаты по «{query}» ({len(results)} узлов):\n"]
        for i, node_id in enumerate(sorted_ids, 1):
            node = self._load_node(node_id)
            if node:
                type_emoji = {"note": "📝", "fact": "💬", "lecture": "📖", 
                             "meeting": "🤝", "flashcard": "🎴", "idea": "💡"}
                emoji = type_emoji.get(node.node_type, "📄")
                tags = f" #{' #'.join(node.tags[:3])}" if node.tags else ""
                preview = node.content[:80].replace("\n", " ")
                n_links = len(node.links) + len(node.auto_links)
                lines.append(
                    f"  {i}. {emoji} **{node.title}**{tags}\n"
                    f"     {preview}... ({n_links} связей)"
                )
        
        return "\n".join(lines)
    
    def get_related(self, query: str, depth: int = 1) -> str:
        """
        Показать связанные узлы (обход графа).
        «Что связано с Python?»
        """
        # Найти стартовый узел
        start_nodes = self._find_by_keyword(query)
        if not start_nodes:
            return f"❌ Не найден узел для «{query}»"
        
        visited = set()
        related = []
        
        # BFS обход графа
        queue = [(nid, 0) for nid in start_nodes[:3]]
        while queue and len(related) < 15:
            node_id, d = queue.pop(0)
            if node_id in visited or d > depth:
                continue
            visited.add(node_id)
            
            node = self._load_node(node_id)
            if node:
                related.append((node, d))
                # Добавить соседей
                all_links = set(node.links + node.auto_links)
                for link_id in all_links:
                    if link_id not in visited:
                        queue.append((link_id, d + 1))
        
        if not related:
            return f"🔍 Нет связей для «{query}»"
        
        lines = [f"🕸 Граф знаний для «{query}»:\n"]
        for node, d in related:
            indent = "  " + "  " * d
            prefix = "→" if d > 0 else "●"
            lines.append(f"{indent}{prefix} {node.title} ({node.node_type})")
        
        return "\n".join(lines)
    
    # ═══ Визуализация графа ═══
    
    def get_graph_ascii(self, max_nodes: int = 20) -> str:
        """ASCII-визуализация графа знаний."""
        nodes = self._load_all_nodes(limit=max_nodes)
        
        if not nodes:
            return "📊 Граф пуст. Скажи «запомни: ...» чтобы начать."
        
        lines = [f"🕸 Граф знаний ТАРС ({len(nodes)} узлов):\n"]
        
        # Группируем по типу
        by_type = defaultdict(list)
        for n in nodes:
            by_type[n.node_type].append(n)
        
        type_names = {
            "note": "📝 Заметки", "fact": "💬 Факты из диалогов",
            "lecture": "📖 Лекции", "meeting": "🤝 Встречи",
            "flashcard": "🎴 Карточки", "idea": "💡 Идеи",
        }
        
        for ntype, type_nodes in sorted(by_type.items()):
            name = type_names.get(ntype, f"📄 {ntype}")
            lines.append(f"\n  {name} ({len(type_nodes)}):")
            for n in type_nodes[:5]:
                n_links = len(n.links) + len(n.auto_links)
                link_str = f" ←→ {n_links}" if n_links > 0 else ""
                lines.append(f"    ├─ {n.title}{link_str}")
        
        # Статистика
        total_links = sum(len(n.links) + len(n.auto_links) for n in nodes)
        lines.append(f"\n  📊 Всего: {len(nodes)} узлов, {total_links} связей")
        lines.append(f"  📂 Типов: {len(by_type)}")
        lines.append(f"  🏷 Тегов: {len(self._get_all_tags(nodes))}")
        
        return "\n".join(lines)
    
    def get_graph_data(self) -> Dict:
        """Данные графа для визуализации (JSON для WebUI)."""
        nodes = self._load_all_nodes(limit=200)
        
        node_data = []
        edge_data = []
        
        for n in nodes:
            node_data.append({
                "id": n.id, "label": n.title[:30],
                "type": n.node_type, "tags": n.tags,
            })
            for link_id in set(n.links + n.auto_links):
                edge_data.append({"from": n.id, "to": link_id})
        
        return {"nodes": node_data, "edges": edge_data}
    
    # ═══ Ручные связи ═══
    
    def link(self, topic1: str, topic2: str) -> str:
        """Ручная связь между темами."""
        nodes1 = self._find_by_keyword(topic1)
        nodes2 = self._find_by_keyword(topic2)
        
        if not nodes1:
            return f"❌ Не найдено: «{topic1}»"
        if not nodes2:
            return f"❌ Не найдено: «{topic2}»"
        
        n1 = self._load_node(nodes1[0])
        n2 = self._load_node(nodes2[0])
        
        if n1 and n2:
            if n2.id not in n1.links:
                n1.links.append(n2.id)
                self._update_node(n1)
            if n1.id not in n2.links:
                n2.links.append(n1.id)
                self._update_node(n2)
            return f"🔗 Связано: «{n1.title}» ↔ «{n2.title}»"
        
        return "❌ Ошибка связывания"
    
    # ═══ Ежедневные заметки ═══
    
    def daily_note(self, content: str = None) -> str:
        """Daily note — как в Obsidian."""
        today = datetime.now().strftime("%Y-%m-%d")
        title = f"📅 Дневник {today}"
        
        # Проверяем есть ли уже
        existing = self._find_by_title(title)
        
        if existing:
            node = self._load_node(existing[0])
            if node and content:
                node.content += f"\n\n{datetime.now().strftime('%H:%M')} — {content}"
                node.modified = datetime.now().isoformat()
                self._update_node(node)
                return f"📝 Добавлено в дневник: {content[:50]}..."
            elif node:
                return f"📅 Дневник за {today}:\n\n{node.content}"
        
        if content:
            self.add_note(
                title, f"{datetime.now().strftime('%H:%M')} — {content}",
                tags=["daily", today], node_type="note", source="daily"
            )
            return f"📅 Дневник создан: {content[:50]}..."
        
        return f"📅 Нет записей за {today}. Скажи что хочешь записать."
    
    # ═══ Статистика ═══
    
    def stats(self) -> str:
        """Статистика графа знаний."""
        try:
            conn = sqlite3.connect(str(_KG_DB))
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM nodes")
            total = c.fetchone()[0]
            c.execute("SELECT node_type, COUNT(*) FROM nodes GROUP BY node_type")
            by_type = c.fetchall()
            c.execute("SELECT COUNT(*) FROM nodes WHERE created > ?",
                      (datetime.now().replace(hour=0, minute=0).isoformat(),))
            today = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM nodes WHERE created > ?",
                      ((datetime.now() - timedelta(days=7)).isoformat(),))
            week = c.fetchone()[0]
            conn.close()
        except Exception:
            return "❌ Ошибка статистики"
        
        lines = [
            f"📊 Граф знаний ТАРС:",
            f"  Всего узлов: {total}",
            f"  Сегодня: +{today}",
            f"  За неделю: +{week}",
            f"\n  По типам:"
        ]
        
        type_names = {
            "note": "📝 Заметки", "fact": "💬 Факты",
            "lecture": "📖 Лекции", "meeting": "🤝 Встречи",
            "flashcard": "🎴 Карточки", "idea": "💡 Идеи",
        }
        for ntype, count in by_type:
            name = type_names.get(ntype, ntype)
            lines.append(f"    {name}: {count}")
        
        return "\n".join(lines)
    
    # ═══ Internal Methods ═══
    
    def _init_db(self):
        """Инициализация SQLite базы."""
        conn = sqlite3.connect(str(_KG_DB))
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT,
                node_type TEXT DEFAULT 'note',
                tags TEXT DEFAULT '[]',
                created TEXT,
                modified TEXT,
                links TEXT DEFAULT '[]',
                auto_links TEXT DEFAULT '[]',
                source TEXT DEFAULT '',
                keywords TEXT DEFAULT '[]'
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_type ON nodes(node_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_created ON nodes(created)")
        conn.commit()
        conn.close()
    
    def _save_node(self, node: KnowledgeNode):
        conn = sqlite3.connect(str(_KG_DB))
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (node.id, node.title, node.content, node.node_type,
             json.dumps(node.tags), node.created, node.modified,
             json.dumps(node.links), json.dumps(node.auto_links),
             node.source, json.dumps(node.keywords))
        )
        conn.commit()
        conn.close()
    
    def _update_node(self, node: KnowledgeNode):
        node.modified = datetime.now().isoformat()
        self._save_node(node)
    
    def _load_node(self, node_id: str) -> Optional[KnowledgeNode]:
        try:
            conn = sqlite3.connect(str(_KG_DB))
            c = conn.cursor()
            c.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = c.fetchone()
            conn.close()
            if row:
                return self._row_to_node(row)
        except Exception:
            pass
        return None
    
    def _load_all_nodes(self, limit: int = 100) -> List[KnowledgeNode]:
        nodes = []
        try:
            conn = sqlite3.connect(str(_KG_DB))
            c = conn.cursor()
            c.execute("SELECT * FROM nodes ORDER BY modified DESC LIMIT ?", (limit,))
            for row in c.fetchall():
                nodes.append(self._row_to_node(row))
            conn.close()
        except Exception:
            pass
        return nodes
    
    def _row_to_node(self, row) -> KnowledgeNode:
        n = KnowledgeNode.__new__(KnowledgeNode)
        n.id, n.title, n.content, n.node_type = row[0], row[1], row[2], row[3]
        n.tags = json.loads(row[4] or "[]")
        n.created, n.modified = row[5], row[6]
        n.links = json.loads(row[7] or "[]")
        n.auto_links = json.loads(row[8] or "[]")
        n.source = row[9] or ""
        n.keywords = json.loads(row[10] or "[]")
        return n
    
    def _find_related(self, node: KnowledgeNode) -> List[Tuple[str, float]]:
        """Найти похожие узлы по ключевым словам."""
        if not node.keywords:
            return []
        
        scores: Dict[str, float] = {}
        for kw in node.keywords:
            if kw in self._keyword_index:
                for other_id in self._keyword_index[kw]:
                    if other_id != node.id:
                        scores[other_id] = scores.get(other_id, 0) + 1.0
        
        return sorted(scores.items(), key=lambda x: -x[1])
    
    def _find_by_keyword(self, keyword: str) -> List[str]:
        """Найти узлы по ключевому слову."""
        keyword = keyword.lower().strip()
        result = set()
        
        # Из индекса
        for kw, ids in self._keyword_index.items():
            if keyword in kw or kw in keyword:
                result.update(ids)
        
        # Из SQLite
        try:
            conn = sqlite3.connect(str(_KG_DB))
            c = conn.cursor()
            c.execute(
                "SELECT id FROM nodes WHERE title LIKE ? OR content LIKE ?",
                (f"%{keyword}%", f"%{keyword}%")
            )
            for (nid,) in c.fetchall():
                result.add(nid)
            conn.close()
        except Exception:
            pass
        
        return list(result)
    
    def _find_by_title(self, title: str) -> List[str]:
        """Найти узлы по заголовку."""
        try:
            conn = sqlite3.connect(str(_KG_DB))
            c = conn.cursor()
            c.execute("SELECT id FROM nodes WHERE title = ?", (title,))
            return [r[0] for r in c.fetchall()]
        except Exception:
            return []
    
    def _add_auto_link(self, node_id: str, link_to: str):
        """Добавить авто-связь к существующему узлу."""
        node = self._load_node(node_id)
        if node and link_to not in node.auto_links:
            node.auto_links.append(link_to)
            self._save_node(node)
    
    def _extract_keywords(self, text: str, n: int = 10) -> List[str]:
        """Извлечь ключевые слова."""
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'а', 'но', 'что', 'это', 'как',
            'я', 'мы', 'он', 'она', 'они', 'не', 'да', 'нет', 'для',
            'из', 'от', 'до', 'за', 'при', 'или', 'то', 'у', 'к', 'же',
            'the', 'a', 'an', 'is', 'are', 'was', 'to', 'of', 'and', 'in',
            'бы', 'ли', 'ну', 'ещё', 'ещё', 'так', 'вот', 'всё', 'тоже',
            'уже', 'тут', 'там', 'кто', 'чем', 'где', 'когда', 'если',
        }
        
        words = re.findall(r'[а-яёa-z]{3,}', text.lower())
        freq = defaultdict(int)
        for w in words:
            if w not in stop_words:
                freq[w] += 1
        
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:n]]
    
    def _extract_facts(self, user_msg: str, tars_response: str) -> List[Tuple[str, str]]:
        """
        Извлечь факты из диалога.
        Не каждый диалог содержит факты — фильтруем шум.
        """
        facts = []
        
        # Условия для создания факта:
        # 1. Вопрос "что такое / как работает / объясни"
        triggers = [
            "что такое", "как работает", "объясни", "расскажи",
            "в чём разница", "зачем", "почему", "what is", "how does",
        ]
        
        msg_lower = user_msg.lower()
        if any(t in msg_lower for t in triggers):
            # Извлечь тему из вопроса
            title = user_msg[:80]
            for prefix in triggers:
                if prefix in msg_lower:
                    idx = msg_lower.index(prefix) + len(prefix)
                    topic = user_msg[idx:].strip().rstrip("?.")
                    if topic:
                        title = topic[:80]
                    break
            
            facts.append((
                title,
                tars_response[:500]
            ))
        
        # 2. Пользователь явно просит запомнить
        remember_triggers = ["запомни", "запиши", "заметка", "remember"]
        if any(t in msg_lower for t in remember_triggers):
            content = user_msg
            for t in remember_triggers:
                content = content.lower().replace(t, "").strip()
            if content and len(content) > 5:
                facts.append((content[:80], content))
        
        return facts
    
    def _rebuild_index(self):
        """Перестроить keyword index из базы."""
        self._keyword_index.clear()
        try:
            conn = sqlite3.connect(str(_KG_DB))
            c = conn.cursor()
            c.execute("SELECT id, keywords FROM nodes")
            for node_id, kw_json in c.fetchall():
                keywords = json.loads(kw_json or "[]")
                for kw in keywords:
                    self._keyword_index.setdefault(kw, set()).add(node_id)
            conn.close()
            logger.info(f"KnowledgeGraph: index rebuilt, {len(self._keyword_index)} keywords")
        except Exception as e:
            logger.debug(f"Index rebuild error: {e}")
    
    def _get_all_tags(self, nodes: List[KnowledgeNode]) -> Set[str]:
        tags = set()
        for n in nodes:
            tags.update(n.tags)
        return tags


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    kg = KnowledgeGraph()
    
    # Тест: добавление заметок
    print(kg.add_note("Python основы", "Python — интерпретируемый язык...", ["python", "programming"]))
    print(kg.add_note("Машинное обучение", "ML использует Python для...", ["ml", "python"]))
    print(kg.add_note("Нейросети", "Нейросети — это подмножество ML...", ["ml", "neural"]))
    print()
    
    # Поиск
    print(kg.search("python"))
    print()
    
    # Связи
    print(kg.get_related("python"))
    print()
    
    # Граф
    print(kg.get_graph_ascii())
    print()
    
    # Статистика
    print(kg.stats())
