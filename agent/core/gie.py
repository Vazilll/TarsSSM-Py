import asyncio
import logging
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import torch
import os
try:
    from brain.mamba2.model import TarsMamba2LM as TarsBrain
except ImportError:
    TarsBrain = None
try:
    from brain.rrn import RrnCore
except ImportError:
    RrnCore = None
try:
    from brain.reflexes import ReflexDispatcher
except ImportError:
    ReflexDispatcher = None
from memory.titans import TitansMemory
from memory.store import TarsStorage
from agent.executor import ActionEngine
from agent.moira import MoIRA
try:
    from brain.mamba2.active_inference import BeliefState, ExpectedFreeEnergy
except ImportError:
    BeliefState = None
    ExpectedFreeEnergy = None
try:
    from agent.learning_helper import LearningHelper
except ImportError:
    LearningHelper = None
try:
    from agent.knowledge_graph import KnowledgeGraph
except ImportError:
    KnowledgeGraph = None
try:
    from agent.file_helper import FileHelper
except ImportError:
    FileHelper = None

class GieAgent:
    """
    GIE (General Intelligence Executive) — Центральный координатор TARS v3.
    
    3-Tier Pipeline:
      Tier 1:  ReflexCore     — мгновенные паттерны (MinGRU, <1ms)
      Tier 1.5: RRN           — рекурсивный рефлекс (Relational Memory) 
      Tier 2:  Mamba-2 Brain  — глубокое мышление (SSD + IDME)
      Router:  MoIRA          — маршрутизация к инструменту
      Exec:    ActionEngine   — выполнение действия
    """
    
    MAX_CONVERSATION = 50  # Cap conversation history to prevent unbounded growth
    MAX_QUERY_LEN = 4096   # Input validation
    CONVERSATION_FILE = "data/tars_conversation.json"
    
    def __init__(self, brain=None, moira: MoIRA = None, 
                 memory: Any = None, titans: TitansMemory = None):
        self.brain = brain
        self.rrn = RrnCore()
        self.reflex = ReflexDispatcher() if ReflexDispatcher else None
        self.moira = moira
        self.memory = memory
        self.titans = titans
        self.storage = TarsStorage()
        self.executor = ActionEngine()
        self.logger = logging.getLogger("Tars.GIE")
        
        # Ембеддинг модель для _text_to_vec (ленивая инициализация)
        self._embedding_model = None
        
        # ═══ Active Inference (Friston, 2006-2026) ═══
        self.belief_state = BeliefState(d_state=128) if BeliefState else None
        
        # ═══ Proactive Systems ═══
        self.learning_helper = LearningHelper() if LearningHelper else None
        self.knowledge_graph = KnowledgeGraph() if KnowledgeGraph else None
        self.file_helper = FileHelper() if FileHelper else None
        # Explicit None for removed subsystems (replaces dangerous __getattr__)
        self.notifier = None
        self.pomodoro = None
        self.schedule = None
        
        # Состояние сессии
        self.state = {
            "history": [],
            "last_thought": None,
            "session_goals": [],
            "total_processed": 0,
            "cumulative_free_energy": 0.0,
            "conversation": [],
        }
        
        # Load persisted conversation
        self._load_conversation()

    async def execute_goal(self, goal: str, fast_callback=None):
        """Главный цикл обработки цели."""
        # Input validation
        if len(goal) > self.MAX_QUERY_LEN:
            goal = goal[:self.MAX_QUERY_LEN]
            self.logger.warning(f"Goal truncated to {self.MAX_QUERY_LEN} chars")
        
        self.state["total_processed"] += 1
        self.state["session_goals"].append(goal)
        self.logger.info(f"GIE: Цель #{self.state['total_processed']} → {goal[:60]}...")

        proactive_hints = ""

        reflex_result = None
        if self.reflex is not None:
            try:
                reflex_ctx = self.reflex.dispatch(goal)
                if reflex_ctx.can_handle_fast and reflex_ctx.fast_response:
                    reflex_result = {"response": reflex_ctx.fast_response, "action": reflex_ctx.intent}
            except Exception as e:
                self.logger.debug(f"Reflex dispatch error: {e}")
        if reflex_result:
            response = reflex_result["response"]
            self.logger.info(f"GIE: Рефлекс [{reflex_result['action']}]: {response[:40]}...")
            # ═══ Fix #5: Reflex тоже сохраняет в память (Total Memory) ═══
            await self.storage.remember(f"[USER] {goal}")
            await self.storage.remember(f"[TARS/reflex] {response}")
            
            # Простые операции (состояние, приветствие) лучше вернуть сразу
            if reflex_result['action'] in ['greet', 'status', 'identity', 'time', 'shutdown', 'acknowledge']:
                return {"text": response, "tokens": 0, "duration": 0.0, "tps": 0.0}
            
            if fast_callback:
                await fast_callback(response)
                # продолжать глубокое обдумывание после выдачи шутки/рефлекса
            else:
                # Fix #4: Сохраняем в conversation history
                self.state["conversation"].append({
                    "user": goal, "tars": response,
                    "time": datetime.now().isoformat(), "tier": "reflex"
                })
                return {"text": response, "tokens": 0, "duration": 0.0, "tps": 0.0}

        # ═══ Stage 1: RRN Recursive Reflex (System 1) ═══
        # RRN сам решает: если MinGRU может ответить — ответит,
        # если нет — вернёт None и мы идём в глубокий анализ (System 2)
        quick_result = await self.rrn.fast_reply(goal)
        if quick_result is not None:
            quick_resp = quick_result["text"]
            self.logger.info(f"GIE: RRN (Light Model) Think: {quick_resp[:40]}...")
            if not quick_result.get("is_garbage", False):
                # ═══ Fix #5: RRN тоже сохраняет в память ═══
                await self.storage.remember(f"[USER] {goal}")
                await self.storage.remember(f"[TARS/rrn] {quick_resp}")
                if fast_callback:
                    await fast_callback(quick_resp)
                else:
                    self.state["conversation"].append({
                        "user": goal, "tars": quick_resp,
                        "time": datetime.now().isoformat(), "tier": "rrn"
                    })
                    return {"text": quick_resp, "tokens": 0, "duration": 0.0, "tps": 0.0}
            else:
                self.logger.info(f"GIE: RRN сгенерировал шум. Переход к глубокому анализу...")
        
        self.logger.info("GIE: Переход к глубокому анализу (Thinking Table)...")

        # ═══ Stage 2: Relational Grounding ═══
        relational_map = await self.rrn.precompute_grounding(goal, self.memory, self.titans)
        
        # Персоналия
        persona = await self.storage.retrieve_memories(goal)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # ═══ Fix #4: Инъекция последних диалогов в контекст (Total Memory) ═══
        recent_conv = ""
        for turn in self.state["conversation"][-5:]:
            recent_conv += f"User: {turn['user']}\nTARS: {turn['tars'][:200]}\n---\n"
        
        # Контекст для мозга (полный, с историей)
        full_context = (
            f"Время: {current_time}\n"
            f"История диалога:\n{recent_conv}\n" if recent_conv else f"Время: {current_time}\n"
            f"Контекст: {relational_map}\n"
            f"Память: {persona}\n"
            f"Текущий запрос: {goal}"
        )

        # Вектор от Titans для нейронного grounding
        # ═══ Prefetch: запускаем RAG параллельно с подготовкой контекста ═══
        # Вместо последовательного await — создаём task, который доставит
        # данные через ProgressiveMemoryManager когда будет готов.
        recall_vec = None
        rag_future = None
        if self.titans:
            async def _prefetch_rag():
                """Prefetch RAG data (runs concurrently with context preparation)."""
                try:
                    goal_vec = self._text_to_vec(goal)
                    rv = await self.titans.get_recall(goal_vec)
                    return {"memory_vec": rv, "rag_state": None}
                except Exception as e:
                    self.logger.debug(f"RAG prefetch error: {e}")
                    return None
            rag_future = asyncio.create_task(_prefetch_rag())
        
        # ═══ Uncertainty-Driven Auto-RAG (запускаем параллельно) ═══
        auto_rag_future = None
        if self.titans:
            async def _auto_rag():
                """Auto-RAG: if surprised, search web."""
                try:
                    goal_vec = self._text_to_vec(goal)
                    recall_surprise = await self.titans.update(goal_vec)
                    if recall_surprise.get("surprised", False):
                        self.logger.info("GIE: 🔍 Высокий surprise → автоматический RAG search")
                        from tools import web_search_sync
                        return web_search_sync(goal, max_results=3)
                except Exception as e:
                    self.logger.debug(f"Auto-RAG error: {e}")
                return None
            auto_rag_future = asyncio.create_task(_auto_rag())

        # Early Exit check: если RRN уже нашел четкий ответ в короткой памяти
        if "answer" in relational_map.lower() or "решение:" in relational_map.lower():
             self.logger.info("GIE: Обнаружено готовое решение в RRN. Попытка Early Exit...")
             # Подаем сигнал мозгу что можно заканчивать сразу
             full_context += "\nSystem-Hint: Solution found in grounding. Summarize and exit."

        # ═══ Gather prefetched results (with timeout — не блокируем навечно) ═══
        if rag_future is not None:
            try:
                rag_result = await asyncio.wait_for(rag_future, timeout=2.0)
                if rag_result and isinstance(rag_result, dict):
                    recall_vec = rag_result.get("memory_vec")
            except asyncio.TimeoutError:
                self.logger.debug("GIE: RAG prefetch timeout (2s) — model proceeds without")
            except Exception as e:
                self.logger.debug(f"RAG gather error: {e}")
        
        if auto_rag_future is not None:
            try:
                rag_result = await asyncio.wait_for(auto_rag_future, timeout=3.0)
                if rag_result:
                    full_context += f"\n\nНайдено в интернете:\n{rag_result[:500]}"
                    self.logger.info(f"GIE: RAG добавил {len(rag_result)} символов контекста")
            except asyncio.TimeoutError:
                self.logger.debug("GIE: Auto-RAG timeout (3s) — skipped")
            except Exception as e:
                self.logger.debug(f"Auto-RAG gather error: {e}")

        total_tokens = 0
        total_duration = 0.0
        tool_history = []
        MAX_LOOPS = 10
        MAX_TOOL_REPEAT = 3

        for step in range(MAX_LOOPS):
            self.logger.info(f"GIE: Шаг мышления {step + 1}/{MAX_LOOPS}")

            # ═══ Stage 3: Mamba-2 Brain (Tier 2 + IDME) ═══
            # brain.think возвращает dict с logits, state, p_value и т.д.
            try:
                if self.brain is not None:
                    # ═══ Fix #3: full_context передаётся мозгу (не только goal) ═══
                    # Кодируем ПОЛНЫЙ контекст в токены (BPE через TarsTokenizer)
                    brain_input = f"{full_context}\n\nОтветь: {goal}"
                    from brain.tokenizer import TarsTokenizer
                    _gie_tok = TarsTokenizer(mode="auto")
                    brain_token_ids = _gie_tok.encode(brain_input)[:1024]
                    goal_tokens = torch.tensor(
                        [brain_token_ids],
                        dtype=torch.long
                    )
                    logits, think_stats = self.brain.think(goal_tokens, memory_vec=recall_vec)
                    thought = goal  # Мысль = сам запрос (генерация в generate_mamba)
                    stats = {"tokens": goal_tokens.shape[1], "duration": think_stats.get("total_ms", 0) / 1000}
                    p_value = think_stats.get("final_p", 2.0)
                else:
                    thought = f"Мозг не загружен. Ответ на основе RRN: {goal}"
                    stats = {}
                    p_value = 2.0
                self.state["last_thought"] = thought
                total_tokens += stats.get("tokens", 0)
                total_duration += stats.get("duration", 0.0)
            except Exception as e:
                self.logger.error(f"GIE: Сбой в Mamba-2 Brain: {e}")
                thought = f"Системный сбой при генерации мысли: {e}"
                stats = {}
                p_value = 0.5
            
            
            # ═══ Stage 4: MoIRA Routing ═══
            thought_text = str(thought)
            thought_vec = self._text_to_vec(thought_text).unsqueeze(1)
            try:
                tool, params, confidence = await self.moira.route(thought_vec, thought_text)
            except Exception as e:
                self.logger.error(f"GIE: Сбой в MoIRA: {e}")
                tool, params, confidence = "FinalAnswer", {"answer": "Ошибка маршрутизации инструмента."}, 1.0

            # Продвинутая защита от петель: проверяем комбинацию инструмента и ВСЕХ параметров
            import json
            tool_signature = f"{tool}:{json.dumps(params, sort_keys=True)}"
            tool_history.append(tool_signature)
            
            if tool_history.count(tool_signature) > MAX_TOOL_REPEAT:
                self.logger.warning(f"GIE: Обнаружена петля для {tool} с идентичными параметрами. Принудительный выход.")
                tool = "FinalAnswer"
                confidence = 1.0
                params = {"answer": thought_text}
                
            # Фундаментальная математика TARS (Chiculaev-Kadymov Theorem)
            # Если интеграл мысли расходится (p <= 1.0), физические действия заблокированы
            if tool not in ["FinalAnswer", "Idle"] and p_value <= 1.0:
                self.logger.warning(f"GIE: [Integral Auditor] Сценарий расходится (p={p_value:.2f} <= 1.0). Выполнение действия {tool} заблокировано в целях безопасности.")
                tool = "FinalAnswer"
                params = {"answer": f"Действие заблокировано системой безопасности (Integral Auditor): сходимость p={p_value:.2f} недостаточна для безопасного выполнения в среде ОС."}
                confidence = 1.0
            
            # ═══ Финальный ответ ═══
            if tool == "FinalAnswer" or (confidence < 0.3 and step > 1):
                # Проверка сходимости OmegaCore. Если не сошёлся (p <= 1.0) и это не последний шаг, рекурсируем.
                if p_value <= 1.0 and step < MAX_LOOPS - 1:
                    # Fix: use a retry counter instead of mutating goal string.
                    # Previous approach prepended "Re-evaluate:" on each loop,
                    # filling the context window with garbage prefixes.
                    retry_key = f"_retry_count_{id(goal)}"
                    retry_count = getattr(self, retry_key, 0)
                    if retry_count >= 1:
                        self.logger.warning(f"GIE: Повторная сходимость не достигнута. Принимаю текущий результат.")
                    else:
                        setattr(self, retry_key, retry_count + 1)
                        self.logger.warning(f"GIE: Сходимость не достигнута (p={p_value:.2f}). Инициирую 1 дополнительный цикл размышления...")
                        full_context += "\nSystem-Hint: Предыдущая итерация не сошлась. Уточни параметры и попробуй снова."
                        continue # Skip FinalAnswer and action execution, just think again
                    
                # Обучение Titans
                if self.titans:
                    success_vec = self._text_to_vec(f"{goal} {thought_text}")
                    await self.titans.update(success_vec)
                
                # ═══ Fix #1: Сохраняем ПОЛНЫЙ ответ (Total Memory) ═══
                await self.storage.remember(f"[USER] {goal}")
                await self.storage.remember(f"[TARS/brain] {thought_text}")
                
                # ═══ Fix #4: Обновляем conversation history ═══
                self.state["conversation"].append({
                    "user": goal, "tars": thought_text,
                    "time": datetime.now().isoformat(), "tier": "brain"
                })
                
                # ═══ Active Inference: обновляем beliefs ═══
                if self.belief_state is not None:
                    try:
                        obs_vec = self._text_to_vec(thought_text)
                        belief_result = self.belief_state.update(obs_vec)
                        self.state["cumulative_free_energy"] += belief_result["free_energy"].item()
                        self.logger.info(
                            f"GIE: BeliefState F={belief_result['free_energy'].item():.3f} "
                            f"(surprise={belief_result['surprise'].item():.3f})"
                        )
                    except Exception as e:
                        self.logger.debug(f"BeliefState update error: {e}")
                
                # ═══ Knowledge Graph: авто-наполнение ═══
                if self.knowledge_graph:
                    try:
                        self.knowledge_graph.add_from_dialog(goal, thought_text)
                    except Exception as e:
                        self.logger.debug(f"KnowledgeGraph add error: {e}")
                
                # Добавляем проактивные подсказки к ответу
                final_text = thought_text
                if proactive_hints:
                    final_text = f"{thought_text}\n\n---\n{proactive_hints}"
                
                return {
                    "text": final_text,
                    "tokens": total_tokens,
                    "duration": total_duration,
                    "tps": total_tokens / total_duration if total_duration > 0 else 0
                }

            # ═══ Stage 5: Action ═══
            observation = await self._act(tool, params)
            
            # Обновление состояния
            log_entry = {"step": step, "tool": tool, "obs": observation[:200]}
            self.state["history"].append(log_entry)
            full_context += f"\nДействие: {tool} → {observation[:150]}"
            
            if "Error" in observation or "failed" in observation.lower():
                full_context += "\nСистема: Действие не удалось, нужна альтернативная стратегия."

            # Sleep Phase при Idle
            if tool == "Idle":
                await self.sleep_phase()

        # Если все шаги исчерпаны
        return {
            "text": str(self.state["last_thought"]) or "Не удалось завершить задачу.",
            "tokens": total_tokens,
            "duration": total_duration,
            "tps": total_tokens / total_duration if total_duration > 0 else 0
        }

    async def sleep_phase(self):
        """Консолидация памяти (Sleep Phase)."""
        await self.rrn.sleep_consolidation(self.state["history"], self.memory)
        self.state["history"] = []
        self.logger.info("GIE: Память консолидирована.")

    async def _act(self, tool: str, params: Dict[str, Any]) -> str:
        """Маппинг команд MoIRA → ActionEngine."""
        action_map = {
            "Python": "execute_script",
            "Terminal": "run_command",
            "Browser": "open_url",
            "Vision": "analyze_workspace",
            "Click": "click",
            "Type": "type",
        }
        
        # ═══ Sanitize params: whitelist per tool ═══
        allowed_keys = {
            "execute_script": {"code"},
            "run_command": {"command"},
            "open_url": {"url"},
            "analyze_workspace": set(),
            "click": {"x", "y", "element"},
            "type": {"text", "element"},
        }
        cmd = action_map.get(tool, tool.lower())
        if cmd in allowed_keys:
            params = {k: v for k, v in params.items() if k in allowed_keys[cmd]}
        
        try:
            result = await self.executor.execute(cmd, params)
            return result
        except Exception as e:
            return f"Error: {e}"


    def _text_to_vec(self, text: str) -> torch.Tensor:
        """
        Семантическое преобразование текста в вектор.
        
        Uses LEANN's embedding model (MiniLM/384d) when available.
        Falls back to deterministic char-hash only if no model loaded.
        """
        # Try real embedding via LEANN's model
        if self.memory is not None:
            try:
                leann = self.memory if hasattr(self.memory, '_get_embedding') else getattr(self.memory, 'leann', None)
                if leann is not None and hasattr(leann, '_get_embedding'):
                    emb_result = leann._get_embedding(text)
                    # _get_embedding returns (int8_vec, scale) or np.ndarray
                    import numpy as np
                    if isinstance(emb_result, tuple):
                        int8_vec, scale = emb_result
                        vec_np = int8_vec.astype(np.float32) * scale
                    else:
                        vec_np = np.array(emb_result, dtype=np.float32)
                    # Pad or truncate to 1024 for compatibility with MoIRA
                    target_dim = 1024
                    if len(vec_np) < target_dim:
                        vec_np = np.pad(vec_np, (0, target_dim - len(vec_np)))
                    else:
                        vec_np = vec_np[:target_dim]
                    return torch.tensor(vec_np, dtype=torch.float32).unsqueeze(0)
            except Exception as e:
                self.logger.debug(f"_text_to_vec embedding fallback: {e}")
        
        # Deterministic char-hash fallback (when no embedding model available)
        vec = torch.zeros(1, 1024)
        for i, ch in enumerate(text[:512]):
            idx = ord(ch) % 1024
            vec[0, idx] += (ord(ch) / 255.0) * ((-1) ** i) * 0.1
        norm = vec.norm()
        return vec / norm if norm > 0 else vec
    
    def _load_conversation(self):
        """Load persisted conversation from disk."""
        try:
            conv_path = Path(self.CONVERSATION_FILE)
            if conv_path.exists():
                with open(conv_path, 'r', encoding='utf-8') as f:
                    self.state["conversation"] = json.load(f)
                self.logger.info(f"GIE: Loaded {len(self.state['conversation'])} conversation entries")
        except Exception as e:
            self.logger.warning(f"GIE: Could not load conversation: {e}")
            self.state["conversation"] = []
    
    def _save_conversation(self):
        """Persist conversation to disk (last 500 entries)."""
        try:
            conv_path = Path(self.CONVERSATION_FILE)
            conv_path.parent.mkdir(parents=True, exist_ok=True)
            # ═══ Rolling window: keep last 500 entries ═══
            conversation = self.state["conversation"][-500:]
            with open(conv_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=1)
        except Exception as e:
            self.logger.warning(f"GIE: Could not save conversation: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GIE Agent module loaded. Use test_system.py to run full integration test.")
