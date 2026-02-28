import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
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
    from agent.routine_detector import RoutineDetector
except ImportError:
    RoutineDetector = None
try:
    from agent.learning_helper import LearningHelper
except ImportError:
    LearningHelper = None
try:
    from agent.reminders import ReminderService
except ImportError:
    ReminderService = None
try:
    from agent.system_monitor import SystemMonitor
except ImportError:
    SystemMonitor = None
try:
    from agent.meeting_scribe import MeetingScribe
except ImportError:
    MeetingScribe = None
try:
    from agent.notifier import TarsNotifier
except ImportError:
    TarsNotifier = None
try:
    from agent.pomodoro import PomodoroTimer
except ImportError:
    PomodoroTimer = None
try:
    from agent.schedule import StudentSchedule
except ImportError:
    StudentSchedule = None
try:
    from agent.lecture_summarizer import LectureSummarizer
except ImportError:
    LectureSummarizer = None
try:
    from agent.knowledge_graph import KnowledgeGraph
except ImportError:
    KnowledgeGraph = None
try:
    from agent.clipboard_manager import ClipboardManager
except ImportError:
    ClipboardManager = None
try:
    from agent.expense_tracker import ExpenseTracker
except ImportError:
    ExpenseTracker = None
try:
    from agent.quiz_generator import QuizGenerator
except ImportError:
    QuizGenerator = None
try:
    from agent.habit_tracker import HabitTracker
except ImportError:
    HabitTracker = None
try:
    from agent.daily_dashboard import DailyDashboard
except ImportError:
    DailyDashboard = None
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
        
        # ═══ Active Inference (Friston, 2006-2026) ═══
        # Байесовское обновление internal beliefs после каждого наблюдения.
        # Free Energy = complexity + surprise → минимизация определяет поведение.
        self.belief_state = BeliefState(d_state=128) if BeliefState else None
        
        # ═══ Proactive Systems ═══
        self.routine_detector = RoutineDetector() if RoutineDetector else None
        self.learning_helper = LearningHelper() if LearningHelper else None
        self.reminders = ReminderService() if ReminderService else None
        self.system_monitor = SystemMonitor() if SystemMonitor else None
        self.meeting_scribe = MeetingScribe() if MeetingScribe else None
        
        # ═══ Central Notifier (ТАРС пишет первым) ═══
        self.notifier = TarsNotifier(
            reminders=self.reminders,
            monitor=self.system_monitor,
            routine_detector=self.routine_detector,
            learning_helper=self.learning_helper,
            meeting_scribe=self.meeting_scribe,
        ) if TarsNotifier else None
        
        # ═══ Student Features ═══
        self.pomodoro = PomodoroTimer() if PomodoroTimer else None
        self.schedule = StudentSchedule() if StudentSchedule else None
        self.summarizer = LectureSummarizer() if LectureSummarizer else None
        self.knowledge_graph = KnowledgeGraph() if KnowledgeGraph else None
        self.clipboard = ClipboardManager() if ClipboardManager else None
        self.expenses = ExpenseTracker() if ExpenseTracker else None
        
        # ═══ Phase 10: Learning + Consumer ═══
        self.quiz = QuizGenerator(
            learning_helper=self.learning_helper,
            knowledge_graph=self.knowledge_graph,
        ) if QuizGenerator else None
        self.habits = HabitTracker() if HabitTracker else None
        self.file_helper = FileHelper() if FileHelper else None
        self.dashboard = DailyDashboard(
            schedule=self.schedule, reminders=self.reminders,
            pomodoro=self.pomodoro, learning_helper=self.learning_helper,
            habit_tracker=self.habits, expenses=self.expenses,
            knowledge_graph=self.knowledge_graph,
            system_monitor=self.system_monitor,
        ) if DailyDashboard else None
        
        # Состояние сессии
        self.state = {
            "history": [],
            "last_thought": None,
            "session_goals": [],
            "total_processed": 0,
            "cumulative_free_energy": 0.0,
            # ═══ Fix #4: Полная история диалога (Total Memory) ═══
            # Каждый элемент: {"user": str, "tars": str, "time": str, "tier": str}
            "conversation": [],
        }

    async def execute_goal(self, goal: str, fast_callback=None):
        """Главный цикл обработки цели."""
        self.state["total_processed"] += 1
        self.state["session_goals"].append(goal)
        self.logger.info(f"GIE: Цель #{self.state['total_processed']} → {goal[:60]}...")

        # ═══ Proactive: собираем уведомления из всех подсистем ═══
        proactive_hints = ""
        if self.notifier:
            # Утреннее приветствие (раз в день)
            greeting = self.notifier.get_morning_greeting()
            if greeting:
                proactive_hints += f"\n{greeting}"
            
            # Все pending уведомления
            notifications = self.notifier.collect_notifications()
            if notifications:
                proactive_hints += f"\n{self.notifier.format_notifications(notifications)}"

        reflex_result = None
        if self.reflex is not None:
            try:
                reflex_ctx = self.reflex.dispatch(goal)
                if reflex_ctx.can_handle_fast and reflex_ctx.fast_response:
                    reflex_result = {"response": reflex_ctx.fast_response, "action": reflex_ctx.intent}
            except Exception:
                pass
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
        recall_vec = None
        if self.titans:
            try:
                goal_vec = self._text_to_vec(goal)
                recall_vec = await self.titans.get_recall(goal_vec)
            except Exception:
                pass

        # Early Exit check: если RRN уже нашел четкий ответ в короткой памяти
        if "answer" in relational_map.lower() or "решение:" in relational_map.lower():
             self.logger.info("GIE: Обнаружено готовое решение в RRN. Попытка Early Exit...")
             # Подаем сигнал мозгу что можно заканчивать сразу
             full_context += "\nSystem-Hint: Solution found in grounding. Summarize and exit."

        # ═══ Uncertainty-Driven Auto-RAG ═══
        # Если Titans surprise высокий → тема новая для модели → ищем в интернете
        if self.titans and recall_vec is not None:
            try:
                recall_surprise = await self.titans.update(
                    self._text_to_vec(goal)
                )
                if recall_surprise.get("surprised", False):
                    self.logger.info("GIE: 🔍 Высокий surprise → автоматический RAG search")
                    try:
                        from agent.knowledge_injector import KnowledgeInjector
                        injector = KnowledgeInjector(
                            leann=self.storage._hub.leann if hasattr(self.storage, '_hub') else None,
                            titans=self.titans
                        )
                        rag_result = injector.handle_tool("search_web", goal)
                        if rag_result and "[Ошибка" not in rag_result:
                            full_context += f"\n\nНайдено в интернете:\n{rag_result[:500]}"
                            self.logger.info(f"GIE: RAG добавил {len(rag_result)} символов контекста")
                    except Exception as e:
                        self.logger.debug(f"GIE: Auto-RAG failed: {e}")
            except Exception:
                pass

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
                    # Кодируем ПОЛНЫЙ контекст в токены (cp1251 byte-level)
                    brain_input = f"{full_context}\n\nОтветь: {goal}"
                    goal_tokens = torch.tensor(
                        [list(brain_input.encode('cp1251', errors='replace')[:1024])],
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
                    if "Re-evaluate" in goal:
                        self.logger.warning(f"GIE: Повторная сходимость не достигнута. Принимаю текущий результат.")
                    else:
                        self.logger.warning(f"GIE: Сходимость не достигнута (p={p_value:.2f}). Инициирую 1 дополнительный цикл размышления...")
                        full_context += "\nSystem-Hint: Предыдущая итерация не сошлась. Уточни параметры и попробуй снова."
                        goal = f"Re-evaluate: {goal}" # Force recursion
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
                    except Exception:
                        pass
                
                # ═══ Learning Helper: автоматические flashcards ═══
                if self.learning_helper and self.learning_helper.should_create_card(goal):
                    try:
                        self.learning_helper.auto_create_card(goal, thought_text)
                        self.logger.info("GIE: 📝 Авто-карточка создана для повторения")
                    except Exception:
                        pass
                
                # ═══ Routine Detector: логирование для паттерн-анализа ═══
                if self.routine_detector:
                    try:
                        self.routine_detector.log_conversation(goal, thought_text, tier="brain")
                    except Exception:
                        pass
                
                # ═══ Knowledge Graph: авто-наполнение графа знаний ═══
                if self.knowledge_graph:
                    try:
                        self.knowledge_graph.add_from_dialog(goal, thought_text)
                    except Exception:
                        pass
                
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
        
        cmd = action_map.get(tool, tool.lower())
        try:
            result = await self.executor.execute(cmd, params)
            return result
        except Exception as e:
            return f"Error: {e}"


    @staticmethod
    def _text_to_vec(text: str) -> torch.Tensor:
        """Детерминированное преобразование текста в вектор [1, 1024]."""
        vec = torch.zeros(1, 1024)
        for i, ch in enumerate(text[:512]):
            idx = ord(ch) % 1024
            vec[0, idx] += (ord(ch) / 255.0) * ((-1) ** i) * 0.1
        norm = vec.norm()
        return vec / norm if norm > 0 else vec


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GIE Agent module loaded. Use test_system.py to run full integration test.")
