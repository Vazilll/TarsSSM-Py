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
    from brain.reflex import ReflexCore
except ImportError:
    ReflexCore = None
from memory.titans import TitansMemory
from memory.store import TarsStorage
from agent.executor import ActionEngine
from agent.moira import MoIRA

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
        self.reflex = ReflexCore()
        self.moira = moira
        self.memory = memory
        self.titans = titans
        self.storage = TarsStorage()
        self.executor = ActionEngine()
        self.logger = logging.getLogger("Tars.GIE")
        
        # Состояние сессии
        self.state = {
            "history": [],
            "last_thought": None,
            "session_goals": [],
            "total_processed": 0,
        }

    async def execute_goal(self, goal: str, fast_callback=None):
        """Главный цикл обработки цели."""
        self.state["total_processed"] += 1
        self.state["session_goals"].append(goal)
        self.logger.info(f"GIE: Цель #{self.state['total_processed']} → {goal[:60]}...")

        # ═══ Stage 0: Рефлекс (System 0) ═══
        reflex_result = await self.reflex.react(goal)
        if reflex_result:
            response = reflex_result["response"]
            self.logger.info(f"GIE: Рефлекс [{reflex_result['action']}]: {response[:40]}...")
            await self.storage.remember(f"Reflex: {goal} → {response[:50]}")
            
            # Простые операции (состояние, приветствие) лучше вернуть сразу
            if reflex_result['action'] in ['greet', 'status', 'identity', 'time', 'shutdown', 'acknowledge']:
                return {"text": response, "tokens": 0, "duration": 0.0, "tps": 0.0}
            
            if fast_callback:
                await fast_callback(response)
                # продолжать глубокое обдумывание после выдачи шутки/рефлекса
            else:
                return {"text": response, "tokens": 0, "duration": 0.0, "tps": 0.0}

        # ═══ Stage 1: RRN Recursive Reflex (System 1) ═══
        # RRN сам решает: если MinGRU может ответить — ответит,
        # если нет — вернёт None и мы идём в глубокий анализ (System 2)
        quick_result = await self.rrn.fast_reply(goal)
        if quick_result is not None:
            quick_resp = quick_result["text"]
            self.logger.info(f"GIE: RRN (Light Model) Think: {quick_resp[:40]}...")
            if not quick_result.get("is_garbage", False):
                if fast_callback:
                    await fast_callback(quick_resp)
                else:
                    return {"text": quick_resp, "tokens": 0, "duration": 0.0, "tps": 0.0}
            else:
                self.logger.info(f"GIE: RRN сгенерировал шум. Переход к глубокому анализу...")
        
        self.logger.info("GIE: Переход к глубокому анализу (Thinking Table)...")

        # ═══ Stage 2: Relational Grounding ═══
        relational_map = await self.rrn.precompute_grounding(goal, self.memory, self.titans)
        
        # Персоналия
        persona = await self.storage.retrieve_memories(goal)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Контекст для мозга
        full_context = (
            f"Время: {current_time}\n"
            f"Цель: {goal}\n"
            f"{relational_map}\n"
            f"Персоналия: {persona}\n"
            f"Последние действия: {self.state['history'][-2:]}"
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
                    # Кодируем текст в токены (UTF-8 byte-level)
                    goal_tokens = torch.tensor([list(goal.encode('utf-8', errors='ignore')[:512])], dtype=torch.long)
                    result = self.brain.think(goal_tokens, memory_vec=recall_vec)
                    thought = goal  # Мысль = сам запрос (генерация в generate_mamba)
                    stats = {"tokens": goal_tokens.shape[1], "duration": 0.0}
                    p_value = result.get("p_value", 2.0)
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
                
                await self.storage.remember(f"Задача: {goal} → {thought_text[:100]}")
                
                return {
                    "text": thought_text,
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
