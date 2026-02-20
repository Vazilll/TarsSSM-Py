import asyncio
import logging
from typing import Optional, Dict, Any, List
import torch
import json
import os
from brain.aussm import TarsBrain
from brain.reflex import ReflexCore
from memory.titans import TitansMemory
from memory.store import TarsStorage
from agent.executor import ActionEngine
from agent.moira import MoIRA

class GieAgent:
    """
    Ultimate GIE Agent.
    Центральный координатор с поддержкой памяти состояний и глубокой интеграции.
    """
    def __init__(self, brain: TarsBrain = None, moira: MoIRA = None, memory: Any = None, titans: TitansMemory = None):
        self.brain = brain
        self.reflex = ReflexCore()
        self.moira = moira
        self.memory = memory
        self.titans = titans
        self.storage = TarsStorage()
        self.executor = ActionEngine()
        self.logger = logging.getLogger("Tars.GIE")
        
        # Внутреннее состояние (Working Memory / State Persistence)
        self.state = {
            "history": [],
            "last_thought": None,
            "session_goals": [],
            "current_context_vector": torch.zeros(1, 1024)
        }

    async def execute_goal(self, goal: str):
        """ Главный цикл с глубокой рефлексией. """
        self.logger.info(f"GIE: Принята комплексная цель -> {goal}")
        self.state["session_goals"].append(goal)

        # 0. Мгновенная реакция (Reflex Core)
        quick_resp = await self.reflex.react(goal)
        if quick_resp:
            return quick_resp

        # 1. Сбор контекста (Grounding)
        # Достаем краткосрочные факты из LEANN
        kb_context = await self.memory.get_context(goal) if self.memory else ""
        # Достаем глубокие знания из Titans
        recall_vec = await self.titans.get_recall(torch.randn(1, 1024)) if self.titans else None
        # Достаем персоналию из Mem0
        persona = await self.storage.retrieve_memories(goal)
        
        full_context = f"Goal: {goal}\nKnowledge: {kb_context}\nPersona: {persona}\nHistory: {self.state['history'][-3:]}"

        for step in range(10): # Увеличен лимит шагов для сложных задач
            self.logger.info(f"GIE: Когнитивный шаг {step + 1}")

            # 2. Мышление (AUSSM Brain)
            # Передаем не только текст, но и вектор прошлого состояния
            thought = await self.brain.think(full_context, memory_context=recall_vec)
            self.state["last_thought"] = thought
            
            # 3. Маршрутизация (MoIRA)
            # В реальности AUSSM выдает вектор для MoIRA
            tool, params, confidence = await self.moira.route(torch.randn(1, 1, 1024))
            
            if tool == "FinalAnswer" and confidence > 0.8:
                self.logger.info("GIE: Задача решена финально.")
                # Обучение Titans на основе успеха
                if self.titans:
                    await self.titans.update(torch.randn(1, 1024))
                await self.storage.remember(f"Успех: {goal} -> {thought}")
                return str(thought) if thought else "Задача выполнена, но мысль не сформирована."

            # 4. Действие (ActionEngine)
            observation = await self._act(tool, params)
            
            # 5. Рефлексия и обновление состояния
            log_entry = {"step": step, "tool": tool, "obs": observation}
            self.state["history"].append(log_entry)
            full_context += f"\nAction: {tool} result: {observation}"
            
            # Самокоррекция: если результат — ошибка, меняем стратегию
            if "Error" in observation:
                full_context += "\nSystem: Action failed, need alternative strategy."

    async def _act(self, tool: str, params: Dict[str, Any]) -> str:
        """ Адаптивное выполнение действий через ActionEngine. """
        # Маппинг высокоуровневых команд MoIRA в низкоуровневые ActionEngine
        action_map = {
            "Python": "execute_script",
            "Terminal": "run_command",
            "Browser": "open_url",
            "Vision": "analyze_workspace", # Обновлено для соответствия Sensory
            "Click": "click",
            "Type": "type"
        }
        
        cmd = action_map.get(tool, tool.lower())
        result = await self.executor.execute(cmd, params)
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Пример инициализации (в реальности через Hub)
    # gie = GieAgent(...)
