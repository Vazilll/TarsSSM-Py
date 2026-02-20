import asyncio
import logging
import torch
import os
from agent.gie import GieAgent
from agent.moira import MoIRA
from memory.leann import TarsMemory
from memory.titans import TitansMemory
from brain.aussm import TarsBrain
from sensory.voice import TarsVoice

async def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Tars.Verify")
    logger.info("=== НАЧАЛО СКВОЗНОЙ ПРОВЕРКИ TARS ===")

    # 1. Инициализация всех компонентов (Склеивание фрагментов)
    memory = TarsMemory()
    titans = TitansMemory()
    brain = TarsBrain()
    moira = MoIRA()
    voice = TarsVoice()
    
    gie = GieAgent(brain=brain, moira=moira, memory=memory, titans=titans)

    # 2. Тестовый ввод (Симуляция сообщения из интерфейса)
    user_goal = "Тарс, запомни мой любимый напиток - кофе, и открой браузер."
    logger.info(f"Input: {user_goal}")

    # 3. Обработка через GIE (Цикл Мысль -> Маршрутизация -> Действие)
    # Здесь задействуется AUSSM с C++ ядрами и Titans
    response = await gie.execute_goal(user_goal)
    logger.info(f"Brain Response: {response}")

    # 4. Проверка озвучки
    # Озвучиваем результат в наушники
    await voice.speak(response)

    logger.info("=== ВЕРИФИКАЦИЯ ЗАВЕРШЕНА. ВСЕ СИСТЕМЫ СКЛЕЕНЫ УСПЕШНО ===")

if __name__ == "__main__":
    asyncio.run(main())
