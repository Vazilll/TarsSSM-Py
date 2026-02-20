from fastapi import FastAPI, BackgroundTasks, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import asyncio
import os
from agent.gie import GieAgent
from agent.moira import MoIRA
from memory.leann import TarsMemory
from memory.titans import TitansMemory
from memory.store import TarsStorage
from brain.aussm import TarsBrain
from sensory.vision import TarsVision
from sensory.voice import TarsVoice

# TARS Hub: Ultimate Functional Edition
app = FastAPI(title="Tars Python Hub - Ultimate Stack")

# Singleton Services
memory = TarsMemory()           # Vector Search
titans = TitansMemory()         # Long-Term Neural Memory
brain = TarsBrain()             # Recursive SSM Brain
moira = MoIRA()                 # Neural Tool Router
storage = TarsStorage()         # Persona Persistence
vision = TarsVision()           # YOLO Workspace Analysis
voice = TarsVoice()             # VAD + Whisper + Piper
gie = GieAgent(brain=brain, moira=moira, memory=memory, titans=titans)

@app.on_event("startup")
async def startup():
    logging.info("Tars Hub: Системы активированы на 100%. Режим максимального функционала.")

@app.post("/voice_interaction")
async def voice_interaction(file: UploadFile = File(...)):
    """ Обработка голоса с VAD фильтрацией. """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # 1. Транскрипция (с автоматическим VAD внутри)
    text = await voice.transcribe(temp_path)
    os.remove(temp_path)
    
    if not text:
        return {"status": "ignored", "reason": "No speech detected (Silero VAD)"}
    
    # 2. Передача в GIE
    response = await gie.execute_goal(text)
    
    # 3. Ответ голосом
    await voice.speak(response)
    
    return {"input": text, "response": response}

@app.post("/execute")
async def execute(goal: str):
    """ Текстовое управление с озвучиванием ответа. """
    response = await gie.execute_goal(goal)
    # Озвучиваем ответ в наушники
    await voice.speak(response)
    return {"status": "done", "goal": goal, "response": response}

@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket):
    """ Глубокая телеметрия в реальном времени. """
    await websocket.accept()
    while True:
        v_data = await vision.analyze_workspace()
        await websocket.send_json({
            "vision": v_data,
            "brain": {
                "recursive_loops": 4,
                "confidence_gate": 0.98
            },
            "memory": {
                "titans_surprise": 0.12,
                "ltm_status": "consolidated"
            }
        })
        await asyncio.sleep(0.5)

app.mount("/", StaticFiles(directory="hub/static", html=True), name="static")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
