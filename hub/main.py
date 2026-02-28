from fastapi import FastAPI, BackgroundTasks, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import asyncio
import os
import queue
import threading
from agent.gie import GieAgent
from agent.moira import MoIRA
from memory.leann import TarsMemory
from memory.titans import TitansMemory
from memory.store import TarsStorage
try:
    from brain.mamba2.model import TarsMamba2LM as TarsBrain
except ImportError:
    TarsBrain = None
from brain.reflexes.reflex_dispatcher import ReflexDispatcher
from sensory.vision import TarsVision
from sensory.voice import TarsVoice

# TARS Hub: Ultimate Functional Edition
app = FastAPI(title="Tars Python Hub - Ultimate Stack")

# Singleton Services
memory = TarsMemory()           # Vector Search
titans = TitansMemory()         # Long-Term Neural Memory
try:
    brain = TarsBrain.load_pretrained()[0] if TarsBrain else None
except Exception:
    brain = None
moira = MoIRA()                 # Neural Tool Router
storage = TarsStorage()         # Persona Persistence
vision = TarsVision()           # YOLO Workspace Analysis
voice = TarsVoice()             # VAD + Whisper + Piper + IntonationSensor
dispatcher = ReflexDispatcher(memory=memory)  # 7 сенсоров параллельно
gie = GieAgent(brain=brain, moira=moira, memory=memory, titans=titans)

@app.on_event("startup")
async def startup():
    logging.info("Tars Hub: Системы активированы на 100%. Режим максимального функционала.")

@app.post("/voice_interaction")
async def voice_interaction(file: UploadFile = File(...)):
    """
    Обработка голоса через волновую архитектуру:
      Audio → Whisper + IntonationSensor → текст + эмоция
      → ReflexDispatcher (7 сенсоров) → ReflexContext
      → GIE (brain.think с контекстом) → ответ
      → Piper TTS (адаптивный к эмоции) → озвучка
    """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # 1. Транскрипция + Интонация (параллельно)
    text, intonation_data = await voice.transcribe_with_intonation(temp_path)
    os.remove(temp_path)
    
    if not text:
        return {"status": "ignored", "reason": "No speech detected (Silero VAD)"}
    
    # 2. ReflexDispatcher: 7 сенсоров с данными интонации
    ctx = dispatcher.dispatch(text, intonation_data=intonation_data)
    
    if ctx.can_handle_fast and ctx.fast_response:
        # Рефлекс обработал — мгновенный ответ
        response = ctx.fast_response
        emotion = ctx.voice_emotion if ctx.has_voice_data else "neutral"
    else:
        # Передаём в GIE с полным контекстом
        response = await gie.execute_goal(text)
        emotion = ctx.voice_emotion if ctx.has_voice_data else ctx.dominant_emotion
    
    # 3. Ответ голосом (адаптивный Piper)
    await voice.speak(response, emotion=emotion)
    
    # 4. История для ContextSensor
    dispatcher.add_to_history(text, response, ctx.intent)
    
    return {
        "input": text,
        "response": response,
        "reflex": ctx.summary_line(),
        "intonation": intonation_data,
        "supplement": ctx.is_supplement,
    }

@app.post("/execute")
async def execute(goal: str):
    """
    Текстовое управление через ReflexDispatcher.
    Голос = сенсорика, текст = сенсорика. Одна матрица.
    """
    ctx = dispatcher.dispatch(goal)
    
    if ctx.can_handle_fast and ctx.fast_response:
        response = ctx.fast_response
    else:
        response = await gie.execute_goal(goal)
    
    # Озвучиваем ответ
    await voice.speak(response, emotion=ctx.dominant_emotion)
    
    dispatcher.add_to_history(goal, response, ctx.intent)
    return {"status": "done", "goal": goal, "response": response, "reflex": ctx.summary_line()}

@app.websocket("/ws/voice_stream")
async def voice_stream(websocket: WebSocket):
    """
    WebSocket: потоковый голос с supplement injection.
    
    Поток:
      1. Клиент шлёт аудио-чанки
      2. VAD → Whisper → текст + интонация
      3. Первый сегмент → brain.think() запускается
      4. Следующие сегменты → supplement_queue → injection между волнами
      5. Результат → обратно клиенту
    """
    await websocket.accept()
    sup_queue = queue.Queue()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("type") == "audio_segment":
                # Голосовой сегмент
                text = data.get("text", "")
                intonation = data.get("intonation", {})
                is_first = data.get("is_first", True)
                
                if is_first:
                    # Первый сегмент — полный dispatch + think
                    ctx = dispatcher.dispatch(text, intonation_data=intonation)
                    response = await gie.execute_goal(text)
                    await websocket.send_json({
                        "type": "response",
                        "text": response,
                        "reflex": ctx.summary_line(),
                    })
                else:
                    # Дополнение — кладём в очередь
                    sup_queue.put({
                        "text": text,
                        "tokens": None,  # токенизация на стороне brain
                        "intonation": intonation,
                    })
                    await websocket.send_json({
                        "type": "supplement_received",
                        "text": text,
                    })
            
            elif data.get("type") == "stop":
                break
    except Exception as e:
        logging.error(f"Voice stream error: {e}")
    finally:
        await websocket.close()

@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket):
    """ Глубокая телеметрия в реальном времени. """
    await websocket.accept()
    while True:
        v_data = await vision.analyze_workspace()
        await websocket.send_json({
            "vision": v_data,
            "brain": {
                "model_loaded": brain is not None and hasattr(brain, 'blocks'),
                "n_layers": getattr(brain, 'n_layers', 0),
                "d_model": getattr(brain, 'd_model', 0),
                "params": sum(p.numel() for p in brain.parameters()) if brain is not None else 0,
            },
            "memory": {
                "leann_docs": len(memory.leann.texts) if hasattr(memory, 'leann') else 0,
                "storage_facts": len(storage._hub._fact_log) if hasattr(storage, '_hub') else 0,
            },
            "session": {
                "goals_processed": gie.state.get("total_processed", 0),
                "history_size": len(gie.state.get("history", [])),
            },
            "dispatcher": dispatcher.get_stats(),
        })
        await asyncio.sleep(1.0)

app.mount("/", StaticFiles(directory="hub/static", html=True), name="static")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)

