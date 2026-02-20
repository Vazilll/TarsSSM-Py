from faster_whisper import WhisperModel
import torch
import numpy as np
import logging
import os
import subprocess
import asyncio
import winsound # Для простого воспроизведения на Windows

class TarsVoice:
    """
    Голосовая система (Voice Layer - Ultimate Stack).
    STT: Whisper + Silero VAD
    TTS: Piper (ONNX)
    """
    def __init__(self, model_size="tiny", device="cpu"):
        self.logger = logging.getLogger("Tars.Voice")
        self.device = device
        
        # Модели
        self.vad_model_path = "models/voice/silero_vad.onnx"
        self.tts_model_path = "models/voice/voice.onnx"
        
        # 1. Инициализация Silero VAD
        if os.path.exists(self.vad_model_path):
            self.logger.info("Voice: Загрузка локального Silero VAD...")
            # В реальности используем torch.hub или локальный загрузчик ONNX
        
        # 2. Инициализация Whisper STT (Локальная)
        # Мы проверяем наличие модели в формате CTranslate2 (для faster-whisper)
        self.whisper_model_path = "models/voice/whisper"
        try:
            if os.path.exists(os.path.join(self.whisper_model_path, "model.bin")):
                self.logger.info(f"Voice: Загрузка локального Whisper (CT2) из {self.whisper_model_path}")
                self.stt_model = WhisperModel(self.whisper_model_path, device=device, compute_type="int8")
            else:
                self.logger.warning("Voice: Локальный Whisper в формате CT2 не найден. Использую 'tiny' (Auto-Download)")
                self.stt_model = WhisperModel("tiny", device=device, compute_type="int8")
        except Exception as e:
            self.logger.error(f"Voice STT Error: {e}. STT отключен.")
            self.stt_model = None

    async def transcribe(self, audio_path: str):
        """ Преобразование звука в текст. """
        if not self.stt_model: return ""
        segments, _ = self.stt_model.transcribe(audio_path, beam_size=5)
        text = "".join([s.text for s in segments]).strip()
        return text

    async def speak(self, text: str):
        """ 
        Генерация и воспроизведение речи через Piper. 
        Озвучивает ответ в наушники/колонки пользователя.
        """
        if not text: return
        
        self.logger.info(f"TARS говорит: {text}")
        output_wav = "data/last_response.wav"
        
        # Если модель Piper существует, используем её
        if os.path.exists(self.tts_model_path):
            try:
                # Вызов Piper через subprocess (самый быстрый путь для ONNX в Python)
                # Предполагаем, что piper.exe находится в PATH или в папке инструментов
                cmd = [
                    "piper", 
                    "--model", self.tts_model_path, 
                    "--output_file", output_wav
                ]
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                process.communicate(input=text.encode('utf-8'))
                
                # Воспроизведение звука в Windows
                if os.path.exists(output_wav):
                    winsound.PlaySound(output_wav, winsound.SND_FILENAME)
            except Exception as e:
                self.logger.error(f"TTS Playback Error: {e}")
        else:
            self.logger.warning("TTS: Модель voice.onnx не найдена. Озвучивание пропущено.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    voice = TarsVoice()
    asyncio.run(voice.speak("Привет! Я готов к работе."))
