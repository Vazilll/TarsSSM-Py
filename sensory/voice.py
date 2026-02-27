"""
voice.py — TARS v3 Voice System.

Architecture:
  VAD:  Silero VAD (ONNX) — voice activity detection
  STT:  Whisper Tiny (faster-whisper) — speech recognition
  TTS:  Piper (ONNX, piper-tts) — speech synthesis

Flow:
  1. Silero VAD listens mic (~1MB model)
  2. Speech detected -> record audio
  3. Speech ended -> send to Whisper
  4. Whisper transcribes -> return text

Dependencies: faster-whisper, sounddevice, numpy, piper-tts (optional)
"""
import logging
import os
import asyncio
import numpy as np

logger = logging.getLogger("Tars.Voice")


class TarsVoice:
    """
    TARS v3 Voice System.
    STT: Whisper Tiny + Silero VAD
    TTS: Piper ONNX (piper-tts Python package)
    """

    # Русские модели Piper (от best → fallback)
    PIPER_MODELS = [
        ("ru_RU-irina-medium.onnx",
         "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx"),
        ("ru_RU-irina-medium.onnx.json",
         "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json"),
    ]

    def __init__(self, model_size="tiny", device="cpu", tts_enabled=False):
        self.device = device
        self.stt_model = None
        self.vad_model = None
        self.tts_enabled = tts_enabled  # TTS выключен по умолчанию
        self.piper_voice = None  # Lazy-loaded
        self.whisper_hotwords = ""
        self.whisper_prompt = "Разговор с ИИ TARS на русском языке."

        # Пути к моделям
        self.voice_dir = os.path.join("models", "voice")
        self.tts_model_path = os.path.join(self.voice_dir, "ru_RU-irina-medium.onnx")
        self.tts_config_path = os.path.join(self.voice_dir, "ru_RU-irina-medium.onnx.json")

        # ── 0. Whisper Context (из обучающего корпуса) ──
        import json as _json
        ctx_path = os.path.join(self.voice_dir, "whisper_context.json")
        if os.path.exists(ctx_path):
            try:
                with open(ctx_path, 'r', encoding='utf-8') as f:
                    ctx = _json.load(f)
                self.whisper_hotwords = " ".join(ctx.get("hotwords", [])[:100])
                prompt = ctx.get("initial_prompt", "")
                if prompt:
                    self.whisper_prompt = f"ТАРС — русскоязычный ИИ. {prompt[:200]}"
                logger.info(f"Voice: Whisper контекст загружен ({ctx.get('keywords_count', 0)} слов)")
            except Exception as e:
                logger.warning(f"Voice: whisper_context.json: {e}")

        # -- 1. Whisper Tiny STT --
        try:
            from faster_whisper import WhisperModel
            whisper_path = os.path.join(self.voice_dir, "whisper_tiny")
            if os.path.exists(os.path.join(whisper_path, "model.bin")):
                logger.info(f"Voice: Whisper Tiny загружен из {whisper_path}")
                self.stt_model = WhisperModel(whisper_path, device=device, compute_type="int8")
            else:
                logger.info("Voice: Скачивание Whisper Tiny...")
                self.stt_model = WhisperModel("tiny", device=device, compute_type="int8")
        except ImportError:
            logger.warning("Voice: faster-whisper не установлен. STT отключён.")
        except Exception as e:
            logger.error(f"Voice STT Error: {e}")

        # -- 2. Silero VAD --
        vad_path = os.path.join(self.voice_dir, "silero_vad.onnx")
        if os.path.exists(vad_path):
            try:
                import onnxruntime
                self.vad_model = onnxruntime.InferenceSession(vad_path)
                logger.info("Voice: Silero VAD загружен (ONNX)")
            except ImportError:
                logger.warning("Voice: onnxruntime не установлен. VAD отключён.")
            except Exception as e:
                logger.error(f"Voice VAD Error: {e}")
        else:
            logger.info("Voice: Silero VAD не найден. Используется energy-based VAD.")

        # -- 3. Piper TTS (подготовка) --
        self._init_piper_tts()

    def _init_piper_tts(self):
        """Подготовка Piper TTS (загрузка модели, если есть)."""
        if os.path.exists(self.tts_model_path) and os.path.exists(self.tts_config_path):
            try:
                from piper import PiperVoice
                self.piper_voice = PiperVoice.load(
                    self.tts_model_path,
                    config_path=self.tts_config_path,
                    use_cuda=False,
                )
                logger.info("Voice: Piper TTS загружен (ru_RU-irina-medium)")
            except ImportError:
                logger.info("Voice: piper-tts не установлен. pip install piper-tts")
            except Exception as e:
                logger.warning(f"Voice: Piper load error: {e}")
        else:
            logger.info(
                f"Voice: Piper модель не найдена ({self.tts_model_path}). "
                f"Скачайте: python -c \"from sensory.voice import TarsVoice; TarsVoice.download_piper_model()\""
            )

    @staticmethod
    def download_piper_model(voice_dir="models/voice"):
        """Скачивает русскую модель Piper (irina-medium, ~30MB)."""
        os.makedirs(voice_dir, exist_ok=True)
        try:
            import urllib.request
            for name, url in TarsVoice.PIPER_MODELS:
                dst = os.path.join(voice_dir, name)
                if os.path.exists(dst):
                    print(f"  ✅ {name}: уже есть")
                    continue
                print(f"  📥 Скачивание {name}...")
                urllib.request.urlretrieve(url, dst)
                size_mb = os.path.getsize(dst) / 1024 / 1024
                print(f"  ✅ {name}: {size_mb:.1f} MB")
            print("  🎤 Piper TTS готов к использованию!")
        except Exception as e:
            print(f"  ❌ Ошибка скачивания: {e}")

    async def transcribe(self, audio_path: str) -> str:
        """Транскрипция аудиофайла → текст (русский)."""
        if not self.stt_model:
            return ""
        try:
            kw = dict(
                beam_size=1,
                language="ru",
                initial_prompt=self.whisper_prompt,
                temperature=0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=400),
            )
            if self.whisper_hotwords:
                kw["hotwords"] = self.whisper_hotwords

            segments, info = self.stt_model.transcribe(audio_path, **kw)
            text = "".join([s.text for s in segments]).strip()
            logger.debug(f"STT: lang={info.language} prob={info.language_probability:.2f}")
            return text
        except Exception as e:
            logger.error(f"STT Error: {e}")
            return ""

    def _detect_speech_silero(self, audio_chunk: np.ndarray, state, sr=16000):
        """Проверка голосовой активности через Silero VAD."""
        if self.vad_model is None:
            # Fallback: energy-based
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            return energy > 0.02, state, float(energy)

        try:
            import torch
            sr_tensor = np.array(sr, dtype=np.int64)
            ort_inputs = {
                "input": torch.from_numpy(audio_chunk).unsqueeze(0).numpy(),
                "sr": sr_tensor,
                "state": state
            }
            vad_out_all = self.vad_model.run(None, ort_inputs)
            prob = float(vad_out_all[0][0][0])
            new_state = vad_out_all[1]
            return prob > 0.4, new_state, prob
        except Exception as e:
            logger.debug(f"VAD Error: {e}")
            energy = np.sqrt(np.mean(audio_chunk ** 2))
            return energy > 0.02, state, float(energy)

    async def listen_stream(self) -> str:
        """
        Слушает микрофон через Silero VAD.
        Whisper НЕ работает постоянно — только когда VAD обнаружил речь.
        """
        try:
            import sounddevice as sd
        except ImportError:
            logger.warning("Voice: sounddevice не установлен. pip install sounddevice")
            return None

        rate = 16000
        chunk_size = 512
        max_listen_sec = 30
        max_silence_chunks = 25  # ~0.8 сек тишины = конец фразы

        # VAD state
        vad_state = np.zeros((2, 1, 128), dtype=np.float32)

        frames = []
        is_recording = False
        silent_chunks = 0

        logger.info("Voice: Ожидание голоса (Silero VAD)...")

        try:
            total_chunks = int(rate / chunk_size * max_listen_sec)
            for _ in range(total_chunks):
                # Запись чанка
                audio_chunk = sd.rec(chunk_size, samplerate=rate, channels=1, dtype='float32')
                sd.wait()
                audio_chunk = audio_chunk.flatten()

                # VAD
                is_speech, vad_state, prob = self._detect_speech_silero(audio_chunk, vad_state, rate)

                # Визуализация
                peak = np.abs(audio_chunk).max()
                filled = int(min(peak * 20, 20))
                meter = "[" + "#" * filled + "-" * (20 - filled) + "]"
                import sys
                sys.stdout.write(f"\rMic: {meter} VAD:{prob:.2f} ")
                sys.stdout.flush()

                if is_speech:
                    if not is_recording:
                        print(f"\nVoice: [Слушаю]...")
                        is_recording = True
                    silent_chunks = 0
                    frames.append((audio_chunk * 32767).astype(np.int16).tobytes())
                elif is_recording:
                    frames.append((audio_chunk * 32767).astype(np.int16).tobytes())
                    silent_chunks += 1
                    if silent_chunks > max_silence_chunks:
                        print("\nVoice: Обработка...")
                        break

                await asyncio.sleep(0.01)

            print("")

            if frames:
                # Сохраняем WAV и отправляем в Whisper
                import wave
                temp_file = "data/temp_record.wav"
                os.makedirs("data", exist_ok=True)
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(rate)
                    wf.writeframes(b"".join(frames))

                text = await self.transcribe(temp_file)
                if text:
                    print(f"Voice: '{text}'")
                    return text

            return None

        except Exception as e:
            logger.error(f"Voice Stream Error: {e}")
            return None

    async def speak(self, text: str):
        """Синтез речи через Piper TTS (ONNX) или системный fallback."""
        if not text:
            return

        logger.info(f"TARS: {text}")

        if not self.tts_enabled:
            logger.debug("Voice: TTS отключён (tts_enabled=False)")
            return

        output_wav = "data/last_response.wav"
        os.makedirs("data", exist_ok=True)

        # 1. Piper TTS (Python-пакет piper-tts)
        if self.piper_voice is not None:
            try:
                import wave
                with wave.open(output_wav, 'wb') as wav_file:
                    self.piper_voice.synthesize(text, wav_file)

                if os.path.exists(output_wav):
                    self._play_audio(output_wav)
                    return
            except Exception as e:
                logger.error(f"TTS Piper Error: {e}")

        # 2. Piper CLI (внешний бинарник, legacy)
        import shutil
        tts_model_legacy = os.path.join(self.voice_dir, "voice.onnx")
        if os.path.exists(tts_model_legacy) and shutil.which("piper"):
            try:
                import subprocess
                cmd = ["piper", "--model", tts_model_legacy, "--output_file", output_wav]
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                process.communicate(input=text.encode('utf-8'))

                if os.path.exists(output_wav):
                    self._play_audio(output_wav)
                    return
            except Exception as e:
                logger.error(f"TTS Piper CLI Error: {e}")

        # 3. Системный fallback (pyttsx3)
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.say(text)
            engine.runAndWait()
            del engine
        except ImportError:
            logger.info("TTS: Piper и pyttsx3 недоступны. Озвучивание пропущено.")
        except Exception as e:
            logger.error(f"TTS Error: {e}")

    def _play_audio(self, wav_path: str):
        """Воспроизводит WAV-файл (кроссплатформенно)."""
        try:
            import sys as _sys
            if _sys.platform == "win32":
                import winsound
                winsound.PlaySound(wav_path, winsound.SND_FILENAME)
            else:
                import subprocess
                subprocess.run(["aplay", wav_path], capture_output=True)
        except Exception as e:
            logger.debug(f"Audio playback error: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    voice = TarsVoice(tts_enabled=True)
    asyncio.run(voice.speak("Привет! Я готов к работе."))
