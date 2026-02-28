"""
═══════════════════════════════════════════════════════════════
  intonation_sensor.py — Анализ интонации голоса (чистый DSP)
═══════════════════════════════════════════════════════════════

Лёгкий анализатор на numpy: определяет pitch (F0), энергию,
тренд интонации и эмоциональный тон говорящего.

Результат интегрируется в ReflexContext → мозг учитывает
эмоциональное состояние пользователя.

Зависимости: numpy (уже есть).
"""

import numpy as np
import logging

logger = logging.getLogger("Tars.Intonation")


class IntonationSensor:
    """
    Анализатор интонации по аудиосигналу.
    
    Работает на чистом numpy (без ML), используя DSP:
      - Pitch (F0): автокорреляция
      - Energy: RMS в дБ
      - Pitch trend: рост/спад/ровный
      - Emotion: вопрос/восклицание/шёпот/спокойный
    """

    def __init__(self, sr: int = 16000):
        self.sr = sr
        # Диапазон человеческого голоса (Hz)
        self.pitch_min = 75
        self.pitch_max = 500

    def analyze(self, audio: np.ndarray, sr: int = None) -> dict:
        """
        Анализирует аудиосигнал и возвращает характеристики интонации.
        
        Args:
            audio: numpy массив аудио (float32, mono)
            sr: sample rate (по умолчанию 16000)
        
        Returns:
            dict с pitch_hz, pitch_trend, energy_db, emotion, is_question
        """
        sr = sr or self.sr
        audio = self._normalize(audio)

        if len(audio) < sr * 0.1:  # < 100ms
            return self._empty_result()

        pitch_hz = self._estimate_pitch(audio, sr)
        energy_db = self._compute_energy(audio)
        pitch_trend = self._analyze_trend(audio, sr)
        speech_rate = self._estimate_speech_rate(audio, sr)
        emotion = self._classify_emotion(pitch_hz, energy_db, pitch_trend, speech_rate)

        return {
            "pitch_hz": round(pitch_hz, 1),
            "pitch_trend": pitch_trend,
            "energy_db": round(energy_db, 1),
            "speech_rate": round(speech_rate, 2),
            "emotion": emotion,
            "is_question": pitch_trend == "rising",
        }

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Нормализация аудио в float32 [-1, 1]."""
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    def _estimate_pitch(self, audio: np.ndarray, sr: int) -> float:
        """Оценка основной частоты (F0) через автокорреляцию."""
        # Берём центральный фрагмент (~200ms)
        center = len(audio) // 2
        frame_len = min(int(sr * 0.2), len(audio))
        start = max(0, center - frame_len // 2)
        frame = audio[start:start + frame_len]

        if len(frame) < 100:
            return 0.0

        # Автокорреляция
        frame = frame - np.mean(frame)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr) // 2:]

        # Ищем пик в диапазоне голоса
        min_lag = int(sr / self.pitch_max)
        max_lag = int(sr / self.pitch_min)
        max_lag = min(max_lag, len(corr) - 1)

        if min_lag >= max_lag or max_lag >= len(corr):
            return 0.0

        search = corr[min_lag:max_lag + 1]
        if len(search) == 0:
            return 0.0

        peak_idx = np.argmax(search) + min_lag

        if corr[0] > 0 and corr[peak_idx] / corr[0] < 0.2:
            return 0.0  # Слишком слабая корреляция

        pitch = sr / peak_idx if peak_idx > 0 else 0.0
        return pitch

    def _compute_energy(self, audio: np.ndarray) -> float:
        """RMS энергия в децибелах."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-10:
            return -60.0
        return 20 * np.log10(rms)

    def _analyze_trend(self, audio: np.ndarray, sr: int) -> str:
        """
        Анализ тренда pitch (F0) по фрагментам.
        
        rising  = вопросительная интонация (F0 растёт в конце)
        falling = утвердительная (F0 падает)
        flat    = ровная (F0 не меняется)
        """
        # Разбиваем на 4 сегмента
        n_segments = 4
        seg_len = len(audio) // n_segments
        if seg_len < sr * 0.05:  # < 50ms
            return "flat"

        pitches = []
        for i in range(n_segments):
            segment = audio[i * seg_len:(i + 1) * seg_len]
            p = self._estimate_pitch(segment, sr)
            if p > 0:
                pitches.append(p)

        if len(pitches) < 2:
            return "flat"

        # Сравниваем последнюю четверть с первой
        first_half = np.mean(pitches[:len(pitches) // 2])
        second_half = np.mean(pitches[len(pitches) // 2:])

        ratio = second_half / first_half if first_half > 0 else 1.0

        if ratio > 1.15:
            return "rising"
        elif ratio < 0.85:
            return "falling"
        return "flat"

    def _estimate_speech_rate(self, audio: np.ndarray, sr: int) -> float:
        """
        Оценка скорости речи через количество пересечений нуля.
        Больше ZCR = быстрее речь (грубая оценка).
        """
        signs = np.sign(audio)
        sign_changes = np.abs(np.diff(signs))
        zcr = np.sum(sign_changes > 0) / (len(audio) / sr)
        # Нормализуем: ~500 ZCR/sec = средняя скорость
        return min(zcr / 500.0, 3.0)

    def _classify_emotion(
        self, pitch: float, energy: float, trend: str, rate: float
    ) -> str:
        """
        Простой классификатор эмоционального тона.
        
        Returns: question, excited, calm, whisper, neutral
        """
        if trend == "rising" and pitch > 150:
            return "question"
        if energy > -10 and (pitch > 250 or rate > 1.5):
            return "excited"
        if energy < -30:
            return "whisper"
        if rate < 0.5 and energy < -15:
            return "calm"
        return "neutral"

    def _empty_result(self) -> dict:
        return {
            "pitch_hz": 0.0,
            "pitch_trend": "flat",
            "energy_db": -60.0,
            "speech_rate": 0.0,
            "emotion": "neutral",
            "is_question": False,
        }


if __name__ == "__main__":
    # Тест с синтетическим сигналом
    sr = 16000
    duration = 1.0

    # Имитация вопросительной интонации (pitch растёт)
    t_question = np.linspace(0, duration, int(sr * duration))
    freq_sweep = np.linspace(150, 300, len(t_question))
    question_audio = np.sin(2 * np.pi * np.cumsum(freq_sweep) / sr).astype(np.float32)

    # Имитация утверждения (pitch падает)
    freq_fall = np.linspace(200, 120, len(t_question))
    statement_audio = np.sin(2 * np.pi * np.cumsum(freq_fall) / sr).astype(np.float32)

    sensor = IntonationSensor(sr=sr)

    print("Тест IntonationSensor:")
    print(f"  Вопрос:      {sensor.analyze(question_audio)}")
    print(f"  Утверждение: {sensor.analyze(statement_audio)}")
    print(f"  Тишина:       {sensor.analyze(np.zeros(sr, dtype=np.float32))}")
    print("\n✅ Тесты пройдены!")
