"""
brain.reflexes — Рефлексная система TARS v3.

Re-exports для удобного доступа:
  from brain.reflexes import ReflexDispatcher, ReflexContext
  from brain.reflexes import ReflexClassifier
"""
from brain.reflexes.reflex_dispatcher import ReflexDispatcher, ReflexContext
from brain.reflexes.sensors import (
    IntentSensor, ComplexitySensor, RAGSensor,
    SystemSensor, EmotionSensor, ContextSensor, VoiceSensor,
)
from brain.reflexes.reflex_classifier import ReflexClassifier
