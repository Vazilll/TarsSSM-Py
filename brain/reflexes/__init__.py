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

# Re-export ReflexClassifier из brain/ (обратная совместимость)
try:
    from brain.reflex_classifier import ReflexClassifier
except ImportError:
    ReflexClassifier = None
