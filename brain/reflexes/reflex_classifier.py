"""
Прокси-модуль: перенаправляет на brain/reflex_classifier.py.

Позволяет импортировать ReflexClassifier как:
  from brain.reflexes.reflex_classifier import ReflexClassifier
  from brain.reflex_classifier import ReflexClassifier  (legacy)
"""
from brain.reflex_classifier import *  # noqa: F401,F403
from brain.reflex_classifier import ReflexClassifier  # explicit re-export
