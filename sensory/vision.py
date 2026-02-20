import cv2
import numpy as np
import bettercam
from ultralytics import YOLO
import logging

class TarsVision:
    """
    Система зрения (Vision Layer).
    Использует YOLOv11 для распознавания интерфейсов и BetterCam для захвата экрана в реальном времени.
    """
    def __init__(self, model_path="models/vision/yolo26n.pt"):
        self.logger = logging.getLogger("Tars.Vision")
        try:
            # Модель глубокого обучения для детекции объектов (кнопки, иконки, окна)
            self.model = YOLO(model_path)
            self.logger.info(f"Vision: Модель YOLO загружена из {model_path}")
        except Exception as e:
            self.logger.warning(f"Vision: Модель YOLO не найдена. Работаем в режиме захвата без ИИ.")
            self.model = None
            
        # Захват экрана на Windows с экстремально высокой частотой кадров (BetterCam)
        self.camera = bettercam.create()
        self.logger.info("Vision: BetterCam инициализирован.")

    def capture_screen(self):
        """ Скриншот рабочего стола в формате BGR для OpenCV. """
        frame = self.camera.grab()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return None

    async def detect_objects(self, frame):
        """ Поиск UI-элементов на кадре через YOLO. """
        if self.model and frame is not None:
            results = self.model(frame, verbose=False)
            # Возвращаем список координат [x1, y1, x2, y2, точность, класс]
            return results[0].boxes.data.tolist() 
        return []

    async def analyze_workspace(self):
        """ Комплексный анализ экрана для GIE агента. """
        frame = self.capture_screen()
        if frame is None:
            return {"status": "error", "message": "Не удалось захватить экран"}
        
        detections = await self.detect_objects(frame)
        
        return {
            "status": "online",
            "det_count": len(detections),
            "summary": f"Обнаружено {len(detections)} UI-элементов."
        }
