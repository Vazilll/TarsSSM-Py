"""
vision.py — TARS Vision Layer (кроссплатформенный)

Conditional imports: BetterCam на Windows, PIL.ImageGrab как fallback.
YOLO загружается лениво (только если модель существует).
"""
import logging
import numpy as np

logger = logging.getLogger("Tars.Vision")

# ═══ Conditional imports для кроссплатформенности ═══
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
    logger.info("Vision: cv2 not available — detection disabled")

try:
    import bettercam
    _HAS_BETTERCAM = True
except ImportError:
    _HAS_BETTERCAM = False
    logger.debug("Vision: bettercam not available (Windows-only)")

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False
    logger.debug("Vision: ultralytics not available — YOLO disabled")


class TarsVision:
    """
    Система зрения (Vision Layer).
    Использует YOLOv11 для распознавания интерфейсов.
    
    Screen capture:
      - BetterCam (Windows, высокая FPS)
      - PIL.ImageGrab (fallback, кроссплатформенный)
    """
    def __init__(self, model_path="models/vision/yolo26n.pt"):
        self.logger = logging.getLogger("Tars.Vision")
        self.model = None
        self.camera = None
        
        # ═══ YOLO: загрузка только если файл существует ═══
        if _HAS_YOLO:
            import os
            if os.path.exists(model_path):
                try:
                    self.model = YOLO(model_path)
                    self.logger.info(f"Vision: YOLO loaded from {model_path}")
                except Exception as e:
                    self.logger.warning(f"Vision: YOLO load failed: {e}")
            else:
                self.logger.info(f"Vision: YOLO model not found at {model_path}")
        
        # ═══ Screen capture: BetterCam (Windows) → PIL fallback ═══
        if _HAS_BETTERCAM:
            try:
                self.camera = bettercam.create()
                self.logger.info("Vision: BetterCam initialized (high FPS)")
            except Exception as e:
                self.logger.warning(f"Vision: BetterCam failed: {e}")
        
        if self.camera is None:
            self.logger.info("Vision: Using PIL.ImageGrab fallback for screen capture")

    def capture_screen(self):
        """Скриншот рабочего стола. Returns BGR numpy array or None."""
        # BetterCam path
        if self.camera is not None:
            frame = self.camera.grab()
            if frame is not None and _HAS_CV2:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        
        # PIL fallback
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            frame = np.array(img)
            if _HAS_CV2:
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            self.logger.error(f"Vision: Screen capture failed: {e}")
            return None

    async def detect_objects(self, frame):
        """Поиск UI-элементов на кадре через YOLO."""
        if self.model and frame is not None:
            results = self.model(frame, verbose=False)
            return results[0].boxes.data.tolist()
        return []

    async def extract_text(self, frame):
        """Чтение текста с экрана (OCR)."""
        try:
            import pytesseract
            text = pytesseract.image_to_string(frame, lang='rus+eng')
            return text.strip()
        except ImportError:
            self.logger.warning("Vision: pytesseract not installed — OCR disabled")
            return ""
        except Exception as e:
            self.logger.error(f"Vision OCR Error: {e}")
            return ""

    async def analyze_workspace(self):
        """Комплексный анализ экрана для GIE агента."""
        frame = self.capture_screen()
        if frame is None:
            return {"status": "error", "message": "Не удалось захватить экран"}
        
        detections = await self.detect_objects(frame)
        text_content = await self.extract_text(frame)
        
        return {
            "status": "online",
            "det_count": len(detections),
            "text": text_content[:500] + "..." if len(text_content) > 500 else text_content,
            "summary": f"Обнаружено {len(detections)} UI-элементов. Текст с экрана: {text_content[:100]}..."
        }
