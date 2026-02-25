import pyautogui
import pygetwindow as gw
import logging
import time
import pyperclip
import os
import io
import contextlib

class ActionEngine:
    """
    Advanced ActionEngine - Оркестратор действий в Windows.
    Реализует сложную автоматизацию: окна, ввод, скрипты.
    """
    def __init__(self):
        self.logger = logging.getLogger("Tars.Actions")
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.3 # Оптимальная задержка для реалистичности

    async def execute(self, action_name: str, params: dict):
        """ Выполнение сложной команды. """
        self.logger.info(f"Actions: Цель '{action_name}'...")
        
        try:
            if action_name == "click":
                return self._click(params)
            elif action_name == "type":
                return self._type(params)
            elif action_name == "focus_window":
                return self._focus(params)
            elif action_name == "open_url":
                url = params.get('url')
                if url:
                    os.system(f"start {url}")
                    return f"Browser opened: {url}"
                return "Error: No URL provided."
            elif action_name == "execute_script":
                # Запуск Python кода, сгенерированного мозгом, с перехватом stdout/stderr
                code = params.get("code", "")
                output_capture = io.StringIO()
                try:
                    with contextlib.redirect_stdout(output_capture), contextlib.redirect_stderr(output_capture):
                        exec(code, globals())
                except Exception as e:
                    import traceback
                    output_capture.write(f"\nExecution error:\n{traceback.format_exc()}")
                
                result_output = output_capture.getvalue()
                if not result_output.strip():
                    return "Script executed successfully (no output)."
                return f"Script output:\n{result_output}"
            elif action_name == "analyze_workspace":
                from sensory.vision import TarsVision
                vision = TarsVision()
                result = await vision.analyze_workspace()
                return str(result)
            elif action_name == "run_command":
                cmd = params.get("command", "")
                import subprocess
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                return f"Command executed. Output: {result.stdout}\nErrors: {result.stderr}"
            elif action_name == "keyboard_combo":
                pyautogui.hotkey(*params.get("keys", []))
                return "Hotkey combo sent."
            elif action_name == "vision_click":
                return await self._vision_click(params)
            else:
                return f"Error: Command {action_name} not found."
        except Exception as e:
            self.logger.error(f"Actions failure: {e}")
            return f"Action failed: {e}"

    def _click(self, params):
        x, y = params.get("x"), params.get("y")
        if x is not None and y is not None:
            pyautogui.click(x, y)
            return f"Clicked at {x}, {y}"
        return "Error: Missing coordinates."

    def _type(self, params):
        text = params.get("text", "")
        if params.get("use_clipboard", False):
            pyperclip.copy(text)
            pyautogui.hotkey('ctrl', 'v')
            return "Text pasted from clipboard."
        pyautogui.write(text, interval=0.05)
        return "Text typed."

    def _focus(self, params):
        title = params.get("title")
        windows = gw.getWindowsWithTitle(title)
        if windows:
            windows[0].activate()
            if params.get("maximize", False):
                windows[0].maximize()
            return f"Window '{title}' focused."
        return f"Window '{title}' not found."

    async def _vision_click(self, params):
        """ Продвинутый OS-уровень: Клик на основе компьютерного зрения """
        target_class = params.get("target_class")
        
        # Ленивый импорт во избежание циклических зависимостей
        from sensory.vision import TarsVision
        vision = TarsVision()
        frame = vision.capture_screen()
        
        if frame is None:
            return "Error: Не удалось получить доступ к экрану для Vision Click (BetterCam error)."
            
        detections = await vision.detect_objects(frame)
        for det in detections:
            # YOLO results [x1, y1, x2, y2, точность, класс]
            x1, y1, x2, y2, conf, cls = det
            # Сравниваем классы с учетом того, что cls может быть float
            if str(cls) == str(target_class) or str(int(cls)) == str(target_class):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                pyautogui.click(cx, cy)
                return f"Успешный клик по классу {target_class} (Координаты: {cx}, {cy}, YOLO Уверенность: {conf:.2f})"
        
        return f"Warning: Элемент '{target_class}' не найден на экране для клика."

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ActionEngine()
    # Пример: engine.execute("open_url", {"url": "https://google.com"})
