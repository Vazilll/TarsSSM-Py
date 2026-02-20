import pyautogui
import pygetwindow as gw
import logging
import time
import pyperclip
import os

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
                # Запуск Python кода, сгенерированного мозгом
                code = params.get("code", "")
                exec(code)
                return "Script executed successfully."
            elif action_name == "keyboard_combo":
                pyautogui.hotkey(*params.get("keys", []))
                return "Hotkey combo sent."
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ActionEngine()
    # Пример: engine.execute("open_url", {"url": "https://google.com"})
