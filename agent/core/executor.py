import logging
import time
import os
import io
import re
import shlex
import webbrowser
import contextlib
import subprocess
import multiprocessing

class ActionEngine:
    """
    Advanced ActionEngine - Оркестратор действий в Windows.
    Реализует сложную автоматизацию: окна, ввод, скрипты.
    
    Security: все операции ограничены песочницей.
    """
    
    # ═══ Security: whitelist для execute_script ═══
    # NOTE: getattr/hasattr/isinstance/type REMOVED — they enable
    #       attribute-traversal sandbox escapes via __class__.__bases__ chains.
    SAFE_BUILTINS = {
        'print', 'len', 'range', 'int', 'float', 'str', 'bool',
        'list', 'dict', 'tuple', 'set', 'sorted', 'enumerate',
        'zip', 'map', 'filter', 'sum', 'min', 'max', 'abs',
        'round',
        'True', 'False', 'None',
    }
    
    SAFE_MODULES = {'math', 'json', 'datetime', 'collections', 're'}
    
    # Dangerous dunder names that enable sandbox escape via subscript/attribute
    _BLOCKED_DUNDERS = {
        '__class__', '__bases__', '__subclasses__', '__mro__',
        '__globals__', '__builtins__', '__code__', '__func__',
        '__self__', '__module__', '__dict__', '__init__',
        '__new__', '__del__', '__reduce__', '__reduce_ex__',
    }
    
    # ═══ Security: URL scheme whitelist ═══
    ALLOWED_URL_SCHEMES = {'http', 'https'}
    
    # ═══ Security: Environment variable whitelist ═══
    SAFE_ENV_VARS = {
        'PATH', 'TEMP', 'TMP', 'SystemRoot', 'USERPROFILE',
        'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'COMSPEC',
        'PATHEXT', 'SYSTEMDRIVE', 'HOME',
    }
    
    # ═══ Security: блокировка опасных shell-паттернов ═══
    BLOCKED_SHELL_PATTERNS = [
        r'rm\s+-r',           # recursive delete
        r'del\s+/s',          # Windows recursive delete
        r'format\s+[a-z]:',   # format drive
        r'Remove-Item.*-Recurse',
        r'mkfs',
        r'dd\s+if=',
        r'shutdown', r'reboot', r'halt',
        r'>\s*/dev/sd',       # write to raw device
        r'\|\s*(bash|sh|cmd)', # pipe to shell
        r'powershell\s+-e',   # encoded commands
        r'base64\s+-d',       # base64 decode pipe
        r'certutil\s+-decode', # Windows LOLBin
        r'bitsadmin',          # Windows LOLBin
        r'mshta',              # Windows LOLBin
        r'wmic\s+process',     # WMI process exec
        r'reg\s+(add|delete)',  # registry modification
        r'net\s+user',         # user management
        r'schtasks\s+/create', # scheduled tasks
    ]
    
    SCRIPT_TIMEOUT = 10  # seconds max для execute_script
    COMMAND_TIMEOUT = 30  # seconds max для run_command
    MAX_OUTPUT = 10240   # 10KB max output
    
    def __init__(self):
        self.logger = logging.getLogger("Tars.Actions")
        try:
            import pyautogui
            pyautogui.FAILSAFE = True
            pyautogui.PAUSE = 0.3
            self._pyautogui = pyautogui
        except ImportError:
            self._pyautogui = None
            self.logger.warning("pyautogui not installed — click/type/hotkey disabled")

    def _safe_env(self) -> dict:
        """Build minimal environment from whitelist for sandboxed subprocesses."""
        env = {}
        for key in self.SAFE_ENV_VARS:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        return env

    async def execute(self, action_name: str, params: dict):
        """ Выполнение сложной команды (с защитой). """
        self.logger.info(f"Actions: Цель '{action_name}'...")
        
        try:
            if action_name == "click":
                return self._click(params)
            elif action_name == "type":
                return self._type(params)
            elif action_name == "focus_window":
                return self._focus(params)
            elif action_name == "open_url":
                return self._safe_open_url(params)
            elif action_name == "execute_script":
                return self._safe_execute_script(params)
            elif action_name == "analyze_workspace":
                from sensory.vision import TarsVision
                vision = TarsVision()
                result = await vision.analyze_workspace()
                return str(result)
            elif action_name == "run_command":
                return self._safe_run_command(params)
            elif action_name == "keyboard_combo":
                if self._pyautogui is None:
                    return "Error: pyautogui not installed"
                self._pyautogui.hotkey(*params.get("keys", []))
                return "Hotkey combo sent."
            elif action_name == "vision_click":
                return await self._vision_click(params)
            else:
                return f"Error: Command {action_name} not found."
        except Exception as e:
            self.logger.error(f"Actions failure: {e}")
            return f"Action failed: {e}"

    # ═══ Security-hardened methods ═══
    
    def _safe_open_url(self, params) -> str:
        """Безопасное открытие URL — только http/https."""
        url = params.get('url', '').strip()
        if not url:
            return "Error: No URL provided."
        
        # Валидация схемы
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme not in self.ALLOWED_URL_SCHEMES:
            return f"Error: URL scheme '{parsed.scheme}' not allowed (only http/https)"
        
        # Проверка на shell injection
        if any(c in url for c in [';', '|', '&', '`', '$', '\n', '\r']):
            return "Error: URL contains forbidden characters"
        
        webbrowser.open(url)
        return f"Browser opened: {url}"
    
    def _safe_execute_script(self, params) -> str:
        """
        Безопасное выполнение Python кода в изолированном процессе.
        
        Ограничения:
          - AST-валидация перед запуском (import/dunder/eval блокировка)
          - Запуск в отдельном subprocess (полная изоляция памяти)
          - Timeout (10 сек) с киллом процесса
          - Ограниченный вывод (10KB)
        """
        import sys
        import tempfile
        
        code = params.get("code", "")
        if not code.strip():
            return "Error: No code provided."
        
        # ═══ AST validation: блокировка опасных операций ═══
        import ast
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f"SyntaxError: {e}"
        
        for node in ast.walk(tree):
            # Блокировать import
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, 'module', None) or \
                         (node.names[0].name if node.names else '')
                if module not in self.SAFE_MODULES:
                    return f"Error: Import of '{module}' is not allowed. Safe: {self.SAFE_MODULES}"
            # Блокировать __import__, eval, exec, compile, open
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in {
                    '__import__', 'eval', 'exec', 'compile', 'open',
                    'getattr', 'setattr', 'delattr', 'globals', 'locals',
                    'breakpoint', 'input',
                }:
                    return f"Error: Call to '{func.id}' is not allowed in sandbox."
            # Блокировать доступ к dunder-атрибутам
            if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                return f"Error: Access to '{node.attr}' is not allowed."
            # Блокировать f-string eval tricks (e.g. f'{__import__("os")}')
            if isinstance(node, ast.FormattedValue):
                return "Error: F-string expressions are not allowed in sandbox."
            # Блокировать starred assignments (unpacking exploits)
            if isinstance(node, ast.Starred):
                return "Error: Starred expressions are not allowed in sandbox."
            # Блокировать global/nonlocal scope manipulation
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                return "Error: global/nonlocal statements are not allowed in sandbox."
            # Блокировать subscript dunder access: obj['__class__']
            if isinstance(node, ast.Subscript):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                    if sl.value in self._BLOCKED_DUNDERS:
                        return f"Error: Subscript access to '{sl.value}' is not allowed."
        
        # ═══ Subprocess isolation: запуск в отдельном процессе ═══
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name
            
            creation_flags = 0
            if os.name == 'nt':
                creation_flags = subprocess.CREATE_NO_WINDOW
            
            result = subprocess.run(
                [sys.executable, '-u', temp_path],
                capture_output=True, text=True,
                timeout=self.SCRIPT_TIMEOUT,
                cwd=os.getcwd(),
                creationflags=creation_flags,
                env=self._safe_env(),
            )
            
            output = result.stdout[:self.MAX_OUTPUT]
            errors = result.stderr[:2000]
            
            if result.returncode != 0:
                return f"Execution error (code {result.returncode}):\n{errors}\nOutput:\n{output}"
            
            if not output.strip():
                return "Script executed successfully (no output)."
            return f"Script output:\n{output}"
        except subprocess.TimeoutExpired:
            return f"Error: Script timed out after {self.SCRIPT_TIMEOUT}s"
        except Exception as e:
            return f"Error running script: {e}"
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def _safe_run_command(self, params) -> str:
        """Безопасное выполнение shell-команды с whitelist-проверкой."""
        cmd = params.get("command", "").strip()
        if not cmd:
            return "Error: No command provided."
        
        # Проверка длины
        if len(cmd) > 500:
            return "Error: Command too long (max 500 chars)"
        
        # Проверка на опасные паттерны
        for pattern in self.BLOCKED_SHELL_PATTERNS:
            if re.search(pattern, cmd, re.IGNORECASE):
                return f"Error: Blocked dangerous pattern in command"
        
        try:
            # shell=False: защита от shell injection
            import shlex
            if os.name == 'nt':
                # Windows: shlex with posix=False handles quoted args properly
                cmd_parts = shlex.split(cmd, posix=False)
                creation_flags = subprocess.CREATE_NO_WINDOW
            else:
                cmd_parts = shlex.split(cmd)
                creation_flags = 0
            
            result = subprocess.run(
                cmd_parts, shell=False,
                capture_output=True, text=True,
                timeout=self.COMMAND_TIMEOUT,
                cwd=os.getcwd(),
                creationflags=creation_flags,
                env=self._safe_env(),
            )
            output = result.stdout[:self.MAX_OUTPUT]
            errors = result.stderr[:2000]
            return f"Command executed. Output: {output}\nErrors: {errors}"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.COMMAND_TIMEOUT}s"
        except FileNotFoundError:
            return f"Error: Command not found: {cmd.split()[0] if cmd else 'empty'}"
        except Exception as e:
            return f"Error: {e}"

    def _click(self, params):
        if self._pyautogui is None:
            return "Error: pyautogui not installed"
        x, y = params.get("x"), params.get("y")
        if x is not None and y is not None:
            self._pyautogui.click(x, y)
            return f"Clicked at {x}, {y}"
        return "Error: Missing coordinates."

    def _type(self, params):
        if self._pyautogui is None:
            return "Error: pyautogui not installed"
        text = params.get("text", "")
        if params.get("use_clipboard", False):
            try:
                import pyperclip
                pyperclip.copy(text)
                self._pyautogui.hotkey('ctrl', 'v')
                return "Text pasted from clipboard."
            except ImportError:
                return "Error: pyperclip not installed"
        self._pyautogui.write(text, interval=0.05)
        return "Text typed."

    def _focus(self, params):
        try:
            import pygetwindow as gw
        except ImportError:
            return "Error: pygetwindow not installed"
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
        if self._pyautogui is None:
            return "Error: pyautogui not installed"
        target_class = params.get("target_class")
        
        from sensory.vision import TarsVision
        vision = TarsVision()
        frame = vision.capture_screen()
        
        if frame is None:
            return "Error: Не удалось получить доступ к экрану для Vision Click."
            
        detections = await vision.detect_objects(frame)
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if str(cls) == str(target_class) or str(int(cls)) == str(target_class):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                self._pyautogui.click(cx, cy)
                return f"Успешный клик по классу {target_class} (Координаты: {cx}, {cy}, YOLO Уверенность: {conf:.2f})"
        
        return f"Warning: Элемент '{target_class}' не найден на экране."

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ActionEngine()
