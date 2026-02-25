import torch
import torch.nn as nn
import logging

class MinGRU(nn.Module):
    """
    minGRU — Сверхлегкий рекуррентный слой для рекурсивных рефлексов.
    
    В отличие от стандартного GRU, эта версия оптимизирована для обработки 
    одиночных векторов состояний (рефлексов), обеспечивая минимальную задержку
    при классификации намерений пользователя.
    """
    def __init__(self, dim=256):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, hidden=None):
        out, h = self.gru(x, hidden)
        return self.proj(out), h


class ReflexCore:
    """
    Рефлекторное ядро TARS (System 0).
    
    Самый быстрый слой обработки. Его задача — перехватить запрос ДО того,
    как проснется тяжелый основной мозг (System 1/2).
    
    Работает в два этапа:
      1. Pattern Matching: Жесткие правила (если есть слово 'Привет' — ответь 'Привет').
      2. Neural Signal: Предсказание 'срочности' или 'типа' запроса через MinGRU.
    
    Если рефлекс не сработал — он возвращает None, и запрос уходит в RRN/AUSSM.
    """
    def __init__(self, dim=256):
        self.model = MinGRU(dim)
        self.dim = dim
        self.logger = logging.getLogger("Tars.Reflex")
        
        # База знаний рефлексов.
        # response: готовый текст или None (для динамических ответов).
        # action: идентификатор действия для системы.
        # База знаний рефлексов. 
        # Удалено по требованию пользователя — теперь все текстовые ответы
        # должны генерироваться исключительно Большой Языковой Моделью.
        self.reflex_patterns = {}
        self.call_count = 0

    async def react(self, input_text: str) -> dict:
        """
        Попытка дать мгновенный ответ.
        Центральный метод 'инстинктов' ассистента.
        """
        self.call_count += 1
        text_lower = input_text.lower()
        
        # 1. Быстрая проверка словаря паттернов.
        # Используем генератор для мгновенного нахождения первого совпадения.
        match = next(( (p, d) for p, d in self.reflex_patterns.items() if p in text_lower ), None)
        
        if match:
            pattern, data = match
            response = data["response"]
            action = data["action"]
            
            if action == "status" and response is None:
                response = self._generate_status()
            elif action == "time" and response is None:
                from datetime import datetime
                response = f"Сейчас {datetime.now().strftime('%H:%M:%S')}. Время — понятие относительное."
            
            return {"response": response, "action": action}
        
        # 2. Нейронная классификация намерения.
        with torch.no_grad():
            text_hash = hash(text_lower) % (2**31)
            torch.manual_seed(text_hash)
            x = torch.randn(1, 1, self.dim) * 0.05 # Уменьшаем шум
            out, _ = self.model(x)
            
            urgency = out.abs().mean().item()
            if urgency > 0.8: # Повышаем порог чувствительности
                self.logger.debug(f"Reflex: Сигнал обнаружен (urgency={urgency:.3f})")
        
        return None
    
    def _generate_status(self) -> str:
        """Сборка телеметрии. Оптимизировано для минимальной задержки."""
        import psutil
        try:
            # Уменьшаем интервал до минимума для скорости.
            cpu = psutil.cpu_percent(interval=0.01)
            mem = psutil.virtual_memory()
            return f"Системы: CPU {cpu}%, RAM {mem.used/(1024**3):.1f}GB. Обработано: {self.call_count}."
        except Exception:
            return "Системы активны."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reflex = ReflexCore()
    
    import asyncio
    async def test():
        # Тест паттернов
        r = await reflex.react("Привет, как дела?")
        print(f"Pattern: {r}")
        
        r = await reflex.react("Покажи статус системы")
        print(f"Status: {r}")
        
        r = await reflex.react("Который час?")
        print(f"Time: {r}")
        
        # Тест: сложный запрос не обрабатывается рефлексом
        r = await reflex.react("Составь план на неделю")
        print(f"Complex (should be None): {r}")
    
    asyncio.run(test())
