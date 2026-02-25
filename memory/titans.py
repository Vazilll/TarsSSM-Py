import torch
import torch.nn as nn
import torch.optim as optim
import logging

class LongTermMemory(nn.Module):
    """
    Нейронная Долговременная Память (Neural LTM Layer).
    
    В отличие от LEANN (где знания лежат в базе), здесь знания 'запекаются'
    в сами веса нейронов. Это аналог человеческой памяти, где информация
    меняет структуру синапсов.
    """
    def __init__(self, dim=1024):
        super().__init__()
        # Двухслойный скрытый базис для хранения ассоциаций.
        self.layer1 = nn.Linear(dim, dim * 2)
        self.silu = nn.SiLU()
        self.layer2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.silu(self.layer1(x))
        return self.norm(self.layer2(h))

class TitansMemory:
    """
    Реализация памяти Titans.
    
    Ключевая особенность: Surprise-Based Learning (Обучение на основе удивления).
    Пока входящие данные предсказуемы (ошибка мала), модель ничего не учит.
    Но если система встречает нечто новое и неожиданное (сюрприз выше порога),
    она моментально запускает цикл градиентного обучения, чтобы 'запомнить' этот факт.
    """
    def __init__(self, dim=1024, learning_rate=2e-4):
        self.ltm = LongTermMemory(dim)
        # Оптимизатор Adam позволяет быстро корректировать веса 'на лету'.
        self.optimizer = optim.Adam(self.ltm.parameters(), lr=learning_rate)
        self.criteria = nn.HuberLoss() # Устойчива к выбросам (аномалиям).
        self.logger = logging.getLogger("Tars.Titans")
        
        # Порог удивления. Если ошибка предсказания выше 0.45 — факт считается важным.
        self.surprise_threshold = 0.45

    async def get_recall(self, context_embedding: torch.Tensor):
        """ Извлечение (вспоминание) знаний на основе текущей ситуации. """
        with torch.no_grad():
            recall = self.ltm(context_embedding)
        return recall

    async def update(self, entry_embedding: torch.Tensor):
        """
        Динамическое обновление весов памяти. 
        TARS обучается каждую секунду, если видит что-то новое.
        """
        # 1. Проверяем, насколько текущая память справляется с новыми данными.
        prediction = self.ltm(entry_embedding)
        
        # 2. Оцениваем 'уровень удивления' (метрика L2).
        with torch.no_grad():
            loss_val = torch.norm(entry_embedding - prediction).item()
        
        if loss_val > self.surprise_threshold:
            self.logger.info(f"Titans: Сюрприз! ({loss_val:.3f}). Консолидация новых знаний...")
            
            # 3. Быстрая консолидация (Fast Weight Update).
            # Мы делаем 3 итерации обучения прямо в рантайме.
            for _ in range(3):
                self.optimizer.zero_grad()
                pred = self.ltm(entry_embedding)
                loss = self.criteria(pred, entry_embedding)
                loss.backward()
                self.optimizer.step()
            
            return True
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    titans = TitansMemory()
    data = torch.randn(1, 1024)
    import asyncio
    async def test():
        await titans.update(data)
        await titans.get_recall(data)
    asyncio.run(test())
