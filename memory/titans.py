import torch
import torch.nn as nn
import torch.optim as optim
import logging

class LongTermMemory(nn.Module):
    """
    Нейронная долговременная память (LTM Layer).
    Знания 'запекаются' прямо в веса.
    """
    def __init__(self, dim=1024):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim * 2)
        self.silu = nn.SiLU()
        self.layer2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.silu(self.layer1(x))
        return self.norm(self.layer2(h))

class TitansMemory:
    """
    Память Titans (Deep implementation).
    Использует Surprise-based learning для вечной адаптации.
    """
    def __init__(self, dim=1024, learning_rate=2e-4):
        self.ltm = LongTermMemory(dim)
        self.optimizer = optim.Adam(self.ltm.parameters(), lr=learning_rate)
        self.criteria = nn.HuberLoss() # Более стабильна чем MSE
        self.logger = logging.getLogger("Tars.Titans")
        self.surprise_threshold = 0.45

    async def get_recall(self, context_embedding: torch.Tensor):
        """ Извлечение знаний на основе текущего контекста. """
        with torch.no_grad():
            recall = self.ltm(context_embedding)
        return recall

    async def update(self, entry_embedding: torch.Tensor):
        """
        Обновление весов памяти при высоком уровне 'сюрприза'.
        """
        # 1. Предсказание на основе текущих весов
        prediction = self.ltm(entry_embedding)
        
        # 2. Вычисление ошибки (сюрприза)
        with torch.no_grad():
            loss_val = torch.norm(entry_embedding - prediction).item()
        
        if loss_val > self.surprise_threshold:
            self.logger.info(f"Titans: Сюрприз ({loss_val:.3f}) > порога. Изучаю новые данные...")
            
            # 3. Цикл быстрой консолидации (3 шага градиента)
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
