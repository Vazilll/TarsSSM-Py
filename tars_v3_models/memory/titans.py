"""
titans.py — Titans Surprise-Based Learning Memory.

Нейронная долговременная память (LTM). Знания 'запекаются' в веса нейронов.
Если модель встречает что-то новое (surprise > threshold) → быстрая консолидация (3 шага градиентного обучения).

Размерность: 384d (выровнена с LEANN/MiniLM эмбеддингами).
"""
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
    
    Размерность 384d — совпадает с LEANN эмбеддингами (all-MiniLM-L6-v2).
    """
    def __init__(self, dim=384):
        super().__init__()
        # Двухслойный скрытый базис для хранения ассоциаций
        self.layer1 = nn.Linear(dim, dim * 2)
        self.silu = nn.SiLU()
        self.layer2 = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.silu(self.layer1(x))
        return self.norm(self.layer2(h))


class TitansMemory:
    """
    Реализация памяти Titans (Surprise-Based Learning).
    
    Ключевая особенность:
      - Пока входящие данные предсказуемы — ничего не учим
      - Если surprise > threshold → 3 шага градиентного обучения (Fast Weight Update)
    
    Связь с пайплайном:
      - Получает 384d эмбеддинги от LEANN/MiniLM
      - Если вход из Mamba-2 (768d+) → проецируется через brain_proj
      - get_recall() возвращает 384d вектор для инъекции в TarsBlock
    """
    def __init__(self, dim=384, brain_dim=768, learning_rate=2e-4):
        self.dim = dim
        self.ltm = LongTermMemory(dim)
        self.optimizer = optim.Adam(self.ltm.parameters(), lr=learning_rate)
        self.criteria = nn.HuberLoss()  # Устойчива к выбросам
        self.logger = logging.getLogger("Tars.Titans")
        
        # Проекция из пространства мозга (768d) → пространство памяти (384d)
        self.brain_proj = nn.Linear(brain_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.brain_proj.weight)
        
        # Порог удивления
        self.surprise_threshold = 0.45
        
        # Статистика
        self.total_updates = 0
        self.total_surprises = 0

    def project_from_brain(self, brain_vec: torch.Tensor) -> torch.Tensor:
        """Проецирует вектор из Mamba-2 (768d) → память (384d)."""
        with torch.no_grad():
            return self.brain_proj(brain_vec)

    async def get_recall(self, context_embedding: torch.Tensor) -> torch.Tensor:
        """
        Извлечение (вспоминание) знаний на основе текущей ситуации.
        
        Args:
            context_embedding: [B, 384] — вектор контекста (от LEANN или brain_proj)
        Returns:
            [B, 384] — вспомненные ассоциации
        """
        with torch.no_grad():
            recall = self.ltm(context_embedding)
        return recall

    async def update(self, entry_embedding: torch.Tensor) -> dict:
        """
        Динамическое обновление весов памяти.
        
        Args:
            entry_embedding: [B, 384] — новый вектор для запоминания
        Returns:
            dict: surprised (bool), loss (float), total_surprises (int)
        """
        self.total_updates += 1
        
        # 1. Проверяем, насколько предсказуем новый вход
        prediction = self.ltm(entry_embedding)
        
        with torch.no_grad():
            loss_val = torch.norm(entry_embedding - prediction).item()
        
        result = {"surprised": False, "loss": loss_val, "total_surprises": self.total_surprises}
        
        if loss_val > self.surprise_threshold:
            self.total_surprises += 1
            self.logger.info(
                f"Titans: Сюрприз #{self.total_surprises}! "
                f"(loss={loss_val:.3f} > {self.surprise_threshold}). Консолидация..."
            )
            
            # 2. Fast Weight Update (3 итерации)
            for _ in range(3):
                self.optimizer.zero_grad()
                pred = self.ltm(entry_embedding)
                loss = self.criteria(pred, entry_embedding)
                loss.backward()
                self.optimizer.step()
            
            result["surprised"] = True
            result["total_surprises"] = self.total_surprises
        
        return result

    def get_stats(self) -> dict:
        """Статистика памяти."""
        return {
            "total_updates": self.total_updates,
            "total_surprises": self.total_surprises,
            "surprise_rate": self.total_surprises / max(1, self.total_updates),
            "ltm_params": sum(p.numel() for p in self.ltm.parameters()),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    titans = TitansMemory(dim=384, brain_dim=768)
    data = torch.randn(1, 384)
    brain_data = torch.randn(1, 768)
    
    # Тест проекции из мозга
    projected = titans.project_from_brain(brain_data)
    print(f"Brain 768d → Memory 384d: {projected.shape}")
    
    import asyncio
    async def test():
        result = await titans.update(data)
        print(f"Update: {result}")
        recall = await titans.get_recall(data)
        print(f"Recall shape: {recall.shape}")
        print(f"Stats: {titans.get_stats()}")
    asyncio.run(test())
