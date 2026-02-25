"""
mole.py — MoLE (Mixture of LoRA Experts) для TARS v3.

8 экспертов-адаптеров (LoRA rank=8), каждый ~16K параметров.
Sparse Top-2 роутер активирует ровно двух экспертов на каждом раунде.

Эксперты:
  0: general   — Общее восприятие запроса
  1: analyzer  — Детальный разбор контекста
  2: critic    — Поиск логических ошибок
  3: creative  — Креативные идеи, метафоры
  4: math      — Математические операции
  5: code      — Программирование
  6: memory    — Работа с памятью и фактами
  7: action    — Выполнение действий в ОС
"""
import torch
import torch.nn as nn
import logging

class LoRAAdapter(nn.Module):
    """
    Адаптер LoRA (Low-Rank Adaptation).
    delta_W = B @ A * scale.  ~16K параметров при dim=1024, rank=8.
    """
    def __init__(self, dim=1024, rank=8):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, dim) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(dim, rank))
        self.scale = 1.0 / rank

    def get_delta(self):
        """Возвращает полную матрицу поправки [dim x dim]."""
        return (self.lora_B @ self.lora_A) * self.scale


EXPERT_NAMES = [
    "general", "analyzer", "critic", "creative",
    "math", "code", "memory", "action"
]


class MoleManager:
    """
    Менеджер MoLE (Mixture of LoRA Experts).

    Sparse Top-2: на каждом раунде активируются ровно 2 эксперта.
    Их delta-веса суммируются и накладываются на базовую Mamba-2.
    """
    def __init__(self, brain_model=None, dim=1024, rank=8, n_experts=8):
        self.brain = brain_model
        self.dim = dim
        self.logger = logging.getLogger("Tars.MoLE")

        self.experts = nn.ModuleDict({
            name: LoRAAdapter(dim, rank) for name in EXPERT_NAMES[:n_experts]
        })

        # Простой роутер: Linear → Top-2
        self.router = nn.Linear(dim, n_experts)
        self.active_experts = ["general", "analyzer"]
        self.logger.info(f"MoLE: {len(self.experts)} экспертов (LoRA rank={rank})")

    def route(self, hidden_vec: torch.Tensor) -> list:
        """
        Sparse Top-2 routing.
        hidden_vec: [B, dim] — текущий вектор мысли.
        Returns: list of 2 expert names.
        """
        with torch.no_grad():
            scores = self.router(hidden_vec.mean(dim=0, keepdim=True))  # [1, n_experts]
            top2 = scores.topk(2, dim=-1).indices[0].tolist()

        names = list(self.experts.keys())
        selected = [names[i] for i in top2 if i < len(names)]
        if len(selected) < 2:
            selected = ["general", "analyzer"]

        self.active_experts = selected
        return selected

    def get_combined_delta(self, hidden_vec: torch.Tensor = None) -> torch.Tensor:
        """
        Возвращает суммарную delta = sum(expert_i.get_delta()) для активных экспертов.
        Если hidden_vec передан — сначала роутит.
        """
        if hidden_vec is not None:
            self.route(hidden_vec)

        delta = torch.zeros(self.dim, self.dim)
        for name in self.active_experts:
            if name in self.experts:
                delta = delta + self.experts[name].get_delta()

        return delta

    def switch_expert(self, loop_index: int):
        """Legacy: переключение по индексу цикла (для совместимости)."""
        order = list(self.experts.keys())
        idx = min(loop_index, len(order) - 1)
        name = order[idx]
        self.active_experts = [name, order[min(idx + 1, len(order) - 1)]]

        if self.brain and hasattr(self.brain, 'layers'):
            delta = self.get_combined_delta()
            for layer in self.brain.layers:
                layer.active_expert_delta = delta

        self.logger.debug(f"MoLE: Эксперты {self.active_experts} (цикл {loop_index})")
        return self.active_experts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mole = MoleManager(dim=1024, rank=8, n_experts=8)

    # Тест роутинга
    fake_hidden = torch.randn(1, 1024)
    selected = mole.route(fake_hidden)
    print(f"Selected experts: {selected}")

    delta = mole.get_combined_delta()
    print(f"Combined delta norm: {delta.norm().item():.6f}")

    # Тест legacy
    for i in range(5):
        active = mole.switch_expert(i)
        print(f"Loop {i}: experts={active}")
