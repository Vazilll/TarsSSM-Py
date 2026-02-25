import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict, Any, List
import asyncio

class MoIRA(nn.Module):
    """
    MoIRA — Нейронный роутер инструментов.
    
    Маршрутизирует «мысли» мозга к конкретным действиям.
    Использует обучаемые эмбеддинги инструментов (не рандом),
    и извлекает параметры действия из текста мысли.
    """
    def __init__(self, dim=1024):
        super().__init__()
        self.logger = logging.getLogger("Tars.MoIRA")
        
        # Проекция мысли в пространство инструментов
        self.router_head = nn.Sequential(
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )
        
        # Обучаемые эмбеддинги инструментов (инициализация ортогональная)
        self.tool_names = ["Python", "Terminal", "Browser", "Vision", "FinalAnswer"]
        tool_embs = torch.empty(len(self.tool_names), 256)
        nn.init.orthogonal_(tool_embs)
        self.tool_embeddings = nn.Parameter(tool_embs)
        
        # Голова уверенности (отдельная от cosine similarity) 
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Ключевые слова для извлечения параметров из текста мысли
        self.keyword_map = {
            "Python": ["код", "скрипт", "python", "вычисл", "программ", "функци"],
            "Terminal": ["команд", "терминал", "запуст", "файл", "директор", "путь"],
            "Browser": ["поиск", "сайт", "интернет", "url", "http", "браузер", "google"],
            "Vision": ["экран", "окно", "увид", "смотр", "визуал", "скриншот"],
            "FinalAnswer": ["ответ", "готов", "результат", "итог", "завершен", "integral core", "анализ", "сходимость", "интеграл", "объясни", "почему", "как", "кто", "что"]
        }

    async def route(self, thought_vector: torch.Tensor, thought_text: str = "") -> Tuple[str, Dict[str, Any], float]:
        """
        Выбор инструмента на основе вектора мысли + текста.
        
        1. Нейронный роутинг (cosine similarity с обучаемыми эмбеддингами)
        2. Текстовое уточнение (если есть текст мысли)
        3. Извлечение параметров
        """
        with torch.no_grad():
            query_vec = self.router_head(thought_vector.mean(dim=1))  # [B, 256]
            
            # Cosine similarity с каждым инструментом
            similarities = torch.cosine_similarity(
                query_vec.unsqueeze(1),  # [B, 1, 256]
                self.tool_embeddings.unsqueeze(0),  # [1, N, 256]
                dim=-1
            )  # [B, N]
            
            # Softmax для нормализации
            probs = torch.softmax(similarities * 5.0, dim=-1)  # temperature=0.2
            best_idx = probs.argmax(dim=-1).item()
            best_tool = self.tool_names[best_idx]
            neural_confidence = probs[0, best_idx].item()
        
        # Текстовое уточнение (если мозг вернул текст)
        if thought_text:
            text_lower = thought_text.lower()
            text_scores = {}
            for tool_name, keywords in self.keyword_map.items():
                score = sum(1 for kw in keywords if kw in text_lower)
                text_scores[tool_name] = score
            
            # Если текст явно указывает на инструмент — переопределяем
            max_text_tool = max(text_scores, key=text_scores.get)
            if text_scores[max_text_tool] >= 2:  # Минимум 2 совпадения
                best_tool = max_text_tool
                neural_confidence = max(neural_confidence, 0.85)
        
        # Извлечение параметров первого инструмента
        params = self._extract_params(best_tool, thought_text)
        
        # Если уверенность роутера низкая (< 0.4) для технических инструментов, 
        # принудительно идем в FinalAnswer. Это защищает от ложных срабатываний Vision/Terminal.
        if neural_confidence < 0.4 and best_tool != "FinalAnswer":
            self.logger.warning(f"MoIRA: Low confidence ({neural_confidence:.3f}) for '{best_tool}'. Routing to FinalAnswer.")
            best_tool = "FinalAnswer"
            neural_confidence = 1.0
            # Переопределяем параметры для FinalAnswer, сохраняя текст мозга
            params = self._extract_params(best_tool, thought_text)

        self.logger.info(f"MoIRA: '{best_tool}' (conf={neural_confidence:.3f})")
        return best_tool, params, neural_confidence
    
    def _extract_params(self, tool: str, thought_text: str) -> Dict[str, Any]:
        """Извлечение параметров действия из текста мысли."""
        params = {"source": "neural"}
        
        if tool == "Terminal":
            # Ищем команду в тексте
            for marker in ["run:", "выполни:", "команда:"]:
                if marker in thought_text.lower():
                    idx = thought_text.lower().index(marker) + len(marker)
                    params["command"] = thought_text[idx:].strip().split("\n")[0]
                    break
            if "command" not in params:
                params["command"] = "echo TARS: No specific command extracted"
        
        elif tool == "Browser":
            # Ищем URL
            import re
            urls = re.findall(r'https?://\S+', thought_text)
            if urls:
                params["url"] = urls[0]
            else:
                # Формируем поисковый запрос
                query = thought_text[:100].replace("\n", " ").strip()
                params["url"] = f"https://www.google.com/search?q={query}"
        
        elif tool == "Python":
            # Ищем код между тройными кавычками или после маркера
            import re
            # Пытаемся найти блок ```python ... ``` или просто ``` ... ```
            match = re.search(r'```(?:python|py)?\s*(.*?)```', thought_text, re.DOTALL | re.IGNORECASE)
            if match:
                params["code"] = match.group(1).strip()
            else:
                params["code"] = ""
        
        elif tool == "Vision":
            params["action"] = "analyze_workspace"
        
        elif tool == "FinalAnswer":
            params["answer"] = thought_text
        
        return params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    moira = MoIRA()
    
    async def test():
        thought = torch.randn(1, 1, 1024)
        
        # Тест нейронного роутинга
        tool, p, s = await moira.route(thought)
        print(f"Neural: {tool} (conf={s:.3f})")
        
        # Тест текстового уточнения
        tool, p, s = await moira.route(thought, "Нужно выполнить поиск в интернете по URL https://example.com")
        print(f"Text-guided: {tool} (conf={s:.3f}), params={p}")
        
        tool, p, s = await moira.route(thought, "Запусти команду: dir /b в терминале")
        print(f"Terminal: {tool} (conf={s:.3f}), params={p}")
    
    asyncio.run(test())
