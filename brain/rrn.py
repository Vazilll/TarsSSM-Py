"""
Ядро TARS RRN — Рекурсивная Реляционная Сеть (Recursive Relational Network)
=============================================
Гибридная архитектура, объединяющая три мощных подхода:
  1. Relational Memory (DeepMind RMC): Механизм внимания между ячейками памяти.
  2. TRM (Tiny Recursive Model): Эффективная рекурсия одним блоком (System 1.5).
  3. Message Passing: Динамический обмен информацией между сущностями в графе мыслей.

Принцип работы:
  Входной вектор (запрос)
    → Проекция в реляционное пространство
    → [Рекурсивный Цикл: от 4 до 8 итераций]
        → Message Passing: Обмен "сообщениями" между объектами памяти.
        → Relational Memory: Слой Multi-Head Attention для поиска скрытых связей.
        → GRU Gate: Плавное обновление вектора состояния.
        → Early Exit (IA/ACT): Ранний выход, если мысль стабилизировалась.
    → Финальный "обогащенный" контекст для основного мозга.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import asyncio
import math
from typing import Optional, List
from pathlib import Path


# ═══════════════════════════════════════════════════════════════
# 1. RELATIONAL MEMORY CORE (из DeepMind relational-rnn-pytorch)
# ═══════════════════════════════════════════════════════════════

class RelationalMemoryCore(nn.Module):
    """
    Ядро реляционной памяти (Relational Memory Core, RMC).
    Математическая реализация из работ DeepMind. 
    
    В отличие от обычной памяти (простой вектор), RMC представляет память как
    набор 'слотов' (матрицу), где каждый слот взаимодействует с другими через
    Self-Attention. Это позволяет модели понимать отношения между фактами.
    """
    def __init__(self, mem_slots=4, head_size=64, num_heads=4, 
                 forget_bias=1.0, input_bias=0.0, gate_style='unit'):
        super().__init__()
        self.mem_slots = mem_slots
        self.head_size = head_size
        self.num_heads = num_heads
        self.mem_size = head_size * num_heads  # Полный размер слота памяти
        
        # QKV проекции для Multi-Head Attention
        self.qkv_size = 3 * head_size  # query + key + value
        self.total_qkv = self.qkv_size * num_heads
        self.qkv_proj = nn.Linear(self.mem_size, self.total_qkv)
        self.qkv_norm = nn.LayerNorm(self.total_qkv)
        
        # Post-attention MLP (2 слоя как в оригинале)
        self.attn_mlp = nn.Sequential(
            nn.Linear(self.mem_size, self.mem_size),
            nn.ReLU(),
            nn.Linear(self.mem_size, self.mem_size)
        )
        self.norm1 = nn.LayerNorm(self.mem_size)
        self.norm2 = nn.LayerNorm(self.mem_size)
        
        # Входная проекция
        self.input_proj = nn.Linear(self.mem_size, self.mem_size)
        
        # Гейты (как в LSTM: input_gate + forget_gate)
        self.gate_style = gate_style
        if gate_style == 'unit':
            gate_size = self.mem_size
        elif gate_style == 'memory':
            gate_size = 1
        else:
            gate_size = 0
            
        if gate_size > 0:
            self.input_gate_proj = nn.Linear(self.mem_size, gate_size * 2)
            self.memory_gate_proj = nn.Linear(self.mem_size, gate_size * 2)
            self.forget_bias = nn.Parameter(torch.tensor(forget_bias))
            self.input_bias = nn.Parameter(torch.tensor(input_bias))
        
    def initial_state(self, batch_size):
        """Инициализация памяти как единичная матрица (каждый слот уникален)."""
        init = torch.eye(self.mem_slots).unsqueeze(0).expand(batch_size, -1, -1)
        if self.mem_size > self.mem_slots:
            pad = torch.zeros(batch_size, self.mem_slots, self.mem_size - self.mem_slots)
            init = torch.cat([init, pad], dim=-1)
        elif self.mem_size < self.mem_slots:
            init = init[:, :, :self.mem_size]
        return init
    
    def multihead_attention(self, memory):
        """Multi-Head Self-Attention из 'Attention is All You Need'."""
        B, N, _ = memory.shape
        
        # Оптимизированный подход: разделяем на Q, K, V после LayerNorm
        qkv = self.qkv_norm(self.qkv_proj(memory))  # [B, N, num_heads * 3 * head_size]
        qkv = qkv.view(B, N, self.num_heads, 3, self.head_size)
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        
        # [B, num_heads, N, head_size]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = self.head_size ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # Merge heads: [B, N, mem_size]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.mem_size)
        return out
    
    def forward_step(self, input_vec, memory):
        """
        Один шаг реляционной памяти.
        input_vec: [B, mem_size] — входной вектор
        memory:    [B, mem_slots, mem_size] — текущая память
        Returns:   (output, new_memory)
        """
        # Проецируем вход и конкатенируем с памятью
        inp = self.input_proj(input_vec).unsqueeze(1)  # [B, 1, mem_size]
        mem_plus_input = torch.cat([memory, inp], dim=1)  # [B, mem_slots+1, mem_size]
        
        # Multi-Head Self-Attention
        attended = self.multihead_attention(mem_plus_input)
        mem_plus_input = self.norm1(mem_plus_input + attended)
        
        # Post-attention MLP
        mlp_out = self.attn_mlp(mem_plus_input)
        mem_plus_input = self.norm2(mem_plus_input + mlp_out)
        
        # Отрезаем входной слот — остаётся обновлённая память
        next_memory = mem_plus_input[:, :-1, :]
        
        # Применяем гейты (как в LSTM)
        if self.gate_style in ('unit', 'memory'):
            gate_inp = self.input_gate_proj(inp)
            gate_mem = self.memory_gate_proj(memory)
            gates = gate_mem + gate_inp
            i_gate, f_gate = gates.chunk(2, dim=-1)
            i_gate = torch.sigmoid(i_gate + self.input_bias)
            f_gate = torch.sigmoid(f_gate + self.forget_bias)
            next_memory = i_gate * torch.tanh(next_memory) + f_gate * memory
        
        # Выход — плоский вектор из всех слотов
        output = next_memory.view(next_memory.shape[0], -1)
        return output, next_memory


# ═══════════════════════════════════════════════════════════════
# 2. MESSAGE PASSING (из recurrent-relational-networks)
# ═══════════════════════════════════════════════════════════════

class MessagePassingLayer(nn.Module):
    """
    Слой обмена "сообщениями" между узлами графа знаний.
    
    Архитектура RRN (Recurrent Relational Networks).
    Каждый слот памяти или объект в запросе рассматривается как узел.
    Узлы 'разговаривают' друг с другом, передавая векторы признаков.
    Это критично для решения задач, требующих многошаговой логики (Grounding).
    """
    def __init__(self, node_dim, msg_dim=256):
        super().__init__()
        # Улучшенный Message Passing Layer (Gated MLP)
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2, msg_dim),
            nn.GELU(),
            nn.Linear(msg_dim, msg_dim),
            nn.GELU(),
            nn.Linear(msg_dim, node_dim * 2) # Выход для Gate и Message
        )
    
    def forward(self, nodes, edges):
        """
        nodes: [B, N, D] — узлы графа
        edges: list of (i, j) — рёбра 
        Returns: [B, N, D] — обновлённые узлы
        """
        B, N, D = nodes.shape
        
        if not edges:
            # Если рёбер нет — полносвязный граф (все со всеми)
            edges = [(i, j) for i in range(N) for j in range(N) if i != j]
        
        # Собираем сообщения с динамическим гейтированием (Gated Message Passing)
        messages = torch.zeros_like(nodes)
        for (i, j) in edges:
            # Сообщение от узла i к узлу j
            pair = torch.cat([nodes[:, i, :], nodes[:, j, :]], dim=-1)  # [B, 2*D]
            out = self.message_fn(pair)  # [B, 2*D]
            msg, gate = out.chunk(2, dim=-1)
            gate = torch.sigmoid(gate)
            messages[:, j, :] = messages[:, j, :] + (msg * gate)
        
        return messages


# ═══════════════════════════════════════════════════════════════
# 3. RECURSIVE REASONING BLOCK (из TinyRecursiveModels TRM)
# ═══════════════════════════════════════════════════════════════

class RecursiveReasoningBlock(nn.Module):
    """
    Рекурсивный блок рассуждений (TRM подход).
    Один и тот же блок прогоняется N раз, уточняя ответ.
    
    Каждый проход:
      1. Инъекция входа (Input Injection — как residual)
      2. Self-Attention (необязательно)  
      3. SwiGLU MLP для нелинейной трансформации
      4. RMS Norm для стабилизации
    """
    def __init__(self, dim, expansion=2.0):
        super().__init__()
        self.dim = dim
        inter = int(dim * expansion)
        
        # SwiGLU MLP (как в TRM и LLaMA)
        self.gate_proj = nn.Linear(dim, inter, bias=False)
        self.up_proj = nn.Linear(dim, inter, bias=False)
        self.down_proj = nn.Linear(inter, dim, bias=False)
        
        self.rms_eps = 1e-5
        
    def rms_norm(self, x):
        variance = x.float().square().mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.rms_eps)).to(x.dtype)
    
    def forward(self, hidden, input_injection):
        """
        hidden: текущее скрытое состояние [B, D]
        input_injection: оригинальный вход [B, D] (residual)
        """
        # Input injection (TRM: z = z + input)
        x = hidden + input_injection
        
        # SwiGLU
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_out = self.down_proj(gate * up)
        
        # RMS Norm + Residual
        x = self.rms_norm(x + mlp_out)
        return x


# ═══════════════════════════════════════════════════════════════
# 4. CONFIDENCE HEAD (Adaptive Computation — ранний выход)
# ═══════════════════════════════════════════════════════════════

class ConfidenceHead(nn.Module):
    """
    Голова уверенности. Решает: «достаточно ли я подумал?»
    Если confidence > threshold — ранний выход из рекурсии.
    Вдохновлено TRM halt mechanism и ACT (Adaptive Computation Time).
    """
    def __init__(self, dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.head(x))


# ═══════════════════════════════════════════════════════════════
# 5. UNIFIED RRN CORE — Главное ядро TARS
# ═══════════════════════════════════════════════════════════════

class TarsRRN(nn.Module):
    """
    Унифицированное ядро RRN для TARS.
    
    Это 'промежуточный мозг' (System 1.5). 
    Он глубже рефлексов, но быстрее основного SSM-ядра.
    
    Главная особенность: Адаптивная глубина рекурсии.
    Использует голову уверенности (Confidence Head) для оценки
    качества 'прожарки' мысли на каждом шаге.
    """
    def __init__(self, dim=256, mem_slots=4, num_heads=4, 
                 max_steps=8, confidence_threshold=0.9):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        
        # Компоненты гибридного ядра.
        head_size = dim // num_heads
        self.rmc = RelationalMemoryCore(mem_slots=mem_slots, head_size=head_size, num_heads=num_heads)
        self.msg_pass = MessagePassingLayer(node_dim=dim)
        self.reason_block = RecursiveReasoningBlock(dim=dim)
        self.confidence = ConfidenceHead(dim)
        self.gru_cell = nn.GRUCell(dim, dim)
        self.z_init = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x, n_steps=None, return_all=False):
        """
        Основной цикл рекурсивного рассуждения.
        """
        B = x.shape[0]
        steps = n_steps or self.max_steps # Определяем потолок шагов.
        
        # Инициализируем 'белый лист' сознания.
        z = self.z_init.unsqueeze(0).expand(B, -1)
        memory = self.rmc.initial_state(B).to(x.device)
        
        actual_steps = 0
        all_outputs = [] if return_all else None
        for step in range(steps):
            actual_steps = step + 1
            
            # 1. Этап Уточнения (Reasoning): инъекция входа.
            z = self.reason_block(z, x)
            
            # 2. Этап Отношений (Relational Attention): работа с памятью.
            _, memory = self.rmc.forward_step(z, memory)
            
            # 3. Этап Сообщений (Message Passing): уточнение связей.
            mem_updated = self.msg_pass(memory, edges=[])
            memory = memory + 0.1 * mem_updated
            
            # 4. Этап Интеграции (GRU): слияние памяти и мысли.
            z = self.gru_cell(memory.mean(dim=1), z)
            
            if return_all:
                all_outputs.append(z.detach().clone())
            
            # 5. Ранний выход (Early Exit): если уверенность > порога (напр. 0.9).
            conf = self.confidence(z)
            if n_steps is None and conf.mean().item() > self.confidence_threshold:
                break
        
        if return_all:
            return z, conf, actual_steps, all_outputs
        return z, conf, actual_steps


# ═══════════════════════════════════════════════════════════════
# 6. RRN CORE — Интерфейс для TARS (GIE / Voice / Sleep)
# ═══════════════════════════════════════════════════════════════

class RrnCore:
    """
    Спинной мозг ТАРС — маршрутизатор нервной системы.
    
    3 режима работы:
      Mode 1 — Рефлекс:  fast_reply()  (<50ms, MinGRU alone)
      Mode 2 — Действие:  dispatch()    (<200ms, Synapses parallel)
      Mode 3 — Глубокий:  precompute_grounding() (500ms+, Synapses → summarize → Brain)
    
    Также:
      sleep_consolidation() — ночная консолидация памяти
    """
    # Порог для рефлекса (Mode 1)
    REFLEX_THRESHOLD = 0.85
    # Порог для действия (Mode 2) vs глубокое мышление (Mode 3)
    ACTION_THRESHOLD = 0.50
    
    def __init__(self, model_path="models/llm/rrn_small.gguf", dim=256):
        self.model_path = model_path
        self.logger = logging.getLogger("Tars.RRN")
        self.llm = None
        self.working_memory = []
        
        # Нейронное ядро RRN
        self.neural_core = TarsRRN(dim=dim, mem_slots=4, num_heads=4, max_steps=8)
        self.input_proj = nn.Linear(1024, dim)   # Проекция из SSM-пространства
        self.output_proj = nn.Linear(dim, 1024)  # Обратная проекция (обученная, не repeat!)
        
        param_count = sum(p.numel() for p in self.neural_core.parameters())
        self.logger.info(f"Spine: Нейронное ядро ({param_count:,} params)")
        
        # MinGRU для рефлексов
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            from brain.min_gru.mingru_lm import MinGRU_LM
            self.mingru_lm = MinGRU_LM(dim=256, num_tokens=256, num_layers=4)
            
            # Загрузить обученные веса если есть
            weights_path = Path(__file__).parent.parent / "models" / "mingru" / "mingru_best.pt"
            if weights_path.exists():
                ckpt = torch.load(str(weights_path), map_location='cpu', weights_only=False)
                self.mingru_lm.load_state_dict(ckpt['model_state_dict'], strict=False)
                self.logger.info(f"Spine: MinGRU weights loaded (epoch {ckpt.get('epoch', '?')})")
            
            self.mingru_lm.to(self.device)
            self.mingru_lm.eval()
            self.has_mingru = True
            self.logger.info("Spine: MinGRU reflex engine active")
        except Exception as e:
            self.has_mingru = False
            self.logger.warning(f"Spine: MinGRU not available: {e}")
        
        # LEANN (lazy-load)
        self._leann = None
        self._leann_loaded = False
        
        # Synapse Pool (lazy-load)
        self._synapse_pool = None
    
    @property
    def leann(self):
        """Lazy-load LEANN memory."""
        if not self._leann_loaded:
            self._leann_loaded = True
            try:
                from memory.leann import LeannIndex
                self._leann = LeannIndex()
                self._leann.load()
                self.logger.info(f"Spine: LEANN loaded ({len(self._leann.texts)} docs)")
            except Exception as e:
                self.logger.warning(f"Spine: LEANN not available: {e}")
        return self._leann
    
    @property
    def synapse_pool(self):
        """Lazy-load SynapsePool."""
        if self._synapse_pool is None:
            from tools.micro_agents import SynapsePool
            self._synapse_pool = SynapsePool(leann=self.leann)
            self.logger.info("Spine: SynapsePool initialized (5 synapses)")
        return self._synapse_pool
    
    def _detect_mode(self, query: str, confidence: float) -> int:
        """
        Определить режим работы:
          1 = Рефлекс (confidence > 85%)
          2 = Действие (есть action-triggers, confidence > 50%)
          3 = Глубокое мышление (всё остальное)
        """
        if confidence > self.REFLEX_THRESHOLD:
            return 1
        
        # Проверяем action-triggers для Mode 2
        q_lower = query.lower()
        action_triggers = [
            "найди", "открой", "запусти", "выполни", "удали", "создай",
            "find", "open", "run", "execute", "delete", "create",
            "pip", "python", "git",
        ]
        if confidence > self.ACTION_THRESHOLD and any(t in q_lower for t in action_triggers):
            return 2
        
        return 3
    
    async def process(self, query: str) -> dict:
        """
        Главная точка входа Spine.
        Определяет mode и обрабатывает соответственно.
        
        Returns:
            {
                "mode": 1|2|3,
                "confidence": float,
                "response": str or None,       # Mode 1: ответ, Mode 2-3: None
                "context": str or None,         # Mode 3: context для Brain
                "synapse_results": list or None, # Mode 2-3: результаты синапсов
                "rrn_vector": Tensor or None,    # Mode 3: context_vector [1, 1024]
            }
        """
        # 1. RRN classification
        with torch.no_grad():
            x = self._text_to_vec(query)
            x_proj = self.input_proj(x)
            z, conf, steps = self.neural_core(x_proj, n_steps=2)
            confidence = conf.item()
        
        mode = self._detect_mode(query, confidence)
        self.logger.info(f"Spine: mode={mode}, conf={confidence:.2f}, steps={steps}")
        
        # ═══ Mode 1: Рефлекс ═══
        if mode == 1:
            reply = await self._reflex_reply(query, z, confidence)
            if reply:
                return {
                    "mode": 1, "confidence": confidence,
                    "response": reply, "context": None,
                    "synapse_results": None, "rrn_vector": None,
                }
            # Fallback to mode 3 if reflex failed
            mode = 3
        
        # ═══ Mode 2: Действие (синапсы) ═══
        if mode == 2:
            results = await self.synapse_pool.fire_for_query(query)
            summary = self.synapse_pool.summarize_results(results)
            return {
                "mode": 2, "confidence": confidence,
                "response": summary if summary else None,
                "context": summary,
                "synapse_results": results, "rrn_vector": None,
            }
        
        # ═══ Mode 3: Глубокое мышление (Synapses → summarize → Brain) ═══
        # 3a. Синапсы собирают данные параллельно
        results = await self.synapse_pool.fire_for_query(query)
        synapse_context = self.synapse_pool.summarize_results(results)
        
        # 3b. RRN глубокий проход (4-8 шагов) для grounding
        with torch.no_grad():
            z_deep, conf_deep, deep_steps = self.neural_core(x_proj, n_steps=None)
        
        # 3c. Проецируем RRN вектор → 1024d через обученную проекцию
        rrn_context_vec = self.output_proj(z_deep)  # [1, 1024]
        
        # 3d. Формируем текстовый контекст для Brain
        full_context = (
            f"[Synapse Data]\n{synapse_context}\n"
            f"[RRN] steps={deep_steps}, confidence={conf_deep.item():.2f}\n"
            f"[Working Memory] {self.working_memory[-3:]}"
        )
        
        self.working_memory.append(f"Deep: {query[:40]} → mode=3, steps={deep_steps}")
        
        return {
            "mode": 3, "confidence": conf_deep.item(),
            "response": None,
            "context": full_context,
            "synapse_results": results,
            "rrn_vector": rrn_context_vec,
        }
    
    async def _reflex_reply(self, prompt: str, z, confidence: float) -> Optional[str]:
        """Mode 1: быстрый ответ через MinGRU."""
        if not self.has_mingru:
            return None
        
        try:
            from brain.min_gru.generate import generate_text
            
            loop = asyncio.get_event_loop()
            mg_prompt = f"Вопрос: {prompt}\nОтвет:"
            
            # Контекст из RRN через обученную проекцию (не repeat!)
            context_vec = self.output_proj(z)  # [1, 1024]
            
            raw_reply = await loop.run_in_executor(None,
                lambda: generate_text(self.mingru_lm, start_text=mg_prompt,
                                      max_length=80, temperature=0.7,
                                      device=self.device, context_vec=context_vec)
            )
            
            # Извлекаем ответ
            if "Ответ:" in raw_reply:
                reply = raw_reply.split("Ответ:")[-1].strip()
            else:
                reply = raw_reply[len(mg_prompt):].strip()
            
            # Фильтр мусора
            if len(reply) < 3:
                return None
            cyrillic_count = sum(1 for c in reply if '\u0400' <= c <= '\u04FF')
            if len(reply) > 5 and cyrillic_count < len(reply) * 0.2:
                self.logger.info(f"Spine: MinGRU garbage ('{reply[:20]}'), escalating")
                return None
            
            # Сохранить в LEANN для долгосрочной памяти
            if self.leann and confidence > 0.7:
                try:
                    self.leann.add_document(f"Q: {prompt}\nA: {reply}")
                except Exception:
                    pass
            
            self.working_memory.append(f"Reflex: {prompt[:30]} → {reply[:30]}")
            return reply
            
        except Exception as e:
            self.logger.warning(f"Spine reflex error: {e}")
            return None
    
    async def fast_reply(self, prompt: str) -> Optional[dict]:
        """
        Совместимый интерфейс: рефлекс (System 1).
        Возвращает dict с text/confidence или None.
        """
        result = await self.process(prompt)
        if result["response"]:
            return {
                "text": result["response"],
                "confidence": result["confidence"],
                "mode": result["mode"],
            }
        return None

    async def precompute_grounding(self, query: str, memory_engine, titans_engine=None) -> str:
        """
        Relational Grounding (System 1.5).
        
        Делает 4–8 шагов рекурсии для построения реляционной карты:
        связывает запрос с LEANN и Titans перед передачей в основной мозг.
        """
        self.logger.info(f"RRN: Grounding для '{query[:30]}...'")
        
        # 1. Нейронный Grounding (глубокий проход)
        with torch.no_grad():
            x = self._text_to_vec(query)
            x_proj = self.input_proj(x)
            z, conf, steps = self.neural_core(x_proj, n_steps=None)  # Адаптивно!
            self.logger.info(f"RRN: Grounding за {steps} шагов (conf={conf.item():.2f})")
        
        # 2. Поиск в LEANN (векторная память)
        leann_data = ""
        if memory_engine:
            mems = await memory_engine.retrieve_memories(query)
            leann_data = " | ".join(mems) if mems else ""
        
        # 3. Поиск в Titans (нейронная ассоциативная память)
        titans_data = ""
        if titans_engine:
            try:
                # Используем выход RRN как вектор запроса для Titans
                rrn_vec = self.output_proj(z)  # [1, 1024] — обогащённый вектор
                recall = await titans_engine.get_recall(rrn_vec)
                titans_data = f"Neural Recall: {recall.mean().item():.4f}"
            except Exception as e:
                self.logger.debug(f"Titans recall error: {e}")
        
        # 4. Формируем реляционную карту
        rel_map = (
            f"[LEANN Knowledge]: {leann_data}\n"
            f"[Titans {titans_data}]\n"
            f"[RRN State]: steps={steps}, confidence={conf.item():.2f}\n"
            f"[Working Memory]: {self.working_memory[-3:]}"
        )
        
        self.working_memory.append(f"Grounding: {query[:40]} -> steps={steps}")
        return rel_map

    async def sleep_consolidation(self, history: list, memory_engine):
        """
        Night Clean (Sleep Phase).
        Консолидирует опыт дня через рекурсию (глубокий анализ).
        """
        if not history or not memory_engine: 
            return
        self.logger.info(f"RRN: Фаза сна. Анализ {len(history)} событий...")
        
        summary_parts = []
        for h in history:
            if isinstance(h, dict):
                tool = h.get('tool', 'unknown')
                obs = h.get('obs', '')
                if 'Error' not in str(obs):
                    summary_parts.append(f"{tool}: {obs}")
        
        if summary_parts:
            final_doc = f"Consolidated Experience: {' | '.join(summary_parts)}"
            memory_engine.add_document(final_doc)
            self.logger.info(f"RRN: Архивировано {len(summary_parts)} записей в LEANN.")
        
        # Очищаем рабочую память
        self.working_memory = self.working_memory[-5:]
        self.logger.info("RRN: Фаза сна завершена.")

    def _text_to_vec(self, text: str) -> torch.Tensor:
        """
        Продвинутая Фурье-векторизация текста (Positional Encoding).
        Обеспечивает строгое выравнивание семантического базиса с AUSSM.
        """
        vec = torch.zeros(1, 1024) # [1, 1024]
        if not text: return vec
        
        chars = [ord(c) for c in text[:256]]
        for i in range(len(chars) - 1):
            val = (chars[i] * 31 + chars[i+1]) % 1024
            # Фурье базис
            freq = 1.0 / (10000 ** ((2 * (val % 512)) / 1024))
            vec[0, val] += math.sin(i * freq)
            vec[0, (val + 1) % 1024] += math.cos(i * freq)
            
        norm = vec.norm()
        return vec / norm if norm > 1e-8 else vec


# ═══════════════════════════════════════════════════════════════
# ТЕСТ
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== TARS RRN Core Test ===\n")
    
    # Тест нейронного ядра
    core = TarsRRN(dim=256, mem_slots=4, num_heads=4, max_steps=8)
    param_count = sum(p.numel() for p in core.parameters())
    print(f"Параметров в нейронном ядре: {param_count:,}")
    
    x = torch.randn(1, 256)
    
    # Рефлекс (2 шага)
    z, conf, steps = core(x, n_steps=2)
    print(f"\n[Reflex Mode] steps={steps}, confidence={conf.item():.4f}, output_shape={z.shape}")
    
    # Адаптивный режим
    z, conf, steps = core(x, n_steps=None)
    print(f"[Adaptive Mode] steps={steps}, confidence={conf.item():.4f}, output_shape={z.shape}")
    
    # Полный проход
    z, conf, steps, all_out = core(x, n_steps=8, return_all=True)
    print(f"[Full Mode] steps={steps}, confidence={conf.item():.4f}")
    for i, o in enumerate(all_out):
        step_conf = core.confidence(o).item()
        print(f"  Step {i+1}: norm={o.norm().item():.4f}, conf={step_conf:.4f}")
    
    # Тест RrnCore (полный интерфейс)
    print("\n=== RrnCore Integration Test ===")
    rrn = RrnCore()
    
    async def test():
        reply = await rrn.fast_reply("Привет, как дела?")
        print(f"Fast Reply: {reply}")
        
        grounding = await rrn.precompute_grounding("Найди файл конфигурации", None, None)
        print(f"Grounding:\n{grounding}")
    
    asyncio.run(test())
