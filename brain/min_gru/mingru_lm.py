import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .mingru import MinGRU
from .utils import exists, default, count_parameters

# UniversalLinear для 1.58-bit квантизации (fallback → nn.Linear)
try:
    from brain.mamba2.bitnet import UniversalLinear as _Linear
except ImportError:
    from torch.nn import Linear as _Linear

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            _Linear(dim, self.dim_inner),
            nn.GELU(),  
            _Linear(self.dim_inner, dim)
        )

    def forward(self, x):
        return self.net(x)

class CausalDepthWiseConv1d(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim),
            nn.Conv1d(dim, dim, kernel_size = 1)
        )
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * (self.gamma + 1)

class MinGRU_Layers(nn.Module):
    def __init__(self, dim, num_tokens, shared_emb=None):
        super().__init__()
        # Используем общий эмбеддинг если передан, иначе создаём свой
        if shared_emb is not None:
            self.emb = shared_emb
        else:
            self.emb = nn.Embedding(num_tokens, dim)
        self.casual_depth = CausalDepthWiseConv1d(dim=dim, kernel_size=3)
        self.rms_norm = RMSNorm(dim)
        self.gru = MinGRU(dim)
        self.ff = FeedForward(dim)
        
        self.norm = RMSNorm(dim)
        self.to_logits = _Linear(dim, num_tokens, bias=False)

    def forward(self, inputs, labels=None, is_first_layer=True, prev_hiddens=None):
        if is_first_layer:
            x = self.emb(inputs)
        else:
            x = self.emb(inputs.argmax(dim=-1))
        
        if exists(prev_hiddens):
            x = x[:, -1:]

        next_prev_hiddens = []
        prev_hiddens = iter(default(prev_hiddens, []))

        x = self.rms_norm(x)
        prev_hidden = next(prev_hiddens, None)

        min_gru_out, next_hidden = self.gru(x, prev_hidden, return_next_prev_hidden=True)

        x = min_gru_out + x
        next_prev_hiddens.append(next_hidden)
        x = self.ff(x) + x
        logits = self.to_logits(self.norm(x))

        if labels is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), labels)
        else:
            loss = None

        return loss, logits, next_prev_hiddens

class MinGRU_LM(nn.Module):
    def __init__(self, dim, num_tokens, num_layers, context_dim=1024):
        super().__init__()
        self.dim = dim
        
        # ═══ Единый эмбеддинг для всех слоёв (экономия памяти в num_layers раз) ═══
        self.shared_embedding = nn.Embedding(num_tokens, dim)
        
        self.layers = nn.ModuleList([
            MinGRU_Layers(dim, num_tokens, shared_emb=self.shared_embedding)
            for _ in range(num_layers)
        ])
        
        # Тайным весом привязываем lm_head к эмбеддингу (weight tying)
        # Каждый слой уже имеет to_logits, но мы привязываем первый к shared_embedding
        self.layers[0].to_logits.weight = self.shared_embedding.weight
        
        # ═══ QuantBridge: int8 LEANN → float32 context ═══
        # Адаптивный мост между int8 памятью (384-dim) и MinGRU (dim)
        try:
            from .quant_bridge import QuantBridge
            self.quant_bridge = QuantBridge(
                leann_dim=384, context_dim=context_dim
            )
            self._has_bridge = True
        except Exception:
            self._has_bridge = False
        
        # Проекция контекста → MinGRU hidden dim
        self.context_proj = _Linear(context_dim, dim)

    def forward(self, inputs, labels=None, context_vec=None):
        """
        Args:
            inputs: [B, L] token indices
            labels: [B, L] target indices (для обучения)
            context_vec: один из вариантов:
                - [B, 1024] float tensor (из Ω-SSM)
                - tuple (int8_vecs, scales) из LEANN (автоматически через QuantBridge)
                - None (без контекста)
        """
        total_loss = 0
        hidden_states = [None] * len(self.layers)
        
        # Если контекст — raw LEANN int8, пропустить через QuantBridge
        if context_vec is not None:
            if isinstance(context_vec, tuple) and self._has_bridge:
                int8_vecs, scales = context_vec
                context_vec = self.quant_bridge(int8_vecs, scales)
            
            ctx = self.context_proj(context_vec)  # [B, dim]
            hidden_states[0] = [ctx.unsqueeze(1)]  # [B, 1, dim]
        
        current_input = inputs

        for i, layer in enumerate(self.layers):
            loss, logits, next_hiddens = layer(
                inputs=current_input,
                labels=labels,
                is_first_layer=(i == 0),
                prev_hiddens=hidden_states[i]
            )
            
            if loss is not None:
                total_loss += loss
                
            current_input = logits
            hidden_states[i] = next_hiddens

        if labels is not None:
            return total_loss / len(self.layers), logits
        return logits

    def from_leann_memory(self, inputs, int8_vecs, scales, labels=None):
        """
        Удобный метод: генерация с контекстом напрямую из LEANN.
        
        Args:
            inputs: [B, L] token indices
            int8_vecs: numpy int8[B, 384] или torch int8[B, 384]
            scales: numpy float32[B] или torch float32[B]
        """
        return self.forward(inputs, labels=labels, context_vec=(int8_vecs, scales))

