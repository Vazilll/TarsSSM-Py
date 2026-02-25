import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .mingru import MinGRU
from .utils import exists, default, count_parameters

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, self.dim_inner),
            nn.GELU(),  
            nn.Linear(self.dim_inner, dim)
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
        self.to_logits = nn.Linear(dim, num_tokens, bias=False)

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
        
        # Проекция контекста из Ω-SSM (1024-dim) → MinGRU hidden (256-dim)
        self.context_proj = nn.Linear(context_dim, dim)

    def forward(self, inputs, labels=None, context_vec=None):
        """
        Args:
            inputs: [B, L] token indices
            labels: [B, L] target indices (для обучения)
            context_vec: [B, 1024] вектор из Ω-SSM/RRN для кондиционирования генерации
        """
        total_loss = 0
        hidden_states = [None] * len(self.layers)
        
        # Если есть контекст из Ω-SSM, инжектируем как начальное скрытое состояние
        if context_vec is not None:
            ctx = self.context_proj(context_vec)  # [B, dim]
            hidden_states[0] = [ctx.unsqueeze(1)]  # [B, 1, dim] — prev_hidden для первого слоя
        
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

