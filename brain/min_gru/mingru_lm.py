import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from .mingru import MinGRU
from .utils import exists, default, count_parameters

# UniversalLinear для 1.58-bit квантизации (fallback → nn.Linear)
try:
    from brain.mamba2.bitnet import UniversalLinear as _Linear
except ImportError:
    from torch.nn import Linear as _Linear

class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN — стандарт в LLaMA/Mistral/PaLM.
    
    Вместо Linear→GELU→Linear используем:
    gate = SiLU(Linear(x))
    value = Linear(x)  
    out = Linear(gate ⊙ value)
    
    Hidden dim = 2/3 * mult * dim (чтобы 3 матрицы ≈ тот же param count).
    """
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        # 2/3 scaling чтобы 3 матрицы ≈ param count of 2 матрицы с GELU
        self.dim_inner = int(dim * mult * 2 / 3)
        self.w_gate = _Linear(dim, self.dim_inner, bias=False)
        self.w_value = _Linear(dim, self.dim_inner, bias=False)
        self.w_out = _Linear(self.dim_inner, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.w_gate(x))  # SiLU = x * sigmoid(x) = Swish
        value = self.w_value(x)
        return self.dropout(self.w_out(gate * value))

# Alias для обратной совместимости
FeedForward = SwiGLUFeedForward

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

class MinGRUBlock(nn.Module):
    """Single MinGRU block: Conv1d → RMSNorm → MinGRU → FF (all residual).
    
    Works on hidden representations directly — no per-layer embedding/logits.
    """
    def __init__(self, dim, num_layers=1, dropout=0.1, expansion_factor=1.5, num_heads=4):
        super().__init__()
        self.conv = CausalDepthWiseConv1d(dim=dim, kernel_size=4)  # 4 = лучше для UTF-8 (рус. буквы = 2 байта)
        self.conv_norm = RMSNorm(dim)
        self.gru_norm = RMSNorm(dim)
        # Multi-Head MinGRU: разбиваем dim на num_heads голов
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.gru_heads = nn.ModuleList([
            MinGRU(head_dim, expansion_factor=expansion_factor)
            for _ in range(num_heads)
        ])
        self.gru_dropout = nn.Dropout(dropout)
        self.ff = SwiGLUFeedForward(dim, dropout=dropout)
        self.ff_norm = RMSNorm(dim)
        
        # Residual scaling: 1/√num_layers (GPT-2 style, stabilizes deep nets)
        self.res_scale = 1.0 / math.sqrt(num_layers)

    def forward(self, x, prev_hidden=None):
        """
        Args:
            x: [B, L, dim] hidden representation
            prev_hidden: optional cached hidden state for incremental decoding
        Returns:
            (output, next_hidden)
        """
        # Causal depthwise conv for local context
        # Always applied (causal padding handles L=1 correctly)
        x = self.conv(self.conv_norm(x)) * self.res_scale + x
        
        # Multi-Head MinGRU recurrence
        normed = self.gru_norm(x)
        B, L, D = normed.shape
        head_dim = D // self.num_heads
        
        # Split into heads: [B, L, D] → list of [B, L, head_dim]
        head_inputs = normed.split(head_dim, dim=-1)
        
        # Split prev_hidden per head using actual inner dims (includes expansion_factor)
        if prev_hidden is not None:
            # Each head's hidden dim_inner = int(head_dim * expansion_factor), not head_dim
            inner_dims = [h.to_hidden.out_features if hasattr(h, 'to_hidden') else head_dim 
                          for h in self.gru_heads]
            prev_per_head = list(prev_hidden.split(inner_dims, dim=-1))
        else:
            prev_per_head = [None] * self.num_heads
        
        head_outputs = []
        head_hiddens = []
        for i, (head, h_in, h_prev) in enumerate(zip(self.gru_heads, head_inputs, prev_per_head)):
            h_out, h_next = head(h_in, h_prev, return_next_prev_hidden=True)
            head_outputs.append(h_out)
            head_hiddens.append(h_next)
        
        gru_out = torch.cat(head_outputs, dim=-1)  # [B, L, D]
        next_hidden = torch.cat(head_hiddens, dim=-1)  # [B, 1, sum(inner_dims)]
        x = self.gru_dropout(gru_out) * self.res_scale + x
        
        # Feedforward
        x = self.ff(self.ff_norm(x)) * self.res_scale + x
        
        return x, next_hidden


class MinGRU_LM(nn.Module):
    def __init__(self, dim, num_tokens, num_layers, context_dim=1024, dropout=0.1, expansion_factor=1.5):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
        
        # ═══ Единый эмбеддинг (shared with LM head via weight tying) ═══
        self.embedding = nn.Embedding(num_tokens, dim)
        self.emb_dropout = nn.Dropout(dropout)
        
        # ═══ MinGRU blocks (operate on hidden dim, no per-layer logits) ═══
        self.blocks = nn.ModuleList([
            MinGRUBlock(dim, num_layers=num_layers, dropout=dropout, expansion_factor=expansion_factor)
            for _ in range(num_layers)
        ])
        
        # ═══ Output head ═══
        self.out_norm = RMSNorm(dim)
        self.lm_head = _Linear(dim, num_tokens, bias=False)
        # Weight tying: lm_head shares weights with embedding
        self.lm_head.weight = self.embedding.weight
        
        # ═══ QuantBridge: int8 LEANN → float32 context ═══
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
        
        # ═══ Proper initialization ═══
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for name, p in self.named_parameters():
            if p.dim() < 2:
                continue  # skip biases and scalars
            if 'embedding' in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'lm_head' in name:
                # Output projection scaled by depth (GPT-2 style)
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
            elif 'conv' in name:
                continue  # Conv1d has its own default init
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, labels=None, context_vec=None, prev_hiddens=None,
                return_hiddens=False):
        """
        Args:
            inputs: [B, L] token indices
            labels: [B, L] target indices (для обучения)
            context_vec: один из вариантов:
                - [B, context_dim] float tensor (из Ω-SSM)
                - tuple (int8_vecs, scales) из LEANN (автоматически через QuantBridge)
                - None (без контекста)
            prev_hiddens: list of hidden states for incremental decoding (None = full sequence)
            return_hiddens: if True, always return (logits, next_hiddens)
        Returns:
            if labels: (loss, logits)
            if prev_hiddens or return_hiddens: (logits, next_hiddens)
            else: logits
        """
        incremental = prev_hiddens is not None or return_hiddens
        
        # ═══ Embedding ═══
        x = self.emb_dropout(self.embedding(inputs))  # [B, L, dim]
        
        # ═══ Context injection (first position bias) ═══
        if context_vec is not None:
            if isinstance(context_vec, tuple) and self._has_bridge:
                int8_vecs, scales = context_vec
                context_vec = self.quant_bridge(int8_vecs, scales)
            ctx = self.context_proj(context_vec)  # [B, dim]
            # Add context as bias to first token position
            x[:, 0, :] = x[:, 0, :] + ctx
        
        # ═══ MinGRU blocks ═══
        next_hiddens = []
        if prev_hiddens is None:
            prev_hiddens = [None] * self.num_layers
        
        for i, block in enumerate(self.blocks):
            x, next_hidden = block(x, prev_hiddens[i])
            next_hiddens.append(next_hidden)
        
        # ═══ Output projection (only from final layer) ═══
        logits = self.lm_head(self.out_norm(x))  # [B, L, num_tokens]
        
        if labels is not None:
            loss = F.cross_entropy(logits.transpose(1, 2), labels)
            return loss, logits
        
        if incremental:
            return logits, next_hiddens
        
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
