"""
═══════════════════════════════════════════════════════════════
  TARS — Grand Unified Config (Agent 4)
═══════════════════════════════════════════════════════════════

ALL hyperparameters in ONE place. Every module reads from here.
No magic numbers scattered across files.

Usage:
    from config import TarsConfig
    cfg = TarsConfig()                         # defaults
    cfg = TarsConfig.from_file("config.json")  # from file
    cfg.to_file("config.json")                 # save

    # Access:
    cfg.d_model          # 2048
    cfg.use_cpp_core     # False (True → C++ inference engine)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TarsConfig:
    """Grand Unified Config — all TARS HELIX LITE hyperparameters."""

    # ═══════════════════════════════════════
    # MODEL ARCHITECTURE (HELIX LITE)
    # ═══════════════════════════════════════
    d_model: int = 1024          # hidden dimension (HELIX LITE)
    n_layers: int = 20           # 20 uniform HelixBlocks (TARS-Base 380M)
    vocab_size: int = 48256      # 48K text + 256 tool/control tokens
    d_state: int = 64            # SSM state dimension
    headdim: int = 64            # per-head dimension (16 heads × 64)
    d_conv: int = 4              # causal conv1d kernel size
    expand: int = 2              # SSD inner expansion factor
    ngroups: int = 1             # grouped SSD (B, C sharing)
    chunk_size: int = 64         # SSD parallel scan chunk
    dim_ff: int = 2816           # SwiGLU FFN hidden dim (~2.75× d_model)

    # WKV-7 (RWKV-7 GatedDeltaNet)
    wkv_low_rank: int = 24       # WKV Low-Rank r=24

    # ═══════════════════════════════════════
    # QUANTIZATION & RUNTIME
    # ═══════════════════════════════════════
    quant_mode: str = "ternary"  # "fp32", "fp16", "ternary" (1.58-bit)
    use_cpp_core: bool = False   # True → tars_core C++ engine
    smoothquant_alpha: float = 0.5  # SmoothQuant balance factor
    int8_dynamic_quant: bool = True  # INT8 dynamic quantization on CPU

    # ═══════════════════════════════════════
    # LoRA (single personality adapter for LITE)
    # ═══════════════════════════════════════
    n_experts: int = 1           # LITE: 1 personality LoRA (Phase 3: MoLE 4+)
    expert_rank: int = 16        # LoRA rank=16 for single expert
    expert_alpha: float = 32.0   # LoRA alpha = 2 × rank
    top_k_experts: int = 1       # LITE: always use the 1 expert
    lora_dropout: float = 0.0    # LoRA dropout

    # ═══════════════════════════════════════
    # FUSION (SSD ↔ WKV scalar gate)
    # ═══════════════════════════════════════
    # LITE: simple scalar gate y = σ(w)·ssd + (1-σ)·wkv
    # Phase 2+: upgrade to WuNeng bottleneck if ablation shows benefit
    fusion_mode: str = "scalar_gate"    # "scalar_gate" or "bottleneck"
    fusion_bottleneck_dim: int = 192    # only used if fusion_mode="bottleneck"
    fusion_init_bias: float = 0.0       # sigmoid(0) = 0.5 → 50/50 split

    # ═══════════════════════════════════════
    # DOUBT ENGINE (3 linear heads)
    # ═══════════════════════════════════════
    doubt_d_model: int = 128     # DoubtEngine hidden dim
    coherence_flag: float = 0.5  # coherence < 0.5 → FLAG
    coherence_block: float = 0.2 # coherence < 0.2 → BLOCK
    safety_flag: float = 0.6     # safety < 0.6 → FLAG
    safety_block: float = 0.3    # safety < 0.3 → BLOCK
    repeat_flag: float = 0.7     # repeat > 0.7 → FLAG
    repeat_block: float = 0.9    # repeat > 0.9 → BLOCK

    # ═══════════════════════════════════════
    # TRAINING
    # ═══════════════════════════════════════
    lr: float = 3e-4             # peak learning rate
    lr_min: float = 3e-5         # minimum LR (cosine decay end)
    batch_size: int = 4          # micro-batch size
    grad_accum_steps: int = 8    # effective batch = batch_size * grad_accum
    warmup_steps: int = 1000     # LR warmup steps
    max_steps: int = 100000      # total training steps
    max_seq_len: int = 4096      # training sequence length
    weight_decay: float = 0.01   # AdamW weight decay
    grad_clip: float = 1.0       # gradient clipping norm
    muon_quintic: bool = True    # Muon optimizer with quintic NS

    # ═══════════════════════════════════════
    # UMOT LOSS (simplified for LITE)
    # ═══════════════════════════════════════
    loss_lm_weight: float = 1.0        # language modeling (CE) loss
    loss_lora_reg_weight: float = 0.01 # LoRA regularization
    loss_doubt_weight: float = 0.1     # DoubtEngine contrastive

    # ═══════════════════════════════════════
    # MEMORY (LITE: SDM only)
    # ═══════════════════════════════════════
    sdm_dim: int = 384           # memory embedding dimension
    sdm_n_slots: int = 30000     # 30K SDM slots
    sdm_dtype: str = "int8"      # SDM storage dtype
    compaction_threshold: float = 0.8  # trigger compaction at 80% fill

    # ═══════════════════════════════════════
    # INFERENCE
    # ═══════════════════════════════════════
    max_new_tokens: int = 2048   # max generation length
    temperature: float = 0.8     # sampling temperature
    top_p: float = 0.95          # nucleus sampling
    top_k: int = 50              # top-k sampling
    min_p: float = 0.05          # min-p filtering
    repetition_penalty: float = 1.1   # repetition penalty

    # ═══════════════════════════════════════
    # TOKENIZER
    # ═══════════════════════════════════════
    tokenizer_path: str = ""     # path to tokenizer model (auto-detect if empty)
    bpe_vocab_size: int = 48256  # Qwen BPE + 256 tool tokens

    # ═══════════════════════════════════════
    # RoPE
    # ═══════════════════════════════════════
    rope_base: int = 500_000     # θ base for RoPE
    rope_max_seq_len: int = 32768  # max supported context length

    # ═══════════════════════════════════════
    # PERSONALITY
    # ═══════════════════════════════════════
    personality_name: str = "tars"   # identity profile
    humor_level: float = 0.75        # 0-1, default 75% (from Interstellar)

    # ═══════════════════════════════════════
    # INFRASTRUCTURE
    # ═══════════════════════════════════════
    models_dir: str = "models"       # directory for saved models
    data_dir: str = "data"           # directory for datasets
    log_level: str = "INFO"          # logging level
    log_max_size_mb: int = 50        # max log file size before rotation
    disk_min_free_gb: float = 1.0    # warn when disk < 1GB free
    disk_block_free_gb: float = 0.5  # block writes when disk < 500MB
    ram_pressure_pct: float = 85.0   # trigger GC when RAM > 85%

    # ═══════════════════════════════════════
    # WEIGHT FORMAT
    # ═══════════════════════════════════════
    weight_format: str = "safetensors"  # "safetensors" | "pt" | "gguf"

    # ═══════════════════════════════════════
    # SERIALIZATION
    # ═══════════════════════════════════════

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_file(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "TarsConfig":
        """Create config from dictionary (ignores unknown keys)."""
        import inspect
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_file(cls, path: str) -> "TarsConfig":
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_file_or_default(cls, path: Optional[str] = None) -> "TarsConfig":
        """Load from file if exists, else return defaults."""
        if path and os.path.exists(path):
            return cls.from_file(path)
        # Try standard locations
        for candidate in ["config.json", "tars_config.json",
                          os.path.join("models", "tars_v3", "config.json")]:
            if os.path.exists(candidate):
                try:
                    return cls.from_file(candidate)
                except Exception:
                    pass
        return cls()

    def __repr__(self) -> str:
        lines = [f"TarsConfig("]
        for k, v in self.to_dict().items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)

    def summary(self) -> str:
        """One-line summary of key params."""
        params_m = (self.d_model * self.d_model * self.n_layers * 4) / 1e6
        return (
            f"TARS d={self.d_model} L={self.n_layers} "
            f"d_state={self.d_state} vocab={self.vocab_size} "
            f"experts={self.n_experts}×top{self.top_k_experts} "
            f"quant={self.quant_mode} "
            f"cpp={'ON' if self.use_cpp_core else 'OFF'} "
            f"~{params_m:.0f}M params"
        )


# ═══════════════════════════════════════
# Global singleton (lazy)
# ═══════════════════════════════════════
_global_config: Optional[TarsConfig] = None


def get_config(path: Optional[str] = None) -> TarsConfig:
    """Get global config singleton."""
    global _global_config
    if _global_config is None:
        _global_config = TarsConfig.from_file_or_default(path)
    return _global_config


def set_config(config: TarsConfig) -> None:
    """Set global config singleton."""
    global _global_config
    _global_config = config
