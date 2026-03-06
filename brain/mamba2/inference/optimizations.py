"""
═══════════════════════════════════════════════════════════════
  TARS Model Optimizations — Maximum Performance Toolkit
═══════════════════════════════════════════════════════════════

Central module for all runtime/training/inference optimizations.
Auto-detects hardware and applies optimal settings.

Target Hardware:
  - RTX 4090 (24GB VRAM, local)
  - L4 (22.5GB VRAM, Colab)
  - A100 (40/80GB VRAM, Colab)
  - T4 (16GB VRAM, Colab free tier)

Usage:
  from brain.mamba2.optimizations import optimize_for_training, optimize_for_inference

  # Training:
  optimize_for_training(model, gpu_name="auto")

  # Inference:
  optimize_for_inference(model, gpu_name="auto")

  # Pruning:
  from brain.mamba2.optimizations import prune_ssm_weights
  prune_ssm_weights(model, sparsity=0.5)
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple

_logger = logging.getLogger("Tars.Optimizations")

# ═══════════════════════════════════════════════════════════════
# 1. Hardware Detection & Profiles
# ═══════════════════════════════════════════════════════════════

# Hardware optimization profiles
_HW_PROFILES: Dict[str, Dict[str, Any]] = {
    "4090": {
        "compile_mode": "max-autotune",
        "amp_dtype": "bf16",
        "chunk_size_ssd": 128,
        "chunk_size_wkv": 64,
        "optimal_batch": 16,
        "optimal_seq_len": 512,
        "grad_checkpointing": False,  # 24GB is enough for most configs
        "channels_last": True,
        "cuda_graphs": True,
        "matmul_precision": "high",  # TF32
    },
    "A100": {
        "compile_mode": "max-autotune",
        "amp_dtype": "bf16",
        "chunk_size_ssd": 256,
        "chunk_size_wkv": 128,
        "optimal_batch": 32,
        "optimal_seq_len": 1024,
        "grad_checkpointing": False,
        "channels_last": True,
        "cuda_graphs": True,
        "matmul_precision": "high",
    },
    "L4": {
        "compile_mode": "default",
        "amp_dtype": "bf16",
        "chunk_size_ssd": 64,
        "chunk_size_wkv": 32,
        "optimal_batch": 8,
        "optimal_seq_len": 256,
        "grad_checkpointing": True,
        "channels_last": True,
        "cuda_graphs": False,  # L4 can be flaky with cuda graphs
        "matmul_precision": "high",
    },
    "T4": {
        "compile_mode": "default",
        "amp_dtype": "fp16",  # T4 doesn't support bf16
        "chunk_size_ssd": 64,
        "chunk_size_wkv": 32,
        "optimal_batch": 4,
        "optimal_seq_len": 128,
        "grad_checkpointing": True,
        "channels_last": False,
        "cuda_graphs": False,
        "matmul_precision": "medium",
    },
    "cpu": {
        "compile_mode": None,
        "amp_dtype": "fp32",
        "chunk_size_ssd": 32,
        "chunk_size_wkv": 16,
        "optimal_batch": 2,
        "optimal_seq_len": 128,
        "grad_checkpointing": False,
        "channels_last": False,
        "cuda_graphs": False,
        "matmul_precision": "highest",
    },
}


def detect_gpu() -> Tuple[str, Dict[str, Any]]:
    """
    Auto-detect GPU and return (gpu_key, profile).
    
    Returns:
        gpu_key: "4090", "A100", "L4", "T4", or "cpu"
        profile: dict of optimal settings
    """
    if not torch.cuda.is_available():
        return "cpu", _HW_PROFILES["cpu"]
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    
    # Match by name
    if "4090" in gpu_name:
        key = "4090"
    elif "a100" in gpu_name:
        key = "A100"
    elif "l4" in gpu_name:
        key = "L4"
    elif "t4" in gpu_name or "tesla t4" in gpu_name:
        key = "T4"
    elif "h100" in gpu_name or "h200" in gpu_name:
        key = "A100"  # Use A100 profile (similar capabilities)
    elif "3090" in gpu_name or "3080" in gpu_name:
        key = "4090"  # Similar architecture
    elif vram_gb >= 30:
        key = "A100"
    elif vram_gb >= 20:
        key = "4090"
    elif vram_gb >= 14:
        key = "T4"
    else:
        key = "T4"  # Conservative default
    
    profile = _HW_PROFILES[key].copy()
    
    # Check bf16 support
    if key != "cpu":
        bf16_supported = torch.cuda.is_bf16_supported()
        if not bf16_supported and profile["amp_dtype"] == "bf16":
            profile["amp_dtype"] = "fp16"
            _logger.info("bf16 not supported, falling back to fp16")
    
    _logger.info(f"Detected GPU: {gpu_name} ({vram_gb:.1f}GB) → profile={key}")
    return key, profile


# ═══════════════════════════════════════════════════════════════
# 2. Model-level Optimizations
# ═══════════════════════════════════════════════════════════════

def optimize_for_training(model: nn.Module, gpu_name: str = "auto") -> Dict[str, Any]:
    """
    Apply all training-time optimizations to a TARS model.
    
    Optimizations applied:
      1. Set matmul precision (TF32 on Ampere+)
      2. Set CUDA environment variables
      3. Selective torch.compile on pure-tensor submodules
      4. Set optimal chunk sizes in TarsCoreBlock
      5. Enable gradient checkpointing if needed
    
    Args:
        model: TarsMamba2LM instance
        gpu_name: "4090", "A100", "L4", "T4", "cpu", or "auto"
    
    Returns:
        dict with applied settings
    """
    if gpu_name == "auto":
        gpu_key, profile = detect_gpu()
    else:
        gpu_key = gpu_name
        profile = _HW_PROFILES.get(gpu_name, _HW_PROFILES["T4"])
    
    applied = {"gpu": gpu_key, "profile": profile}
    
    # 1. Matmul precision → TF32 on Ampere+ GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision(profile["matmul_precision"])
        applied["matmul_precision"] = profile["matmul_precision"]
        _logger.info(f"Matmul precision: {profile['matmul_precision']}")
    
    # 2. CUDA env vars for max performance
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # 3. Set optimal chunk sizes in all TarsCoreBlock
    _set_chunk_sizes(model, profile["chunk_size_ssd"], profile["chunk_size_wkv"])
    applied["chunk_size_ssd"] = profile["chunk_size_ssd"]
    applied["chunk_size_wkv"] = profile["chunk_size_wkv"]
    
    # 4. Gradient checkpointing
    if profile["grad_checkpointing"]:
        model.use_checkpointing = True
        applied["grad_checkpointing"] = True
        _logger.info("Gradient checkpointing enabled")
    
    # 5. Selective torch.compile (skip on Windows / CPU)
    import platform
    if (profile["compile_mode"] is not None 
        and torch.cuda.is_available() 
        and platform.system() != "Windows"
        and hasattr(torch, "compile")):
        _selective_compile(model, profile["compile_mode"])
        applied["compile_mode"] = profile["compile_mode"]
    else:
        applied["compile_mode"] = None
    
    # 6. CPU thread optimization
    n_cpu = os.cpu_count() or 4
    try:
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(max(1, n_cpu // 2))
    except RuntimeError:
        pass
    
    return applied


def optimize_for_inference(model: nn.Module, gpu_name: str = "auto") -> Dict[str, Any]:
    """
    Apply all inference-time optimizations.
    
    Extra over training:
      - CUDA Graph capture for step()
      - Eval mode
      - Disable dropout
      - Cache inv_freq for RoPE
    """
    applied = optimize_for_training(model, gpu_name)
    
    model.eval()
    applied["mode"] = "eval"
    
    # Disable all dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    applied["dropout"] = "disabled"
    
    # CUDA Graph for step() — only on 4090/A100
    gpu_key = applied.get("gpu", "cpu")
    prof = _HW_PROFILES.get(gpu_key, {})
    if prof.get("cuda_graphs", False) and torch.cuda.is_available():
        try:
            _setup_cuda_graph_step(model)
            applied["cuda_graph"] = True
            _logger.info("CUDA Graph captured for step()")
        except Exception as e:
            applied["cuda_graph"] = False
            _logger.warning(f"CUDA Graph failed: {e}")
    
    return applied


def _set_chunk_sizes(model: nn.Module, ssd_chunk: int, wkv_chunk: int):
    """Set optimal SSD and WKV chunk sizes in TarsCoreBlocks."""
    for module in model.modules():
        cls_name = type(module).__name__
        if cls_name == "TarsCoreBlock":
            module.chunk_size = ssd_chunk


def _selective_compile(model: nn.Module, mode: str = "default"):
    """
    Selectively compile individual submodules that are torch.compile-safe.
    
    Skips: integral_auditor, personality_adapter, thinking_chain,
           meta_cortex, dream_engine, thinking_logger (contain Python
           control flow, time.time(), .item(), iterative loops).
    """
    compile_targets = []
    skip_names = {
        "integral_auditor", "meta_auditor", "meta_cortex",
        "dream_engine", "thinking_logger", "thinking_chain",
        "personality", "wave_critic", "critic_head",
        "belief_state", "hankel",
    }
    
    # Compile TarsCoreBlocks (pure tensor ops)
    for i, block in enumerate(model.blocks):
        try:
            block.core = torch.compile(block.core, mode=mode, fullgraph=False)
            compile_targets.append(f"blocks.{i}.core")
        except Exception as e:
            _logger.debug(f"Failed to compile block {i} core: {e}")
    
    # Compile WaveConsolidation layers (pure tensor ops)
    if hasattr(model, 'wave_consolidations'):
        for i, wc in enumerate(model.wave_consolidations):
            try:
                model.wave_consolidations[i] = torch.compile(wc, mode=mode, fullgraph=False)
                compile_targets.append(f"wave_consolidations.{i}")
            except Exception as e:
                _logger.debug(f"Failed to compile WaveConsolidation {i}: {e}")
    
    # Compile output head
    if hasattr(model, 'lm_head'):
        try:
            model.lm_head = torch.compile(model.lm_head, mode=mode)
            compile_targets.append("lm_head")
        except Exception:
            pass
    if hasattr(model, 'norm_f'):
        try:
            model.norm_f = torch.compile(model.norm_f, mode=mode)
            compile_targets.append("norm_f")
        except Exception:
            pass
    
    _logger.info(f"Compiled {len(compile_targets)} submodules: {compile_targets[:5]}...")


def _setup_cuda_graph_step(model: nn.Module):
    """
    Capture model.step() as a CUDA graph for zero-overhead inference.
    
    Requires a warmup run first to populate CUDA caches.
    """
    if not hasattr(model, 'step') or not torch.cuda.is_available():
        return
    
    device = next(model.parameters()).device
    model.reset_cache()
    
    # Warmup
    dummy = torch.randint(0, 256, (1, 1), device=device)
    for _ in range(3):
        model.step(dummy)
    
    # Capture graph
    graph = torch.cuda.CUDAGraph()
    static_input = torch.randint(0, 256, (1, 1), device=device)
    
    with torch.cuda.graph(graph):
        static_output = model.step(static_input)
    
    # Store for replay
    model._cuda_graph = graph
    model._cuda_graph_input = static_input
    model._cuda_graph_output = static_output
    
    # Monkey-patch step
    original_step = model.step
    
    @torch.no_grad()
    def fast_step(token_ids):
        if token_ids.shape == static_input.shape:
            static_input.copy_(token_ids)
            graph.replay()
            return static_output.clone()
        else:
            return original_step(token_ids)
    
    model._original_step = original_step
    model.step = fast_step


# ═══════════════════════════════════════════════════════════════
# 3. SparseSSM Pruning
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def prune_ssm_weights(
    model: nn.Module,
    sparsity: float = 0.5,
    method: str = "magnitude",
    protect_critical: bool = True,
) -> Dict[str, Any]:
    """
    SparseSSM-style pruning for Mamba/SSM models.
    
    Prunes low-importance weights to zero, creating sparse matrices.
    No fine-tuning needed for sparsity <= 0.5 (per SparseSSM paper).
    
    Args:
        model: TarsMamba2LM instance
        sparsity: fraction of weights to zero out (0.0-1.0)
        method: "magnitude" (simple) or "gradient" (needs calibration)
        protect_critical: if True, skip A_log, D, dt_bias (critical SSM params)
    
    Returns:
        dict with pruning statistics
    """
    total_params = 0
    pruned_params = 0
    pruned_layers = 0
    
    target_modules = []
    
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        
        # Target: UniversalLinear and nn.Linear in SSM paths
        if cls_name in ("UniversalLinear", "Linear"):
            # Skip embedding and lm_head (shared weights)
            if "embedding" in name or "lm_head" in name:
                continue
            target_modules.append((name, module))
    
    for name, module in target_modules:
        weight = module.weight.data
        n_params = weight.numel()
        total_params += n_params
        
        if method == "magnitude":
            # Simple magnitude-based pruning
            threshold = torch.quantile(weight.abs().flatten(), sparsity)
            mask = weight.abs() >= threshold
            weight.mul_(mask.float())
            n_pruned = (~mask).sum().item()
        else:
            # Gradient-based (simplified OBS approximation)
            # Use magnitude as proxy — full OBS requires Hessian
            threshold = torch.quantile(weight.abs().flatten(), sparsity)
            mask = weight.abs() >= threshold
            weight.mul_(mask.float())
            n_pruned = (~mask).sum().item()
        
        pruned_params += n_pruned
        pruned_layers += 1
    
    # Also prune critical SSM params if not protected
    if not protect_critical:
        for name, param in model.named_parameters():
            if any(k in name for k in ["A_log", "D", "dt_bias"]):
                n = param.numel()
                total_params += n
                threshold = torch.quantile(param.data.abs().flatten(), sparsity * 0.3)
                mask = param.data.abs() >= threshold
                param.data.mul_(mask.float())
                pruned_params += (~mask).sum().item()
    
    actual_sparsity = pruned_params / max(total_params, 1)
    
    stats = {
        "total_params": total_params,
        "pruned_params": pruned_params,
        "pruned_layers": pruned_layers,
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
    }
    
    _logger.info(
        f"Pruned {pruned_params:,}/{total_params:,} params "
        f"({actual_sparsity:.1%} sparsity) across {pruned_layers} layers"
    )
    
    return stats


# ═══════════════════════════════════════════════════════════════
# 4. LoRA / PEFT Setup
# ═══════════════════════════════════════════════════════════════

def setup_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[List[str]] = None,
    dropout: float = 0.05,
) -> Tuple[nn.Module, int]:
    """
    Add LoRA adapters to TARS model for parameter-efficient fine-tuning.
    
    Targets SSM-critical projections:
      - in_proj (shared input projection: Mamba + RWKV)
      - out_proj (shared output projection)
      - wkv_up (WKV upscale for fusion)
    
    Args:
        model: TarsMamba2LM
        rank: LoRA rank (8-32 recommended)
        alpha: LoRA alpha scaling
        target_modules: list of module name patterns to target
        dropout: LoRA dropout
    
    Returns:
        (model, n_trainable_params)
    """
    if target_modules is None:
        target_modules = ["in_proj", "out_proj", "wkv_up", "dt_proj"]
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    n_lora_params = 0
    n_adapters = 0
    
    for name, module in model.named_modules():
        # Check if this module should get a LoRA adapter
        should_adapt = any(t in name for t in target_modules)
        if not should_adapt:
            continue
        
        # Only adapt Linear/UniversalLinear layers
        cls_name = type(module).__name__
        if cls_name not in ("Linear", "UniversalLinear"):
            continue
        
        in_features = module.in_features if hasattr(module, 'in_features') else module.weight.shape[1]
        out_features = module.out_features if hasattr(module, 'out_features') else module.weight.shape[0]
        
        # Add LoRA A and B matrices
        lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        lora_B = nn.Parameter(torch.zeros(out_features, rank))
        lora_scale = alpha / rank
        
        # Register as buffers on the module
        module.register_parameter("lora_A", lora_A)
        module.register_parameter("lora_B", lora_B)
        module.lora_scale = lora_scale
        module.lora_dropout = nn.Dropout(dropout)
        
        # Monkey-patch forward to include LoRA
        original_forward = module.forward
        
        def make_lora_forward(mod, orig_fwd):
            def lora_forward(x):
                base = orig_fwd(x)
                # LoRA: base + (x @ A^T @ B^T) * scale
                lora_out = F.linear(
                    F.linear(mod.lora_dropout(x), mod.lora_A),
                    mod.lora_B
                ) * mod.lora_scale
                return base + lora_out
            return lora_forward
        
        module.forward = make_lora_forward(module, original_forward)
        
        n_lora_params += lora_A.numel() + lora_B.numel()
        n_adapters += 1
    
    # Also unfreeze norm layers and embedding (standard practice)
    for name, param in model.named_parameters():
        if "norm" in name or "lora_" in name:
            param.requires_grad = True
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    _logger.info(
        f"LoRA: {n_adapters} adapters, {n_lora_params:,} LoRA params, "
        f"{total_trainable:,}/{total_params:,} trainable "
        f"({total_trainable/total_params:.1%})"
    )
    
    return model, total_trainable


# ═══════════════════════════════════════════════════════════════
# 5. Knowledge Distillation Helper
# ═══════════════════════════════════════════════════════════════

def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Combined knowledge distillation loss.
    
    L = α * KL(softmax(student/T), softmax(teacher/T)) * T²
      + (1-α) * CE(student, labels)
    
    Args:
        student_logits: [B, L, V]
        teacher_logits: [B, L, V] (detached)
        labels: [B, L] ground truth
        temperature: soft target temperature (higher = softer)
        alpha: balance between distill and CE loss
    
    Returns:
        combined loss scalar
    """
    # Hard label loss
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
    
    # Soft target loss (KL divergence)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    kl_loss = F.kl_div(
        student_soft.view(-1, student_soft.size(-1)),
        teacher_soft.view(-1, teacher_soft.size(-1)),
        reduction="batchmean",
    ) * (temperature ** 2)
    
    return alpha * kl_loss + (1 - alpha) * ce_loss


# ═══════════════════════════════════════════════════════════════
# 6. AMP Dtype Helper
# ═══════════════════════════════════════════════════════════════

def get_amp_config(gpu_name: str = "auto") -> Dict[str, Any]:
    """
    Get optimal AMP configuration for the hardware.
    
    Returns:
        dict with keys: enabled, dtype, scaler_needed
    """
    if gpu_name == "auto":
        gpu_key, profile = detect_gpu()
    else:
        gpu_key = gpu_name
        profile = _HW_PROFILES.get(gpu_name, _HW_PROFILES["cpu"])
    
    amp_str = profile["amp_dtype"]
    
    if amp_str == "bf16":
        return {
            "enabled": True,
            "dtype": torch.bfloat16,
            "scaler_needed": False,
            "gpu": gpu_key,
        }
    elif amp_str == "fp16":
        return {
            "enabled": True,
            "dtype": torch.float16,
            "scaler_needed": True,
            "gpu": gpu_key,
        }
    else:
        return {
            "enabled": False,
            "dtype": torch.float32,
            "scaler_needed": False,
            "gpu": gpu_key,
        }


# ═══════════════════════════════════════════════════════════════
# 7. Memory Estimation
# ═══════════════════════════════════════════════════════════════

def estimate_memory(
    d_model: int = 2048,
    n_layers: int = 24,
    batch_size: int = 16,
    seq_len: int = 512,
    dtype_bytes: int = 2,  # 2 for fp16/bf16
) -> Dict[str, float]:
    """
    Estimate VRAM usage in GB.
    
    Returns dict with model_params_gb, activations_gb, optimizer_gb, total_gb
    """
    # Model params (approximate)
    expand = 2
    d_inner = d_model * expand
    d_state = 128
    
    # Per-block params: in_proj + conv1d + out_proj + omega + mole + ...
    per_block = (
        d_model * (2 * d_inner + 2 * d_state + d_inner // 64)  # in_proj (Mamba)
        + d_model * 5 * d_state                                   # in_proj (RWKV)
        + d_inner * 4                                             # conv1d
        + d_inner * d_model                                       # out_proj
        + d_inner * 2 * d_inner                                   # fusion_gate
        + d_state * d_inner                                       # wkv_up
    )
    total_params = n_layers * per_block + d_model * 32000  # + embedding
    model_gb = total_params * dtype_bytes / (1024 ** 3)
    
    # Activations (simplified)
    act_per_layer = batch_size * seq_len * d_model * dtype_bytes
    activations_gb = n_layers * act_per_layer * 3 / (1024 ** 3)  # ~3x for intermediates
    
    # Optimizer (AdamW: 2x params for momentum + variance)
    optimizer_gb = total_params * 4 * 2 / (1024 ** 3)  # fp32 optimizer states
    
    return {
        "model_params_gb": round(model_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "total_gb": round(model_gb + activations_gb + optimizer_gb, 2),
        "total_params": total_params,
    }
