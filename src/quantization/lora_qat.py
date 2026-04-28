"""
LoRA-QAT: LoRA adapters trained on top of a quantized-and-frozen base model.

Pipeline
--------

    1. Load      -- load SmolLM2-1.7B weights from HuggingFace Hub (FP32).

    2. LoRA wrap -- apply peft.get_peft_model() with a LoraConfig targeting the
                    attention projection layers (q_proj, k_proj, v_proj, o_proj).
                    PEFT replaces each target nn.Linear with a lora.Linear that
                    holds the original weight in .base_layer and trainable adapter
                    matrices in .lora_A / .lora_B.

    3. Quantize base -- walk every layer and freeze it in its quantized state:
                        * For PEFT lora.Linear: quantize base_layer.weight to the
                          dequantized FP32 approximation (round-trip via quantize_weight),
                          then set requires_grad=False. The LoRA adapters remain trainable.
                        * For remaining nn.Linear (MLP gate/up/down projections):
                          quantize weight in-place and freeze.
                        * Excluded layers (lm_head, embed_tokens): left untouched.

    4. Train     -- the trainer optimises only the LoRA adapter parameters
                    (lora_A + lora_B for each target layer, ~1-2% of total params).
                    Base weights do not accumulate gradients.

    5. Merge     -- peft_model.merge_and_unload() fuses adapters into base weights:
                        merged_weight = base_weight + (lora_B @ lora_A) * scaling
                    Returns a plain nn.Module with no PEFT machinery. The merged
                    weights are FP32 (quantized base + FP32 LoRA contribution).

    6. Export    -- GGUF export (src/utils/export.py) re-applies real PTQ to the
                    merged weights before packing for llama.cpp deployment.

Why quantize the base before training LoRA?
-------------------------------------------
Training the LoRA adapters against the quantized base teaches them to compensate
for the specific quantization error in that base. If we trained on the FP32 base
and only quantized afterwards, the adapters would have been optimised for a weight
distribution that no longer exists at inference time.

PEFT lora.Linear structure (after get_peft_model)
---------------------------------------------------
Each target nn.Linear becomes:

    lora.Linear
    ├── .base_layer     nn.Linear   original weight (to be quantized & frozen)
    ├── .lora_A         ModuleDict  {"default": nn.Linear(in, rank, bias=False)}
    ├── .lora_B         ModuleDict  {"default": nn.Linear(rank, out, bias=False)}
    └── .scaling        dict        {"default": alpha / rank}

Forward pass:
    out = base_layer(x) + lora_B["default"](lora_A["default"](dropout(x))) * scaling

Module-tree caveat: named_modules() yields lora.Linear AND its base_layer AND its
adapter linears as separate entries. Without care, a plain "for nn.Linear" loop would
try to freeze the adapter linears. This module uses id-based tracking to avoid that.

Quantization method
-------------------
Base weights are stored as their dequantized FP32 approximation:

    weight_int, scale = quantize_weight(w, bits, per_channel)   # from ptq.py
    w_dequant = weight_int.float() * scale
    base_layer.weight.data.copy_(w_dequant)
    base_layer.weight.requires_grad_(False)

This is semantically equivalent to fake quantization but without the STE (no
gradients needed since base weights are frozen). The in-place modification keeps
PEFT's module structure intact, so merge_and_unload() works correctly afterwards.

Public API
----------
    ParameterCount                  -- named tuple: total, trainable, frozen counts
    LoRAQATResult                   -- full setup summary for logging
    build_lora_model(config, device)-- full setup pipeline, returns (peft_model, result)
    get_trainable_parameters(model) -- list of LoRA params for the optimizer
    count_parameters(model)         -- count total / trainable / frozen params
    merge_lora_into_base(model)     -- merge adapters, return plain nn.Module
    save_lora_checkpoint(model, dir)-- save PEFT adapter weights to disk
    load_lora_checkpoint(base, dir) -- restore PeftModel from adapter checkpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from src.quantization.ptq import (
    DEFAULT_EXCLUDE_LAYERS,
    SUPPORTED_BITS,
    quantize_weight,
)
from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_is_excluded(name: str, exclude_layers: Sequence[str]) -> bool:
    """Return True if any component of the dotted module path is in exclude_layers."""
    parts = set(name.split("."))
    return any(excl in parts for excl in exclude_layers)


def _is_lora_layer(module: nn.Module) -> bool:
    """Duck-type check for a PEFT lora.Linear module.

    Checks for the three attributes PEFT adds to every LoRA-wrapped linear:
    base_layer, lora_A, and lora_B. This is version-agnostic and works across
    peft>=0.6.0 which changed the internal class hierarchy.
    """
    return (
        hasattr(module, "base_layer")
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and hasattr(module, "scaling")
    )


def _collect_lora_child_ids(model: nn.Module) -> set[int]:
    """Return the Python ids of all nn.Linear sub-modules owned by LoRA wrappers.

    PEFT's lora.Linear stores the original nn.Linear as .base_layer and adapter
    matrices inside .lora_A / .lora_B ModuleDicts. named_modules() yields all of
    these as separate entries. This function collects their ids so that the
    quantization loop can skip them and only process top-level LoRA wrappers and
    truly-plain nn.Linear layers.
    """
    child_ids: set[int] = set()
    for module in model.modules():
        if _is_lora_layer(module):
            child_ids.add(id(module.base_layer))
            for adapter_linear in module.lora_A.values():
                child_ids.add(id(adapter_linear))
            for adapter_linear in module.lora_B.values():
                child_ids.add(id(adapter_linear))
    return child_ids


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

@dataclass
class ParameterCount:
    """Parameter counts for a LoRA-QAT model.

    Attributes:
        total:     All parameters (base + adapters), regardless of grad.
        trainable: Parameters with requires_grad=True (LoRA adapters only).
        frozen:    Parameters with requires_grad=False (quantized base).
    """
    total: int
    trainable: int
    frozen: int

    @property
    def trainable_fraction(self) -> float:
        """Fraction of total parameters that are trainable."""
        return self.trainable / max(self.total, 1)

    def __str__(self) -> str:
        return (
            f"Parameters: total={self.total:,}  "
            f"trainable={self.trainable:,} ({self.trainable_fraction:.2%})  "
            f"frozen={self.frozen:,}"
        )


def count_parameters(model: nn.Module) -> ParameterCount:
    """Count total, trainable, and frozen parameters in the model.

    Args:
        model: Any nn.Module (PeftModel or plain model).

    Returns:
        ParameterCount with total, trainable, and frozen counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParameterCount(total=total, trainable=trainable, frozen=total - trainable)


# ---------------------------------------------------------------------------
# Setup result
# ---------------------------------------------------------------------------

@dataclass
class LoRAQATResult:
    """Summary of a completed LoRA-QAT model setup.

    Returned by build_lora_model() for logging at training start.

    Attributes:
        bits:                   Bit-width used for base weight quantization.
        rank:                   LoRA rank (r).
        alpha:                  LoRA alpha scaling factor.
        target_modules:         Module names that received LoRA adapters.
        parameter_count:        Total / trainable / frozen parameter counts.
        num_lora_layers:        Number of attention layers that received LoRA adapters.
        num_quantized_non_lora: Other nn.Linear layers quantized and frozen (MLP etc.).
        num_excluded:           Layers left at FP32 (lm_head, embed_tokens).
    """
    bits: int
    rank: int
    alpha: int
    target_modules: list[str]
    parameter_count: ParameterCount
    num_lora_layers: int
    num_quantized_non_lora: int
    num_excluded: int

    @property
    def lora_scaling(self) -> float:
        """LoRA output scaling: alpha / rank."""
        return self.alpha / self.rank

    def summary(self) -> str:
        lines = [
            f"LoRA-QAT setup summary:",
            f"  Base quantization : INT{self.bits}",
            f"  LoRA rank / alpha : {self.rank} / {self.alpha}"
            f"  (scaling={self.lora_scaling:.3f})",
            f"  Target modules    : {', '.join(self.target_modules)}",
            f"  LoRA layers       : {self.num_lora_layers}",
            f"  Quantized base    : {self.num_quantized_non_lora} non-LoRA layers",
            f"  Excluded (FP32)   : {self.num_excluded} layers",
            f"  {self.parameter_count}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base weight quantization
# ---------------------------------------------------------------------------

def _apply_base_quantization(
    model: nn.Module,
    bits: int,
    exclude_layers: Sequence[str],
    per_channel: bool,
) -> tuple[int, int, int]:
    """Quantize and freeze all base weights in the model.

    Handles three categories of layers:

    1. PEFT lora.Linear (target_modules wrapped by PEFT):
       Quantize base_layer.weight to its dequantized FP32 approximation and
       freeze it. The adapter matrices (lora_A, lora_B) are left trainable.

    2. Plain nn.Linear (non-target base layers -- MLP gate/up/down, etc.):
       Quantize weight in-place and freeze. These layers have no LoRA adapters.

    3. Excluded layers (lm_head, embed_tokens) and LoRA adapter linears:
       Skipped entirely.

    This function uses id-based tracking (_collect_lora_child_ids) to avoid
    accidentally processing the nn.Linear modules inside LoRA wrappers (base_layer,
    lora_A, lora_B) as if they were standalone base layers.

    Args:
        model:          The PeftModel after get_peft_model() has been applied.
        bits:           Bit-width for quantization (4 or 8).
        exclude_layers: Module name components to leave at FP32.
        per_channel:    Per-output-channel scaling (True) or per-tensor (False).

    Returns:
        (num_lora_layers, num_non_lora_quantized, num_excluded) counts.
    """
    lora_child_ids = _collect_lora_child_ids(model)

    num_lora = 0
    num_non_lora = 0
    num_excluded = 0

    for name, module in model.named_modules():
        # -- Category 1: PEFT LoRA wrapper --
        if _is_lora_layer(module):
            if _name_is_excluded(name, exclude_layers):
                num_excluded += 1
                logger.debug("LoRA-QAT: skip excluded LoRA layer: %s", name)
                continue

            base = module.base_layer
            w_int, scale = quantize_weight(base.weight.data, bits, per_channel)
            w_dequant = (w_int.float() * scale).to(base.weight.dtype)
            base.weight.data.copy_(w_dequant)
            base.weight.requires_grad_(False)
            if base.bias is not None:
                base.bias.requires_grad_(False)

            num_lora += 1
            logger.debug(
                "LoRA-QAT: quantized base_layer of LoRA layer %s (INT%d)", name, bits
            )

        # -- Category 2: plain nn.Linear not inside a LoRA wrapper --
        elif isinstance(module, nn.Linear) and id(module) not in lora_child_ids:
            if _name_is_excluded(name, exclude_layers):
                num_excluded += 1
                logger.debug("LoRA-QAT: skip excluded base layer: %s", name)
                continue

            w_int, scale = quantize_weight(module.weight.data, bits, per_channel)
            w_dequant = (w_int.float() * scale).to(module.weight.dtype)
            module.weight.data.copy_(w_dequant)
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.requires_grad_(False)

            num_non_lora += 1
            logger.debug(
                "LoRA-QAT: quantized non-LoRA base layer %s (INT%d)", name, bits
            )

    logger.info(
        "LoRA-QAT base quantization: %d LoRA base layers + %d non-LoRA layers "
        "quantized (INT%d); %d excluded",
        num_lora, num_non_lora, bits, num_excluded,
    )
    return num_lora, num_non_lora, num_excluded


# ---------------------------------------------------------------------------
# Main setup pipeline
# ---------------------------------------------------------------------------

def build_lora_model(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, LoRAQATResult]:
    """Full LoRA-QAT setup: load FP32 model, apply PEFT LoRA, quantize base.

    Executes the three-step setup:
        1. Load SmolLM2-1.7B from HuggingFace Hub (FP32 weights).
        2. Apply peft.get_peft_model() with LoraConfig from the YAML.
        3. Quantize and freeze all base weights; leave LoRA adapters trainable.

    After this call the returned model is ready for training with the standard
    trainer loop. Only parameters returned by get_trainable_parameters() should
    be passed to the optimizer.

    Args:
        config: Parsed ExperimentConfig. Must have .lora_config and .quantize_config.
        device: Target device for training. The model is moved here after setup.

    Returns:
        (peft_model, result) where peft_model is a PeftModel ready for training
        and result is a LoRAQATResult suitable for logging.

    Raises:
        ValueError: If config.lora_config is None (not a lora_qat config).
        ImportError: If peft is not installed.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The peft library is required for LoRA-QAT. "
            "Install it with: pip install peft>=0.10.0"
        ) from exc

    from transformers import AutoModelForCausalLM  # type: ignore[import]

    if config.lora_config is None:
        raise ValueError(
            f"build_lora_model requires config.lora_config — "
            f"method={config.method!r} has no 'lora_config' section."
        )

    lc = config.lora_config
    qc = config.quantize_config
    per_channel = qc.granularity == "per_channel"

    # 1. Load pretrained FP32 model
    logger.info(
        "Loading %s from cache %s ...", config.model.name, config.model.cache_dir
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=torch.float32,
    )
    logger.info(
        "Base model loaded: %d parameters",
        sum(p.numel() for p in base_model.parameters()),
    )

    # 2. Apply PEFT LoRA adapters to target attention projection layers
    peft_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lc.rank,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        target_modules=lc.target_modules,
        bias="none",           # do not add bias adapters
        inference_mode=False,  # enable adapter training
    )

    peft_model = get_peft_model(base_model, peft_lora_config)
    logger.info(
        "PEFT LoRA applied: rank=%d, alpha=%d, dropout=%.2f, targets=%s",
        lc.rank, lc.alpha, lc.dropout, lc.target_modules,
    )

    # 3. Quantize all base weights and freeze them
    num_lora, num_non_lora, num_excluded = _apply_base_quantization(
        peft_model,
        bits=config.target_bits,
        exclude_layers=qc.exclude_layers,
        per_channel=per_channel,
    )

    # Move to training device after quantization (quantization runs on CPU)
    peft_model.to(device)
    peft_model.train()

    # Build result summary
    param_count = count_parameters(peft_model)
    result = LoRAQATResult(
        bits=config.target_bits,
        rank=lc.rank,
        alpha=lc.alpha,
        target_modules=list(lc.target_modules),
        parameter_count=param_count,
        num_lora_layers=num_lora,
        num_quantized_non_lora=num_non_lora,
        num_excluded=num_excluded,
    )
    logger.info(result.summary())
    return peft_model, result


# ---------------------------------------------------------------------------
# Optimizer parameter access
# ---------------------------------------------------------------------------

def get_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return only the LoRA adapter parameters (lora_A and lora_B weights).

    Pass this list to the optimizer instead of model.parameters() to ensure
    only LoRA adapters are updated; base weights stay frozen.

    The filter uses requires_grad rather than name-matching so it remains
    correct if users add additional trainable components (e.g., task-specific
    heads) that should also be trained.

    Args:
        model: PeftModel returned by build_lora_model().

    Returns:
        List of nn.Parameter with requires_grad=True.
    """
    return [p for p in model.parameters() if p.requires_grad]


def get_trainable_named_parameters(
    model: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """Return (name, parameter) pairs for all trainable LoRA parameters.

    Useful for logging which adapters are training and for selective
    optimizer configuration (e.g., different LR for different adapter groups).

    Args:
        model: PeftModel returned by build_lora_model().

    Returns:
        List of (dotted_name, parameter) for all requires_grad=True params.
    """
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Merge and export
# ---------------------------------------------------------------------------

def merge_lora_into_base(peft_model: nn.Module) -> nn.Module:
    """Merge LoRA adapter weights into the quantized base and return a plain model.

    Calls peft_model.merge_and_unload() which:
        1. For each lora.Linear: merged_weight = base_weight + (lora_B @ lora_A) * scaling
        2. Replaces lora.Linear with a plain nn.Linear holding the merged weight.
        3. Returns the model stripped of all PEFT infrastructure.

    The merged weight is the sum of:
        - The dequantized base weight (FP32 approximation of the quantized original)
        - The LoRA low-rank correction learned during training

    After merging, the model is a plain HuggingFace CausalLM. The GGUF export
    step (src/utils/export.py) then re-applies real PTQ to pack the merged
    weights into INT4/INT8 for llama.cpp deployment.

    Args:
        peft_model: PeftModel returned by build_lora_model() after training.

    Returns:
        Plain nn.Module (HuggingFace CausalLM) with merged FP32 weights.
        All LoRA infrastructure is removed; the model behaves as a standard
        pretrained model with improved quantization-adapted weights.
    """
    logger.info("Merging LoRA adapters into base weights ...")
    merged = peft_model.merge_and_unload()
    n_params = sum(p.numel() for p in merged.parameters())
    n_trainable = sum(p.numel() for p in merged.parameters() if p.requires_grad)
    logger.info(
        "Merge complete: %d total params, %d trainable (all base, no adapters remain)",
        n_params, n_trainable,
    )
    return merged


# ---------------------------------------------------------------------------
# Checkpoint saving and loading
# ---------------------------------------------------------------------------

def save_lora_checkpoint(
    model: nn.Module,
    output_dir: Union[str, Path],
    adapter_name: str = "default",
) -> None:
    """Save the LoRA adapter weights to disk in PEFT format.

    Saves only the adapter matrices (lora_A, lora_B) and their config, not the
    full base model weights. This results in a tiny checkpoint (a few MB vs
    several GB for the full model), making it easy to share and version.

    The saved directory can be passed directly to load_lora_checkpoint() or to
    PeftModel.from_pretrained() to restore the adapter on top of a fresh base.

    Layout of output_dir after saving:
        adapter_config.json     -- LoraConfig serialised as JSON
        adapter_model.safetensors  -- adapter weight tensors (lora_A, lora_B)

    Args:
        model:        PeftModel after training (before merge).
        output_dir:   Directory to write the adapter checkpoint.
        adapter_name: PEFT adapter name (default: "default").
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    logger.info("LoRA adapter checkpoint saved -> %s", output_dir)

    # Log adapter size for reference
    adapter_files = list(output_dir.iterdir())
    total_bytes = sum(f.stat().st_size for f in adapter_files if f.is_file())
    logger.info(
        "Adapter checkpoint size: %.1f MB (%d files)",
        total_bytes / 1e6, len(adapter_files),
    )


def load_lora_checkpoint(
    base_model_name: str,
    adapter_dir: Union[str, Path],
    device: torch.device,
    cache_dir: Optional[str] = None,
) -> nn.Module:
    """Restore a PeftModel from a saved adapter checkpoint.

    Loads the FP32 base model fresh from HuggingFace Hub, then layers the saved
    LoRA adapters on top. Note: this restores the model to its pre-merge state
    (quantized base + LoRA adapters), not the merged version.

    After loading, the base weights must be re-quantized if the checkpoint was
    saved before quantization was applied (i.e., if using save_pretrained on the
    raw adapter without first quantizing). For production use, save a full merged
    checkpoint or re-apply build_lora_model() then load adapter weights manually.

    Args:
        base_model_name: HuggingFace model identifier (e.g. "HuggingFaceTB/SmolLM2-1.7B").
        adapter_dir:     Directory produced by save_lora_checkpoint().
        device:          Device to load the model onto.
        cache_dir:       Local cache for base model weights.

    Returns:
        PeftModel with loaded adapters, moved to device and set to eval mode.
    """
    try:
        from peft import PeftModel  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The peft library is required. Install with: pip install peft>=0.10.0"
        ) from exc

    from transformers import AutoModelForCausalLM  # type: ignore[import]

    logger.info("Loading base model %s for adapter restore ...", base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
    )

    logger.info("Loading LoRA adapter from %s ...", adapter_dir)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.to(device)
    peft_model.eval()
    logger.info("LoRA checkpoint loaded successfully.")
    return peft_model


# ---------------------------------------------------------------------------
# Introspection utilities
# ---------------------------------------------------------------------------

def lora_layer_names(model: nn.Module) -> list[str]:
    """Return the dotted paths of all PEFT LoRA-wrapped layers in the model."""
    return [name for name, mod in model.named_modules() if _is_lora_layer(mod)]


def adapter_parameter_summary(model: nn.Module) -> str:
    """Format a per-adapter-layer parameter count table for logging.

    Useful at training start to confirm which layers received adapters and
    how many parameters each contributes.

    Returns:
        Multi-line string with one row per LoRA layer.
    """
    rows: list[str] = ["LoRA adapter parameter breakdown:"]
    total_adapter_params = 0

    for name, mod in model.named_modules():
        if not _is_lora_layer(mod):
            continue
        layer_params = 0
        for adapter_linear in list(mod.lora_A.values()) + list(mod.lora_B.values()):
            layer_params += adapter_linear.weight.numel()
        total_adapter_params += layer_params
        rows.append(f"  {name:<60} {layer_params:>10,} params")

    rows.append(f"  {'TOTAL':<60} {total_adapter_params:>10,} params")
    return "\n".join(rows)
