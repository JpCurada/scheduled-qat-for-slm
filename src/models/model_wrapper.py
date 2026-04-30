"""
Model wrapper for SmolLM2-1.7B — loads the pretrained model and dispatches
to the correct quantization setup based on config.method.

This module is the single entry point for all training scripts and notebooks.
Call build_model_for_training(config, device) and get back a QuantizedModelWrapper
that holds the model, optional controller, and architecture metadata regardless
of which quantization method is selected.

Supported methods
-----------------
  ptq           -- Post-training quantization (no training loop, weights quantized in-place).
  standard_qat  -- Standard QAT: fake quant at target precision from epoch 0.
  scheduled_qat -- Scheduled QAT: precision reduced gradually via a schedule.
  lora_qat      -- LoRA-QAT: quantized frozen base + trainable LoRA adapters.

Architecture introspection
--------------------------
SmolLM2-1.7B is a Llama2-style model. build_model_for_training() calls
inspect_model() after loading to capture the HuggingFace model.config fields.
The resulting ModelInfo provides:
  - Layer counts, hidden sizes, vocab size, total parameters.
  - Memory estimates for FP32 / FP16 / INT8 / INT4 (approximate).
  - Per-layer breakdown via list_linear_layers().

Public API
----------
  ModelInfo                         -- architecture + memory metadata
  LayerInfo                         -- per-linear-layer metadata
  QuantizedModelWrapper             -- container returned to the trainer
  load_base_model(config, device)   -- raw FP32 load (no quantization)
  inspect_model(model, config)      -- extract ModelInfo from loaded model
  list_linear_layers(model, ...)    -- enumerate all nn.Linear layers
  build_model_for_training(config, device, total_steps)
                                    -- dispatch factory (main entry point)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Expected SmolLM2-1.7B architecture values (used as fallbacks if HuggingFace
# config attributes have different names across model revisions).
_SMOLLM2_DEFAULTS = {
    "num_hidden_layers": 24,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "vocab_size": 49152,
    "max_position_embeddings": 2048,
}

# Bytes per parameter for each precision level.
_BYTES_PER_PARAM = {
    "fp32": 4,
    "fp16": 2,
    "int8": 1,
    "int4": 0.5,
}

# Map config compute_dtype string -> torch dtype.
_COMPUTE_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_compute_dtype(config: ExperimentConfig) -> torch.dtype:
    """Resolve config.training.compute_dtype to a torch dtype.

    Defaults to torch.float32 when training is None or the field is missing,
    matching the legacy load behaviour. A string of "fp32" / "fp16" / "bf16"
    selects float32, float16 or bfloat16 respectively.
    """
    if config.training is None:
        return torch.float32
    dtype_str = getattr(config.training, "compute_dtype", "fp32") or "fp32"
    return _COMPUTE_DTYPE_MAP[dtype_str.lower()]


def maybe_enable_gradient_checkpointing(model: nn.Module, config: ExperimentConfig) -> bool:
    """Turn on HuggingFace gradient checkpointing if the config requested it.

    Required when training a 1.7B model on a 14-16 GB T4: trades ~30% more
    compute for ~40% less activation memory by recomputing the forward pass
    of each transformer block during backward.

    Disables HF's KV cache because it is incompatible with checkpointing
    (HF would otherwise emit a warning every step).

    Returns True iff checkpointing was activated, False otherwise.
    """
    if config.training is None or not getattr(config.training, "gradient_checkpointing", False):
        return False
    if not hasattr(model, "gradient_checkpointing_enable"):
        logger.warning("gradient_checkpointing requested but model has no enable method")
        return False
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    logger.info("Gradient checkpointing enabled (use_cache=False)")
    return True


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """Metadata for a single nn.Linear layer in the model.

    Attributes:
        name:        Full dotted module path (e.g. "model.layers.0.self_attn.q_proj").
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        has_bias:    Whether the layer has a bias term.
        is_excluded: True if the layer is excluded from quantization
                     (lm_head, embed_tokens).
        is_quantized: True after quantization has been applied.
        param_count: Total number of parameters (weight + bias).
    """
    name: str
    in_features: int
    out_features: int
    has_bias: bool
    is_excluded: bool
    is_quantized: bool = False

    @property
    def param_count(self) -> int:
        count = self.in_features * self.out_features
        if self.has_bias:
            count += self.out_features
        return count

    @property
    def shape_str(self) -> str:
        return f"({self.out_features}, {self.in_features})"


@dataclass
class ModelInfo:
    """Architecture metadata and memory estimates for SmolLM2-1.7B.

    Populated by inspect_model() after loading the HuggingFace model.
    Memory estimates are approximate (parameter bytes only, no activations).

    Attributes:
        model_name:           HuggingFace model identifier.
        num_hidden_layers:    Number of transformer blocks.
        hidden_size:          Model hidden dimension.
        intermediate_size:    FFN intermediate dimension.
        num_attention_heads:  Number of attention heads.
        num_key_value_heads:  Number of KV heads (GQA if < num_attention_heads).
        vocab_size:           Vocabulary size.
        max_position_embeddings: Maximum sequence length.
        total_params:         Total number of parameters.
        total_linear_layers:  Number of nn.Linear modules.
        quantizable_layers:   Layers eligible for fake quantization.
        excluded_layers:      Layers excluded from quantization.
    """
    model_name: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    total_params: int
    total_linear_layers: int
    quantizable_layers: int
    excluded_layers: int

    def memory_estimate_gb(self, precision: str = "fp32") -> float:
        """Approximate model memory in GB for the given precision.

        Args:
            precision: One of "fp32", "fp16", "int8", "int4".

        Returns:
            Estimated memory in GB (parameters only, no activations).
        """
        precision = precision.lower()
        if precision not in _BYTES_PER_PARAM:
            raise ValueError(
                f"Unknown precision {precision!r}. "
                f"Valid options: {sorted(_BYTES_PER_PARAM)}"
            )
        return self.total_params * _BYTES_PER_PARAM[precision] / 1e9

    def summary(self) -> str:
        """Multi-line architecture and memory summary for logging."""
        lines = [
            f"Model: {self.model_name}",
            f"  Layers:          {self.num_hidden_layers} transformer blocks",
            f"  Hidden size:     {self.hidden_size}",
            f"  FFN size:        {self.intermediate_size}",
            f"  Attention heads: {self.num_attention_heads} "
            f"(KV heads: {self.num_key_value_heads})",
            f"  Vocab size:      {self.vocab_size:,}",
            f"  Max seq length:  {self.max_position_embeddings}",
            f"  Total params:    {self.total_params:,}",
            f"  Linear layers:   {self.total_linear_layers} total, "
            f"{self.quantizable_layers} quantizable, "
            f"{self.excluded_layers} excluded",
            "  Memory estimates:",
            f"    FP32: {self.memory_estimate_gb('fp32'):.2f} GB",
            f"    FP16: {self.memory_estimate_gb('fp16'):.2f} GB",
            f"    INT8: {self.memory_estimate_gb('int8'):.2f} GB",
            f"    INT4: {self.memory_estimate_gb('int4'):.2f} GB",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# QuantizedModelWrapper
# ---------------------------------------------------------------------------

class QuantizedModelWrapper:
    """Container that the trainer receives from build_model_for_training().

    Holds the model, any lifecycle controller, model architecture info,
    and the original config. The trainer should use:
        - wrapper.model           -- the nn.Module to train
        - wrapper.controller      -- call on_epoch_start() / on_step() each step
        - wrapper.info            -- ModelInfo for logging
        - wrapper.config          -- ExperimentConfig for hyperparameters

    Controllers are method-specific:
        standard_qat  -> StandardQATController
        scheduled_qat -> ScheduledQATController
        lora_qat      -> None (no epoch-level state changes)
        ptq           -> None (no training)

    Attributes:
        model:      The (possibly quantized) nn.Module.
        controller: Lifecycle manager or None.
        info:       Architecture metadata.
        config:     Parsed YAML experiment config.
        extra:      Method-specific result object (PTQResult, StandardQATResult, etc.)
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        info: ModelInfo,
        controller: Optional[Any] = None,
        extra: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.info = info
        self.controller = controller
        self.extra = extra

    @property
    def method(self) -> str:
        return self.config.method

    @property
    def target_bits(self) -> int:
        return self.config.target_bits

    def describe(self) -> str:
        """One-line description for training logs."""
        ctrl_desc = self.controller.describe() if self.controller is not None else "no controller"
        return (
            f"QuantizedModelWrapper method={self.method}  "
            f"bits=INT{self.target_bits}  {ctrl_desc}"
        )


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def load_base_model(
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Load SmolLM2-1.7B from HuggingFace Hub in FP32 (or specified dtype).

    This is a thin wrapper around AutoModelForCausalLM.from_pretrained.
    It does NOT inject any quantization nodes. Use it when you need the raw
    pretrained model (e.g. for the FP32 baseline or KL divergence reference).

    Args:
        config: ExperimentConfig with model.name and model.cache_dir.
        device: Target device.
        dtype:  PyTorch dtype. Defaults to float32 for QAT shadow weights.

    Returns:
        nn.Module in train mode on the specified device.
    """
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    logger.info("Loading %s (dtype=%s) ...", config.model.name, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=dtype,
    )
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Base model loaded: %d parameters (%.2f GB FP32)",
        total_params,
        total_params * 4 / 1e9,
    )
    return model


def inspect_model(
    model: nn.Module,
    config: ExperimentConfig,
    exclude_layers: Optional[tuple[str, ...]] = None,
) -> ModelInfo:
    """Extract architecture metadata from a loaded SmolLM2-1.7B model.

    Reads fields from model.config (the HuggingFace PretrainedConfig) with
    graceful fallback to known SmolLM2-1.7B defaults if a field is absent.
    Also counts all nn.Linear modules and splits them into quantizable vs
    excluded buckets.

    Args:
        model:         Loaded nn.Module (may or may not have fake quant injected).
        config:        ExperimentConfig (used for model name and exclude_layers).
        exclude_layers: Override the exclude list. Defaults to config.quantize_config
                        exclude_layers or ("lm_head", "embed_tokens").

    Returns:
        ModelInfo populated from the live model.
    """
    from src.quantization.fake_quantize import DEFAULT_EXCLUDE_LAYERS  # avoid circular at top

    if exclude_layers is None:
        qc_exclude = getattr(config.quantize_config, "exclude_layers", None)
        exclude_layers = tuple(qc_exclude) if qc_exclude else DEFAULT_EXCLUDE_LAYERS

    # Read HuggingFace PretrainedConfig (stored as model.config after from_pretrained).
    hf_cfg = getattr(model, "config", None)

    def _get(attr: str) -> int:
        if hf_cfg is not None and hasattr(hf_cfg, attr):
            return int(getattr(hf_cfg, attr))
        return _SMOLLM2_DEFAULTS.get(attr, 0)

    total_params = sum(p.numel() for p in model.parameters())

    # Count linear layers by category.
    total_linears = 0
    excluded_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_linears += 1
            parts = set(name.split("."))
            if any(excl in parts for excl in exclude_layers):
                excluded_count += 1

    return ModelInfo(
        model_name=config.model.name,
        num_hidden_layers=_get("num_hidden_layers"),
        hidden_size=_get("hidden_size"),
        intermediate_size=_get("intermediate_size"),
        num_attention_heads=_get("num_attention_heads"),
        num_key_value_heads=_get("num_key_value_heads"),
        vocab_size=_get("vocab_size"),
        max_position_embeddings=_get("max_position_embeddings"),
        total_params=total_params,
        total_linear_layers=total_linears,
        quantizable_layers=total_linears - excluded_count,
        excluded_layers=excluded_count,
    )


def list_linear_layers(
    model: nn.Module,
    exclude_layers: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> list[LayerInfo]:
    """Return a LayerInfo for every nn.Linear in the model.

    Layers whose name contains any token from exclude_layers (split on '.')
    are marked as excluded. FakeQuantizeLinear and QuantizedLinear wrappers
    are detected to set is_quantized.

    Args:
        model:         The nn.Module to enumerate.
        exclude_layers: Layer name tokens that mark a layer as excluded.

    Returns:
        List of LayerInfo ordered by module traversal order.
    """
    # Import conditionally to avoid forcing all methods to be loaded.
    try:
        from src.quantization.fake_quantize import FakeQuantizeLinear
        _fq_class: tuple = (FakeQuantizeLinear,)
    except ImportError:
        _fq_class = ()

    try:
        from src.quantization.ptq import QuantizedLinear
        _ptq_class: tuple = (QuantizedLinear,)
    except ImportError:
        _ptq_class = ()

    quantized_classes = _fq_class + _ptq_class

    layers: list[LayerInfo] = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.Linear, *quantized_classes)):
            continue

        parts = set(name.split("."))
        is_excluded = any(excl in parts for excl in exclude_layers)
        is_quantized = isinstance(module, quantized_classes) if quantized_classes else False

        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            has_bias = module.bias is not None
        else:
            # FakeQuantizeLinear / QuantizedLinear expose these attributes.
            in_f = getattr(module, "in_features", 0)
            out_f = getattr(module, "out_features", 0)
            has_bias = getattr(module, "bias", None) is not None

        layers.append(LayerInfo(
            name=name,
            in_features=in_f,
            out_features=out_f,
            has_bias=has_bias,
            is_excluded=is_excluded,
            is_quantized=is_quantized,
        ))

    return layers


# ---------------------------------------------------------------------------
# Dispatch factory
# ---------------------------------------------------------------------------

def build_model_for_training(
    config: ExperimentConfig,
    device: torch.device,
    total_steps: Optional[int] = None,
) -> QuantizedModelWrapper:
    """Load SmolLM2-1.7B and set up quantization based on config.method.

    This is the single entry point for all training scripts. It dispatches
    to the method-specific builder, captures architecture metadata, and
    returns a QuantizedModelWrapper.

    Method dispatch:
        "ptq"           -> run_ptq (no training, immediate quantization)
        "standard_qat"  -> build_standard_qat_model
        "scheduled_qat" -> build_scheduled_qat_model (requires total_steps)
        "lora_qat"      -> build_lora_model

    Args:
        config:      Parsed ExperimentConfig (from load_config()).
        device:      Target device for model and training.
        total_steps: Total training steps (required for "scheduled_qat" so
                     the controller can convert step → fractional epoch).
                     Ignored for other methods. Computed as:
                     (num_train_tokens / seq_len / effective_batch_size) * epochs

    Returns:
        QuantizedModelWrapper with model, controller, info, config, extra.

    Raises:
        ValueError: If config.method is not one of the four supported values,
                    or if total_steps is None for scheduled_qat.
    """
    method = config.method

    logger.info(
        "build_model_for_training: method=%s  bits=INT%d  device=%s",
        method, config.target_bits, device,
    )

    if method == "ptq":
        return _build_ptq(config, device)
    elif method == "standard_qat":
        return _build_standard_qat(config, device)
    elif method == "scheduled_qat":
        if total_steps is None:
            raise ValueError(
                "total_steps is required for scheduled_qat. "
                "Compute it as: (train_tokens / seq_len / effective_batch) * epochs"
            )
        return _build_scheduled_qat(config, device, total_steps)
    elif method == "lora_qat":
        return _build_lora_qat(config, device)
    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            f"Valid methods: ptq, standard_qat, scheduled_qat, lora_qat"
        )


# ---------------------------------------------------------------------------
# Method-specific builders (private)
# ---------------------------------------------------------------------------

def _build_ptq(config: ExperimentConfig, device: torch.device) -> QuantizedModelWrapper:
    """Build a PTQ model (quantized weights, no training)."""
    from src.quantization.ptq import run_ptq  # type: ignore[import]

    model, result = run_ptq(config, device, calibrate=True)
    info = inspect_model(model, config)
    logger.info(info.summary())

    return QuantizedModelWrapper(
        model=model,
        config=config,
        info=info,
        controller=None,
        extra=result,
    )


def _build_standard_qat(
    config: ExperimentConfig,
    device: torch.device,
) -> QuantizedModelWrapper:
    """Build a Standard QAT model with FakeQuantizeLinear layers."""
    from src.quantization.standard_qat import build_standard_qat_model  # type: ignore[import]

    model, controller, result = build_standard_qat_model(config, device)
    info = inspect_model(model, config)
    logger.info(info.summary())

    return QuantizedModelWrapper(
        model=model,
        config=config,
        info=info,
        controller=controller,
        extra=result,
    )


def _build_scheduled_qat(
    config: ExperimentConfig,
    device: torch.device,
    total_steps: int,
) -> QuantizedModelWrapper:
    """Build a Scheduled QAT model with a precision schedule controller."""
    from src.quantization.scheduled_qat import build_scheduled_qat_model  # type: ignore[import]

    model, controller, result = build_scheduled_qat_model(config, device, total_steps)
    info = inspect_model(model, config)
    logger.info(info.summary())

    return QuantizedModelWrapper(
        model=model,
        config=config,
        info=info,
        controller=controller,
        extra=result,
    )


def _build_lora_qat(
    config: ExperimentConfig,
    device: torch.device,
) -> QuantizedModelWrapper:
    """Build a LoRA-QAT model with frozen quantized base and trainable adapters."""
    from src.quantization.lora_qat import build_lora_model  # type: ignore[import]

    model, result = build_lora_model(config, device)
    info = inspect_model(model, config)
    logger.info(info.summary())

    return QuantizedModelWrapper(
        model=model,
        config=config,
        info=info,
        controller=None,
        extra=result,
    )
