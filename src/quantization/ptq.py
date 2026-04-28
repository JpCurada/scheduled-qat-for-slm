"""
Post-Training Quantization (PTQ) for SmolLM2-1.7B.

PTQ converts a pretrained FP32 model to a lower-precision representation without
any further training. The process has three steps:

    1. Load — load the pretrained FP32 model from disk or HuggingFace Hub.
    2. Calibrate — run a small set of real inputs through the model to collect
       per-layer weight quantization error statistics. For weight-only PTQ with
       the symmetric min-max formula, calibration data is used for diagnostics
       (quantization SNR per layer) rather than changing how scales are computed.
    3. Quantize — walk every eligible nn.Linear layer, compute a per-channel
       scale from the weight's absolute maximum, round weights to integers in
       [-q_max, q_max], and store them as a QuantizedLinear module.

The resulting model stores integer weights (torch.int8) alongside float32 per-
channel scales. During inference it dequantizes on the fly:
    w_fp32 = weight_int.float() * scale

This matches the GGUF runtime behaviour — llama.cpp stores weights as integers
and dequantizes them at compute time. The GGUF export step (src/utils/export.py)
reads the integer weights and scales directly.

Formula (from project spec, same as used in fake_quantize.py):
    scale      = max(abs(weight)) / (2^(bits-1) - 1)    per-channel
    quantized  = round(weight / scale).clamp(-q_max, q_max)
    dequantized = quantized * scale                      inference path

Excluded layers (kept at FP32 throughout):
    lm_head      -- output projection; directly affects token probabilities
    embed_tokens -- input embedding table (nn.Embedding, not Linear)

Public API:
    quantize_weight(weight, bits, per_channel)  -- quantize a single weight tensor
    QuantizedLinear                             -- module storing int8 weights + scale
    apply_ptq(model, bits, ...)                 -- replace Linear layers in-place
    run_calibration(model, calib_loader, ...)   -- collect per-layer error stats
    quantization_error(weight, ql)              -- MSE + SNR for a single layer
    run_ptq(config, device)                     -- full pipeline: load → calibrate → quantize
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)

# Symmetric integer ranges: [-q_max, q_max], matching fake_quantize.py.
SUPPORTED_BITS = {4, 8}
_Q_MAX = {4: 7, 8: 127}   # 2^(bits-1) - 1

DEFAULT_EXCLUDE_LAYERS: tuple[str, ...] = ("lm_head", "embed_tokens")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_is_excluded(name: str, exclude_layers: Sequence[str]) -> bool:
    """Return True if any component of the dotted module path is in exclude_layers.

    Matches complete path segments only, so "lm_head" matches "model.lm_head"
    but not "lm_head_proj".
    """
    parts = set(name.split("."))
    return any(excl in parts for excl in exclude_layers)


# ---------------------------------------------------------------------------
# Core quantization op  (weight tensor → integers + scale)
# ---------------------------------------------------------------------------

def quantize_weight(
    weight: torch.Tensor,
    bits: int,
    per_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel (or per-tensor) scale and quantize a weight matrix.

    Applies the symmetric min-max formula:
        scale     = max(abs(weight)) / q_max        q_max = 2^(bits-1) - 1
        quantized = round(weight / scale).clamp(-q_max, q_max)

    Unlike fake_quantize_tensor(), this function returns actual integer values
    (dtype torch.int8) alongside the float32 scale tensor. There is no
    dequantization step here -- the QuantizedLinear module does that at
    inference time.

    INT4 values occupy only 4 bits of information (range [-7, 7]) but are
    stored as int8 (1 byte each) in Python/PyTorch. Packing two INT4 values
    into a single byte happens during GGUF export, not here.

    Args:
        weight:      FP32 weight tensor of shape (out_features, in_features).
        bits:        Target bit-width: 4 or 8.
        per_channel: True (default) → one scale per output channel (row).
                     False          → one scale for the entire tensor.

    Returns:
        weight_int: torch.int8 tensor, same shape as weight, values in
                    [-q_max, q_max].
        scale:      float32 tensor. Shape (out_features, 1) for per-channel,
                    scalar for per-tensor. Multiply by weight_int to recover
                    the dequantized approximation.

    Raises:
        ValueError: If bits is not 4 or 8.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")

    q_max = _Q_MAX[bits]
    w = weight.float()   # work in fp32 regardless of model dtype

    if per_channel:
        reduce_dims = tuple(range(1, w.dim()))
        scale = w.abs().amax(dim=reduce_dims, keepdim=True) / q_max
    else:
        scale = w.abs().amax() / q_max

    scale = scale.clamp(min=1e-8)   # guard zero-weight rows

    weight_int = (w / scale).round().clamp(-q_max, q_max).to(torch.int8)
    return weight_int, scale.float()


# ---------------------------------------------------------------------------
# QuantizedLinear module
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """nn.Linear replacement that stores real integer weights for PTQ inference.

    Unlike FakeQuantizeLinear (which keeps FP32 shadow weights for training),
    QuantizedLinear holds actual torch.int8 weights alongside float32 per-channel
    scales. The original FP32 weights are discarded after quantization, reducing
    memory footprint. Dequantization happens in forward() at inference time:

        w_fp32 = weight_int.float() * scale
        output = F.linear(x, w_fp32, bias)

    This matches what llama.cpp does: weights are stored as integers and
    dequantized to fp16 on the fly during matrix multiplication.

    The bias (if present) is kept as a full-precision nn.Parameter -- bias
    values are small in magnitude and do not benefit meaningfully from
    quantization.

    Attributes:
        weight_int:  Registered buffer, torch.int8, shape (out, in).
        scale:       Registered buffer, float32, shape (out, 1) per-channel.
        bias:        nn.Parameter (float32) or None.
        bits:        Bit-width used when quantizing (4 or 8).
        per_channel: Whether per-channel (True) or per-tensor (False) scaling.
    """

    def __init__(
        self,
        linear: nn.Linear,
        bits: int,
        per_channel: bool = True,
    ) -> None:
        super().__init__()
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")

        weight_int, scale = quantize_weight(linear.weight.data, bits, per_channel)

        # Register as buffers (not Parameters) -- these are not trainable.
        self.register_buffer("weight_int", weight_int)
        self.register_buffer("scale", scale)

        if linear.bias is not None:
            self.bias: Optional[nn.Parameter] = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize on the fly: int8 → fp32 → match input dtype for the matmul.
        w = self.weight_int.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, self.bias)

    def dequantized_weight(self) -> torch.Tensor:
        """Return the dequantized FP32 weight. Used by export.py and diagnostics."""
        return self.weight_int.float() * self.scale

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, "
            f"per_channel={self.per_channel}"
        )


# ---------------------------------------------------------------------------
# Quantization error diagnostics
# ---------------------------------------------------------------------------

@dataclass
class LayerQuantError:
    """Per-layer quantization error statistics.

    Attributes:
        name:      Dotted module path (e.g. "model.layers.3.mlp.gate_proj").
        mse:       Mean squared error between original and dequantized weight.
        snr_db:    Signal-to-noise ratio in decibels.
                   snr_db = 20 * log10(||weight|| / ||weight - dequant||)
                   Higher is better. >40 dB is generally considered good.
        max_error: Maximum absolute element-wise difference.
        bits:      Bit-width used.
    """
    name: str
    mse: float
    snr_db: float
    max_error: float
    bits: int


def quantization_error(
    name: str,
    original_weight: torch.Tensor,
    quantized_linear: QuantizedLinear,
) -> LayerQuantError:
    """Compute quantization error statistics for a single layer.

    Compares the original FP32 weight to the dequantized approximation stored
    in quantized_linear. Useful for identifying which layers suffer most from
    quantization (low SNR layers may benefit from keeping at higher precision).

    Args:
        name:             Module name for reporting.
        original_weight:  The FP32 weight before quantization.
        quantized_linear: The QuantizedLinear that replaced this layer.

    Returns:
        LayerQuantError with MSE, SNR (dB), and max absolute error.
    """
    w_orig = original_weight.float()
    w_dq = quantized_linear.dequantized_weight()

    diff = w_orig - w_dq
    mse = diff.pow(2).mean().item()
    max_err = diff.abs().amax().item()

    orig_norm = w_orig.norm().item()
    diff_norm = diff.norm().item()
    if diff_norm < 1e-12:
        snr_db = float("inf")
    else:
        snr_db = 20.0 * (torch.log10(torch.tensor(orig_norm / diff_norm))).item()

    return LayerQuantError(
        name=name, mse=mse, snr_db=snr_db, max_error=max_err, bits=quantized_linear.bits
    )


# ---------------------------------------------------------------------------
# Model-level quantization: apply_ptq
# ---------------------------------------------------------------------------

@dataclass
class PTQResult:
    """Summary of a completed PTQ run.

    Attributes:
        num_quantized:     Number of Linear layers converted to QuantizedLinear.
        num_excluded:      Number of Linear layers left at FP32 (excluded by name).
        bits:              Bit-width used.
        per_channel:       Whether per-channel scaling was used.
        layer_errors:      Per-layer quantization error statistics (populated by
                           run_calibration after apply_ptq).
        elapsed_seconds:   Wall-clock time for the quantization pass.
    """
    num_quantized: int
    num_excluded: int
    bits: int
    per_channel: bool
    layer_errors: list[LayerQuantError] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def mean_snr_db(self) -> float:
        """Mean signal-to-noise ratio across all quantized layers."""
        if not self.layer_errors:
            return float("nan")
        finite = [e.snr_db for e in self.layer_errors if e.snr_db != float("inf")]
        return sum(finite) / len(finite) if finite else float("inf")

    @property
    def worst_layer(self) -> Optional[LayerQuantError]:
        """Layer with the lowest SNR (most degraded by quantization)."""
        if not self.layer_errors:
            return None
        return min(self.layer_errors, key=lambda e: e.snr_db)

    def summary(self) -> str:
        lines = [
            f"PTQ INT{self.bits} summary:",
            f"  Quantized layers : {self.num_quantized}",
            f"  Excluded (FP32)  : {self.num_excluded}",
            f"  Granularity      : {'per-channel' if self.per_channel else 'per-tensor'}",
            f"  Elapsed          : {self.elapsed_seconds:.1f}s",
        ]
        if self.layer_errors:
            lines += [
                f"  Mean SNR         : {self.mean_snr_db:.1f} dB",
                f"  Worst layer      : {self.worst_layer.name} "
                f"({self.worst_layer.snr_db:.1f} dB)",
            ]
        return "\n".join(lines)


def apply_ptq(
    model: nn.Module,
    bits: int,
    exclude_layers: Sequence[str] = DEFAULT_EXCLUDE_LAYERS,
    per_channel: bool = True,
) -> PTQResult:
    """Walk the model and replace eligible nn.Linear layers with QuantizedLinear.

    Each eligible layer is replaced in-place. The original FP32 weight tensor is
    read, quantized to int8, and then released -- QuantizedLinear only stores the
    integer weights and scales. This reduces memory by approximately bits/32 for
    the weight parameters (e.g. 8× for INT4, 4× for INT8).

    The operation is performed on CPU to avoid OOM errors for large models; move
    the resulting model to the target device after this call.

    Args:
        model:          The pretrained FP32 model to quantize in-place.
        bits:           Target bit-width: 4 or 8.
        exclude_layers: Module name components to keep at FP32.
                        Defaults to ("lm_head", "embed_tokens").
        per_channel:    True (default) for per-output-channel scaling.

    Returns:
        PTQResult with counts and per-layer error statistics.

    Raises:
        ValueError: If bits is not 4 or 8.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")

    t0 = time.perf_counter()
    num_quantized = 0
    num_excluded = 0
    layer_errors: list[LayerQuantError] = []

    def _replace_recursive(module: nn.Module, prefix: str) -> None:
        nonlocal num_quantized, num_excluded
        for child_name, child in list(module.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear):
                if _name_is_excluded(full_name, exclude_layers):
                    num_excluded += 1
                    logger.debug("PTQ skip (excluded): %s", full_name)
                else:
                    # Capture original weight before replacement for error stats.
                    orig_weight = child.weight.data.clone()
                    ql = QuantizedLinear(child, bits, per_channel)
                    setattr(module, child_name, ql)

                    err = quantization_error(full_name, orig_weight, ql)
                    layer_errors.append(err)
                    num_quantized += 1
                    logger.debug(
                        "PTQ quantized: %-55s SNR=%6.1f dB  MSE=%.2e",
                        full_name, err.snr_db, err.mse,
                    )
            else:
                _replace_recursive(child, full_name)

    _replace_recursive(model, prefix="")
    elapsed = time.perf_counter() - t0

    result = PTQResult(
        num_quantized=num_quantized,
        num_excluded=num_excluded,
        bits=bits,
        per_channel=per_channel,
        layer_errors=layer_errors,
        elapsed_seconds=elapsed,
    )
    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# Calibration pass
# ---------------------------------------------------------------------------

@dataclass
class CalibrationStats:
    """Per-layer activation statistics collected during the calibration pass.

    These are collected via forward hooks and represent the distribution of
    activations seen on the calibration corpus. Used for diagnostics and to
    support future activation quantization.

    Attributes:
        name:        Module path.
        act_min:     Minimum activation value observed across all calibration inputs.
        act_max:     Maximum activation value observed.
        act_abs_max: Maximum absolute activation value (used for symmetric scale).
    """
    name: str
    act_min: float
    act_max: float
    act_abs_max: float


@torch.no_grad()
def run_calibration(
    model: nn.Module,
    calib_loader: DataLoader,
    device: torch.device,
    collect_activation_stats: bool = True,
) -> list[CalibrationStats]:
    """Run calibration inputs through the model and collect per-layer statistics.

    Calibration serves two purposes:
        1. Diagnostics — verify the quantized model runs without errors on real
           inputs and log per-layer activation ranges.
        2. Infrastructure — forward hooks capture output activation statistics
           for each QuantizedLinear, enabling future activation quantization
           without changing the public API.

    For weight-only PTQ (this project's primary use case), the calibration pass
    does not change any model parameters -- scales are computed directly from
    weight values in apply_ptq(). However, running calibration provides a
    per-layer activation SNR profile which can guide decisions about which
    layers to keep at higher precision.

    Args:
        model:                    The quantized model (output of apply_ptq()).
        calib_loader:             DataLoader over the calibration subset
                                  (build_calibration_loader() from data_loader.py).
        device:                   Device for inference.
        collect_activation_stats: If True, attach forward hooks to QuantizedLinear
                                  layers to record output activation ranges.

    Returns:
        List of CalibrationStats, one per QuantizedLinear layer, sorted by name.
        Empty list if collect_activation_stats=False.
    """
    model.eval()
    model.to(device)

    # --- Register forward hooks on each QuantizedLinear ---
    stats_accum: dict[str, dict] = {}   # name -> {"min": ..., "max": ...}
    hooks: list[torch.utils.hooks.RemovableHook] = []

    if collect_activation_stats:
        for name, module in model.named_modules():
            if isinstance(module, QuantizedLinear):
                stats_accum[name] = {"min": float("inf"), "max": float("-inf")}

                def _make_hook(layer_name: str):
                    def _hook(mod, inp, out: torch.Tensor):
                        v_min = out.detach().min().item()
                        v_max = out.detach().max().item()
                        stats_accum[layer_name]["min"] = min(
                            stats_accum[layer_name]["min"], v_min
                        )
                        stats_accum[layer_name]["max"] = max(
                            stats_accum[layer_name]["max"], v_max
                        )
                    return _hook

                hooks.append(module.register_forward_hook(_make_hook(name)))

    # --- Forward pass over calibration data ---
    num_batches = 0
    for batch in calib_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model(input_ids=input_ids, attention_mask=attention_mask)
        num_batches += 1

    # Remove all hooks
    for h in hooks:
        h.remove()

    logger.info("Calibration complete: %d batches processed", num_batches)

    if not collect_activation_stats:
        return []

    # --- Assemble CalibrationStats ---
    result: list[CalibrationStats] = []
    for name in sorted(stats_accum):
        s = stats_accum[name]
        if s["min"] == float("inf"):
            continue  # layer never produced output (shouldn't happen)
        abs_max = max(abs(s["min"]), abs(s["max"]))
        result.append(
            CalibrationStats(
                name=name,
                act_min=s["min"],
                act_max=s["max"],
                act_abs_max=abs_max,
            )
        )
        logger.debug(
            "Activation stats %-55s  min=%+.3f  max=%+.3f  abs_max=%.3f",
            name, s["min"], s["max"], abs_max,
        )

    return result


# ---------------------------------------------------------------------------
# Full PTQ pipeline
# ---------------------------------------------------------------------------

def run_ptq(
    config: ExperimentConfig,
    device: torch.device,
    calibrate: bool = True,
) -> tuple[nn.Module, PTQResult]:
    """Full PTQ pipeline: load pretrained model, quantize weights, run calibration.

    Executes the three-step PTQ process:
        1. Load SmolLM2-1.7B from HuggingFace Hub (or local cache).
        2. Quantize all eligible nn.Linear layers with apply_ptq().
        3. Optionally run the calibration dataset through the model to collect
           activation statistics and validate correct end-to-end inference.

    The returned model is ready for evaluation with src/utils/metrics.py or
    export with src/utils/export.py. It is on CPU after this call; move it
    to the target device with model.to(device) if needed for evaluation.

    Args:
        config:    Parsed PTQ ExperimentConfig. Must have .calibration section.
                   Uses config.target_bits, config.quantize_config, and
                   config.calibration for all settings.
        device:    Device for the calibration forward pass.
        calibrate: Run the calibration pass after quantizing (default True).
                   Set False to skip calibration when re-loading a checkpoint.

    Returns:
        (model, ptq_result) where model is the quantized nn.Module and
        ptq_result is a PTQResult with error statistics.

    Raises:
        ValueError: If config.calibration is None (not a PTQ config).
    """
    from transformers import AutoModelForCausalLM  # type: ignore[import]
    from src.utils.data_loader import build_calibration_loader

    if config.calibration is None:
        raise ValueError(
            f"run_ptq requires config.calibration — "
            f"method={config.method!r} config has no 'calibration' section."
        )

    qc = config.quantize_config
    per_channel = qc.granularity == "per_channel"

    # 1. Load pretrained model
    logger.info("Loading %s from cache %s ...", config.model.name, config.model.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=torch.float32,   # PTQ always starts from full FP32
    )
    model.eval()
    logger.info("Model loaded: %d parameters", sum(p.numel() for p in model.parameters()))

    # 2. Quantize weights (CPU; no device needed for weight-only quantization)
    logger.info(
        "Applying PTQ: INT%d, %s, exclude=%s",
        config.target_bits, qc.granularity, qc.exclude_layers,
    )
    ptq_result = apply_ptq(
        model,
        bits=config.target_bits,
        exclude_layers=qc.exclude_layers,
        per_channel=per_channel,
    )

    # 3. Calibration pass
    if calibrate:
        logger.info(
            "Running calibration: %d samples from %s/%s ...",
            config.calibration.num_samples,
            config.calibration.dataset,
            config.calibration.split,
        )
        calib_loader = build_calibration_loader(config, num_workers=0)
        calib_stats = run_calibration(model, calib_loader, device)

        if calib_stats:
            abs_maxes = [s.act_abs_max for s in calib_stats]
            logger.info(
                "Calibration activation stats: mean_abs_max=%.3f  "
                "min_abs_max=%.3f  max_abs_max=%.3f",
                sum(abs_maxes) / len(abs_maxes),
                min(abs_maxes),
                max(abs_maxes),
            )

    return model, ptq_result
