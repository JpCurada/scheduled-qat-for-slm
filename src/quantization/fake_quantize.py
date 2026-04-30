"""
Fake quantization ops for Quantization-Aware Training (QAT).

Implements symmetric per-channel fake quantization for INT4 and INT8 using the
formula from the project spec:

    scale      = max(abs(weight)) / (2^(bits-1) - 1)
    quantized  = round(weight / scale)          # integers in [-q_max, q_max]
    dequantized = quantized * scale             # FP32 values with reduced precision

"Fake" quantization means the weights stay in FP32 throughout training. Each
forward pass simulates the precision loss of real quantization by quantizing then
immediately dequantizing. The difference from the original weight is the
quantization noise the model learns to be robust to.

The round() operation has zero gradient everywhere (and is undefined at integers),
so gradients cannot flow through it directly. The Straight-Through Estimator (STE)
solves this by replacing the zero gradient with 1 during backprop -- treating
round() as the identity for gradient purposes. This lets the AdamW optimizer update
the underlying FP32 "shadow weights" as if rounding never happened.

Granularity: per-channel (one scale per output channel / row of the weight matrix).
This matches the configs' quantize_config.granularity: per_channel setting and
produces better quality than per-tensor quantization.

Excluded layers (kept at full FP32):
    lm_head      -- output projection; directly affects token probabilities
    embed_tokens -- input embeddings; critical for token representation

Public API:
    fake_quantize_tensor(x, bits, per_channel)  -- quantize a raw tensor
    FakeQuantizeLinear                           -- nn.Linear replacement module
    inject_fake_quantize(model, bits, ...)       -- walk model, inject FakeQuantizeLinear
    remove_fake_quantize(model)                  -- revert to plain nn.Linear
    set_fake_quantize_bits(model, bits)          -- update bits (scheduled QAT)
    set_fake_quantize_enabled(model, enabled)    -- pause/resume fake quant (warmup)
    count_fake_quantize_layers(model)            -- number of injected layers
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Supported precisions and their symmetric integer ranges.
# Symmetric means the range is [-q_max, q_max] (not [-q_max-1, q_max]).
# This avoids asymmetry that would complicate scale computation.
SUPPORTED_BITS = {4, 8}
_Q_MAX = {4: 7, 8: 127}   # 2^(bits-1) - 1

# Default layers to keep at FP32. Matched against the full dotted module path.
DEFAULT_EXCLUDE_LAYERS: tuple[str, ...] = ("lm_head", "embed_tokens")


# ---------------------------------------------------------------------------
# Straight-Through Estimator
# ---------------------------------------------------------------------------

class _STERound(torch.autograd.Function):
    """Round with Straight-Through Estimator gradient.

    Forward:  y = round(x)          -- standard rounding to nearest integer
    Backward: dy/dx = 1             -- gradient passes through unchanged

    The STE approximation is the standard solution for training through
    non-differentiable quantization steps. Without it, the gradient of
    round() is zero almost everywhere, and the optimizer cannot update weights.

    Reference: Bengio et al., "Estimating or Propagating Gradients Through
    Stochastic Neurons for Conditional Computation" (arXiv:1308.3432).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # Identity: pass gradients through unchanged.
        return grad_output


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Apply round() with STE backward pass."""
    return _STERound.apply(x)


# ---------------------------------------------------------------------------
# Core fake quantization op
# ---------------------------------------------------------------------------

def fake_quantize_tensor(
    x: torch.Tensor,
    bits: int,
    per_channel: bool = True,
) -> torch.Tensor:
    """Symmetric fake quantization of a weight tensor.

    Quantizes then immediately dequantizes x, simulating the precision loss
    of storing weights at `bits` bits without actually reducing precision.
    The result has the same dtype as the input but values snapped to the
    FP32 representation of the nearest integer multiple of `scale`.

    Formula (from project spec):
        scale       = max(abs(x)) / q_max          per-channel or per-tensor
        quantized   = round(x / scale)              via STE so gradients flow
        dequantized = clamp(quantized, -q_max, q_max) * scale

    The clamp is technically a no-op when scale is derived from max(abs(x))
    because x / scale <= q_max by construction. It is included as a safety
    net against floating-point rounding that could produce values like 7.0001.

    Args:
        x:           Weight tensor of any shape. The first dimension is treated
                     as the output-channel dimension for per-channel scaling.
        bits:        Target bit-width. Must be 4 or 8.
        per_channel: If True (default), compute one scale per output channel
                     (per row of the weight matrix). If False, use a single
                     scale for the entire tensor.

    Returns:
        Dequantized tensor of the same shape and dtype as x. Gradients flow
        through via the STE applied inside _ste_round().

    Raises:
        ValueError: If bits is not 4 or 8.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")

    q_max = _Q_MAX[bits]

    # Compute scale in FP32 so that very small magnitudes do not underflow
    # when the model weights are stored as FP16. The dequantized result is
    # cast back to the input dtype at the end so the surrounding F.linear
    # call still runs in the model's native dtype.
    orig_dtype = x.dtype
    x_fp32 = x.to(torch.float32) if orig_dtype != torch.float32 else x

    if per_channel:
        # Reduce over every dimension except the first (output-channel dimension).
        # keepdim=True so scale broadcasts correctly over in_features (and any
        # further dimensions for Conv weights, though SmolLM2 only has Linear).
        reduce_dims = tuple(range(1, x_fp32.dim()))
        scale = x_fp32.abs().amax(dim=reduce_dims, keepdim=True) / q_max
    else:
        scale = x_fp32.abs().amax() / q_max

    # Guard against zero-weight rows (e.g. freshly initialised layers where
    # a full row happens to be zero). Without the clamp, scale=0 causes NaN.
    scale = scale.clamp(min=1e-8)

    x_scaled = x_fp32 / scale
    x_rounded = _ste_round(x_scaled)                       # STE: gradient = 1
    x_clamped = x_rounded.clamp(-q_max, q_max)             # safety clamp
    out = x_clamped * scale
    return out.to(orig_dtype) if orig_dtype != torch.float32 else out


# ---------------------------------------------------------------------------
# FakeQuantizeLinear module
# ---------------------------------------------------------------------------

class FakeQuantizeLinear(nn.Module):
    """Drop-in replacement for nn.Linear that fake-quantizes weights in forward.

    The underlying FP32 weight tensor (the "shadow weight") is stored as a
    normal nn.Parameter and updated by the optimizer at every step. During the
    forward pass, the shadow weight is fake-quantized before being used in the
    matrix multiplication, so the activations see only the quantization-degraded
    values and the model learns to compensate.

    Setting enabled=False bypasses fake quantization entirely and behaves
    identically to a plain nn.Linear. This is used during the warmup period
    of scheduled QAT where precision has not yet been reduced.

    Args:
        linear:      The nn.Linear layer to wrap. Its weight and bias parameters
                     are moved into this module (not copied -- same storage).
        bits:        Bit-width for fake quantization (4 or 8).
        per_channel: Use per-channel scaling (True) or per-tensor (False).

    Attributes:
        weight:      FP32 shadow weights -- the trainable parameters.
        bias:        Bias from the original layer (None if the layer had none).
        bits:        Current bit-width. Can be updated by set_fake_quantize_bits().
        per_channel: Scaling granularity.
        enabled:     When False, forward uses raw FP32 weights (no fake quant).
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

        # Reuse the same parameter tensors -- no copy, no extra memory.
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = bits
        self.per_channel = per_channel
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            w = fake_quantize_tensor(self.weight, self.bits, self.per_channel)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, bits={self.bits}, "
            f"per_channel={self.per_channel}, enabled={self.enabled}"
        )


# ---------------------------------------------------------------------------
# Model-level injection / removal helpers
# ---------------------------------------------------------------------------

def _name_is_excluded(name: str, exclude_layers: Sequence[str]) -> bool:
    """Return True if the module's dotted path matches any excluded layer name.

    Matches on complete path components to avoid false positives.
    e.g. "lm_head" matches "lm_head" and "model.lm_head" but not "lm_head_aux".

    Args:
        name:           Full dotted module path (e.g. "model.layers.0.mlp.gate_proj").
        exclude_layers: Layer name components to exclude (e.g. ["lm_head", "embed_tokens"]).
    """
    parts = set(name.split("."))
    return any(excl in parts for excl in exclude_layers)


def _inject_recursive(
    module: nn.Module,
    bits: int,
    per_channel: bool,
    exclude_layers: Sequence[str],
    prefix: str,
    count: list[int],
) -> None:
    """Recursively replace nn.Linear children with FakeQuantizeLinear."""
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Linear) and not _name_is_excluded(full_name, exclude_layers):
            setattr(module, child_name, FakeQuantizeLinear(child, bits, per_channel))
            count[0] += 1
            logger.debug("Injected fake quant -> %s (INT%d)", full_name, bits)
        else:
            # Recurse into sub-modules; skip the excluded layer itself.
            if not _name_is_excluded(full_name, exclude_layers):
                _inject_recursive(child, bits, per_channel, exclude_layers, full_name, count)


def inject_fake_quantize(
    model: nn.Module,
    bits: int,
    exclude_layers: Sequence[str] = DEFAULT_EXCLUDE_LAYERS,
    per_channel: bool = True,
) -> int:
    """Walk the model and replace eligible nn.Linear layers with FakeQuantizeLinear.

    Each eligible layer is replaced in-place: the same weight Parameter tensor
    is reused inside FakeQuantizeLinear, so no additional memory is allocated
    and the optimizer's parameter references remain valid after injection.

    Layers named in exclude_layers (matched on path components) are left as-is
    at full FP32 precision. The defaults are lm_head and embed_tokens per the
    project spec. Note: embed_tokens is nn.Embedding, not nn.Linear, so it
    would not be replaced regardless -- the exclusion primarily guards lm_head.

    Args:
        model:          The model to modify in-place (e.g. SmolLM2-1.7B).
        bits:           Bit-width for all injected fake quant layers (4 or 8).
        exclude_layers: Module name components to keep at FP32.
        per_channel:    Use per-channel scaling (True, default) or per-tensor.

    Returns:
        Number of nn.Linear layers that were replaced.

    Raises:
        ValueError: If bits is not 4 or 8.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")

    count = [0]
    _inject_recursive(model, bits, per_channel, exclude_layers, prefix="", count=count)
    logger.info(
        "Injected fake quantization into %d Linear layers (INT%d, %s)",
        count[0], bits, "per-channel" if per_channel else "per-tensor",
    )
    return count[0]


def remove_fake_quantize(model: nn.Module) -> int:
    """Revert all FakeQuantizeLinear layers back to plain nn.Linear.

    Useful when the training loop needs to evaluate the raw FP32 model, or
    before exporting to GGUF (which applies its own real quantization).

    The original FP32 shadow weights are preserved -- they are simply rewrapped
    in a standard nn.Linear with no behaviour change.

    Args:
        model: The model to modify in-place.

    Returns:
        Number of FakeQuantizeLinear layers that were reverted.
    """
    count = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, FakeQuantizeLinear):
                plain = nn.Linear(
                    child.in_features, child.out_features, bias=child.bias is not None
                )
                plain.weight = child.weight
                plain.bias = child.bias
                setattr(module, child_name, plain)
                count += 1
    logger.info("Removed fake quantization from %d layers (restored to nn.Linear)", count)
    return count


# ---------------------------------------------------------------------------
# Per-attribute setters (used by scheduled QAT and training loops)
# ---------------------------------------------------------------------------

def set_fake_quantize_bits(model: nn.Module, bits: int) -> int:
    """Update the bit-width on every FakeQuantizeLinear layer in the model.

    Called by the scheduled QAT trainer at each precision transition to reduce
    the simulated bit-width without re-injecting or rebuilding layers.

    Args:
        model: Model containing FakeQuantizeLinear layers.
        bits:  New bit-width (4 or 8).

    Returns:
        Number of layers updated.

    Raises:
        ValueError: If bits is not 4 or 8.
    """
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")

    count = 0
    for module in model.modules():
        if isinstance(module, FakeQuantizeLinear):
            module.bits = bits
            count += 1

    logger.info("Updated %d FakeQuantizeLinear layers to INT%d", count, bits)
    return count


def set_fake_quantize_enabled(model: nn.Module, enabled: bool) -> int:
    """Enable or disable fake quantization on all FakeQuantizeLinear layers.

    When enabled=False every FakeQuantizeLinear forwards raw FP32 weights,
    making the model equivalent to the original pretrained model. This is used
    during the warmup period of scheduled QAT (first warmup_epochs at full
    precision) before precision reduction begins.

    Args:
        model:   Model containing FakeQuantizeLinear layers.
        enabled: True to activate fake quant; False to bypass it.

    Returns:
        Number of layers updated.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, FakeQuantizeLinear):
            module.enabled = enabled
            count += 1

    state = "enabled" if enabled else "disabled"
    logger.info("Fake quantization %s on %d layers", state, count)
    return count


# ---------------------------------------------------------------------------
# Introspection utilities
# ---------------------------------------------------------------------------

def count_fake_quantize_layers(model: nn.Module) -> int:
    """Return the number of FakeQuantizeLinear layers currently in the model."""
    return sum(1 for m in model.modules() if isinstance(m, FakeQuantizeLinear))


def get_fake_quantize_config(model: nn.Module) -> dict:
    """Return a summary of the current fake quantization configuration.

    Inspects the first FakeQuantizeLinear found to report the shared settings,
    then counts total layers. Useful for logging at training start.

    Returns:
        Dict with keys: enabled, bits, per_channel, num_layers.
        Returns {"num_layers": 0} if no FakeQuantizeLinear layers exist.
    """
    for m in model.modules():
        if isinstance(m, FakeQuantizeLinear):
            return {
                "enabled": m.enabled,
                "bits": m.bits,
                "per_channel": m.per_channel,
                "num_layers": count_fake_quantize_layers(model),
            }
    return {"num_layers": 0}
