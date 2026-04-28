"""
Standard Quantization-Aware Training (Standard QAT) for SmolLM2-1.7B.

Standard QAT is the simplest QAT strategy and serves as the baseline against
which Scheduled QAT is compared. It applies fake quantization noise at full
target precision from the very first training step (fake_quant_start_epoch=0)
and keeps that precision fixed for the entire training run.

How it differs from Scheduled QAT
-----------------------------------
Standard QAT: inject fake quant at target_bits from epoch fake_quant_start_epoch,
              never change the bit-width.
Scheduled QAT: start at full precision, gradually reduce bit-width over training
               according to a linear/cosine/step schedule.

The hypothesis this project tests is that the scheduled variant produces better
quantized models because the model has time to adapt at each precision level
before the noise increases further. Standard QAT is the ablation baseline: the
same total compute, the same target precision, but without the gradual reduction.

Training dynamics
-----------------
In each forward pass, FakeQuantizeLinear quantizes its weight tensor to the
target bit-width using the STE-backed fake_quantize_tensor() op, then runs the
normal matrix multiply. The optimizer sees the loss gradient and updates the
underlying FP32 "shadow weights" via the STE (which treats round() as identity).
Over many steps the shadow weights shift to values that are more robust to the
quantization grid at target_bits, reducing the loss relative to a raw PTQ model.

fake_quant_start_epoch
-----------------------
Normally 0 (fake quant active from step one). Can be set to a positive value to
give the model a brief warm-up at full FP32 precision before quantization noise
is introduced. This is rarely used in standard QAT (it is more relevant to
scheduled QAT's warmup concept), but the config field is honoured here.

Pipeline
--------
    1. Load       -- AutoModelForCausalLM from HuggingFace Hub (FP32 weights).
    2. Inject     -- inject_fake_quantize() replaces eligible nn.Linear layers
                     with FakeQuantizeLinear (from fake_quantize.py).
    3. Control    -- StandardQATController tracks whether fake quant should be
                     active at the current epoch and calls set_fake_quantize_enabled()
                     when the start epoch is crossed.
    4. Train      -- trainer.py runs the standard AdamW loop; all parameters
                     (FP32 shadow weights) are trainable throughout.
    5. Checkpoint -- save/load full model state dict via torch.save/load.

Public API
----------
    StandardQATResult                   -- setup summary for logging
    StandardQATController               -- stateful per-epoch lifecycle manager
    build_standard_qat_model(config, device)
                                        -- full setup: load + inject + return controller
    save_checkpoint(model, path, meta)  -- save model state dict + metadata
    load_checkpoint(path, device)       -- restore model + metadata from checkpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from src.quantization.fake_quantize import (
    DEFAULT_EXCLUDE_LAYERS,
    count_fake_quantize_layers,
    get_fake_quantize_config,
    inject_fake_quantize,
    set_fake_quantize_enabled,
)
from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StandardQATResult:
    """Summary of a completed Standard QAT model setup.

    Returned by build_standard_qat_model() for logging at training start.

    Attributes:
        bits:                  Target fake-quantization bit-width (4 or 8).
        fake_quant_start_epoch: Epoch at which fake quant is first enabled.
                               0 means active from step 1.
        per_channel:           Whether per-channel scaling is used.
        num_injected:          Number of nn.Linear layers replaced with
                               FakeQuantizeLinear.
        num_excluded:          Layers left at FP32 (lm_head, embed_tokens).
        total_params:          All trainable FP32 shadow weight parameters.
    """
    bits: int
    fake_quant_start_epoch: int
    per_channel: bool
    num_injected: int
    num_excluded: int
    total_params: int

    def summary(self) -> str:
        start = (
            "from step 1"
            if self.fake_quant_start_epoch == 0
            else f"from epoch {self.fake_quant_start_epoch}"
        )
        return (
            f"Standard QAT setup: INT{self.bits} fake quant {start} | "
            f"injected={self.num_injected} layers | "
            f"excluded={self.num_excluded} | "
            f"trainable params={self.total_params:,} | "
            f"granularity={'per-channel' if self.per_channel else 'per-tensor'}"
        )


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class StandardQATController:
    """Stateful lifecycle manager for Standard QAT training.

    The trainer loop calls on_epoch_start(epoch) at the beginning of each
    epoch. The controller decides whether to enable or disable fake quant on
    the model based on fake_quant_start_epoch, and returns a boolean indicating
    whether the state just changed (so the trainer can log the event).

    In Standard QAT fake_quant_start_epoch is almost always 0, so fake quant
    is active from the very first call. The controller is still useful for the
    case where start_epoch > 0 (brief FP32 warm-up before quantization noise).

    Args:
        model:                 The model with FakeQuantizeLinear layers injected.
        bits:                  Target bit-width (informational; already set in layers).
        fake_quant_start_epoch: First epoch at which fake quant should be active.

    Example usage in trainer.py::

        controller = StandardQATController(model, bits=4, fake_quant_start_epoch=0)
        for epoch in range(total_epochs):
            changed = controller.on_epoch_start(epoch)
            if changed:
                logger.info("Fake quant state changed: %s", controller.describe())
            for batch in train_loader:
                ...
    """

    def __init__(
        self,
        model: nn.Module,
        bits: int,
        fake_quant_start_epoch: int,
    ) -> None:
        self._model = model
        self._bits = bits
        self._start_epoch = fake_quant_start_epoch
        self._active = False

        # If start_epoch is 0, activate immediately so the first forward pass
        # already uses fake quant even before on_epoch_start is called.
        if fake_quant_start_epoch == 0:
            set_fake_quantize_enabled(model, True)
            self._active = True
            logger.info("Standard QAT: fake quantization enabled from epoch 0 (INT%d)", bits)

    # ------------------------------------------------------------------
    # Epoch lifecycle
    # ------------------------------------------------------------------

    def on_epoch_start(self, epoch: float) -> bool:
        """Update fake quant enabled/disabled state for this epoch.

        Should be called at the start of every epoch (or at the start of every
        step if using fractional epoch tracking).

        Args:
            epoch: Current fractional epoch (e.g. 1.0 = start of epoch 2).

        Returns:
            True if the fake quant state changed this call (the trainer should
            log this event); False if there was no change.
        """
        should_be_active = epoch >= self._start_epoch

        if should_be_active and not self._active:
            set_fake_quantize_enabled(self._model, True)
            self._active = True
            logger.info(
                "Standard QAT: fake quantization ENABLED at epoch %.3f (INT%d)",
                epoch, self._bits,
            )
            return True

        if not should_be_active and self._active:
            # This branch only fires if start_epoch > 0 and epoch somehow goes
            # backwards (shouldn't happen in normal training, but guard it).
            set_fake_quantize_enabled(self._model, False)
            self._active = False
            logger.warning(
                "Standard QAT: fake quantization unexpectedly DISABLED at epoch %.3f",
                epoch,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if fake quantization is currently enabled on the model."""
        return self._active

    @property
    def bits(self) -> int:
        """Target bit-width (constant for standard QAT)."""
        return self._bits

    @property
    def start_epoch(self) -> int:
        """Epoch at which fake quant was / will be enabled."""
        return self._start_epoch

    def describe(self) -> str:
        """One-line description for training logs."""
        state = "ACTIVE" if self._active else "INACTIVE"
        return (
            f"StandardQAT fake_quant={state}  bits=INT{self._bits}  "
            f"start_epoch={self._start_epoch}"
        )

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialisable controller state for inclusion in training checkpoints.

        The trainer should include controller.state_dict() in the checkpoint
        saved by save_checkpoint(), and call load_state_dict() after restoring
        to resume from the correct fake-quant activation state.
        """
        return {
            "bits": self._bits,
            "start_epoch": self._start_epoch,
            "active": self._active,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore controller state from a checkpoint dict.

        Also synchronises the model's FakeQuantizeLinear layers to match the
        restored active/inactive state.
        """
        self._bits = state["bits"]
        self._start_epoch = state["start_epoch"]
        self._active = state["active"]
        set_fake_quantize_enabled(self._model, self._active)
        logger.info(
            "StandardQATController restored: active=%s bits=INT%d start_epoch=%d",
            self._active, self._bits, self._start_epoch,
        )


# ---------------------------------------------------------------------------
# Setup pipeline
# ---------------------------------------------------------------------------

def build_standard_qat_model(
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[nn.Module, StandardQATController, StandardQATResult]:
    """Full Standard QAT setup: load pretrained model, inject fake quant.

    Executes the two-step setup:
        1. Load SmolLM2-1.7B from HuggingFace Hub (FP32 weights).
        2. Replace eligible nn.Linear layers with FakeQuantizeLinear in-place.
           The replacement reuses the same weight Parameters so the optimizer
           references remain valid and no extra memory is allocated.

    The returned controller manages the fake-quant lifecycle. Pass it to the
    training loop; call controller.on_epoch_start(epoch) at the start of each
    epoch.

    All model parameters (FP32 shadow weights) are trainable. Unlike LoRA-QAT,
    Standard QAT trains the entire model -- the fake quantization noise teaches
    all weights to be robust to quantization simultaneously.

    Args:
        config: Parsed ExperimentConfig with method="standard_qat".
                Uses config.target_bits, config.quantize_config, and
                config.training for all settings.
        device: Target device for training.

    Returns:
        (model, controller, result) where:
            model      -- nn.Module with FakeQuantizeLinear layers injected.
            controller -- StandardQATController for per-epoch lifecycle calls.
            result     -- StandardQATResult for logging at training start.

    Raises:
        ValueError: If config.training is None (not a QAT config).
    """
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    if config.training is None:
        raise ValueError(
            f"build_standard_qat_model requires config.training — "
            f"method={config.method!r} has no 'training' section."
        )

    qc = config.quantize_config
    per_channel = qc.granularity == "per_channel"
    exclude_layers = list(qc.exclude_layers) or list(DEFAULT_EXCLUDE_LAYERS)

    # 1. Load pretrained FP32 model
    logger.info("Loading %s ...", config.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=torch.float32,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %d parameters", total_params)

    # 2. Inject fake quantization nodes
    num_injected = inject_fake_quantize(
        model,
        bits=config.target_bits,
        exclude_layers=exclude_layers,
        per_channel=per_channel,
    )

    # Derive excluded count from total eligible linears
    total_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    num_excluded = total_linears - num_injected

    # 3. Move to device
    model.to(device)
    model.train()

    # 4. Build controller (activates fake quant immediately if start_epoch == 0)
    controller = StandardQATController(
        model=model,
        bits=config.target_bits,
        fake_quant_start_epoch=qc.fake_quant_start_epoch,
    )

    result = StandardQATResult(
        bits=config.target_bits,
        fake_quant_start_epoch=qc.fake_quant_start_epoch,
        per_channel=per_channel,
        num_injected=num_injected,
        num_excluded=num_excluded,
        total_params=total_params,
    )
    logger.info(result.summary())
    return model, controller, result


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    output_path: Union[str, Path],
    controller: Optional[StandardQATController] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> None:
    """Save the full model state dict and optional training metadata.

    Saves a .pt file containing:
        "model_state_dict"   -- all FP32 shadow weights (the trainable tensors)
        "qat_controller"     -- controller.state_dict() if controller is provided
        "meta"               -- extra_meta dict (epoch, step, loss, etc.)

    Args:
        model:       The QAT model (with FakeQuantizeLinear layers).
        output_path: Destination .pt file (e.g. "models/checkpoints/standard_qat_int4.pt").
        controller:  If provided, its state is included so training can resume
                     from the correct fake-quant activation state.
        extra_meta:  Any additional metadata to embed (logged but not used by
                     load_checkpoint beyond passing it back to the caller).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "qat_controller": controller.state_dict() if controller is not None else None,
        "meta": extra_meta or {},
    }
    torch.save(checkpoint, output_path)

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Checkpoint saved -> %s (%.1f MB)", output_path, size_mb)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    device: torch.device,
    controller: Optional[StandardQATController] = None,
) -> dict[str, Any]:
    """Restore a model from a Standard QAT checkpoint.

    Loads the model state dict in-place. If a controller is provided and the
    checkpoint contains controller state, it is restored so training can resume
    at the correct fake-quant activation state.

    Args:
        path:       Path to the .pt checkpoint file.
        model:      The model instance to load into (must have identical
                    architecture -- use build_standard_qat_model first).
        device:     Device to map tensors to during loading.
        controller: If provided and the checkpoint has qat_controller state,
                    controller.load_state_dict() is called to restore it.

    Returns:
        The "meta" dict from the checkpoint (epoch, step, loss, etc.).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model weights loaded from %s", path)

    if controller is not None and checkpoint.get("qat_controller") is not None:
        controller.load_state_dict(checkpoint["qat_controller"])

    meta = checkpoint.get("meta", {})
    if meta:
        logger.info("Checkpoint meta: %s", meta)
    return meta


# ---------------------------------------------------------------------------
# Introspection utilities
# ---------------------------------------------------------------------------

def training_summary(
    model: nn.Module,
    controller: StandardQATController,
    epoch: float,
) -> str:
    """One-line training status for step-level logging.

    Args:
        model:      The QAT model.
        controller: The active StandardQATController.
        epoch:      Current fractional epoch.

    Returns:
        Formatted string describing current fake-quant state and layer counts.
    """
    cfg = get_fake_quantize_config(model)
    state = "ON" if cfg.get("enabled") else "OFF"
    return (
        f"[epoch {epoch:.3f}] "
        f"fake_quant={state}  INT{cfg.get('bits', '?')}  "
        f"layers={cfg.get('num_layers', '?')}  "
        f"{'per-channel' if cfg.get('per_channel') else 'per-tensor'}"
    )
