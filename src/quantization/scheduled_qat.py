"""
Scheduled Quantization-Aware Training (Scheduled QAT) -- core thesis contribution.

Scheduled QAT trains SmolLM2-1.7B with fake quantization noise that increases
gradually over time, giving the model time to adapt at each precision level before
the noise intensifies. This module composes two existing pieces:

    scheduler.py       -- computes the current bit-width as a function of epoch
    fake_quantize.py   -- injects/updates FakeQuantizeLinear layers in the model

and adds the ScheduledQATController that the training loop calls at each step to
keep the model's fake quantization configuration in sync with the schedule.

Standard QAT vs Scheduled QAT
------------------------------

Standard QAT:   bits = target_bits   (constant, active from fake_quant_start_epoch)
Scheduled QAT:  bits = schedule(epoch) (increasing noise from start_bits → target_bits)

The scheduler defines three phases:
    Warmup       -- fake quant DISABLED, model trains at full FP32 precision.
    Active zone  -- fake quant ENABLED, bit-width decreases per the schedule curve.
    Stabilization-- fake quant ENABLED at target_bits, model consolidates.

Controller responsibilities
----------------------------

At every training step the controller:
    1. Converts the current step to a fractional epoch.
    2. Queries scheduler.get_state(epoch) for the new ScheduleState.
    3. Compares to the previous state; if the phase or fake_quant_bits changed:
       a. If entering warmup → disable: set_fake_quantize_enabled(False)
       b. If leaving warmup → enable: set_fake_quantize_enabled(True), set bits
       c. If bits changed   → update: set_fake_quantize_bits(new_bits)
    4. Returns a ScheduledQATEvent if a change occurred (so the trainer can log).

Calling on_step() every step (not just every epoch) is important for continuous
schedules (linear, cosine) where the snapped bit-width can change mid-epoch.
For the step schedule, transitions always fire at exact epoch boundaries, so
calling on_epoch_start() would be sufficient -- but on_step() is correct for all
three strategies.

Training the whole model
------------------------

Like Standard QAT, Scheduled QAT trains all FP32 shadow weights. There is no
layer freezing. The fake quantization noise is the only mechanism that teaches
the model to be robust to reduced precision.

Public API
----------

    ScheduledQATEvent               -- describes a state change that just occurred
    ScheduledQATController          -- per-step lifecycle manager
    build_scheduled_qat_model(...)  -- load + inject fake quant + build controller
    ScheduledQATResult              -- setup summary for logging
    save_checkpoint(...)            -- save model state dict + controller state
    load_checkpoint(...)            -- restore model + controller from checkpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from src.quantization.fake_quantize import (
    DEFAULT_EXCLUDE_LAYERS,
    count_fake_quantize_layers,
    get_fake_quantize_config,
    inject_fake_quantize,
    set_fake_quantize_bits,
    set_fake_quantize_enabled,
)
from src.quantization.scheduler import (
    PrecisionScheduler,
    ScheduleState,
    build_scheduler,
    snap_bits,
)
from src.utils.config_loader import ExperimentConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event dataclass -- describes what just changed
# ---------------------------------------------------------------------------

@dataclass
class ScheduledQATEvent:
    """Describes a precision schedule state change for the training log.

    Returned by ScheduledQATController.on_step() when something changed.
    The trainer logs this and can use it to trigger eval, checkpoint saves, etc.

    Attributes:
        epoch:           Fractional epoch at which the change occurred.
        prev_state:      ScheduleState before the change.
        curr_state:      ScheduleState after the change.
        enabled_changed: True if fake quant was just enabled or disabled.
        bits_changed:    True if the active bit-width (INT8 ↔ INT4) changed.
    """
    epoch: float
    prev_state: ScheduleState
    curr_state: ScheduleState
    enabled_changed: bool
    bits_changed: bool

    def describe(self) -> str:
        """One-line description for training logs."""
        parts: list[str] = [f"[epoch {self.epoch:.3f}] Precision transition:"]
        if self.enabled_changed:
            was = "disabled" if self.prev_state.fake_quant_bits is None else "enabled"
            now = "disabled" if self.curr_state.fake_quant_bits is None else "enabled"
            parts.append(f"fake_quant {was} → {now}")
        if self.bits_changed:
            prev_b = self.prev_state.fake_quant_bits
            curr_b = self.curr_state.fake_quant_bits
            parts.append(
                f"bits INT{prev_b} → INT{curr_b}"
                if prev_b and curr_b
                else f"bits changed ({prev_b} → {curr_b})"
            )
        parts.append(f"[{self.prev_state.phase} → {self.curr_state.phase}]")
        parts.append(
            f"continuous: {self.prev_state.continuous_bits:.2f} → "
            f"{self.curr_state.continuous_bits:.2f}"
        )
        return "  ".join(parts)


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ScheduledQATController:
    """Per-step lifecycle manager for Scheduled QAT training.

    Keeps the model's FakeQuantizeLinear layers in sync with the precision
    schedule. Must be called at every training step (or at minimum at every
    epoch boundary for the step schedule).

    Args:
        model:         The model with FakeQuantizeLinear layers injected.
        scheduler:     A PrecisionScheduler (Linear, Cosine, or Step variant).
        total_steps:   Total number of training steps (for epoch conversion).
        total_epochs:  Total training epochs (for epoch conversion).

    Example usage in trainer.py::

        controller = ScheduledQATController(model, scheduler, total_steps, total_epochs)
        for step, batch in enumerate(train_loader):
            event = controller.on_step(step)
            if event:
                logger.info(event.describe())
            # ... forward, backward, optimizer.step() ...
    """

    def __init__(
        self,
        model: nn.Module,
        scheduler: PrecisionScheduler,
        total_steps: int,
        total_epochs: float,
    ) -> None:
        self._model = model
        self._scheduler = scheduler
        self._total_steps = total_steps
        self._total_epochs = total_epochs
        self._steps_per_epoch = total_steps / total_epochs

        # Initialise state at epoch 0 and apply it immediately.
        initial_state = scheduler.get_state(0.0)
        self._prev_state: ScheduleState = initial_state

        # Apply initial fake-quant configuration.
        if initial_state.fake_quant_bits is None:
            set_fake_quantize_enabled(model, False)
            logger.info(
                "Scheduled QAT: starting in WARMUP phase (fake quant disabled) | "
                "warmup until epoch %.2f",
                scheduler.config.warmup_epochs,
            )
        else:
            set_fake_quantize_enabled(model, True)
            set_fake_quantize_bits(model, initial_state.fake_quant_bits)
            logger.info(
                "Scheduled QAT: starting with fake quant ENABLED (INT%d)",
                initial_state.fake_quant_bits,
            )

    # ------------------------------------------------------------------
    # Step lifecycle
    # ------------------------------------------------------------------

    def on_step(self, step: int) -> Optional[ScheduledQATEvent]:
        """Update fake quantization configuration for this training step.

        Converts the step index to a fractional epoch, queries the scheduler,
        and applies any configuration change to the model's FakeQuantizeLinear
        layers. Call this before the forward pass on every step.

        Args:
            step: Zero-based global training step index.

        Returns:
            ScheduledQATEvent if the fake-quant configuration changed (the
            trainer should log this); None if nothing changed.
        """
        epoch = step / self._steps_per_epoch
        curr_state = self._scheduler.get_state(epoch)

        event = self._apply_state_change(epoch, self._prev_state, curr_state)
        self._prev_state = curr_state
        return event

    def on_epoch_start(self, epoch: float) -> Optional[ScheduledQATEvent]:
        """Convenience wrapper for epoch-level trainers.

        Identical to on_step() but accepts a fractional epoch directly instead
        of a step index. Use this when the trainer loop counts epochs rather
        than steps, or when only epoch-boundary updates are needed (safe for
        the step schedule; may miss mid-epoch transitions for cosine/linear).

        Args:
            epoch: Fractional training epoch.

        Returns:
            ScheduledQATEvent if the configuration changed; None otherwise.
        """
        curr_state = self._scheduler.get_state(epoch)
        event = self._apply_state_change(epoch, self._prev_state, curr_state)
        self._prev_state = curr_state
        return event

    def _apply_state_change(
        self,
        epoch: float,
        prev: ScheduleState,
        curr: ScheduleState,
    ) -> Optional[ScheduledQATEvent]:
        """Compare prev and curr states; update the model if anything changed.

        Handles three distinct transitions:
            FP32 → FP32   : no change (still in warmup or both disabled)
            FP32 → INT*   : enable fake quant + set bit-width
            INT* → INT*   : update bit-width only (e.g. INT8 → INT4)
            INT* → FP32   : disable fake quant (only during warmup, not expected
                            in normal runs since precision only decreases)
        """
        prev_bits = prev.fake_quant_bits
        curr_bits = curr.fake_quant_bits

        enabled_changed = (prev_bits is None) != (curr_bits is None)
        bits_changed = (
            prev_bits is not None
            and curr_bits is not None
            and prev_bits != curr_bits
        )

        if not enabled_changed and not bits_changed:
            return None

        # Apply the state change to the model
        if enabled_changed:
            if curr_bits is None:
                # Entering / remaining in full precision (warmup or unexpected)
                set_fake_quantize_enabled(self._model, False)
            else:
                # Leaving warmup: enable fake quant at the new bit-width
                set_fake_quantize_enabled(self._model, True)
                set_fake_quantize_bits(self._model, curr_bits)

        elif bits_changed:
            # Phase unchanged (both active), but snapped bit-width crossed a threshold
            set_fake_quantize_bits(self._model, curr_bits)

        event = ScheduledQATEvent(
            epoch=epoch,
            prev_state=prev,
            curr_state=curr,
            enabled_changed=enabled_changed,
            bits_changed=bits_changed,
        )
        logger.info(event.describe())
        return event

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> ScheduleState:
        """The most recently computed ScheduleState."""
        return self._prev_state

    @property
    def scheduler(self) -> PrecisionScheduler:
        """The underlying PrecisionScheduler (for schedule trace plotting)."""
        return self._scheduler

    def describe(self) -> str:
        """One-line status for training logs."""
        s = self._prev_state
        fq = f"INT{s.fake_quant_bits}" if s.fake_quant_bits else "FP32"
        return (
            f"ScheduledQAT [{s.phase}] "
            f"epoch={s.epoch:.3f}  "
            f"bits={s.continuous_bits:.2f}  "
            f"fake_quant={fq}"
        )

    def step_to_epoch(self, step: int) -> float:
        """Convert a step index to a fractional epoch."""
        return step / self._steps_per_epoch

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Serialisable controller state for training checkpoints.

        Include this in every checkpoint so training can resume at the exact
        same schedule position if interrupted.
        """
        return {
            "prev_state_epoch": self._prev_state.epoch,
            "prev_state_continuous_bits": self._prev_state.continuous_bits,
            "prev_state_fake_quant_bits": self._prev_state.fake_quant_bits,
            "prev_state_phase": self._prev_state.phase,
            "total_steps": self._total_steps,
            "total_epochs": self._total_epochs,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore controller state from a checkpoint.

        Re-applies the fake-quant configuration that was active when the
        checkpoint was saved, so the model resumes in the correct state.
        """
        restored = ScheduleState(
            epoch=state["prev_state_epoch"],
            continuous_bits=state["prev_state_continuous_bits"],
            fake_quant_bits=state["prev_state_fake_quant_bits"],
            phase=state["prev_state_phase"],
        )
        self._prev_state = restored

        # Re-sync the model to the restored state
        if restored.fake_quant_bits is None:
            set_fake_quantize_enabled(self._model, False)
        else:
            set_fake_quantize_enabled(self._model, True)
            set_fake_quantize_bits(self._model, restored.fake_quant_bits)

        logger.info(
            "ScheduledQATController restored: %s", self.describe()
        )


# ---------------------------------------------------------------------------
# Setup result
# ---------------------------------------------------------------------------

@dataclass
class ScheduledQATResult:
    """Summary of a completed Scheduled QAT model setup.

    Attributes:
        schedule_type:   "linear", "cosine", or "step".
        target_bits:     Final target bit-width (4 or 8).
        start_bits:      Starting bit-width (typically 32).
        warmup_epochs:   Epochs at full precision before schedule begins.
        stab_epochs:     Epochs at target_bits before training ends.
        per_channel:     Whether per-channel scaling is used.
        num_injected:    nn.Linear layers replaced with FakeQuantizeLinear.
        num_excluded:    Layers left at FP32 (lm_head, embed_tokens).
        total_params:    All trainable FP32 shadow weight parameters.
        schedule_summary: Human-readable schedule description from scheduler.
    """
    schedule_type: str
    target_bits: int
    start_bits: int
    warmup_epochs: float
    stab_epochs: float
    per_channel: bool
    num_injected: int
    num_excluded: int
    total_params: int
    schedule_summary: str

    def summary(self) -> str:
        return (
            f"Scheduled QAT ({self.schedule_type}) setup: "
            f"INT{self.start_bits} → INT{self.target_bits} | "
            f"warmup={self.warmup_epochs}ep  stab={self.stab_epochs}ep | "
            f"injected={self.num_injected} layers | "
            f"excluded={self.num_excluded} | "
            f"trainable params={self.total_params:,}\n"
            f"  Schedule: {self.schedule_summary}"
        )


# ---------------------------------------------------------------------------
# Setup pipeline
# ---------------------------------------------------------------------------

def build_scheduled_qat_model(
    config: ExperimentConfig,
    device: torch.device,
    total_steps: int,
) -> tuple[nn.Module, ScheduledQATController, ScheduledQATResult]:
    """Full Scheduled QAT setup: load model, inject fake quant, build controller.

    Executes three steps:
        1. Load SmolLM2-1.7B from HuggingFace Hub (FP32 weights).
        2. Inject FakeQuantizeLinear layers at start_bits (may be disabled
           during warmup -- the controller manages the enabled state).
        3. Build the PrecisionScheduler and ScheduledQATController.

    After this call, the model is ready for training. The trainer must call
    controller.on_step(step) before every forward pass.

    Args:
        config:      ExperimentConfig with method="scheduled_qat".
                     Must have .schedule, .training, and .quantize_config.
        device:      Training device.
        total_steps: Total number of optimizer steps for the full training run.
                     Used by the controller to convert step → fractional epoch.
                     Typically: ceil(train_dataset_size / batch_size) * epochs
                     / gradient_accumulation_steps.

    Returns:
        (model, controller, result) where:
            model      -- nn.Module with FakeQuantizeLinear layers injected.
            controller -- ScheduledQATController; call .on_step(step) every step.
            result     -- ScheduledQATResult for logging at training start.

    Raises:
        ValueError: If config.schedule or config.training is None.
    """
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    if config.schedule is None:
        raise ValueError(
            f"build_scheduled_qat_model requires config.schedule — "
            f"method={config.method!r} has no 'schedule' section."
        )
    if config.training is None:
        raise ValueError(
            f"build_scheduled_qat_model requires config.training — "
            f"method={config.method!r} has no 'training' section."
        )

    sched_cfg = config.schedule
    qc = config.quantize_config
    per_channel = qc.granularity == "per_channel"
    exclude_layers = list(qc.exclude_layers) or list(DEFAULT_EXCLUDE_LAYERS)
    total_epochs = float(config.training.epochs)

    # 1. Load pretrained FP32 model
    logger.info("Loading %s ...", config.model.name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=torch.float32,
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %d parameters", total_params)

    # 2. Inject fake quantization at start_bits.
    #    The controller will disable it during warmup and enable it at the right epoch.
    #    We inject at target_bits for simplicity; the controller adjusts bits dynamically.
    #    During warmup, enabled=False so the actual bit-width value doesn't matter.
    num_injected = inject_fake_quantize(
        model,
        bits=config.target_bits,
        exclude_layers=exclude_layers,
        per_channel=per_channel,
    )
    total_linears = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    num_excluded = total_linears - num_injected

    # Disable fake quant immediately; the controller re-enables at the right epoch.
    set_fake_quantize_enabled(model, False)

    # 3. Build scheduler and controller
    scheduler = build_scheduler(sched_cfg, total_epochs)

    model.to(device)
    model.train()

    controller = ScheduledQATController(
        model=model,
        scheduler=scheduler,
        total_steps=total_steps,
        total_epochs=total_epochs,
    )

    result = ScheduledQATResult(
        schedule_type=sched_cfg.type,
        target_bits=config.target_bits,
        start_bits=sched_cfg.start_bits,
        warmup_epochs=sched_cfg.warmup_epochs,
        stab_epochs=sched_cfg.stabilization_epochs,
        per_channel=per_channel,
        num_injected=num_injected,
        num_excluded=num_excluded,
        total_params=total_params,
        schedule_summary=scheduler.summary(),
    )
    logger.info(result.summary())
    return model, controller, result


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    output_path: Union[str, Path],
    controller: Optional[ScheduledQATController] = None,
    extra_meta: Optional[dict[str, Any]] = None,
) -> None:
    """Save model weights and controller state for resuming Scheduled QAT.

    The checkpoint contains:
        "model_state_dict"    -- FP32 shadow weights of all FakeQuantizeLinear layers
        "sqat_controller"     -- controller.state_dict() (schedule position, phase)
        "meta"                -- caller-supplied metadata (step, epoch, loss, etc.)

    Args:
        model:       The QAT model.
        output_path: Destination .pt file path.
        controller:  ScheduledQATController; its state is embedded so the
                     schedule position is preserved across interruptions.
        extra_meta:  Dict of any additional values to embed in the checkpoint.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "sqat_controller": controller.state_dict() if controller is not None else None,
        "meta": extra_meta or {},
    }
    torch.save(checkpoint, output_path)

    size_mb = output_path.stat().st_size / 1e6
    logger.info("Scheduled QAT checkpoint saved -> %s (%.1f MB)", output_path, size_mb)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    device: torch.device,
    controller: Optional[ScheduledQATController] = None,
) -> dict[str, Any]:
    """Restore a Scheduled QAT checkpoint into an existing model and controller.

    Args:
        path:       Path to the .pt checkpoint file.
        model:      Model instance to load into (build with build_scheduled_qat_model
                    first to ensure matching architecture and injected fake quant).
        device:     Device to map tensors to.
        controller: If provided and the checkpoint has sqat_controller state,
                    the controller is restored to the saved schedule position.

    Returns:
        The "meta" dict from the checkpoint.
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Scheduled QAT model weights loaded from %s", path)

    if controller is not None and checkpoint.get("sqat_controller") is not None:
        controller.load_state_dict(checkpoint["sqat_controller"])

    meta = checkpoint.get("meta", {})
    if meta:
        logger.info("Checkpoint meta: %s", meta)
    return meta
