"""
Precision schedulers for Scheduled QAT -- the core thesis contribution.

Scheduled QAT gradually reduces the simulated bit-width during training rather
than applying full quantization noise from epoch 1 (as Standard QAT does). The
hypothesis is that the model benefits from time to adapt at each precision level
before the noise is increased further. This module implements three schedule
strategies for controlling that reduction.

Three schedule strategies
-------------------------

LINEAR  -- constant rate of precision loss over the active training period.
           Interpolates bit-width as a straight line from start_bits to
           target_bits. Simple and predictable; the model sees a steadily
           increasing quantization signal with no periods of faster or slower
           change.

           bits
            32 |\\
               | \\
               |  \\
             8 |   ------
             4 |          \\____
               +--+--------+--+--> epoch
                 warmup   stab

COSINE  -- slow start, fast middle, slow end. Within each segment (pair of
           consecutive transitions), the bit-width follows a cosine curve:

               bits_hi + ((1 - cos(pi*t)) / 2) * (bits_lo - bits_hi)

           This spends more epochs near the segment endpoints (gentle entry
           into each new precision level, gradual approach to the next) and
           moves quickly through the intermediate values. Borrowed directly
           from cosine learning-rate scheduling.

           bits
            32 |`-.
               |   `.
            16 |     `._
               |        `-.
             8 |            `--.__
             4 |                  `-.____
               +--+--+--+--+--+--+--+--> epoch
                 0.5 1.5  2.0  2.5  3.0

STEP    -- hard drops at defined transition epochs with flat plateaus between.
           The model trains at each bit-width until the next drop fires. All
           transition epochs and their target bit-widths are specified
           explicitly in the YAML config. This gives maximum control over when
           each precision level is introduced.

           bits
            32 |-----.
            16 |      `------.
             8 |              `---.
             4 |                   `------
               +--+------+---+--+--> epoch
                 0.5    1.5  2.0 2.5  3.0

Common structure across all three schedules
-------------------------------------------

Each schedule divides the total training run into three phases:

    [0, warmup_epochs)                    -- full precision (FP32), fake quant DISABLED
    [warmup_epochs, total - stab_epochs)  -- active zone, precision decreases per strategy
    [total - stab_epochs, total]          -- stabilization at target_bits, fake quant fixed

Fake quantization mapping
-------------------------

The schedules return a continuous float ("conceptual precision level"). The
trainer maps this to an actual fake-quantization bit-width via snap_bits():

    continuous_bits > 8   --> None  (disable fake quant, run at full FP32)
    6 < continuous_bits <= 8  --> 8  (INT8 fake quantization)
    continuous_bits <= 6      --> 4  (INT4 fake quantization)

The INT8/INT4 threshold is the midpoint (6.0) between the two supported
discrete levels. The full-precision threshold is 8.0 because at 16 bits the
quantization noise is negligible and training at FP32 is equivalent.

For the step schedule, transitions always land on exact integer bit-widths
(32, 16, 8, 4), so snap_bits() maps them cleanly: 32/16 -> None, 8 -> 8, 4 -> 4.

For linear/cosine, the continuous values cross these thresholds naturally,
creating implicit INT8 and INT4 phases within the active period even when
transitions are not explicitly listed (as in the linear configs).

Public API
----------

    snap_bits(continuous_bits)      -- map continuous float to fake-quant int or None
    ScheduleState                   -- snapshot returned by scheduler.get_state(epoch)
    LinearScheduler                 -- linear interpolation strategy
    CosineScheduler                 -- piecewise cosine interpolation strategy
    StepScheduler                   -- hard-drop plateau strategy
    build_scheduler(config, epochs) -- factory: returns the right scheduler for a config
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.utils.config_loader import ScheduleConfig


# ---------------------------------------------------------------------------
# Fake-quantization bit-width mapping
# ---------------------------------------------------------------------------

# Threshold between "full precision" (fake quant disabled) and INT8.
# At bits > _FP_THRESHOLD the quantization noise is negligible; treat as FP32.
_FP_THRESHOLD: float = 8.0

# Threshold between INT8 and INT4: midpoint between the two supported levels.
_INT4_THRESHOLD: float = 6.0


def snap_bits(continuous_bits: float) -> Optional[int]:
    """Map a continuous bit-width to the nearest supported fake-quant level.

    The precision schedule returns a continuous float representing the
    "conceptual" bit-width at a given training epoch. This function maps that
    float to the actual integer bit-width passed to set_fake_quantize_bits(),
    or None when the model should run at full precision with fake quant disabled.

    Mapping:
        continuous_bits > 8.0          --> None   (full precision, disable fake quant)
        6.0 < continuous_bits <= 8.0   --> 8      (INT8 fake quantization)
        continuous_bits <= 6.0         --> 4      (INT4 fake quantization)

    The INT8/INT4 boundary at 6.0 is the arithmetic midpoint between 4 and 8.
    The full-precision boundary at 8.0 is chosen because at 16 bits (the first
    transition in the step/cosine configs) quantization noise is negligible --
    running at FP32 with fake quant disabled is equivalent.

    For the step schedule, transitions always land on 32, 16, 8, or 4, so the
    mapping is exact: 32/16 -> None, 8 -> 8, 4 -> 4.

    Args:
        continuous_bits: The bit-width returned by a scheduler's get_bits().

    Returns:
        None if the model should run at full precision (fake quant disabled),
        8 for INT8 fake quantization, or 4 for INT4 fake quantization.
    """
    if continuous_bits > _FP_THRESHOLD:
        return None
    elif continuous_bits > _INT4_THRESHOLD:
        return 8
    else:
        return 4


# ---------------------------------------------------------------------------
# Schedule state snapshot
# ---------------------------------------------------------------------------

@dataclass
class ScheduleState:
    """Snapshot of the precision schedule at a specific training epoch.

    Returned by PrecisionScheduler.get_state(). The trainer compares
    consecutive states to detect transitions and call set_fake_quantize_bits()
    or set_fake_quantize_enabled() at the right moment.

    Attributes:
        epoch:            Training epoch at which this state was sampled
                          (fractional, e.g. 1.375 mid-epoch).
        continuous_bits:  Raw bit-width from the schedule curve. May be any
                          float; this is what's plotted in notebooks.
        fake_quant_bits:  Actual fake-quant level: None (disabled), 8, or 4.
                          Obtained by passing continuous_bits through snap_bits().
        phase:            "warmup" | "active" | "stabilization".
    """

    epoch: float
    continuous_bits: float
    fake_quant_bits: Optional[int]
    phase: str

    def describe(self) -> str:
        """One-line description for training logs."""
        fq = f"INT{self.fake_quant_bits}" if self.fake_quant_bits else "FP32"
        return (
            f"epoch={self.epoch:.3f}  bits={self.continuous_bits:.2f}  "
            f"fake_quant={fq}  phase={self.phase}"
        )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PrecisionScheduler:
    """Base class for all precision schedule strategies.

    Subclasses implement get_bits(epoch) which returns the continuous bit-width
    at any training epoch. All other methods are defined here in terms of that.

    The scheduler is stateless: it is a pure function of epoch. The trainer
    holds the scheduler and queries it at each step; no internal state is
    mutated by queries.

    Args:
        config:       Parsed ScheduleConfig from the experiment YAML.
        total_epochs: Total training epochs (from training.epochs in YAML).
    """

    def __init__(self, config: ScheduleConfig, total_epochs: float) -> None:
        self.config = config
        self.total_epochs = total_epochs

        self._active_start: float = config.warmup_epochs
        self._active_end: float = total_epochs - config.stabilization_epochs

        if self._active_start >= self._active_end:
            raise ValueError(
                f"Active training period has zero or negative duration. "
                f"warmup_epochs={config.warmup_epochs}, "
                f"total_epochs={total_epochs}, "
                f"stabilization_epochs={config.stabilization_epochs}. "
                f"Reduce warmup or stabilization, or increase total epochs."
            )

    @property
    def active_duration(self) -> float:
        """Length of the active zone in epochs."""
        return self._active_end - self._active_start

    def _phase(self, epoch: float) -> str:
        """Classify an epoch into warmup | active | stabilization."""
        if epoch < self._active_start:
            return "warmup"
        if epoch >= self._active_end:
            return "stabilization"
        return "active"

    def get_bits(self, epoch: float) -> float:
        """Return the continuous bit-width at the given fractional epoch.

        Must be overridden by subclasses.

        Args:
            epoch: Fractional training epoch (e.g. 1.5 = halfway through epoch 2).

        Returns:
            Continuous bit-width. At warmup boundary this equals start_bits;
            at stabilization boundary this equals target_bits.
        """
        raise NotImplementedError

    def get_state(self, epoch: float) -> ScheduleState:
        """Return a full ScheduleState snapshot at the given epoch.

        Convenience method for the trainer: captures continuous_bits, snapped
        fake_quant_bits, and the current phase in one call.

        Args:
            epoch: Fractional training epoch.

        Returns:
            ScheduleState ready for logging and comparison.
        """
        bits = self.get_bits(epoch)
        return ScheduleState(
            epoch=epoch,
            continuous_bits=bits,
            fake_quant_bits=snap_bits(bits),
            phase=self._phase(epoch),
        )

    def did_fake_quant_change(
        self, prev_epoch: float, curr_epoch: float
    ) -> bool:
        """Return True if the fake-quant level changed between two epochs.

        The trainer calls this after each step to decide whether to update the
        model's fake quantization configuration. Comparing snapped bit-widths
        (not continuous values) means the trainer is only notified when an
        actionable change occurs.

        Args:
            prev_epoch: The epoch at the previous step.
            curr_epoch: The epoch at the current step.

        Returns:
            True if snap_bits(get_bits(prev_epoch)) != snap_bits(get_bits(curr_epoch)).
        """
        return snap_bits(self.get_bits(prev_epoch)) != snap_bits(self.get_bits(curr_epoch))

    def get_schedule_trace(self, num_points: int = 500) -> list[tuple[float, float]]:
        """Return the full precision curve as (epoch, bits) pairs.

        Used by notebooks (04_scheduled_qat.ipynb) to visualise the schedule.
        Returns num_points evenly spaced across [0, total_epochs].

        Args:
            num_points: Resolution of the output curve (default 500).

        Returns:
            List of (epoch, continuous_bits) tuples.
        """
        return [
            (i * self.total_epochs / (num_points - 1), self.get_bits(i * self.total_epochs / (num_points - 1)))
            for i in range(num_points)
        ]

    def summary(self) -> str:
        """Human-readable schedule summary for logging at training start."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Linear schedule
# ---------------------------------------------------------------------------

class LinearScheduler(PrecisionScheduler):
    """Constant-rate precision reduction from start_bits to target_bits.

    The bit-width decreases as a straight line across the active period:

        t    = (epoch - warmup_epochs) / active_duration
        bits = start_bits + t * (target_bits - start_bits)

    Where t runs from 0 at the warmup boundary to 1 at the stabilization
    boundary. The rate of precision loss is constant -- the model sees a
    steadily increasing quantization signal with no faster or slower periods.

    The transitions field in the YAML config is ignored for linear schedules.

    Fake-quant timeline (example: start=32, target=4, active=[0.5, 2.5]):
        epoch < 0.5           -- full precision  (bits=32, fake quant disabled)
        0.5 <= epoch < 1.643  -- full precision  (bits interpolating 32->8, still > 8)
        1.643 <= epoch < 2.0  -- INT8 fake quant (bits interpolating 8->6)
        2.0 <= epoch < 2.5    -- INT4 fake quant (bits interpolating 6->4)
        2.5 <= epoch <= 3.0   -- stabilization   (bits=4, INT4)
    """

    def get_bits(self, epoch: float) -> float:
        """Linearly interpolated bit-width at this epoch."""
        phase = self._phase(epoch)
        if phase == "warmup":
            return float(self.config.start_bits)
        if phase == "stabilization":
            return float(self.config.target_bits)

        t = (epoch - self._active_start) / self.active_duration
        return self.config.start_bits + t * (self.config.target_bits - self.config.start_bits)

    def summary(self) -> str:
        return (
            f"LinearScheduler: INT{self.config.target_bits} target | "
            f"warmup={self.config.warmup_epochs}ep  "
            f"active=[{self._active_start:.2f}, {self._active_end:.2f}]  "
            f"stab={self.config.stabilization_epochs}ep | "
            f"rate={(self.config.start_bits - self.config.target_bits) / self.active_duration:.2f} bits/epoch"
        )


# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------

class CosineScheduler(PrecisionScheduler):
    """Piecewise cosine precision reduction: slow start, fast middle, slow end.

    Within each segment the bit-width follows a cosine curve:

        t             = (epoch - seg_start) / (seg_end - seg_start)
        cosine_factor = (1 - cos(pi * t)) / 2
        bits          = bits_hi + cosine_factor * (bits_lo - bits_hi)

    At t=0: cosine_factor=0, bits=bits_hi (start of segment, slow change).
    At t=0.5: cosine_factor=0.5, bits=midpoint (fastest change).
    At t=1: cosine_factor=1, bits=bits_lo (end of segment, slow change).

    This means the model spends longer near each precision boundary -- a gentle
    introduction to each new noise level and a gentle approach to the next.

    Segmentation
    ~~~~~~~~~~~~
    If transitions are provided (cosine INT4 and INT8 configs), the active
    period is divided into segments by the transition epochs. Within each
    segment the cosine curve runs independently between that segment's bit-
    width endpoints.

    Example (INT4, transitions=[{0.5,16},{1.5,8},{2.0,4}]):
        Segment 1: epoch [0.5, 1.5], bits cosine from 16 -> 8
        Segment 2: epoch [1.5, 2.0], bits cosine from  8 -> 4
        Past 2.0: bits fixed at 4 (before stabilization kicks in at 2.5)

    If no transitions are provided, a single cosine curve spans the entire
    active period from start_bits to target_bits.

    Fake-quant timeline (example: INT4, cosine segments as above):
        epoch < 0.5    -- full precision (bits=32, fake quant disabled)
        0.5-1.5        -- full precision (bits 16->8, still > 8)
        epoch = 1.5    -- INT8 kicks in (bits reaches 8 exactly)
        1.5-1.75       -- INT8 fake quant (bits cosine 8->6)
        epoch ~= 1.75  -- INT4 kicks in (bits crosses 6 at cosine midpoint)
        1.75-2.0       -- INT4 fake quant (bits cosine 6->4)
        2.0-2.5        -- INT4 stabilization (bits fixed at 4)
    """

    def __init__(self, config: ScheduleConfig, total_epochs: float) -> None:
        super().__init__(config, total_epochs)
        self._segments: list[tuple[float, float, float, float]] = self._build_segments()

    def _build_segments(self) -> list[tuple[float, float, float, float]]:
        """Build the list of (seg_start, seg_end, bits_hi, bits_lo) tuples.

        Each tuple defines one piecewise cosine segment. Zero-length segments
        (where seg_start == seg_end, produced when the first transition falls
        exactly on warmup_epochs) are filtered out since they cause division-
        by-zero and represent an instantaneous snap already captured by the
        transition's bit-width becoming the new segment start.
        """
        if not self.config.transitions:
            # Single segment: full active period from start_bits to target_bits.
            return [(
                self._active_start,
                self._active_end,
                float(self.config.start_bits),
                float(self.config.target_bits),
            )]

        # Sort transitions by epoch; they define the segment boundaries.
        sorted_tr = sorted(self.config.transitions, key=lambda t: t.epoch)

        # Build the point list that defines segment endpoints.
        # Prepend the active_start at start_bits so that any gap between
        # warmup_epochs and the first transition epoch is covered.
        # When the first transition IS at warmup_epochs, this creates a zero-
        # length segment that is filtered out below.
        points: list[tuple[float, float]] = [
            (self._active_start, float(self.config.start_bits))
        ]
        for tr in sorted_tr:
            points.append((tr.epoch, float(tr.bits)))

        segments: list[tuple[float, float, float, float]] = []
        for i in range(len(points) - 1):
            seg_start, bits_hi = points[i]
            seg_end, bits_lo = points[i + 1]
            if seg_end > seg_start:            # skip zero-length segments
                segments.append((seg_start, seg_end, bits_hi, bits_lo))

        return segments

    def get_bits(self, epoch: float) -> float:
        """Piecewise cosine bit-width at this epoch."""
        phase = self._phase(epoch)
        if phase == "warmup":
            return float(self.config.start_bits)
        if phase == "stabilization":
            return float(self.config.target_bits)

        # Walk segments in order and find the one that contains this epoch.
        # Because segments are sorted and non-overlapping, the first segment
        # whose seg_end >= epoch is the correct one.
        for seg_start, seg_end, bits_hi, bits_lo in self._segments:
            if epoch <= seg_end:
                seg_dur = seg_end - seg_start
                # Clamp t to [0, 1] as a safety net for floating-point edge cases.
                t = max(0.0, min(1.0, (epoch - seg_start) / seg_dur))
                cosine_factor = (1.0 - math.cos(math.pi * t)) / 2.0
                return bits_hi + cosine_factor * (bits_lo - bits_hi)

        # Epoch is past all segments but before stabilization (can happen when
        # the last transition epoch is earlier than active_end).
        return float(self.config.target_bits)

    def summary(self) -> str:
        seg_strs = [
            f"[{s:.2f}->{e:.2f}: {bh:.0f}->{bl:.0f} bits]"
            for s, e, bh, bl in self._segments
        ]
        return (
            f"CosineScheduler: INT{self.config.target_bits} target | "
            f"warmup={self.config.warmup_epochs}ep  "
            f"stab={self.config.stabilization_epochs}ep | "
            f"segments: {', '.join(seg_strs)}"
        )


# ---------------------------------------------------------------------------
# Step schedule
# ---------------------------------------------------------------------------

class StepScheduler(PrecisionScheduler):
    """Hard precision drops at defined transition epochs with flat plateaus.

    The bit-width stays constant between transitions. At each transition epoch,
    it drops immediately to the specified bit-width. The training log will show
    a clear moment of increased quantization noise at each drop.

    Transitions are defined explicitly in the YAML config and sorted by epoch.
    The config_loader validates that at least one transition is present.

    Plateau structure (example: INT4, transitions=[{0.5,16},{1.5,8},{2.0,4}]):
        epoch  0.0 - 0.5   : bits=32   (warmup, fake quant disabled)
        epoch  0.5 - 1.5   : bits=16   (plateau, still full precision)
        epoch  1.5 - 2.0   : bits=8    (plateau, INT8 fake quant)
        epoch  2.0 - 2.5   : bits=4    (plateau, INT4 fake quant)
        epoch  2.5 - 3.0   : bits=4    (stabilization, INT4)

    Relationship to cosine/linear:
        Step provides maximum control over when each precision level is
        introduced. Cosine and linear produce the same eventual target but
        with a smoother, more gradual reduction curve.
    """

    def __init__(self, config: ScheduleConfig, total_epochs: float) -> None:
        if not config.transitions:
            raise ValueError(
                "StepScheduler requires at least one transition in schedule.transitions. "
                "Add entries like: transitions: [{epoch: 1.0, bits: 8}]"
            )
        super().__init__(config, total_epochs)
        # Pre-sort once; get_bits() will binary-search or linear-scan this list.
        self._sorted_transitions = sorted(config.transitions, key=lambda t: t.epoch)

    def get_bits(self, epoch: float) -> float:
        """Stepped bit-width: last fired transition wins."""
        phase = self._phase(epoch)
        if phase == "warmup":
            return float(self.config.start_bits)
        if phase == "stabilization":
            return float(self.config.target_bits)

        # Walk transitions in ascending order; apply every one whose epoch has
        # passed. The final application gives the current plateau bit-width.
        current_bits = float(self.config.start_bits)
        for tr in self._sorted_transitions:
            if epoch >= tr.epoch:
                current_bits = float(tr.bits)
            else:
                break   # transitions are sorted, so no later one can apply
        return current_bits

    @property
    def transition_epochs(self) -> list[float]:
        """Epochs at which discrete precision drops occur (for logging/plotting)."""
        return [tr.epoch for tr in self._sorted_transitions]

    @property
    def plateau_widths(self) -> list[float]:
        """Duration of each plateau in epochs.

        Returns one value per transition, representing how long the model
        trains at that transition's bit-width before the next drop fires.
        The last entry covers from the final transition to active_end.
        """
        epochs = self.transition_epochs + [self._active_end]
        return [epochs[i + 1] - epochs[i] for i in range(len(epochs) - 1)]

    def summary(self) -> str:
        drops = " -> ".join(
            f"{tr.bits}b@{tr.epoch:.2f}ep" for tr in self._sorted_transitions
        )
        widths = ", ".join(f"{w:.2f}ep" for w in self.plateau_widths)
        return (
            f"StepScheduler: INT{self.config.target_bits} target | "
            f"warmup={self.config.warmup_epochs}ep  "
            f"stab={self.config.stabilization_epochs}ep | "
            f"drops: {drops} | plateau widths: {widths}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_scheduler(
    config: ScheduleConfig,
    total_epochs: float,
) -> PrecisionScheduler:
    """Construct the appropriate PrecisionScheduler from a parsed ScheduleConfig.

    Called by the scheduled QAT trainer at startup after loading the YAML config.

    Args:
        config:       Parsed ScheduleConfig (from ExperimentConfig.schedule).
        total_epochs: Total training epochs (from TrainingConfig.epochs).

    Returns:
        Concrete PrecisionScheduler subclass matching config.type.

    Raises:
        ValueError: If config.type is not "linear", "cosine", or "step".
    """
    if config.type == "linear":
        return LinearScheduler(config, total_epochs)
    if config.type == "cosine":
        return CosineScheduler(config, total_epochs)
    if config.type == "step":
        return StepScheduler(config, total_epochs)
    raise ValueError(
        f"Unknown schedule type {config.type!r}. "
        f"Must be one of: 'linear', 'cosine', 'step'."
    )
