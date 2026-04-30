"""
YAML config loader for scheduled-qat-for-slm experiments.

Parses and validates experiment configs according to the schemas defined in SKILL.md.
Returns typed ExperimentConfig dataclasses ready for use by the trainer and utilities.

Usage:
    from src.utils.config_loader import load_config

    config = load_config("configs/ptq/ptq_int4.yaml")
    print(config.method)          # "ptq"
    print(config.target_bits)     # 4
    print(config.model.name)      # "HuggingFaceTB/SmolLM2-1.7B"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Valid values
# ---------------------------------------------------------------------------

VALID_METHODS = {"ptq", "standard_qat", "scheduled_qat", "lora_qat"}
VALID_BITS = {4, 8, 16, 32}
VALID_SCHEMES = {"symmetric", "asymmetric"}
VALID_GRANULARITIES = {"per_tensor", "per_channel"}
VALID_SCHEDULE_TYPES = {"linear", "cosine", "step"}

# Required top-level sections per method
_REQUIRED_SECTIONS: dict[str, set[str]] = {
    "ptq": {"model", "calibration", "quantize_config", "evaluation", "export"},
    "standard_qat": {"model", "data", "training", "quantize_config", "evaluation", "export", "logging"},
    "scheduled_qat": {"model", "data", "training", "schedule", "quantize_config", "evaluation", "export", "logging"},
    "lora_qat": {"model", "data", "training", "lora_config", "quantize_config", "evaluation", "export", "logging"},
}


# ---------------------------------------------------------------------------
# Dataclasses — one per YAML section
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str
    cache_dir: str


@dataclass
class DataConfig:
    train_dataset: str
    eval_dataset: str
    train_split: str
    eval_split: str
    seq_length: int


@dataclass
class CalibrationConfig:
    dataset: str
    split: str
    num_samples: int
    seq_length: int


@dataclass
class TrainingConfig:
    epochs: float
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    warmup_steps: int
    lr_scheduler: str
    # Memory-efficiency knobs (all default to off so existing configs and
    # high-VRAM setups behave exactly as before).
    #
    # compute_dtype:           Dtype the model weights are loaded in. "fp32"
    #                          (default) keeps the legacy behaviour; "fp16" or
    #                          "bf16" halves weight + grad memory. The
    #                          fake-quantize math runs in this dtype too — the
    #                          STE handles either case correctly.
    # use_amp:                 Wrap forward/backward in torch.cuda.amp.autocast
    #                          and use a GradScaler for FP16. Has no effect on
    #                          BF16 (autocast is fine but no scaler needed).
    # use_8bit_optimizer:      Use bitsandbytes.optim.AdamW8bit. Cuts AdamW
    #                          state from 2*params*4B to ~2*params*1B. Falls
    #                          back to torch.optim.AdamW if bitsandbytes is
    #                          not installed.
    # gradient_checkpointing:  Call model.gradient_checkpointing_enable().
    #                          Re-computes activations during backward to free
    #                          activation memory, ~30% slower but ~40% less
    #                          activation RAM.
    compute_dtype: str = "fp32"
    use_amp: bool = False
    use_8bit_optimizer: bool = False
    gradient_checkpointing: bool = False


@dataclass
class QuantizeConfig:
    scheme: str
    granularity: str
    exclude_layers: list[str]
    fake_quant_start_epoch: int = 0  # standard_qat only; ignored for other methods


@dataclass
class ScheduleTransition:
    epoch: float
    bits: int


@dataclass
class ScheduleConfig:
    """Precision schedule for scheduled_qat.

    - linear: continuous reduction from start_bits to target_bits; transitions unused.
    - cosine:  cosine-curved reduction; transitions define discrete epoch checkpoints.
    - step:    hard drops at each transition epoch; transitions are required.
    """
    type: str
    warmup_epochs: float
    start_bits: int
    target_bits: int
    stabilization_epochs: float
    transitions: list[ScheduleTransition] = field(default_factory=list)


@dataclass
class LoRAConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: list[str]
    freeze_base: bool


@dataclass
class EvaluationConfig:
    primary_metrics: list[str]
    secondary_benchmarks: list[str]
    track_answer_flips: bool


@dataclass
class ExportConfig:
    format: str
    output_dir: str
    merge_lora: bool = False  # lora_qat only: merge adapters into base before GGUF export


@dataclass
class LoggingConfig:
    save_every_steps: int
    eval_every_steps: int
    log_dir: str


@dataclass
class ExperimentConfig:
    method: str
    target_bits: int
    model: ModelConfig
    quantize_config: QuantizeConfig
    evaluation: EvaluationConfig
    export: ExportConfig
    # Method-specific (None when not applicable)
    data: Optional[DataConfig] = None
    calibration: Optional[CalibrationConfig] = None
    training: Optional[TrainingConfig] = None
    schedule: Optional[ScheduleConfig] = None
    lora_config: Optional[LoRAConfig] = None
    logging: Optional[LoggingConfig] = None

    @property
    def experiment_name(self) -> str:
        """Canonical name used in result file naming: {method}_{bits}[_{schedule}].

        Examples:
            ptq_int4
            standard_qat_int8
            scheduled_qat_cosine_int4
            lora_qat_int4
        """
        if self.method == "scheduled_qat" and self.schedule:
            return f"scheduled_qat_{self.schedule.type}_int{self.target_bits}"
        return f"{self.method}_int{self.target_bits}"


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------

def _parse_model(raw: dict) -> ModelConfig:
    return ModelConfig(name=raw["name"], cache_dir=raw["cache_dir"])


def _parse_data(raw: dict) -> DataConfig:
    return DataConfig(
        train_dataset=raw["train_dataset"],
        eval_dataset=raw["eval_dataset"],
        train_split=raw["train_split"],
        eval_split=raw["eval_split"],
        seq_length=int(raw["seq_length"]),
    )


def _parse_calibration(raw: dict) -> CalibrationConfig:
    return CalibrationConfig(
        dataset=raw["dataset"],
        split=raw["split"],
        num_samples=int(raw["num_samples"]),
        seq_length=int(raw["seq_length"]),
    )


def _parse_training(raw: dict) -> TrainingConfig:
    compute_dtype = str(raw.get("compute_dtype", "fp32")).lower()
    if compute_dtype not in ("fp32", "fp16", "bf16"):
        raise ValueError(
            f"training.compute_dtype must be one of fp32, fp16, bf16; got {compute_dtype!r}"
        )
    return TrainingConfig(
        epochs=float(raw["epochs"]),
        batch_size=int(raw["batch_size"]),
        gradient_accumulation_steps=int(raw["gradient_accumulation_steps"]),
        learning_rate=float(raw["learning_rate"]),
        optimizer=raw["optimizer"],
        weight_decay=float(raw["weight_decay"]),
        warmup_steps=int(raw["warmup_steps"]),
        lr_scheduler=raw["lr_scheduler"],
        compute_dtype=compute_dtype,
        use_amp=bool(raw.get("use_amp", False)),
        use_8bit_optimizer=bool(raw.get("use_8bit_optimizer", False)),
        gradient_checkpointing=bool(raw.get("gradient_checkpointing", False)),
    )


def _parse_quantize_config(raw: dict) -> QuantizeConfig:
    scheme = raw["scheme"]
    granularity = raw["granularity"]
    if scheme not in VALID_SCHEMES:
        raise ValueError(f"quantize_config.scheme must be one of {VALID_SCHEMES}, got {scheme!r}")
    if granularity not in VALID_GRANULARITIES:
        raise ValueError(
            f"quantize_config.granularity must be one of {VALID_GRANULARITIES}, got {granularity!r}"
        )
    return QuantizeConfig(
        scheme=scheme,
        granularity=granularity,
        exclude_layers=list(raw.get("exclude_layers", [])),
        fake_quant_start_epoch=int(raw.get("fake_quant_start_epoch", 0)),
    )


def _parse_schedule(raw: dict) -> ScheduleConfig:
    stype = raw["type"]
    if stype not in VALID_SCHEDULE_TYPES:
        raise ValueError(f"schedule.type must be one of {VALID_SCHEDULE_TYPES}, got {stype!r}")
    transitions = [
        ScheduleTransition(epoch=float(t["epoch"]), bits=int(t["bits"]))
        for t in raw.get("transitions", [])
    ]
    if stype == "step" and not transitions:
        raise ValueError("schedule.type=step requires at least one entry in schedule.transitions")
    return ScheduleConfig(
        type=stype,
        warmup_epochs=float(raw["warmup_epochs"]),
        start_bits=int(raw["start_bits"]),
        target_bits=int(raw["target_bits"]),
        stabilization_epochs=float(raw["stabilization_epochs"]),
        transitions=transitions,
    )


def _parse_lora_config(raw: dict) -> LoRAConfig:
    return LoRAConfig(
        rank=int(raw["rank"]),
        alpha=int(raw["alpha"]),
        dropout=float(raw["dropout"]),
        target_modules=list(raw["target_modules"]),
        freeze_base=bool(raw["freeze_base"]),
    )


def _parse_evaluation(raw: dict) -> EvaluationConfig:
    return EvaluationConfig(
        primary_metrics=list(raw["primary_metrics"]),
        secondary_benchmarks=list(raw["secondary_benchmarks"]),
        track_answer_flips=bool(raw["track_answer_flips"]),
    )


def _parse_export(raw: dict) -> ExportConfig:
    return ExportConfig(
        format=raw["format"],
        output_dir=raw["output_dir"],
        merge_lora=bool(raw.get("merge_lora", False)),
    )


def _parse_logging(raw: dict) -> LoggingConfig:
    return LoggingConfig(
        save_every_steps=int(raw["save_every_steps"]),
        eval_every_steps=int(raw["eval_every_steps"]),
        log_dir=raw["log_dir"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Fully parsed and validated ExperimentConfig.

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: A field has an invalid value or the file is empty.
        KeyError: A required section or field is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Config file is empty: {path}")

    method = raw.get("method")
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")

    target_bits = int(raw.get("target_bits", 0))
    if target_bits not in VALID_BITS:
        raise ValueError(f"target_bits must be one of {VALID_BITS}, got {target_bits}")

    missing = _REQUIRED_SECTIONS[method] - set(raw.keys())
    if missing:
        raise KeyError(
            f"Config for method={method!r} is missing required sections: {sorted(missing)}"
        )

    return ExperimentConfig(
        method=method,
        target_bits=target_bits,
        model=_parse_model(raw["model"]),
        quantize_config=_parse_quantize_config(raw["quantize_config"]),
        evaluation=_parse_evaluation(raw["evaluation"]),
        export=_parse_export(raw["export"]),
        data=_parse_data(raw["data"]) if "data" in raw else None,
        calibration=_parse_calibration(raw["calibration"]) if "calibration" in raw else None,
        training=_parse_training(raw["training"]) if "training" in raw else None,
        schedule=_parse_schedule(raw["schedule"]) if "schedule" in raw else None,
        lora_config=_parse_lora_config(raw["lora_config"]) if "lora_config" in raw else None,
        logging=_parse_logging(raw["logging"]) if "logging" in raw else None,
    )
