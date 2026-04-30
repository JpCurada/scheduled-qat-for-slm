"""
Unified training loop for scheduled-qat-for-slm experiments.

Reads any YAML config via config_loader, dispatches to the correct quantization
method (ptq / standard_qat / scheduled_qat / lora_qat), runs the training loop
if needed, evaluates with metrics.py, and saves results.

Training pipeline (QAT methods)
--------------------------------
    1.  Build dataloaders: WikiText-103 train split (training) + test split (eval)
        + validation split (mid-training monitoring).
    2.  Build model via build_model_for_training() — loads FP32 weights and
        injects the method-appropriate quantization.
    3.  Build AdamW optimizer + LR scheduler (cosine or linear with warmup).
    4.  Training loop with gradient accumulation:
            - Per-epoch: call StandardQATController.on_epoch_start() for standard_qat.
            - Per-step:  call ScheduledQATController.on_step() for scheduled_qat.
            - Periodic validation perplexity logged every eval_every_steps steps.
            - Periodic checkpoint saved every save_every_steps steps.
    5.  Final evaluation on the test split: perplexity + KL divergence vs FP32
        baseline (reads from results/baseline/fp32_logits.pt if present).
    6.  Save results/{experiment_name}/training_results.json.

PTQ pipeline (no training)
---------------------------
    1.  Build model via build_model_for_training() — loads + quantizes + calibrates.
    2.  Evaluate perplexity on WikiText-103 test split + KL divergence.
    3.  Save results.

Checkpoints
-----------
    standard_qat / scheduled_qat  ->  models/checkpoints/{name}/step_{N}.pt
    lora_qat                       ->  models/checkpoints/{name}/step_{N}/  (adapter dir)

Results
-------
    results/{experiment_name}/training_results.json

Usage
-----
    python -m src.training.trainer --config configs/scheduled_qat/scheduled_cosine_int4.yaml
    python -m src.training.trainer --config configs/ptq/ptq_int4.yaml
    python -m src.training.trainer --config configs/lora_qat/lora_qat_int4.yaml --run-benchmarks
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.model_wrapper import QuantizedModelWrapper, build_model_for_training
from src.utils.config_loader import ExperimentConfig, load_config
from src.utils.data_loader import build_dataloaders, build_validation_loader
from src.utils.metrics import (
    compute_kl_divergence,
    compute_perplexity,
    run_lm_eval,
)


# Map config compute_dtype -> autocast dtype. fp32 disables autocast entirely.
_AMP_DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _amp_settings(config: ExperimentConfig, device: torch.device) -> tuple[Optional[torch.dtype], bool]:
    """Resolve (autocast_dtype, use_grad_scaler) from the training config.

    Two valid mixed-precision recipes are supported:

    1. FP32 master weights + FP16 autocast + GradScaler
       (compute_dtype="fp32", use_amp=True). Standard PyTorch AMP. Gradients
       are FP32, scaler.unscale_ works correctly. Highest memory of the two.

    2. Low-precision weights (BF16 or FP16), autocast as a no-op
       (compute_dtype="bf16" or "fp16", use_amp ignored for scaler purposes).
       Weights, activations and gradients all live in the configured dtype.
       No GradScaler — it would reject the non-FP32 gradients with
       "Attempting to unscale FP16 gradients". BF16's wide dynamic range
       removes the need for loss scaling; pure FP16 weight training is
       supported but riskier (loss can go to NaN without scaling).

    Returns (autocast_dtype, use_grad_scaler). Either may be None/False.
    """
    if config.training is None:
        return None, False
    tc = config.training
    if device.type != "cuda":
        if getattr(tc, "use_amp", False):
            logger.info("use_amp=True but device is %s; AMP disabled.", device.type)
        return None, False

    weight_dtype_str = getattr(tc, "compute_dtype", "fp32").lower()
    weight_dtype = _AMP_DTYPE_MAP.get(weight_dtype_str)  # None when fp32

    # Recipe 2: weights are already low-precision. Autocast becomes a no-op
    # (or is skipped entirely) and GradScaler is incompatible with non-FP32
    # gradients, so we disable it.
    if weight_dtype is not None:
        return weight_dtype, False

    # Recipe 1: FP32 weights. Autocast only kicks in when use_amp=True; the
    # default amp_dtype is FP16, which needs the scaler.
    if not getattr(tc, "use_amp", False):
        return None, False
    return torch.float16, True

logger = logging.getLogger(__name__)

# Gradient clipping max norm — standard for LLM fine-tuning.
_GRAD_CLIP_NORM = 1.0

# Log a training step summary every this many optimizer steps.
_LOG_EVERY_STEPS = 25

# Default path where the baseline script saves FP32 logits.
_DEFAULT_LOGITS_PATH = "results/baseline/fp32_logits.pt"


# ---------------------------------------------------------------------------
# Optimizer & LR scheduler
# ---------------------------------------------------------------------------

def _build_optimizer(model: nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    """Build AdamW over all parameters that require gradients.

    If training.use_8bit_optimizer is True, uses bitsandbytes.optim.AdamW8bit
    which keeps the (m, v) optimizer state in INT8 — cuts AdamW memory from
    ~13 GB to ~3 GB for SmolLM2-1.7B and is what makes QAT fit on a T4.
    Falls back to vanilla AdamW with a warning if bitsandbytes is not
    importable, so the trainer never crashes on missing optional deps.
    """
    tc = config.training
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(
            "No trainable parameters found. "
            "Check that the model and LoRA config are set up correctly."
        )

    if getattr(tc, "use_8bit_optimizer", False):
        try:
            import bitsandbytes as bnb  # type: ignore[import]
            opt = bnb.optim.AdamW8bit(
                trainable,
                lr=tc.learning_rate,
                weight_decay=tc.weight_decay,
            )
            logger.info(
                "Optimizer: AdamW8bit (bitsandbytes)  lr=%g  weight_decay=%g  trainable_params=%d",
                tc.learning_rate, tc.weight_decay, sum(p.numel() for p in trainable),
            )
            return opt
        except ImportError:
            logger.warning(
                "use_8bit_optimizer=True but bitsandbytes is not installed. "
                "Falling back to torch.optim.AdamW. "
                "Install with: pip install bitsandbytes",
            )

    logger.info("Optimizer: AdamW  lr=%g  weight_decay=%g  trainable_params=%d",
                tc.learning_rate, tc.weight_decay, sum(p.numel() for p in trainable))
    return torch.optim.AdamW(
        trainable,
        lr=tc.learning_rate,
        weight_decay=tc.weight_decay,
    )


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    total_steps: int,
) -> Any:
    """Build a cosine or linear LR scheduler with linear warmup."""
    from transformers import (  # type: ignore[import]
        get_cosine_schedule_with_warmup,
        get_linear_schedule_with_warmup,
    )

    tc = config.training
    warmup = tc.warmup_steps
    sched_type = tc.lr_scheduler.lower()

    if sched_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )
    elif sched_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps,
        )
    else:
        raise ValueError(
            f"Unknown lr_scheduler {sched_type!r}. Supported: cosine, linear."
        )

    logger.info(
        "LR scheduler: %s  warmup=%d steps  total=%d steps",
        sched_type, warmup, total_steps,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _compute_total_steps(train_loader: DataLoader, config: ExperimentConfig) -> int:
    """Total optimizer steps = (batches // grad_accum) * epochs."""
    steps_per_epoch = max(1, len(train_loader) // config.training.gradient_accumulation_steps)
    return int(steps_per_epoch * config.training.epochs)


def _build_ptq_eval_loader(config: ExperimentConfig, batch_size: int = 4) -> DataLoader:
    """Build a WikiText-103 test-split eval loader for the PTQ method.

    PTQ configs have no 'data' section, so we build the loader directly from
    calibration config fields (seq_length) and hardcode the test split.
    """
    from src.utils.data_loader import _load_and_chunk, get_tokenizer  # noqa: PLC0415

    seq_length = config.calibration.seq_length if config.calibration else 2048
    tokenizer = get_tokenizer(config.model.name, config.model.cache_dir)
    ds = _load_and_chunk("wikitext-103-raw-v1", "test", tokenizer, seq_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    wrapper: QuantizedModelWrapper,
    ckpt_dir: Path,
    global_step: int,
    epoch: int,
    loss: float,
) -> None:
    """Save a training checkpoint based on the active quantization method."""
    meta = {"global_step": global_step, "epoch": epoch, "loss": loss}
    method = wrapper.method

    if method == "standard_qat":
        from src.quantization.standard_qat import save_checkpoint  # noqa: PLC0415
        path = ckpt_dir / f"step_{global_step:07d}.pt"
        save_checkpoint(wrapper.model, path, controller=wrapper.controller, extra_meta=meta)

    elif method == "scheduled_qat":
        from src.quantization.scheduled_qat import save_checkpoint  # noqa: PLC0415
        path = ckpt_dir / f"step_{global_step:07d}.pt"
        save_checkpoint(wrapper.model, path, controller=wrapper.controller, extra_meta=meta)

    elif method == "lora_qat":
        from src.quantization.lora_qat import save_lora_checkpoint  # noqa: PLC0415
        adapter_dir = ckpt_dir / f"step_{global_step:07d}"
        save_lora_checkpoint(wrapper.model, str(adapter_dir))

    else:
        logger.warning("No checkpoint handler for method=%s", method)


def _save_final_checkpoint(wrapper: QuantizedModelWrapper, config: ExperimentConfig) -> None:
    """Save the final model checkpoint after training completes."""
    ckpt_dir = Path("models/checkpoints") / config.experiment_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    method = wrapper.method
    meta = {"stage": "final", "experiment": config.experiment_name}

    if method == "standard_qat":
        from src.quantization.standard_qat import save_checkpoint  # noqa: PLC0415
        save_checkpoint(wrapper.model, ckpt_dir / "final.pt", controller=wrapper.controller, extra_meta=meta)

    elif method == "scheduled_qat":
        from src.quantization.scheduled_qat import save_checkpoint  # noqa: PLC0415
        save_checkpoint(wrapper.model, ckpt_dir / "final.pt", controller=wrapper.controller, extra_meta=meta)

    elif method == "lora_qat":
        from src.quantization.lora_qat import save_lora_checkpoint  # noqa: PLC0415
        save_lora_checkpoint(wrapper.model, str(ckpt_dir / "final_adapter"))

    logger.info("Final checkpoint saved -> %s", ckpt_dir)


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _append_training_log(log_path: Path, entry: dict) -> None:
    """Append a JSONL entry to the step-level training log."""
    with log_path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _run_training(
    wrapper: QuantizedModelWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[int, float]:
    """Main QAT training loop.

    Handles gradient accumulation, per-step / per-epoch controller callbacks,
    periodic validation, periodic checkpointing, and step-level logging.

    Args:
        wrapper:       QuantizedModelWrapper with model, controller, config.
        train_loader:  DataLoader over the WikiText-103 training split.
        val_loader:    DataLoader over the WikiText-103 validation split for
                       mid-training monitoring.
        optimizer:     AdamW optimizer.
        lr_scheduler:  Transformers LR scheduler.
        config:        ExperimentConfig.
        device:        Training device.

    Returns:
        (total_optimizer_steps, last_loss): Steps completed and final loss value.
    """
    model = wrapper.model
    controller = wrapper.controller
    method = wrapper.method
    tc = config.training
    lc = config.logging

    grad_accum = tc.gradient_accumulation_steps
    total_epochs = int(tc.epochs)
    eval_every = lc.eval_every_steps if lc else 0
    save_every = lc.save_every_steps if lc else 0

    log_dir = Path(lc.log_dir) if lc else Path("results/logs") / config.experiment_name
    ckpt_dir = Path("models/checkpoints") / config.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step_log_path = log_dir / "training_steps.jsonl"
    per_step_log_path = log_dir / "per_step_loss.jsonl"   # micro view: every optimizer step

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    global_step = 0
    micro_step = 0
    last_loss = 0.0
    t_start = time.time()

    # AMP setup — autocast wraps the forward pass; GradScaler is only used
    # for FP16 (BF16 keeps the FP32 dynamic range so no scaling is needed).
    amp_dtype, use_grad_scaler = _amp_settings(config, device)
    scaler = torch.amp.GradScaler(device.type, enabled=use_grad_scaler)

    def _autocast_ctx():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    model.train()
    optimizer.zero_grad()

    # User-visible logs are routed through print() so they show up reliably in
    # Colab/Kaggle notebook cells (some notebook frontends swallow logger output).
    print(
        f"Training: method={method}  epochs={total_epochs}  grad_accum={grad_accum}  "
        f"batch={tc.batch_size}  effective_batch={tc.batch_size * grad_accum}  "
        f"amp={amp_dtype}  scaler={use_grad_scaler}",
        flush=True,
    )

    for epoch in range(total_epochs):

        # ---- Standard QAT: epoch-level fake quant enable/disable ----
        if method == "standard_qat" and controller is not None:
            changed = controller.on_epoch_start(float(epoch))
            if changed:
                print(f"[epoch {epoch}] {controller.describe()}", flush=True)

        epoch_loss_sum = 0.0
        epoch_micro_steps = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with _autocast_ctx():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                # Divide loss by grad_accum before backward so accumulated gradients
                # are equivalent to a single forward on the full effective batch.
                loss = outputs.loss / grad_accum

            scaler.scale(loss).backward()

            micro_step += 1
            epoch_micro_steps += 1
            epoch_loss_sum += loss.item() * grad_accum  # un-scaled for logging

            is_accumulation_step = micro_step % grad_accum == 0

            if is_accumulation_step:
                # Unscale before clipping so the clip threshold is applied to
                # real gradients, not the scaled ones. No-op when scaler disabled.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=_GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                last_loss = loss.item() * grad_accum
                lr = lr_scheduler.get_last_lr()[0]

                # ---- Per-step loss log (micro view, every optimizer step) ----
                _append_training_log(per_step_log_path, {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": last_loss,
                    "lr": lr,
                })

                # ---- Scheduled QAT: step-level precision update ----
                if method == "scheduled_qat" and controller is not None:
                    event = controller.on_step(global_step)
                    if event is not None:
                        print(f"[step {global_step}] schedule event: {event.describe()}",
                              flush=True)

                # ---- Periodic step-level logging ----
                if global_step % _LOG_EVERY_STEPS == 0:
                    elapsed = time.time() - t_start
                    print(
                        f"[step {global_step} | epoch {epoch + 1}/{total_epochs}] "
                        f"loss={last_loss:.4f}  lr={lr:.2e}  elapsed={elapsed:.0f}s",
                        flush=True,
                    )

                # ---- Periodic validation perplexity ----
                if eval_every > 0 and global_step % eval_every == 0:
                    val_ppl = compute_perplexity(model, val_loader, device)
                    model.train()
                    print(f"[step {global_step}] val_ppl={val_ppl:.4f}", flush=True)
                    _append_training_log(step_log_path, {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": last_loss,
                        "val_ppl": val_ppl,
                        "lr": lr,
                    })

                # ---- Periodic checkpoint ----
                if save_every > 0 and global_step % save_every == 0:
                    _save_checkpoint(wrapper, ckpt_dir, global_step, epoch, last_loss)

        # ---- Flush any partial accumulation at end of epoch ----
        if micro_step % grad_accum != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=_GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_epoch_loss = epoch_loss_sum / max(epoch_micro_steps, 1)
        print(
            f"Epoch {epoch + 1}/{total_epochs} complete: "
            f"avg_loss={avg_epoch_loss:.4f}  global_step={global_step}",
            flush=True,
        )

    print(
        f"Training complete: {global_step} optimizer steps  "
        f"{time.time() - t_start:.0f}s elapsed",
        flush=True,
    )
    return global_step, last_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    baseline_logits_path: Optional[str] = None,
    run_benchmarks: bool = False,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run the full evaluation suite on a trained/quantized model.

    Computes:
        - Perplexity on WikiText-103 test split (always).
        - KL divergence vs FP32 baseline logits if the logits file exists.
        - lm-evaluation-harness benchmarks if run_benchmarks=True.

    Args:
        model:                 The model to evaluate (will be switched to eval mode).
        eval_loader:           DataLoader over the WikiText-103 test split.
        config:                ExperimentConfig (used for benchmark task list).
        device:                Inference device.
        baseline_logits_path:  Path to FP32 logits .pt file for KL divergence.
                               If None or file does not exist, KL divergence is skipped.
        run_benchmarks:        Run the full lm-evaluation-harness benchmark suite.
        output_dir:            Directory to write lm-eval JSON results.

    Returns:
        Dict with keys: perplexity, kl_divergence (if computed), lm_eval (if run).
    """
    results: dict[str, Any] = {}

    print("Running evaluation ...", flush=True)
    ppl = compute_perplexity(model, eval_loader, device)
    results["perplexity"] = ppl
    print(f"Perplexity (WikiText-103 test): {ppl:.4f}", flush=True)

    logits_path = Path(baseline_logits_path) if baseline_logits_path else Path(_DEFAULT_LOGITS_PATH)
    if logits_path.exists():
        print(f"Computing KL divergence vs FP32 baseline ({logits_path}) ...", flush=True)
        kld = compute_kl_divergence(
            model,
            device,
            baseline_logits_path=logits_path,
        )
        results["kl_divergence"] = kld
        print(f"KL divergence: {kld:.6f}", flush=True)
    else:
        print(
            f"WARNING — FP32 baseline logits not found at {logits_path}. "
            "Run notebook 01 first to enable KL divergence. Skipping.",
            flush=True,
        )

    if run_benchmarks:
        tasks = config.evaluation.secondary_benchmarks if config.evaluation else None
        eval_out = output_dir or (Path("results") / config.experiment_name / "lm_eval")
        print(f"Running lm-evaluation-harness: tasks={tasks}", flush=True)
        lm_results = run_lm_eval(
            model_path=config.model.name,
            output_path=eval_out,
            tasks=tasks,
            device=str(device),
        )
        results["lm_eval"] = lm_results
        for task, r in lm_results.items():
            acc = r.get("acc") or 0.0
            err = r.get("acc_stderr") or 0.0
            print(f"  {task:25s} acc={acc:.4f} +/- {err:.4f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _save_results(
    results: dict[str, Any],
    config: ExperimentConfig,
    extra: Optional[dict] = None,
) -> Path:
    """Write training results to results/{experiment_name}/training_results.json.

    Args:
        results:   Metric dict from _evaluate_model().
        config:    ExperimentConfig for naming.
        extra:     Additional fields to merge (e.g. training stats).

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path("results") / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "experiment_name": config.experiment_name,
        "method": config.method,
        "target_bits": config.target_bits,
        **results,
        **(extra or {}),
    }

    out_path = out_dir / "training_results.json"
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Results saved -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def run_experiment(
    config_path: str,
    device_str: str = "cuda",
    baseline_logits: Optional[str] = None,
    run_benchmarks: bool = False,
) -> dict[str, Any]:
    """Full experiment pipeline: load config, quantize, train (if QAT), evaluate.

    This is the main entry point for programmatic use (e.g. from notebooks).
    The CLI calls this via main().

    Args:
        config_path:       Path to a YAML experiment config.
        device_str:        PyTorch device string (e.g. "cuda", "cuda:0", "cpu").
                           Automatically falls back to CPU if CUDA is unavailable.
        baseline_logits:   Path to the FP32 logits .pt file for KL divergence.
                           Defaults to results/baseline/fp32_logits.pt.
        run_benchmarks:    Run the full lm-evaluation-harness suite at the end.

    Returns:
        Dict with all computed metrics and training statistics.
    """
    # ------------------------------------------------------------------ setup
    _configure_logging()

    config = load_config(config_path)
    device = torch.device(
        device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — running on CPU (training will be very slow).")

    print("=" * 70, flush=True)
    print(f"Experiment: {config.experiment_name}  device={device}", flush=True)
    print("=" * 70, flush=True)

    method = config.method

    # ------------------------------------------------------------ PTQ (no loop)
    if method == "ptq":
        eval_loader = _build_ptq_eval_loader(config)
        wrapper = build_model_for_training(config, device)

        results = _evaluate_model(
            wrapper.model,
            eval_loader,
            config,
            device,
            baseline_logits_path=baseline_logits,
            run_benchmarks=run_benchmarks,
        )
        out_path = _save_results(results, config, extra={"method": "ptq"})
        print(f"PTQ experiment complete -> {out_path}", flush=True)
        return results

    # --------------------------------------------------------- QAT (training)
    # 1. Build dataloaders (we need len(train_loader) to compute total_steps).
    train_loader, eval_loader = build_dataloaders(config)
    val_loader = build_validation_loader(config)

    # 2. Compute total optimizer steps (needed for scheduled_qat model init).
    tc = config.training
    steps_per_epoch = max(1, len(train_loader) // tc.gradient_accumulation_steps)
    total_steps = int(steps_per_epoch * tc.epochs)

    print(
        f"Steps: {len(train_loader)} batches/epoch  {tc.gradient_accumulation_steps} accum  "
        f"{steps_per_epoch} steps/epoch  {tc.epochs:.1f} epochs  = {total_steps} total steps",
        flush=True,
    )

    # 3. Build model (dispatches to the right quantization builder).
    wrapper = build_model_for_training(
        config,
        device,
        total_steps=total_steps if method == "scheduled_qat" else None,
    )

    # 4. Build optimizer + LR scheduler.
    optimizer = _build_optimizer(wrapper.model, config)
    lr_scheduler = _build_lr_scheduler(optimizer, config, total_steps)

    # 5. Training loop.
    t0 = time.time()
    global_step, last_loss = _run_training(
        wrapper, train_loader, val_loader, optimizer, lr_scheduler, config, device,
    )
    training_time_s = time.time() - t0

    # 6. Save final checkpoint.
    _save_final_checkpoint(wrapper, config)

    # 7. Final evaluation on the test split.
    results = _evaluate_model(
        wrapper.model,
        eval_loader,
        config,
        device,
        baseline_logits_path=baseline_logits,
        run_benchmarks=run_benchmarks,
        output_dir=Path("results") / config.experiment_name / "lm_eval",
    )

    # 8. Save results.
    extra = {
        "total_steps": global_step,
        "total_epochs": int(tc.epochs),
        "final_loss": last_loss,
        "training_time_seconds": round(training_time_s, 1),
        "steps_per_epoch": steps_per_epoch,
    }
    out_path = _save_results(results, config, extra=extra)
    logger.info("Experiment complete -> %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(level: int = logging.INFO) -> None:
    """Configure root logger if no handlers are attached yet."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
        root.setLevel(level)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train and evaluate a quantized SmolLM2-1.7B model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard QAT INT4
  python -m src.training.trainer --config configs/standard_qat/qat_int4.yaml

  # Scheduled QAT cosine INT4 with benchmarks
  python -m src.training.trainer \\
      --config configs/scheduled_qat/scheduled_cosine_int4.yaml \\
      --run-benchmarks

  # PTQ INT8
  python -m src.training.trainer --config configs/ptq/ptq_int8.yaml

  # LoRA-QAT on CPU
  python -m src.training.trainer \\
      --config configs/lora_qat/lora_qat_int4.yaml \\
      --device cpu
""",
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config.",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help='Torch device string (default: "cuda"; falls back to cpu automatically).',
    )
    p.add_argument(
        "--baseline-logits",
        default=None,
        metavar="PATH",
        help=(
            f"Path to FP32 logits .pt file for KL divergence. "
            f"Defaults to {_DEFAULT_LOGITS_PATH}."
        ),
    )
    p.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run lm-evaluation-harness at the end (MMLU, HellaSwag, ARC, PIQA, GSM8K).",
    )
    return p


def main() -> None:
    """CLI entry point: parse args and run experiment."""
    import sys

    # Make project root importable when invoked as python -m src.training.trainer
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    args = _build_arg_parser().parse_args()
    run_experiment(
        config_path=args.config,
        device_str=args.device,
        baseline_logits=args.baseline_logits,
        run_benchmarks=args.run_benchmarks,
    )


if __name__ == "__main__":
    main()
