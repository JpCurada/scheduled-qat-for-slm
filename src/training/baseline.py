"""
FP32 baseline evaluation — Phase 1 of the experiment pipeline.

Downloads SmolLM2-1.7B (if not cached), evaluates perplexity on the WikiText-103
test set, optionally saves FP32 output logits for KL divergence reference, and
optionally runs the lm-evaluation-harness benchmark suite.

Why this script matters
-----------------------
All KL divergence values across Phase 2 and 3 are computed against the FP32
baseline logits saved here. Running this script once establishes the ground
truth against which every quantized variant is compared. It also produces our
own FP32 accuracy numbers for MMLU/HellaSwag/ARC/PIQA/GSM8K that serve as the
reference for knowledge-retention metrics — these may differ slightly from
HuggingFace's published numbers because we use lm-evaluation-harness while
their published scores use lighteval.

Outputs
-------
    results/baseline/
        baseline_results.json       -- perplexity + lm-eval scores in one file
        fp32_logits.pt              -- saved logits for KL divergence (if --save-logits)
        lm_eval/                    -- raw lm-evaluation-harness JSON output
                                       (if --run-benchmarks)

Usage
-----
    # Minimum: perplexity only (fast, ~10 minutes on a single GPU)
    python -m src.training.baseline --model HuggingFaceTB/SmolLM2-1.7B

    # Full Phase 1: perplexity + logits + benchmarks
    python -m src.training.baseline \\
        --model HuggingFaceTB/SmolLM2-1.7B \\
        --save-logits \\
        --run-benchmarks

    # Custom output directory and smaller logit set (less disk)
    python -m src.training.baseline \\
        --model HuggingFaceTB/SmolLM2-1.7B \\
        --save-logits \\
        --num-logit-samples 32 \\
        --output results/baseline_fp16/

Public API
----------
    run_baseline(model_name, output_dir, ...)  -- programmatic entry point
    main()                                      -- CLI entry point
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import (
    compute_perplexity,
    run_lm_eval,
    save_fp32_logits,
)

logger = logging.getLogger(__name__)

# Default benchmark tasks (full project suite from SKILL.md).
_DEFAULT_TASKS = ["mmlu", "hellaswag", "arc_challenge", "piqa", "gsm8k"]

# Default output directory (KL divergence in trainer.py looks here by default).
_DEFAULT_OUTPUT_DIR = "results/baseline"

# Default model identifier (can be overridden via CLI or API).
_DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-1.7B"

# Default model cache directory.
_DEFAULT_CACHE_DIR = "models/base/"


# ---------------------------------------------------------------------------
# Data loading helpers (no ExperimentConfig needed for baseline)
# ---------------------------------------------------------------------------

def _build_test_loader(
    model_name: str,
    cache_dir: str,
    seq_length: int = 2048,
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """Load the WikiText-103 test split as a DataLoader.

    Uses the same chunking strategy as the QAT experiments so that perplexity
    numbers are computed on identical token sequences. The test split (~246K tokens)
    yields ~120 non-overlapping chunks at seq_length=2048.

    Args:
        model_name:  HuggingFace model identifier for the tokenizer.
        cache_dir:   Local cache directory for the tokenizer.
        seq_length:  Token chunk length (must match the QAT experiment configs).
        batch_size:  Inference batch size.
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader over the WikiText-103 test split.
    """
    from src.utils.data_loader import _load_and_chunk, get_tokenizer  # noqa: PLC0415

    logger.info("Loading WikiText-103 test split (seq_length=%d) ...", seq_length)
    tokenizer = get_tokenizer(model_name, cache_dir)
    ds = _load_and_chunk("wikitext-103-raw-v1", "test", tokenizer, seq_length)
    logger.info("Test split: %d chunks of %d tokens", len(ds), seq_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def _build_train_loader_for_logits(
    model_name: str,
    cache_dir: str,
    seq_length: int = 2048,
    num_samples: int = 128,
    batch_size: int = 4,
    num_workers: int = 4,
) -> DataLoader:
    """Load a subset of the WikiText-103 training split for logit saving.

    Uses the training split (not test) so that the logit reference samples are
    drawn from the same distribution as QAT training data. This ensures KL
    divergence is measured on in-domain text the QAT models have adapted to.

    Args:
        model_name:  HuggingFace model identifier for the tokenizer.
        cache_dir:   Local cache directory.
        seq_length:  Token chunk length (must match QAT configs).
        num_samples: Number of sequences to save as logit reference.
        batch_size:  Inference batch size.
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader over num_samples training sequences.
    """
    from src.utils.data_loader import _load_and_chunk, get_tokenizer  # noqa: PLC0415

    logger.info(
        "Loading WikiText-103 train split for logit reference (n=%d sequences) ...",
        num_samples,
    )
    tokenizer = get_tokenizer(model_name, cache_dir)
    ds = _load_and_chunk(
        "wikitext-103-raw-v1", "train", tokenizer, seq_length,
        num_samples=num_samples,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_fp32_model(
    model_name: str,
    cache_dir: str,
    device: torch.device,
) -> nn.Module:
    """Load SmolLM2-1.7B in FP32 for baseline evaluation.

    Downloads from HuggingFace Hub on first call; subsequent calls read from
    the local cache at cache_dir. Weights are loaded as float32 so the logits
    saved to disk exactly represent the pretrained model's output distribution.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir:  Local cache directory for model weights.
        device:     Target device.

    Returns:
        nn.Module in eval mode on the specified device.
    """
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    logger.info("Loading FP32 model: %s (cache: %s) ...", model_name, cache_dir)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    size_gb = total_params * 4 / 1e9
    logger.info(
        "Model loaded in %.1fs: %d parameters (%.2f GB FP32)",
        time.time() - t0, total_params, size_gb,
    )
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline(
    model_name: str = _DEFAULT_MODEL,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    device_str: str = "cuda",
    seq_length: int = 2048,
    eval_batch_size: int = 4,
    save_logits: bool = True,
    num_logit_samples: int = 128,
    run_benchmarks: bool = False,
    benchmark_tasks: Optional[list[str]] = None,
    benchmark_batch_size: int = 16,
) -> dict:
    """Run FP32 baseline evaluation for Phase 1 of the experiment pipeline.

    Args:
        model_name:           HuggingFace model identifier.
        cache_dir:            Local directory to cache model weights.
        output_dir:           Directory to write all baseline outputs.
        device_str:           PyTorch device string. Falls back to CPU if CUDA
                              is unavailable.
        seq_length:           Token sequence length — must match QAT experiment
                              configs (default 2048 from SmolLM2-1.7B max context).
        eval_batch_size:      Batch size for perplexity evaluation.
        save_logits:          Whether to save FP32 output logits for later KL
                              divergence computation. Recommended: always True.
        num_logit_samples:    Number of training-split sequences to save as
                              logit reference. Storage: ~200 MB per sequence at
                              seq_length=2048, vocab=49152 in float16. Default
                              128 matches PTQ calibration set size.
        run_benchmarks:       Run lm-evaluation-harness on benchmark_tasks.
                              Takes ~1-4 hours depending on GPU speed.
        benchmark_tasks:      Tasks for lm-evaluation-harness. Defaults to the
                              full project suite: mmlu, hellaswag, arc_challenge,
                              piqa, gsm8k.
        benchmark_batch_size: Inference batch size for lm-eval.

    Returns:
        Dict with keys:
            model          -- model identifier
            perplexity     -- FP32 perplexity on WikiText-103 test
            logits_path    -- path to saved logits file (if save_logits=True)
            lm_eval        -- benchmark results dict (if run_benchmarks=True)
    """
    _configure_logging()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — running on CPU. This will be very slow.")

    logger.info("=" * 70)
    logger.info("FP32 Baseline Evaluation")
    logger.info("  model:   %s", model_name)
    logger.info("  device:  %s", device)
    logger.info("  output:  %s", output_path)
    logger.info("=" * 70)

    results: dict = {"model": model_name}

    # 1. Load FP32 model.
    model = _load_fp32_model(model_name, cache_dir, device)

    # 2. Perplexity on WikiText-103 test split.
    test_loader = _build_test_loader(
        model_name, cache_dir,
        seq_length=seq_length,
        batch_size=eval_batch_size,
    )
    logger.info("Computing FP32 perplexity on WikiText-103 test split ...")
    t0 = time.time()
    ppl = compute_perplexity(model, test_loader, device)
    results["perplexity"] = ppl
    logger.info("FP32 Perplexity: %.4f  (%.0fs)", ppl, time.time() - t0)

    # 3. Save FP32 logits for KL divergence reference.
    if save_logits:
        logits_path = output_path / "fp32_logits.pt"
        logit_loader = _build_train_loader_for_logits(
            model_name, cache_dir,
            seq_length=seq_length,
            num_samples=num_logit_samples,
            batch_size=eval_batch_size,
        )
        logger.info(
            "Saving FP32 logits: %d sequences -> %s ...",
            num_logit_samples, logits_path,
        )
        t0 = time.time()
        save_fp32_logits(
            model, logit_loader, logits_path, device,
            num_samples=num_logit_samples,
        )
        file_gb = logits_path.stat().st_size / 1e9
        logger.info(
            "Logits saved: %.2f GB  (%.0fs)",
            file_gb, time.time() - t0,
        )
        results["logits_path"] = str(logits_path)
        results["logits_sequences"] = num_logit_samples
        results["logits_size_gb"] = round(file_gb, 3)

    # 4. lm-evaluation-harness benchmarks.
    if run_benchmarks:
        tasks = benchmark_tasks or _DEFAULT_TASKS
        lm_eval_dir = output_path / "lm_eval"
        logger.info("Running lm-evaluation-harness: tasks=%s", tasks)
        logger.info("(This can take 1-4 hours depending on GPU speed.)")
        t0 = time.time()
        lm_results = run_lm_eval(
            model_path=model_name,
            output_path=lm_eval_dir,
            tasks=tasks,
            batch_size=benchmark_batch_size,
            device=str(device),
        )
        results["lm_eval"] = lm_results
        results["lm_eval_time_seconds"] = round(time.time() - t0, 1)

        logger.info("Benchmark results:")
        for task, r in lm_results.items():
            logger.info(
                "  %-25s acc=%.4f  (+/- %.4f)",
                task, r["acc"] or 0.0, r["acc_stderr"] or 0.0,
            )

    # 5. Save summary JSON.
    summary_path = output_path / "baseline_results.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Baseline results saved -> %s", summary_path)

    return results


# ---------------------------------------------------------------------------
# Convenience: model download only
# ---------------------------------------------------------------------------

def download_model(
    model_name: str = _DEFAULT_MODEL,
    cache_dir: str = _DEFAULT_CACHE_DIR,
) -> None:
    """Download SmolLM2-1.7B weights to the local cache without running eval.

    Equivalent to:
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-1.7B',
                                              cache_dir='models/base/')

    Useful as a first step to pre-fetch the model before starting GPU training.

    Args:
        model_name: HuggingFace model identifier.
        cache_dir:  Local directory to cache the downloaded weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

    logger.info("Downloading model weights: %s -> %s", model_name, cache_dir)
    AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    logger.info("Download complete.")


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
        root.setLevel(level)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FP32 baseline evaluation for Phase 1 of the experiment pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perplexity only (fast)
  python -m src.training.baseline

  # Full Phase 1: perplexity + logits + benchmarks
  python -m src.training.baseline \\
      --model HuggingFaceTB/SmolLM2-1.7B \\
      --save-logits \\
      --run-benchmarks

  # Download model weights only (pre-fetch before training)
  python -m src.training.baseline --download-only

  # Smaller logit set to save disk space (32 sequences instead of 128)
  python -m src.training.baseline --save-logits --num-logit-samples 32

  # Run on CPU (slow but works without GPU)
  python -m src.training.baseline --save-logits --device cpu
""",
    )
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"HuggingFace model identifier (default: {_DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--cache-dir",
        default=_DEFAULT_CACHE_DIR,
        help=f"Local cache directory for model weights (default: {_DEFAULT_CACHE_DIR}).",
    )
    p.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Output directory for results (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help='Torch device string (default: "cuda"; auto-falls back to "cpu").',
    )
    p.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Token sequence length for perplexity and logit evaluation (default: 2048).",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Batch size for perplexity evaluation (default: 4).",
    )
    p.add_argument(
        "--save-logits",
        action="store_true",
        help=(
            "Save FP32 output logits to <output>/fp32_logits.pt for KL divergence. "
            "Required before running Phase 2 experiments."
        ),
    )
    p.add_argument(
        "--num-logit-samples",
        type=int,
        default=128,
        help=(
            "Number of training sequences to save as logit reference. "
            "Storage: ~200 MB per sequence at seq_len=2048. "
            "128 sequences ≈ 25 GB; 32 ≈ 6.4 GB (default: 128)."
        ),
    )
    p.add_argument(
        "--run-benchmarks",
        action="store_true",
        help="Run lm-evaluation-harness (MMLU, HellaSwag, ARC-Challenge, PIQA, GSM8K).",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        metavar="TASK",
        help=f"lm-eval tasks to run (default: {' '.join(_DEFAULT_TASKS)}).",
    )
    p.add_argument(
        "--benchmark-batch-size",
        type=int,
        default=16,
        help="Batch size for lm-evaluation-harness (default: 16).",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the model weights; skip all evaluation.",
    )
    return p


def main() -> None:
    """CLI entry point: parse args and run baseline evaluation."""
    import sys

    # Make project root importable when invoked as python -m src.training.baseline
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    args = _build_arg_parser().parse_args()
    _configure_logging()

    if args.download_only:
        download_model(model_name=args.model, cache_dir=args.cache_dir)
        return

    run_baseline(
        model_name=args.model,
        cache_dir=args.cache_dir,
        output_dir=args.output,
        device_str=args.device,
        seq_length=args.seq_length,
        eval_batch_size=args.eval_batch_size,
        save_logits=args.save_logits,
        num_logit_samples=args.num_logit_samples,
        run_benchmarks=args.run_benchmarks,
        benchmark_tasks=args.tasks,
        benchmark_batch_size=args.benchmark_batch_size,
    )


if __name__ == "__main__":
    main()
