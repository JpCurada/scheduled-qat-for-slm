"""
Evaluation metrics for scheduled-qat-for-slm experiments.

Provides five metric groups:

    1. Perplexity       — compute_perplexity()
       WikiText-103 test-set perplexity via the model's own cross-entropy loss.

    2. FP32 logit I/O   — save_fp32_logits()
       Run the FP32 baseline and persist output logits (float16) to disk once.
       Re-used for KL divergence when evaluating many quantized variants.

    3. KL Divergence    — compute_kl_divergence()
       Forward KL(P_fp32 || P_quant) averaged over tokens. Accepts either
       pre-saved logits (fast for multiple quantized variants) or streaming
       side-by-side comparison (no disk I/O needed).

    4. Answer Flips     — compute_answer_flips()
       Counts correct->wrong (bad) and wrong->correct (good) prediction changes
       between FP32 and quantized model on MCQ benchmarks.

    5. Derived metrics  — compute_knowledge_retention(), compute_efficiency_score()
       Knowledge retention: (quant_acc / fp32_acc) * 100%.
       Efficiency score (Unsloth): (MMLU - 25) / model_size_GB.

    6. lm-eval harness  — run_lm_eval()
       Subprocess/API wrapper around lm-evaluation-harness; returns a parsed
       {task: {acc, acc_stderr}} dict for MMLU, HellaSwag, ARC, PIQA, GSM8K.

CLI usage (Phase 3 evaluation):
    python -m src.utils.metrics \\
        --model models/checkpoints/scheduled_qat_cosine_int4.pt \\
        --baseline-logits results/baseline/fp32_logits.pt \\
        --metrics perplexity kl_divergence \\
        --output results/scheduled_qat_cosine_int4/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute perplexity on a WikiText-103 split.

    Uses the model's built-in cross-entropy loss (available on any HuggingFace
    CausalLM via outputs.loss). Accumulates total NLL and total predicted tokens
    across the full dataloader before computing exp(), which gives a properly
    weighted aggregate rather than an average of per-batch PPL values.

    Each batch must contain input_ids, attention_mask, and labels keys as
    produced by _ChunkDataset in data_loader.py. Labels are a copy of
    input_ids; HuggingFace shifts them internally, so each sequence of length L
    contributes (L - 1) predicted tokens.

    Args:
        model:      HuggingFace CausalLM (or any model returning outputs.loss).
        dataloader: Eval DataLoader -- typically the WikiText-103 test split.
        device:     Device to run inference on.

    Returns:
        Perplexity as a float. Lower is better.
    """
    model.eval()
    model.to(device)

    total_nll = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # outputs.loss is mean NLL per predicted token for this batch.
        # HuggingFace shifts labels internally: each sequence of length L
        # contributes (L - 1) tokens to the loss.
        seq_len = input_ids.size(1)
        n_tokens = input_ids.size(0) * (seq_len - 1)
        total_nll += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_nll / total_tokens)


# ---------------------------------------------------------------------------
# 2. FP32 logit persistence
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_fp32_logits(
    model: nn.Module,
    dataloader: DataLoader,
    output_path: Union[str, Path],
    device: torch.device,
    num_samples: int = 128,
) -> None:
    """Run the FP32 model and save output logits to disk for KL-divergence reference.

    Saves a dict with two tensors:
        input_ids  -- LongTensor (N, seq_len)
        logits     -- HalfTensor (N, seq_len, vocab_size)  stored in float16

    Storage note: at seq_len=2048 and vocab_size=49152 each sequence occupies
    ~200 MB in float16. 128 sequences is ~25 GB; 32 sequences ~6.4 GB; 16 ~3.2 GB.
    Reduce num_samples when disk space is limited. The default of 128 matches the
    calibration set size in the PTQ configs and is sufficient for a reliable
    KL-divergence estimate.

    Args:
        model:       FP32 baseline model.
        dataloader:  DataLoader to source input sequences from (train split).
        output_path: Destination .pt file (e.g. "results/baseline/fp32_logits.pt").
        device:      Device to run inference on.
        num_samples: Maximum number of sequences to collect.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    all_input_ids: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []
    collected = 0

    for batch in dataloader:
        if collected >= num_samples:
            break

        remaining = num_samples - collected
        input_ids = batch["input_ids"][:remaining].to(device)
        attention_mask = batch["attention_mask"][:remaining].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        all_input_ids.append(input_ids.cpu())
        # Store in float16 to halve disk usage vs float32.
        all_logits.append(outputs.logits.half().cpu())
        collected += input_ids.size(0)

    torch.save(
        {
            "input_ids": torch.cat(all_input_ids, dim=0),
            "logits": torch.cat(all_logits, dim=0),
        },
        output_path,
    )
    logger.info("Saved FP32 logits for %d sequences -> %s", collected, output_path)


# ---------------------------------------------------------------------------
# 3. KL Divergence
# ---------------------------------------------------------------------------

def compute_kl_divergence(
    model: nn.Module,
    device: torch.device,
    baseline_logits_path: Optional[Union[str, Path]] = None,
    fp32_model: Optional[nn.Module] = None,
    dataloader: Optional[DataLoader] = None,
    num_samples: int = 128,
) -> float:
    """Compute mean per-token forward KL divergence: KL(P_fp32 || P_quant).

    KL divergence measures how much the quantized model's output distribution
    diverges from the FP32 baseline. A value of 0 means the distributions are
    identical; higher values indicate more distortion. Per "Accuracy is Not All
    You Need" (arxiv.org/pdf/2407.09141), this is the gold-standard metric for
    quantization quality.

    Two operating modes -- exactly one must be provided:

    Mode A (saved logits) -- recommended when evaluating multiple variants:
        Pass baseline_logits_path pointing to the .pt file from save_fp32_logits().
        The quantized model is re-run on the same input_ids used when saving.

    Mode B (streaming) -- one-off comparisons, no disk I/O:
        Pass fp32_model and dataloader. Both models run on each batch simultaneously.
        Requires more GPU memory since two models are loaded at once.

    Args:
        model:                Quantized model to evaluate.
        device:               Device for inference.
        baseline_logits_path: Path to .pt file from save_fp32_logits() [Mode A].
        fp32_model:           FP32 model for streaming comparison [Mode B].
        dataloader:           DataLoader -- required for Mode B.
        num_samples:          Max sequences to process in Mode B.

    Returns:
        Mean per-token KL divergence as a float. Lower is better (0 = identical).

    Raises:
        ValueError: If neither or both source modes are provided.
    """
    if baseline_logits_path is None and fp32_model is None:
        raise ValueError(
            "Provide either baseline_logits_path (Mode A) or fp32_model + dataloader (Mode B)."
        )
    if baseline_logits_path is not None and fp32_model is not None:
        raise ValueError("Provide only one of baseline_logits_path or fp32_model, not both.")
    if fp32_model is not None and dataloader is None:
        raise ValueError("dataloader is required when using fp32_model (Mode B).")

    model.eval()
    model.to(device)

    if baseline_logits_path is not None:
        return _kl_from_saved_logits(model, Path(baseline_logits_path), device)
    return _kl_streaming(model, fp32_model, dataloader, device, num_samples)


@torch.no_grad()
def _kl_from_saved_logits(
    quant_model: nn.Module,
    logits_path: Path,
    device: torch.device,
    batch_size: int = 8,
) -> float:
    """Mode A: KL divergence using pre-saved FP32 logits."""
    checkpoint = torch.load(logits_path, map_location="cpu", weights_only=True)
    fp32_logits: torch.Tensor = checkpoint["logits"].float()   # (N, L, V)
    input_ids: torch.Tensor = checkpoint["input_ids"]          # (N, L)

    N = input_ids.size(0)
    total_kl = 0.0
    total_tokens = 0

    for start in range(0, N, batch_size):
        ids_batch = input_ids[start : start + batch_size].to(device)
        fp32_batch = fp32_logits[start : start + batch_size].to(device)

        quant_logits = quant_model(input_ids=ids_batch).logits.float()  # (B, L, V)

        # KL(P_fp32 || P_quant) = sum p_fp32 * (log p_fp32 - log p_quant)
        # F.kl_div(log_q, p, reduction="sum") computes sum(p*(log_p - log_q)) = KL(p||q)
        fp32_log_probs = F.log_softmax(fp32_batch, dim=-1)
        quant_log_probs = F.log_softmax(quant_logits, dim=-1)
        fp32_probs = fp32_log_probs.exp()

        kl = F.kl_div(quant_log_probs, fp32_probs, reduction="sum")
        total_kl += kl.item()
        total_tokens += ids_batch.numel()

    return total_kl / total_tokens


@torch.no_grad()
def _kl_streaming(
    quant_model: nn.Module,
    fp32_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int,
) -> float:
    """Mode B: KL divergence by running both models simultaneously on each batch."""
    fp32_model.eval()
    fp32_model.to(device)

    total_kl = 0.0
    total_tokens = 0
    collected = 0

    for batch in dataloader:
        if collected >= num_samples:
            break

        remaining = num_samples - collected
        input_ids = batch["input_ids"][:remaining].to(device)
        attention_mask = batch["attention_mask"][:remaining].to(device)

        fp32_logits = fp32_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.float()
        quant_logits = quant_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits.float()

        fp32_log_probs = F.log_softmax(fp32_logits, dim=-1)
        quant_log_probs = F.log_softmax(quant_logits, dim=-1)
        fp32_probs = fp32_log_probs.exp()

        kl = F.kl_div(quant_log_probs, fp32_probs, reduction="sum")
        total_kl += kl.item()
        total_tokens += input_ids.numel()
        collected += input_ids.size(0)

    return total_kl / total_tokens


# ---------------------------------------------------------------------------
# 4. Answer Flips
# ---------------------------------------------------------------------------

@dataclass
class AnswerFlipResult:
    """Answer flip analysis between FP32 and quantized model predictions.

    A flip occurs whenever the two models give different answers to the same
    question. A model can maintain identical overall accuracy while having many
    flips in both directions -- indicating unstable behaviour even when
    aggregate numbers look the same.

    Attributes:
        bad_flips:       Questions answered correctly by FP32 but incorrectly
                         by the quantized model (correct -> wrong).
        good_flips:      Questions answered incorrectly by FP32 but correctly
                         by the quantized model (wrong -> correct).
        total_questions: Total number of evaluated questions.
        fp32_correct:    Questions answered correctly by the FP32 model.
        quant_correct:   Questions answered correctly by the quantized model.
    """

    bad_flips: int
    good_flips: int
    total_questions: int
    fp32_correct: int
    quant_correct: int

    @property
    def bad_flip_rate(self) -> float:
        """Fraction of FP32-correct answers that flipped to wrong."""
        return self.bad_flips / max(self.fp32_correct, 1)

    @property
    def good_flip_rate(self) -> float:
        """Fraction of FP32-wrong answers that flipped to correct."""
        fp32_wrong = self.total_questions - self.fp32_correct
        return self.good_flips / max(fp32_wrong, 1)

    @property
    def net_flips(self) -> int:
        """Net answer change: positive = more good flips than bad."""
        return self.good_flips - self.bad_flips

    @property
    def fp32_accuracy(self) -> float:
        return self.fp32_correct / max(self.total_questions, 1)

    @property
    def quant_accuracy(self) -> float:
        return self.quant_correct / max(self.total_questions, 1)

    def __str__(self) -> str:
        return (
            f"AnswerFlips("
            f"bad={self.bad_flips} [{self.bad_flip_rate:.1%} of FP32-correct], "
            f"good={self.good_flips} [{self.good_flip_rate:.1%} of FP32-wrong], "
            f"net={self.net_flips:+d} | "
            f"FP32={self.fp32_correct}/{self.total_questions} [{self.fp32_accuracy:.1%}], "
            f"Quant={self.quant_correct}/{self.total_questions} [{self.quant_accuracy:.1%}])"
        )


def compute_answer_flips(
    fp32_predictions: Sequence[int],
    quant_predictions: Sequence[int],
    ground_truth: Sequence[int],
) -> AnswerFlipResult:
    """Count answer flips between FP32 and quantized model predictions on MCQ tasks.

    Args:
        fp32_predictions:  Predicted answer indices from the FP32 model.
                           e.g. [0, 3, 1, 2, ...] for 4-choice MCQ (MMLU, ARC).
        quant_predictions: Predicted answer indices from the quantized model.
        ground_truth:      Correct answer indices (same index convention).

    Returns:
        AnswerFlipResult with bad/good flip counts and accuracy statistics.

    Raises:
        ValueError: If input sequences have different lengths.
    """
    n = len(fp32_predictions)
    if len(quant_predictions) != n or len(ground_truth) != n:
        raise ValueError(
            f"All sequences must have equal length -- "
            f"fp32={n}, quant={len(quant_predictions)}, truth={len(ground_truth)}"
        )

    bad_flips = 0
    good_flips = 0
    fp32_correct = 0
    quant_correct = 0

    for fp32_pred, quant_pred, truth in zip(fp32_predictions, quant_predictions, ground_truth):
        fp32_right = fp32_pred == truth
        quant_right = quant_pred == truth

        if fp32_right:
            fp32_correct += 1
        if quant_right:
            quant_correct += 1

        if fp32_right and not quant_right:
            bad_flips += 1
        elif not fp32_right and quant_right:
            good_flips += 1

    return AnswerFlipResult(
        bad_flips=bad_flips,
        good_flips=good_flips,
        total_questions=n,
        fp32_correct=fp32_correct,
        quant_correct=quant_correct,
    )


# ---------------------------------------------------------------------------
# 5. Derived metrics
# ---------------------------------------------------------------------------

def compute_knowledge_retention(
    quantized_accuracy: float,
    fp32_accuracy: float,
) -> float:
    """Compute knowledge retention: fraction of FP32 capability preserved.

    Formula: (quantized_accuracy / fp32_accuracy) * 100

    Pass accuracies on the same scale (both 0-1 or both 0-100). Values above
    100 mean the quantized model outperforms FP32 on that task (possible due
    to regularisation effects of quantization noise).

    Args:
        quantized_accuracy: Task accuracy of the quantized model.
        fp32_accuracy:      Task accuracy of the FP32 baseline.

    Returns:
        Retention as a percentage (0-100+).

    Raises:
        ValueError: If fp32_accuracy is zero.
    """
    if fp32_accuracy == 0.0:
        raise ValueError("fp32_accuracy cannot be zero (would cause division by zero).")
    return (quantized_accuracy / fp32_accuracy) * 100.0


def compute_efficiency_score(
    mmlu_accuracy: float,
    model_size_gb: float,
) -> float:
    """Compute the Unsloth efficiency score: (MMLU - 25) / model_size_GB.

    The -25 baseline subtracts random-chance performance on a 4-choice MCQ
    (25% = random guess), so only non-trivial accuracy contributes. Higher
    scores mean more useful quality per gigabyte of storage.

    Args:
        mmlu_accuracy: MMLU accuracy as a percentage in the range 0-100
                       (e.g. pass 65.0 for 65%, not 0.65).
        model_size_gb: Model file size in gigabytes (GGUF file size on disk).

    Returns:
        Efficiency score (higher is better).

    Raises:
        ValueError: If model_size_gb is not positive.
    """
    if model_size_gb <= 0:
        raise ValueError(f"model_size_gb must be positive, got {model_size_gb}")
    return (mmlu_accuracy - 25.0) / model_size_gb


# ---------------------------------------------------------------------------
# 6. lm-evaluation-harness wrapper
# ---------------------------------------------------------------------------

_DEFAULT_TASKS = ["mmlu", "hellaswag", "arc_challenge", "piqa", "gsm8k"]

# Keys tried in order when parsing lm-eval result dicts.
# The exact suffix varies across lm-eval versions.
_ACC_KEYS = ("acc,none", "acc_norm,none", "acc", "acc_norm")
_ERR_KEYS = (
    "acc_stderr,none",
    "acc_norm_stderr,none",
    "acc_stderr",
    "acc_norm_stderr",
)


def run_lm_eval(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    tasks: Optional[list[str]] = None,
    batch_size: int = 16,
    device: str = "cuda",
    use_python_api: bool = True,
) -> dict[str, dict]:
    """Run lm-evaluation-harness and return parsed benchmark results.

    This is the project's single evaluation entry point for MMLU, HellaSwag,
    ARC-Challenge, PIQA, and GSM8K. All quantized variants are evaluated here
    for a consistent, framework-matched comparison.

    Tries the lm_eval Python API first (no subprocess overhead); falls back to
    the CLI via subprocess if the package's Python API is unavailable.

    Results are written to output_path/lm_eval_results.json (Python API) or
    the file lm_eval generates automatically (CLI mode).

    Args:
        model_path:     Path to a saved checkpoint or a HuggingFace model id.
        output_path:    Directory to write results JSON files.
        tasks:          Tasks to evaluate. Defaults to the full project suite:
                        [mmlu, hellaswag, arc_challenge, piqa, gsm8k].
        batch_size:     Per-device inference batch size.
        device:         Torch device string (e.g. "cuda", "cuda:0", "cpu").
        use_python_api: Attempt the Python API first; set False to always use
                        the CLI subprocess.

    Returns:
        Dict mapping task name to {"acc": float, "acc_stderr": float, "raw": dict}.
        acc is in the 0.0-1.0 range as returned by lm-eval.

    Raises:
        RuntimeError: If both the Python API and the CLI subprocess fail.
    """
    tasks = tasks or _DEFAULT_TASKS
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = str(model_path)

    if use_python_api:
        try:
            return _run_lm_eval_api(model_path, tasks, batch_size, device, output_path)
        except ImportError:
            logger.warning(
                "lm_eval Python API unavailable -- falling back to CLI subprocess."
            )

    return _run_lm_eval_cli(model_path, tasks, batch_size, device, output_path)


def _run_lm_eval_api(
    model_path: str,
    tasks: list[str],
    batch_size: int,
    device: str,
    output_path: Path,
) -> dict[str, dict]:
    """Run lm-eval via its Python API and save raw results to disk."""
    import lm_eval  # type: ignore[import]

    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=tasks,
        batch_size=batch_size,
        device=device,
    )

    results_file = output_path / "lm_eval_results.json"
    with results_file.open("w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("lm-eval results saved -> %s", results_file)

    return _parse_lm_eval_results(results.get("results", {}))


def _run_lm_eval_cli(
    model_path: str,
    tasks: list[str],
    batch_size: int,
    device: str,
    output_path: Path,
) -> dict[str, dict]:
    """Run lm-eval via CLI subprocess and parse the JSON it writes to output_path."""
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path}",
        "--tasks", ",".join(tasks),
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", str(output_path),
    ]

    logger.info("lm_eval CLI: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"lm_eval CLI failed (exit {proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # lm_eval writes one or more JSON files; pick the most recently modified.
    json_files = sorted(output_path.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not json_files:
        raise RuntimeError(
            f"lm_eval CLI exited 0 but wrote no JSON to {output_path}"
        )

    with json_files[-1].open() as f:
        raw = json.load(f)

    return _parse_lm_eval_results(raw.get("results", {}))


def _parse_lm_eval_results(results: dict) -> dict[str, dict]:
    """Normalise lm-eval output into a flat {task: {acc, acc_stderr, raw}} dict.

    lm-eval uses different key suffixes across versions (e.g. "acc,none" in
    v0.4+, plain "acc" in older releases, "acc_norm" for HellaSwag). This
    parser tries common variants in priority order so the rest of the codebase
    does not need to know which version of lm-eval is installed.
    """
    parsed: dict[str, dict] = {}

    for task, task_results in results.items():
        acc = next(
            (task_results[k] for k in _ACC_KEYS if k in task_results),
            None,
        )
        err = next(
            (task_results[k] for k in _ERR_KEYS if k in task_results),
            None,
        )
        parsed[task] = {"acc": acc, "acc_stderr": err, "raw": task_results}

    return parsed


# ---------------------------------------------------------------------------
# CLI entry point  (python -m src.utils.metrics)
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a quantized SmolLM2 checkpoint (Phase 3 pipeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Perplexity + KL divergence against saved FP32 logits
  python -m src.utils.metrics \\
      --model models/checkpoints/scheduled_qat_cosine_int4.pt \\
      --baseline-logits results/baseline/fp32_logits.pt \\
      --metrics perplexity kl_divergence \\
      --output results/scheduled_qat_cosine_int4/

  # Save FP32 baseline logits (run once in Phase 1)
  python -m src.utils.metrics \\
      --model HuggingFaceTB/SmolLM2-1.7B \\
      --metrics save_logits \\
      --output results/baseline/
""",
    )
    p.add_argument("--model", required=True, help="Checkpoint path or HF model id.")
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=["perplexity", "kl_divergence", "lm_eval", "save_logits"],
        default=["perplexity"],
        help="Which metrics to compute.",
    )
    p.add_argument(
        "--baseline-logits",
        default=None,
        help="Path to FP32 logits .pt file (required for kl_divergence).",
    )
    p.add_argument(
        "--output",
        default="results/metrics/",
        help="Directory to write metric results JSON.",
    )
    p.add_argument("--batch-size", type=int, default=8, help="DataLoader batch size.")
    p.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Sequences to use for KL divergence / save_logits.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=_DEFAULT_TASKS,
        help="lm-eval tasks (used with --metrics lm_eval).",
    )
    return p


def _cli_main() -> None:
    """CLI entry point: load model, run requested metrics, write JSON results."""
    import sys

    # Make project root importable when called as python -m src.utils.metrics
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from transformers import AutoModelForCausalLM  # type: ignore[import]

    from src.utils.config_loader import load_config
    from src.utils.data_loader import build_dataloaders, get_tokenizer, _load_and_chunk

    args = _build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    collected_results: dict = {}

    if "save_logits" in args.metrics:
        tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-1.7B", "models/base/")
        ds = _load_and_chunk(
            "wikitext-103-raw-v1", "train", tokenizer, 2048,
            num_samples=args.num_samples,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        save_fp32_logits(
            model, loader,
            output_path / "fp32_logits.pt",
            device,
            num_samples=args.num_samples,
        )
        collected_results["save_logits"] = {"path": str(output_path / "fp32_logits.pt")}

    if "perplexity" in args.metrics or "kl_divergence" in args.metrics:
        config_candidates = sorted(Path("configs").rglob("*.yaml")) if Path("configs").exists() else []
        if not config_candidates:
            logger.error("No YAML configs found in configs/. Cannot build dataloaders.")
        else:
            config = load_config(config_candidates[0])
            _, eval_loader = build_dataloaders(config, num_workers=0)

            if "perplexity" in args.metrics:
                logger.info("Computing perplexity on eval split...")
                ppl = compute_perplexity(model, eval_loader, device)
                collected_results["perplexity"] = ppl
                logger.info("Perplexity: %.4f", ppl)

            if "kl_divergence" in args.metrics:
                if not args.baseline_logits:
                    logger.error("--baseline-logits is required for kl_divergence.")
                else:
                    logger.info("Computing KL divergence...")
                    kld = compute_kl_divergence(
                        model, device,
                        baseline_logits_path=args.baseline_logits,
                        num_samples=args.num_samples,
                    )
                    collected_results["kl_divergence"] = kld
                    logger.info("KL divergence: %.6f", kld)

    if "lm_eval" in args.metrics:
        logger.info("Running lm-evaluation-harness on: %s", args.tasks)
        lm_results = run_lm_eval(
            args.model,
            output_path / "lm_eval",
            tasks=args.tasks,
            batch_size=args.batch_size,
            device=args.device,
        )
        collected_results["lm_eval"] = lm_results
        for task, r in lm_results.items():
            logger.info(
                "  %-20s acc=%.4f +/- %.4f",
                task, r["acc"] or 0.0, r["acc_stderr"] or 0.0,
            )

    summary_path = output_path / "metrics_summary.json"
    with summary_path.open("w") as f:
        json.dump(collected_results, f, indent=2, default=str)
    logger.info("Metrics summary -> %s", summary_path)


if __name__ == "__main__":
    _cli_main()
