"""
WikiText-103 data loader for scheduled-qat-for-slm experiments.

Loads WikiText-103 from HuggingFace datasets library, tokenizes using SmolLM2-1.7B's
GPT-2 BPE tokenizer, and chunks into fixed-length sequences for causal language modeling.

WikiText-103 splits (wikitext-103-raw-v1):
    train       ~103M tokens  — QAT training & PTQ calibration source
    validation   ~218K tokens  — loss monitoring during training (not used for final eval)
    test         ~246K tokens  — final perplexity evaluation (held out until reporting)

Tokenization strategy: all non-empty article texts are concatenated into one long token
stream then split into non-overlapping seq_length chunks. No padding is needed — every
chunk is exactly seq_length tokens. The trailing partial chunk is dropped.

Usage:
    from src.utils.data_loader import build_dataloaders, build_calibration_loader
    from src.utils.config_loader import load_config

    config = load_config("configs/standard_qat/qat_int4.yaml")
    train_loader, eval_loader = build_dataloaders(config)

    ptq_config = load_config("configs/ptq/ptq_int4.yaml")
    calib_loader = build_calibration_loader(ptq_config)
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils.config_loader import ExperimentConfig


_HF_DATASET = "wikitext"


# ---------------------------------------------------------------------------
# Internal dataset wrapper
# ---------------------------------------------------------------------------

class _ChunkDataset(TorchDataset):
    """Wraps a HuggingFace dataset of tokenized chunks.

    Each item returns:
        input_ids      (seq_length,)  — token ids as long tensor
        attention_mask (seq_length,)  — all ones (no padding; chunks are complete)
        labels         (seq_length,)  — copy of input_ids (causal LM objective)
    """

    def __init__(self, hf_dataset) -> None:
        # hf_dataset has set_format("torch") applied — indexing returns tensors
        self._ds = hf_dataset

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids: torch.Tensor = self._ds[idx]["input_ids"]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


# ---------------------------------------------------------------------------
# Core loading logic
# ---------------------------------------------------------------------------

def get_tokenizer(model_name: str, cache_dir: str) -> PreTrainedTokenizerBase:
    """Load SmolLM2-1.7B's GPT-2 BPE tokenizer (vocab size 49152).

    Sets pad_token = eos_token so the tokenizer can be used for batch encoding
    even though padding is not used during training (chunks are always full-length).

    Args:
        model_name: HuggingFace model identifier (e.g. "HuggingFaceTB/SmolLM2-1.7B").
        cache_dir:  Local path to cache downloaded tokenizer files.

    Returns:
        Loaded PreTrainedTokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_and_chunk(
    dataset_config: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    seq_length: int,
    num_samples: Optional[int] = None,
) -> _ChunkDataset:
    """Load a WikiText-103 split, tokenize, and chunk into fixed-length sequences.

    Uses HuggingFace dataset.map() with batched processing, which automatically
    caches tokenized results to disk — repeated calls with the same arguments are
    fast (reads from cache instead of re-tokenizing).

    Args:
        dataset_config: HuggingFace dataset config name (e.g. "wikitext-103-raw-v1").
        split:          Split name: "train", "validation", or "test".
        tokenizer:      Tokenizer to encode text.
        seq_length:     Number of tokens per output chunk.
        num_samples:    If set, return at most this many chunks (PTQ calibration).

    Returns:
        _ChunkDataset ready to wrap in a DataLoader.
    """
    raw = load_dataset(_HF_DATASET, dataset_config, split=split)

    # WikiText-103 raw has many blank lines between article sections; filter them out
    # so they don't produce near-empty tokenized sequences.
    raw = raw.filter(
        lambda example: len(example["text"].strip()) > 0,
        desc=f"Filtering empty lines ({split})",
    )

    # Tokenize all texts. add_special_tokens=False avoids inserting BOS/EOS between
    # articles — the concatenated stream represents continuous text, not discrete items.
    tokenized = raw.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
        ),
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {split}",
    )

    # Concatenate all token sequences and split into non-overlapping seq_length chunks.
    # The trailing partial chunk is dropped to keep all sequences the same length.
    def _group_texts(examples: dict) -> dict:
        all_ids = sum(examples["input_ids"], [])
        total = (len(all_ids) // seq_length) * seq_length
        return {
            "input_ids": [
                all_ids[i : i + seq_length] for i in range(0, total, seq_length)
            ]
        }

    chunked = tokenized.map(
        _group_texts,
        batched=True,
        desc=f"Chunking to {seq_length}-token sequences ({split})",
    )

    if num_samples is not None:
        chunked = chunked.select(range(min(num_samples, len(chunked))))

    # Return tensors directly when indexing (avoids per-item list→tensor conversion)
    chunked.set_format(type="torch", columns=["input_ids"])

    return _ChunkDataset(chunked)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dataloaders(
    config: ExperimentConfig,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Build train and eval DataLoaders for QAT / LoRA-QAT experiments.

    Eval loader uses the config's eval_split (typically "test") — the held-out
    split reserved for final perplexity reporting. Use build_validation_loader()
    for the "validation" split monitored during training.

    Args:
        config:      Parsed experiment config. Must have .data and .training sections.
        num_workers: DataLoader worker processes for parallel prefetching.

    Returns:
        (train_loader, eval_loader) tuple.

    Raises:
        ValueError: If config is missing required .data or .training sections.
    """
    if config.data is None:
        raise ValueError(f"method={config.method!r} config has no 'data' section")
    if config.training is None:
        raise ValueError(f"method={config.method!r} config has no 'training' section")

    tokenizer = get_tokenizer(config.model.name, config.model.cache_dir)
    seq_length = config.data.seq_length
    batch_size = config.training.batch_size

    train_ds = _load_and_chunk(
        config.data.train_dataset,
        config.data.train_split,
        tokenizer,
        seq_length,
    )
    eval_ds = _load_and_chunk(
        config.data.eval_dataset,
        config.data.eval_split,
        tokenizer,
        seq_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,   # keeps all training batches the same size
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,  # evaluate on every token
    )

    return train_loader, eval_loader


def build_validation_loader(
    config: ExperimentConfig,
    num_workers: int = 2,
) -> DataLoader:
    """Build a DataLoader for the WikiText-103 validation split.

    The validation split (218K tokens) is used to monitor loss/perplexity during
    training for early stopping decisions. It is separate from the test split used
    for final evaluation, preventing any leakage from hyperparameter tuning.

    Args:
        config:      Parsed experiment config. Must have .data and .training sections.
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader over the "validation" split.

    Raises:
        ValueError: If config is missing required .data or .training sections.
    """
    if config.data is None:
        raise ValueError(f"method={config.method!r} config has no 'data' section")
    if config.training is None:
        raise ValueError(f"method={config.method!r} config has no 'training' section")

    tokenizer = get_tokenizer(config.model.name, config.model.cache_dir)

    val_ds = _load_and_chunk(
        config.data.train_dataset,  # same HuggingFace dataset, different split
        "validation",
        tokenizer,
        config.data.seq_length,
    )

    return DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_calibration_loader(
    config: ExperimentConfig,
    num_workers: int = 2,
) -> DataLoader:
    """Build a DataLoader for PTQ calibration.

    Loads a small, fixed subset of the training split (config.calibration.num_samples
    sequences) for computing per-channel quantization scales and zero-points.

    Calibration uses batch_size=1 to accumulate activation statistics one sequence
    at a time, matching how PTQ calibration algorithms (min/max, percentile) collect
    range statistics without GPU memory pressure.

    Note on calibration overfitting: calibration and evaluation both draw from
    WikiText-103, but from different splits (train vs. test). Cross-domain benchmarks
    (MMLU, HellaSwag, ARC, PIQA, GSM8K) provide an unbiased quality signal.

    Args:
        config:      Parsed PTQ experiment config. Must have .calibration section.
        num_workers: DataLoader worker processes.

    Returns:
        DataLoader over the calibration subset (no shuffle — deterministic order).

    Raises:
        ValueError: If config is missing the .calibration section.
    """
    if config.calibration is None:
        raise ValueError(f"method={config.method!r} config has no 'calibration' section")

    cal = config.calibration
    tokenizer = get_tokenizer(config.model.name, config.model.cache_dir)

    calib_ds = _load_and_chunk(
        cal.dataset,
        cal.split,
        tokenizer,
        cal.seq_length,
        num_samples=cal.num_samples,
    )

    return DataLoader(
        calib_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
