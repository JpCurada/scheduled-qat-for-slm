"""
PyTorch checkpoint → GGUF export pipeline for edge deployment.

Converts trained QAT/PTQ/LoRA models to GGUF format for on-device inference
via llama.cpp on Android, iOS, and Raspberry Pi.

Why two steps?
--------------
GGUF is llama.cpp's native format. The conversion pipeline always goes through
an intermediate HuggingFace-format directory because:
  1. `convert_hf_to_gguf.py` (from llama.cpp) reads the HF format directly.
  2. A lossless F16 GGUF intermediate preserves full precision, then
     `llama-quantize` quantizes it. Splitting the steps means the F16 GGUF
     can be re-quantized to Q4 or Q8 without re-running the slower conversion.

Checkpoint formats per method
------------------------------
  standard_qat / scheduled_qat  ->  .pt file (state dict + controller meta)
  lora_qat                       ->  adapter directory (PEFT save_pretrained)
  ptq                            ->  .pt file (QuantizedLinear state dict)
  baseline (FP32)                ->  no checkpoint; export directly from HF Hub

Reconstruction per method
--------------------------
  standard_qat / scheduled_qat:
      Load FP32 model → re-inject FakeQuantizeLinear (matching checkpoint shape)
      → load state dict → remove_fake_quantize() → plain nn.Linear with trained
      FP32 shadow weights ready for save_pretrained().

  lora_qat:
      Load base model → PeftModel.from_pretrained(adapter_dir)
      → merge_and_unload() → plain nn.Module.

  ptq:
      Load FP32 model → apply_ptq() (matching checkpoint structure)
      → load state dict → _dequantize_ptq_layers() → plain nn.Linear with
      dequantized FP32 weights (slightly grid-rounded vs originals).

  baseline:
      Load from HF Hub directly; no checkpoint reconstruction needed.

GGUF type mapping (bits → GGUF quantization format)
-----------------------------------------------------
  4   →  Q4_K_M  (best quality 4-bit, mixed precision)
  8   →  Q8_0    (8-bit, near-lossless)
  16  →  F16     (half precision, no quantization loss)
  32  →  F32     (full precision, largest file)

Finding llama.cpp tools
------------------------
The pipeline auto-detects:
  1. convert_hf_to_gguf.py — searches llama_cpp package dir, ./llama.cpp/,
     ../llama.cpp/, ~/llama.cpp/, and any user-specified --llama-cpp-dir.
  2. llama-quantize binary — searches PATH, build dirs near the script, and
     any user-specified --llama-cpp-dir.

If neither tool is found, the pipeline saves the HF-format directory and raises
a RuntimeError with instructions to run the steps manually.

Public API
----------
  GGUFExportResult                        -- export summary dataclass
  export_qat_checkpoint(...)              -- standard_qat / scheduled_qat .pt
  export_lora_adapter(...)                -- lora_qat adapter directory
  export_ptq_checkpoint(...)              -- ptq .pt checkpoint
  export_hf_model(...)                    -- baseline FP32 or any HF model
  export_from_config(config, ckpt, ...)   -- config-driven dispatcher
  find_llama_cpp_tools(llama_cpp_dir)     -- locate convert script + binary

Usage
-----
  python -m src.utils.export \\
      --checkpoint models/checkpoints/scheduled_qat_cosine_int4/final.pt \\
      --method scheduled_qat \\
      --target-bits 4 \\
      --output models/gguf/

  python -m src.utils.export \\
      --checkpoint models/checkpoints/lora_qat_int4/final_adapter \\
      --method lora_qat \\
      --target-bits 4 \\
      --output models/gguf/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default model identifier (all experiments use SmolLM2-1.7B).
_DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-1.7B"
_DEFAULT_CACHE_DIR = "models/base/"

# Map from bit-width to GGUF quantization type string.
# Q4_K_M is the highest-quality 4-bit format in llama.cpp (mixed k-quants).
# Q8_0 is the standard 8-bit format, essentially lossless for LLMs.
_BITS_TO_GGUF_TYPE: dict[int, str] = {
    4:  "Q4_K_M",
    8:  "Q8_0",
    16: "F16",
    32: "F32",
}

# All llama-quantize type strings accepted in the CLI --gguf-type argument.
_VALID_GGUF_TYPES = frozenset({
    "F32", "F16", "BF16",
    "Q8_0",
    "Q6_K",
    "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1",
    "Q4_K_M", "Q4_K_S", "Q4_0", "Q4_1",
    "Q3_K_M", "Q3_K_S", "Q3_K_L",
    "Q2_K",
    "IQ4_XS", "IQ3_XXS",
})

# Candidate script names for the llama.cpp HF→GGUF converter (in priority order).
_CONVERT_SCRIPT_NAMES = [
    "convert_hf_to_gguf.py",  # current llama.cpp (post-2024-02)
    "convert.py",              # legacy llama.cpp name
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GGUFExportResult:
    """Summary of a completed GGUF export operation.

    Attributes:
        gguf_path:            Absolute path to the final .gguf file.
        method:               Quantization method the checkpoint came from.
        source_bits:          Original QAT/PTQ bit-width (4 or 8).
        gguf_type:            GGUF quantization format (e.g. "Q4_K_M").
        file_size_gb:         Size of the .gguf file in gigabytes.
        export_time_seconds:  Total wall-clock time for the export.
        hf_format_dir:        Intermediate HuggingFace-format directory
                              (populated only if keep_hf_dir=True).
        f16_gguf_path:        Intermediate F16 GGUF path (populated only
                              when a two-step convert+quantize was performed).
    """
    gguf_path: Path
    method: str
    source_bits: int
    gguf_type: str
    file_size_gb: float
    export_time_seconds: float
    hf_format_dir: Optional[Path] = None
    f16_gguf_path: Optional[Path] = None

    def summary(self) -> str:
        lines = [
            f"GGUF Export complete:",
            f"  output:      {self.gguf_path}",
            f"  method:      {self.method}",
            f"  source bits: INT{self.source_bits}",
            f"  gguf type:   {self.gguf_type}",
            f"  file size:   {self.file_size_gb:.3f} GB",
            f"  time:        {self.export_time_seconds:.1f}s",
        ]
        if self.hf_format_dir:
            lines.append(f"  hf format:   {self.hf_format_dir}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool detection
# ---------------------------------------------------------------------------

@dataclass
class LlamaCppTools:
    """Paths to the two llama.cpp tools needed for GGUF export.

    Attributes:
        convert_script:  Path to convert_hf_to_gguf.py (or convert.py).
                         None if not found.
        quantize_binary: Path to the llama-quantize executable.
                         None if not found.
    """
    convert_script: Optional[Path] = None
    quantize_binary: Optional[Path] = None

    @property
    def can_convert(self) -> bool:
        return self.convert_script is not None

    @property
    def can_quantize(self) -> bool:
        return self.quantize_binary is not None

    def describe(self) -> str:
        c = str(self.convert_script) if self.convert_script else "NOT FOUND"
        q = str(self.quantize_binary) if self.quantize_binary else "NOT FOUND"
        return f"convert_hf_to_gguf: {c}\nllama-quantize: {q}"


def find_llama_cpp_tools(llama_cpp_dir: Optional[str] = None) -> LlamaCppTools:
    """Locate the llama.cpp convert script and quantize binary.

    Searched in this order:
        1. ``llama_cpp_dir`` (user-specified path).
        2. ``llama_cpp`` Python package installation directory and its parents.
        3. Common local relative paths: ``./llama.cpp/``, ``../llama.cpp/``.
        4. The user's home directory: ``~/llama.cpp/``.
        5. System PATH (for llama-quantize binary only).

    Args:
        llama_cpp_dir: Optional base directory of a local llama.cpp checkout
                       or installation (e.g. ``/opt/llama.cpp``).

    Returns:
        LlamaCppTools with whichever tools were found (None if not found).
    """
    tools = LlamaCppTools()

    # --- Build candidate search directories ---
    candidate_dirs: list[Path] = []

    if llama_cpp_dir:
        candidate_dirs.append(Path(llama_cpp_dir))

    # llama_cpp Python package directory
    try:
        import llama_cpp as _llama_pkg  # type: ignore[import]
        pkg_dir = Path(_llama_pkg.__file__).parent
        candidate_dirs.extend([
            pkg_dir,
            pkg_dir.parent,
            pkg_dir / "scripts",
        ])
    except ImportError:
        pass

    # Common local paths relative to the working directory
    candidate_dirs.extend([
        Path("llama.cpp"),
        Path("../llama.cpp"),
        Path.home() / "llama.cpp",
    ])

    # --- Locate convert_hf_to_gguf.py ---
    for directory in candidate_dirs:
        for script_name in _CONVERT_SCRIPT_NAMES:
            candidate = directory / script_name
            if candidate.exists():
                tools.convert_script = candidate.resolve()
                logger.debug("Found convert script: %s", tools.convert_script)
                break
        if tools.convert_script:
            break

    # --- Locate llama-quantize binary ---
    # 1. System PATH
    quantize_in_path = shutil.which("llama-quantize")
    if quantize_in_path:
        tools.quantize_binary = Path(quantize_in_path)
    else:
        # 2. Build directories near the convert script or candidate dirs
        binary_names = ["llama-quantize", "llama-quantize.exe", "quantize"]
        search_dirs: list[Path] = list(candidate_dirs)
        if tools.convert_script:
            script_parent = tools.convert_script.parent
            search_dirs.extend([
                script_parent / "build" / "bin",
                script_parent / "build",
                script_parent,
            ])
        for directory in search_dirs:
            for name in binary_names:
                candidate = directory / name
                if candidate.exists() and candidate.is_file():
                    tools.quantize_binary = candidate.resolve()
                    logger.debug("Found quantize binary: %s", tools.quantize_binary)
                    break
            if tools.quantize_binary:
                break

    return tools


# ---------------------------------------------------------------------------
# Model reconstruction helpers
# ---------------------------------------------------------------------------

def _load_fp32_base(
    model_name: str,
    cache_dir: str,
    device: torch.device,
) -> nn.Module:
    """Load SmolLM2-1.7B FP32 weights from HuggingFace Hub / local cache."""
    from transformers import AutoModelForCausalLM  # type: ignore[import]

    logger.info("Loading FP32 base model: %s ...", model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
    )
    model.to(device)
    model.eval()
    return model


def _reconstruct_qat_model(
    checkpoint_path: Path,
    model_name: str,
    cache_dir: str,
    target_bits: int,
    device: torch.device,
) -> nn.Module:
    """Reconstruct plain FP32 model from a standard_qat / scheduled_qat checkpoint.

    The checkpoint stores FP32 shadow weights in FakeQuantizeLinear layers.
    Reconstruction:
        1. Load base architecture from HuggingFace.
        2. Re-inject FakeQuantizeLinear (matching the checkpoint's structure).
        3. Load state dict — the trained shadow weights populate the layers.
        4. Remove fake quant — restores FakeQuantizeLinear → nn.Linear while
           keeping the trained FP32 weights as the layer's weight parameter.

    The resulting model has the same architecture as the original pretrained
    SmolLM2-1.7B but with QAT-adapted weights that are more robust to the
    quantization noise that will be applied during GGUF export.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model_name:      HuggingFace model identifier (for architecture).
        cache_dir:       Local cache directory for base model weights.
        target_bits:     Bit-width used during QAT training (4 or 8).
        device:          Device to load tensors onto.

    Returns:
        nn.Module in eval mode with trained FP32 weights and no fake quant nodes.
    """
    from src.quantization.fake_quantize import (  # noqa: PLC0415
        inject_fake_quantize,
        remove_fake_quantize,
    )

    model = _load_fp32_base(model_name, cache_dir, device)

    # Re-inject FakeQuantizeLinear to match the checkpoint's layer structure.
    inject_fake_quantize(model, bits=target_bits)

    logger.info("Loading QAT checkpoint: %s ...", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore nn.Linear layers, keeping the trained FP32 shadow weights.
    n_removed = remove_fake_quantize(model)
    logger.info("Removed %d FakeQuantizeLinear layers → plain nn.Linear", n_removed)

    return model


def _reconstruct_lora_model(
    adapter_dir: Path,
    model_name: str,
    cache_dir: str,
    device: torch.device,
) -> nn.Module:
    """Reconstruct plain FP32 model from a LoRA-QAT adapter directory.

    Loads the quantized-and-frozen base model, attaches the trained LoRA
    adapters, then calls merge_and_unload() to absorb the adapters into
    the base weights. The result is a standard nn.Module at FP32 precision.

    Note: the merged weights are a combination of the dequantized base weights
    and the low-rank adapter matrices. The base weights were quantized to
    ``target_bits`` and frozen during training — the LoRA adapters compensate
    for the quantization error. After merging, the combined FP32 weights are
    more accurate than the quantized-only base.

    Args:
        adapter_dir: Path to the PEFT adapter directory (from save_lora_checkpoint).
        model_name:  HuggingFace model identifier for the base architecture.
        cache_dir:   Local cache directory for base model weights.
        device:      Device to load tensors onto.

    Returns:
        Merged nn.Module in eval mode with FP32 weights.
    """
    from peft import PeftModel  # type: ignore[import]

    base_model = _load_fp32_base(model_name, cache_dir, device)

    logger.info("Loading LoRA adapter: %s ...", adapter_dir)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    logger.info("Merging LoRA adapters into base model ...")
    merged = peft_model.merge_and_unload()
    merged.eval()

    logger.info("LoRA merge complete.")
    return merged


def _reconstruct_ptq_model(
    checkpoint_path: Path,
    model_name: str,
    cache_dir: str,
    target_bits: int,
    device: torch.device,
) -> nn.Module:
    """Reconstruct plain FP32 model from a PTQ checkpoint.

    PTQ weights are stored as int8 integers + per-channel float32 scales in
    QuantizedLinear buffers. Reconstruction dequantizes each layer:
        dequantized_weight = weight_int.float() * scale

    The dequantized weights are slightly different from the original FP32
    weights (they have been rounded to the quantization grid), but this is
    intentional — they represent the PTQ model's effective weight values
    and will be re-quantized by llama.cpp's GGUF converter.

    Args:
        checkpoint_path: Path to the .pt PTQ checkpoint file.
        model_name:      HuggingFace model identifier.
        cache_dir:       Local cache directory for base model weights.
        target_bits:     Bit-width used during PTQ (4 or 8).
        device:          Device to load tensors onto.

    Returns:
        nn.Module in eval mode with dequantized FP32 weights.
    """
    from src.quantization.ptq import (  # noqa: PLC0415
        QuantizedLinear,
        apply_ptq,
    )

    model = _load_fp32_base(model_name, cache_dir, device)

    # Re-apply PTQ to create QuantizedLinear layers matching the checkpoint.
    apply_ptq(model, bits=target_bits)

    logger.info("Loading PTQ checkpoint: %s ...", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Dequantize in-place: replace every QuantizedLinear with a plain nn.Linear
    # carrying the dequantized FP32 weight.
    n_dequantized = _dequantize_ptq_layers(model)
    logger.info("Dequantized %d PTQ layers → plain nn.Linear (FP32)", n_dequantized)

    return model


def _dequantize_ptq_layers(model: nn.Module) -> int:
    """Replace every QuantizedLinear in-place with a plain nn.Linear (FP32).

    Reads weight_int (int8 buffer) and scale (float32 buffer), computes
    dequantized weight = weight_int.float() * scale, and swaps the module.

    Args:
        model: nn.Module containing QuantizedLinear layers (modified in-place).

    Returns:
        Number of layers that were dequantized.
    """
    from src.quantization.ptq import QuantizedLinear  # noqa: PLC0415

    count = 0
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, QuantizedLinear):
                continue

            # Dequantize: float(int8) * scale → FP32 weight
            dq_weight = child_module.dequantized_weight()  # (out, in)

            linear = nn.Linear(
                child_module.in_features,
                child_module.out_features,
                bias=child_module.bias is not None,
            )
            with torch.no_grad():
                linear.weight.copy_(dq_weight)
                if child_module.bias is not None:
                    linear.bias.copy_(child_module.bias)

            setattr(parent_module, child_name, linear)
            count += 1

    return count


# ---------------------------------------------------------------------------
# HuggingFace format persistence
# ---------------------------------------------------------------------------

def _save_hf_format(
    model: nn.Module,
    model_name: str,
    cache_dir: str,
    output_dir: Path,
) -> Path:
    """Save a reconstructed model as HuggingFace format for GGUF conversion.

    Writes:
        output_dir/
            config.json          -- model architecture (from original HF model)
            tokenizer.json       -- GPT-2 BPE tokenizer
            tokenizer_config.json
            special_tokens_map.json
            model.safetensors    -- trained FP32 weights (or sharded if large)

    The HuggingFace format is consumed by ``convert_hf_to_gguf.py`` from
    llama.cpp. The tokenizer is loaded from the original model name (not the
    checkpoint) to ensure vocab consistency.

    Args:
        model:      Plain nn.Module with save_pretrained() (loaded via
                    AutoModelForCausalLM originally).
        model_name: HuggingFace model identifier (for tokenizer).
        cache_dir:  Local cache for tokenizer download.
        output_dir: Directory to write HuggingFace format files.

    Returns:
        Path to output_dir (same as input, for chaining).
    """
    from transformers import AutoTokenizer  # type: ignore[import]

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model in HuggingFace format -> %s ...", output_dir)

    # Save model weights and architecture config.
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # Save tokenizer from the original model (not the checkpoint).
    logger.info("Saving tokenizer (from %s) ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.save_pretrained(str(output_dir))

    size_gb = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) / 1e9
    logger.info("HuggingFace format saved: %.2f GB", size_gb)

    return output_dir


# ---------------------------------------------------------------------------
# GGUF conversion (subprocess calls to llama.cpp tools)
# ---------------------------------------------------------------------------

def _convert_hf_to_f16_gguf(
    hf_dir: Path,
    output_path: Path,
    convert_script: Path,
) -> Path:
    """Run convert_hf_to_gguf.py to produce an F16 GGUF file.

    Always converts to F16 first (lossless representation of FP32 weights in
    GGUF format). The quantize step (if requested) runs separately afterward.
    This two-step approach lets the F16 GGUF be re-quantized to multiple bit
    widths without re-running the slower model conversion.

    Args:
        hf_dir:         HuggingFace-format model directory.
        output_path:    Destination path for the F16 .gguf file.
        convert_script: Absolute path to convert_hf_to_gguf.py.

    Returns:
        Path to the written F16 .gguf file.

    Raises:
        RuntimeError: If the script exits with a non-zero code.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(convert_script),
        str(hf_dir),
        "--outfile", str(output_path),
        "--outtype", "f16",
    ]

    logger.info("Converting HF → F16 GGUF:")
    logger.info("  %s", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed (exit {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"convert_hf_to_gguf.py exited 0 but did not create {output_path}.\n"
            f"STDOUT:\n{proc.stdout}"
        )

    size_gb = output_path.stat().st_size / 1e9
    logger.info("F16 GGUF written: %s (%.2f GB)", output_path, size_gb)
    return output_path


def _quantize_gguf(
    f16_path: Path,
    output_path: Path,
    gguf_type: str,
    quantize_binary: Path,
) -> Path:
    """Run llama-quantize to convert an F16 GGUF to the target quantization type.

    Args:
        f16_path:        Path to the input F16 GGUF file.
        output_path:     Destination path for the quantized GGUF.
        gguf_type:       llama-quantize type string (e.g. "Q4_K_M", "Q8_0").
        quantize_binary: Absolute path to the llama-quantize executable.

    Returns:
        Path to the quantized .gguf file.

    Raises:
        RuntimeError: If the binary exits with a non-zero code.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(quantize_binary),
        str(f16_path),
        str(output_path),
        gguf_type,
    ]

    logger.info("Quantizing F16 GGUF → %s:", gguf_type)
    logger.info("  %s", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed (exit {proc.returncode}).\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"llama-quantize exited 0 but did not create {output_path}.\n"
            f"STDOUT:\n{proc.stdout}"
        )

    size_gb = output_path.stat().st_size / 1e9
    logger.info("%s GGUF written: %s (%.2f GB)", gguf_type, output_path, size_gb)
    return output_path


# ---------------------------------------------------------------------------
# Core conversion pipeline
# ---------------------------------------------------------------------------

def _run_export_pipeline(
    model: nn.Module,
    model_name: str,
    cache_dir: str,
    output_dir: Path,
    experiment_name: str,
    gguf_type: str,
    tools: LlamaCppTools,
    keep_hf_dir: bool = False,
    keep_f16_gguf: bool = True,
) -> GGUFExportResult:
    """Convert a reconstructed FP32 model to GGUF with the specified quant type.

    Pipeline:
        1. Save as HuggingFace format (temp dir or output_dir/hf/).
        2. Run convert_hf_to_gguf.py → F16 GGUF.
        3. If gguf_type != F16: run llama-quantize → quantized GGUF.
        4. Clean up intermediates if not keeping them.

    Args:
        model:           Plain FP32 nn.Module (output of a _reconstruct_* function).
        model_name:      HuggingFace model id (for tokenizer).
        cache_dir:       Local model cache.
        output_dir:      Directory to write the final .gguf file.
        experiment_name: Name stem for output file (e.g. "standard_qat_int4").
        gguf_type:       Target GGUF quantization (e.g. "Q4_K_M").
        tools:           LlamaCppTools from find_llama_cpp_tools().
        keep_hf_dir:     Retain the intermediate HuggingFace format directory.
        keep_f16_gguf:   Retain the intermediate F16 GGUF (useful for re-quantizing).

    Returns:
        GGUFExportResult with paths, size, and timing.

    Raises:
        RuntimeError: If the convert script is not found, or any subprocess fails.
    """
    if not tools.can_convert:
        raise RuntimeError(
            "convert_hf_to_gguf.py was not found. Cannot produce GGUF output.\n\n"
            "Options:\n"
            "  1. Clone llama.cpp and pass --llama-cpp-dir /path/to/llama.cpp\n"
            "  2. Install llama-cpp-python: pip install llama-cpp-python\n"
            "  3. Run manually:\n"
            f"     python convert_hf_to_gguf.py <hf_dir> --outfile <model.gguf> --outtype f16\n"
            f"     llama-quantize <model_f16.gguf> <model_{gguf_type.lower()}.gguf> {gguf_type}\n\n"
            f"Detected tools:\n{tools.describe()}"
        )

    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    gguf_type_upper = gguf_type.upper()
    hf_format_dir: Optional[Path] = None
    f16_gguf_path: Optional[Path] = None

    with tempfile.TemporaryDirectory(prefix="sqat_hf_") as tmp_str:
        tmp_dir = Path(tmp_str)

        # --- Step 1: Save HuggingFace format ---
        hf_dir = (output_dir / "hf" / experiment_name) if keep_hf_dir else (tmp_dir / "hf")
        _save_hf_format(model, model_name, cache_dir, hf_dir)
        if keep_hf_dir:
            hf_format_dir = hf_dir

        # --- Step 2: Convert HF → F16 GGUF ---
        f16_name = f"{experiment_name}_f16.gguf"
        f16_dest = output_dir / f16_name if keep_f16_gguf else tmp_dir / f16_name

        _convert_hf_to_f16_gguf(hf_dir, f16_dest, tools.convert_script)
        if keep_f16_gguf:
            f16_gguf_path = f16_dest

        # --- Step 3: Quantize F16 → target type (if not already F16) ---
        if gguf_type_upper == "F16":
            final_gguf = f16_dest if keep_f16_gguf else output_dir / f16_name
            if not keep_f16_gguf:
                shutil.move(str(f16_dest), str(final_gguf))
            f16_gguf_path = None  # the final file IS the F16, don't double-report it
        else:
            if not tools.can_quantize:
                raise RuntimeError(
                    f"llama-quantize binary was not found. Cannot quantize to {gguf_type}.\n\n"
                    f"The F16 GGUF has been written to:\n  {f16_dest}\n\n"
                    f"Run manually:\n"
                    f"  llama-quantize {f16_dest} <output>.gguf {gguf_type}\n\n"
                    f"Detected tools:\n{tools.describe()}"
                )

            final_name = f"{experiment_name}_{gguf_type.lower()}.gguf"
            final_gguf = output_dir / final_name
            _quantize_gguf(f16_dest, final_gguf, gguf_type_upper, tools.quantize_binary)

    file_size_gb = final_gguf.stat().st_size / 1e9
    elapsed = time.time() - t0

    result = GGUFExportResult(
        gguf_path=final_gguf.resolve(),
        method="",           # caller fills this in
        source_bits=0,       # caller fills this in
        gguf_type=gguf_type_upper,
        file_size_gb=round(file_size_gb, 3),
        export_time_seconds=round(elapsed, 1),
        hf_format_dir=hf_format_dir,
        f16_gguf_path=f16_gguf_path,
    )
    logger.info(result.summary())
    return result


# ---------------------------------------------------------------------------
# Public export functions
# ---------------------------------------------------------------------------

def export_qat_checkpoint(
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = _DEFAULT_MODEL,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    target_bits: int = 4,
    gguf_type: Optional[str] = None,
    llama_cpp_dir: Optional[str] = None,
    device_str: str = "cpu",
    keep_hf_dir: bool = False,
    keep_f16_gguf: bool = True,
) -> GGUFExportResult:
    """Export a standard_qat or scheduled_qat checkpoint to GGUF.

    Reconstructs trained FP32 shadow weights from the checkpoint by removing
    fake quantization nodes, then converts to GGUF via llama.cpp.

    Args:
        checkpoint_path: Path to the .pt checkpoint file (final.pt or step_N.pt).
        output_dir:      Directory to write the .gguf file.
        model_name:      HuggingFace model identifier for architecture + tokenizer.
        cache_dir:       Local cache for base model weights.
        target_bits:     QAT bit-width (4 or 8). Used to determine GGUF type if
                         gguf_type is None.
        gguf_type:       Override GGUF quantization type (e.g. "Q5_K_M").
                         Default: "Q4_K_M" for 4-bit, "Q8_0" for 8-bit.
        llama_cpp_dir:   Path to llama.cpp checkout for tool detection.
        device_str:      Device for model loading ("cpu" recommended for export).
        keep_hf_dir:     Keep intermediate HuggingFace format directory.
        keep_f16_gguf:   Keep intermediate F16 GGUF for later re-quantization.

    Returns:
        GGUFExportResult with final .gguf path and export metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    device = torch.device(device_str)

    resolved_gguf_type = gguf_type or _BITS_TO_GGUF_TYPE.get(target_bits, "Q4_K_M")
    tools = find_llama_cpp_tools(llama_cpp_dir)

    logger.info(
        "Exporting QAT checkpoint: %s → %s (type=%s)",
        checkpoint_path, output_dir, resolved_gguf_type,
    )

    # Detect method from checkpoint keys.
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "sqat_controller" in raw_ckpt:
        method = "scheduled_qat"
    elif "qat_controller" in raw_ckpt:
        method = "standard_qat"
    else:
        method = "qat"  # generic fallback
    del raw_ckpt

    model = _reconstruct_qat_model(
        checkpoint_path, model_name, cache_dir, target_bits, device,
    )

    # Derive experiment name from checkpoint path stem.
    experiment_name = checkpoint_path.parent.name or f"{method}_int{target_bits}"

    result = _run_export_pipeline(
        model=model,
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        gguf_type=resolved_gguf_type,
        tools=tools,
        keep_hf_dir=keep_hf_dir,
        keep_f16_gguf=keep_f16_gguf,
    )
    result.method = method
    result.source_bits = target_bits
    return result


def export_lora_adapter(
    adapter_dir: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = _DEFAULT_MODEL,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    target_bits: int = 4,
    gguf_type: Optional[str] = None,
    llama_cpp_dir: Optional[str] = None,
    device_str: str = "cpu",
    keep_hf_dir: bool = False,
    keep_f16_gguf: bool = True,
) -> GGUFExportResult:
    """Export a lora_qat adapter directory to GGUF.

    Merges LoRA adapters into the base model weights via merge_and_unload(),
    producing a single plain FP32 model, then converts to GGUF.

    Args:
        adapter_dir:   Path to the PEFT adapter directory (from save_lora_checkpoint).
        output_dir:    Directory to write the .gguf file.
        model_name:    HuggingFace model identifier.
        cache_dir:     Local cache for base model weights.
        target_bits:   LoRA-QAT bit-width (determines default GGUF type).
        gguf_type:     Override GGUF quantization type.
        llama_cpp_dir: Path to llama.cpp checkout.
        device_str:    Device for model loading.
        keep_hf_dir:   Keep intermediate HuggingFace format directory.
        keep_f16_gguf: Keep intermediate F16 GGUF.

    Returns:
        GGUFExportResult with final .gguf path and export metadata.
    """
    adapter_dir = Path(adapter_dir)
    output_dir = Path(output_dir)
    device = torch.device(device_str)

    resolved_gguf_type = gguf_type or _BITS_TO_GGUF_TYPE.get(target_bits, "Q4_K_M")
    tools = find_llama_cpp_tools(llama_cpp_dir)

    logger.info(
        "Exporting LoRA adapter: %s → %s (type=%s)",
        adapter_dir, output_dir, resolved_gguf_type,
    )

    model = _reconstruct_lora_model(adapter_dir, model_name, cache_dir, device)
    experiment_name = adapter_dir.name or f"lora_qat_int{target_bits}"

    result = _run_export_pipeline(
        model=model,
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        gguf_type=resolved_gguf_type,
        tools=tools,
        keep_hf_dir=keep_hf_dir,
        keep_f16_gguf=keep_f16_gguf,
    )
    result.method = "lora_qat"
    result.source_bits = target_bits
    return result


def export_ptq_checkpoint(
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_name: str = _DEFAULT_MODEL,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    target_bits: int = 4,
    gguf_type: Optional[str] = None,
    llama_cpp_dir: Optional[str] = None,
    device_str: str = "cpu",
    keep_hf_dir: bool = False,
    keep_f16_gguf: bool = True,
) -> GGUFExportResult:
    """Export a PTQ checkpoint to GGUF.

    Dequantizes int8 QuantizedLinear layers back to FP32, then converts to
    GGUF. The dequantized weights are slightly rounded compared to the original
    pretrained weights — they represent the PTQ model's effective weight values
    that will be re-quantized by llama.cpp's converter.

    Args:
        checkpoint_path: Path to the .pt PTQ checkpoint file.
        output_dir:      Directory to write the .gguf file.
        model_name:      HuggingFace model identifier.
        cache_dir:       Local cache for base model weights.
        target_bits:     PTQ bit-width (4 or 8).
        gguf_type:       Override GGUF quantization type.
        llama_cpp_dir:   Path to llama.cpp checkout.
        device_str:      Device for model loading.
        keep_hf_dir:     Keep intermediate HuggingFace format directory.
        keep_f16_gguf:   Keep intermediate F16 GGUF.

    Returns:
        GGUFExportResult with final .gguf path and export metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    device = torch.device(device_str)

    resolved_gguf_type = gguf_type or _BITS_TO_GGUF_TYPE.get(target_bits, "Q4_K_M")
    tools = find_llama_cpp_tools(llama_cpp_dir)

    logger.info(
        "Exporting PTQ checkpoint: %s → %s (type=%s)",
        checkpoint_path, output_dir, resolved_gguf_type,
    )

    model = _reconstruct_ptq_model(
        checkpoint_path, model_name, cache_dir, target_bits, device,
    )
    experiment_name = checkpoint_path.parent.name or f"ptq_int{target_bits}"

    result = _run_export_pipeline(
        model=model,
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        gguf_type=resolved_gguf_type,
        tools=tools,
        keep_hf_dir=keep_hf_dir,
        keep_f16_gguf=keep_f16_gguf,
    )
    result.method = "ptq"
    result.source_bits = target_bits
    return result


def export_hf_model(
    model_name: str = _DEFAULT_MODEL,
    cache_dir: str = _DEFAULT_CACHE_DIR,
    output_dir: Union[str, Path] = "models/gguf/",
    gguf_type: str = "F16",
    experiment_name: Optional[str] = None,
    llama_cpp_dir: Optional[str] = None,
    device_str: str = "cpu",
    keep_hf_dir: bool = False,
    keep_f16_gguf: bool = True,
) -> GGUFExportResult:
    """Export a HuggingFace model (FP32 baseline or PTQ reference) to GGUF.

    Loads the model directly from HuggingFace Hub / local cache without any
    checkpoint reconstruction. Useful for:
        - FP32 baseline → F16 GGUF (for inference comparison on device)
        - PTQ reference (skipping dequantization, letting llama.cpp re-quantize)

    Args:
        model_name:      HuggingFace model identifier.
        cache_dir:       Local cache directory.
        output_dir:      Directory to write the .gguf file.
        gguf_type:       GGUF quantization type (default F16 for FP32 baseline).
        experiment_name: Stem for the output filename. Defaults to model_name
                         with slashes replaced by underscores.
        llama_cpp_dir:   Path to llama.cpp checkout for tool detection.
        device_str:      Device for model loading (cpu recommended).
        keep_hf_dir:     Keep intermediate HuggingFace format directory.
        keep_f16_gguf:   Keep intermediate F16 GGUF.

    Returns:
        GGUFExportResult with final .gguf path and export metadata.
    """
    output_dir = Path(output_dir)
    device = torch.device(device_str)
    tools = find_llama_cpp_tools(llama_cpp_dir)

    if experiment_name is None:
        experiment_name = model_name.replace("/", "_").replace("-", "_").lower()

    logger.info(
        "Exporting HF model %s → %s (type=%s)", model_name, output_dir, gguf_type,
    )

    model = _load_fp32_base(model_name, cache_dir, device)

    result = _run_export_pipeline(
        model=model,
        model_name=model_name,
        cache_dir=cache_dir,
        output_dir=output_dir,
        experiment_name=experiment_name,
        gguf_type=gguf_type,
        tools=tools,
        keep_hf_dir=keep_hf_dir,
        keep_f16_gguf=keep_f16_gguf,
    )
    result.method = "baseline"
    result.source_bits = 32
    return result


def export_from_config(
    config,
    checkpoint_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    llama_cpp_dir: Optional[str] = None,
    device_str: str = "cpu",
    keep_f16_gguf: bool = True,
) -> GGUFExportResult:
    """Config-driven export dispatcher.

    Reads method, target_bits, model.name, model.cache_dir, and
    export.output_dir from an ExperimentConfig and routes to the correct
    export function. This is the entry point used by the notebooks.

    Args:
        config:          Parsed ExperimentConfig (from load_config()).
        checkpoint_path: Path to the checkpoint file or adapter directory.
        output_dir:      Override export output directory. Defaults to
                         config.export.output_dir.
        llama_cpp_dir:   Path to llama.cpp checkout.
        device_str:      Device for model loading.
        keep_f16_gguf:   Keep the intermediate F16 GGUF file.

    Returns:
        GGUFExportResult from the appropriate export function.

    Raises:
        ValueError: If config.method is not one of the four supported values.
    """
    method = config.method
    target_bits = config.target_bits
    model_name = config.model.name
    cache_dir = config.model.cache_dir
    out_dir = Path(output_dir) if output_dir else Path(config.export.output_dir)
    checkpoint_path = Path(checkpoint_path)

    if method in ("standard_qat", "scheduled_qat"):
        return export_qat_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=out_dir,
            model_name=model_name,
            cache_dir=cache_dir,
            target_bits=target_bits,
            llama_cpp_dir=llama_cpp_dir,
            device_str=device_str,
            keep_f16_gguf=keep_f16_gguf,
        )
    elif method == "lora_qat":
        return export_lora_adapter(
            adapter_dir=checkpoint_path,
            output_dir=out_dir,
            model_name=model_name,
            cache_dir=cache_dir,
            target_bits=target_bits,
            llama_cpp_dir=llama_cpp_dir,
            device_str=device_str,
            keep_f16_gguf=keep_f16_gguf,
        )
    elif method == "ptq":
        return export_ptq_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=out_dir,
            model_name=model_name,
            cache_dir=cache_dir,
            target_bits=target_bits,
            llama_cpp_dir=llama_cpp_dir,
            device_str=device_str,
            keep_f16_gguf=keep_f16_gguf,
        )
    else:
        raise ValueError(
            f"Unknown method {method!r} in config. "
            f"Supported: standard_qat, scheduled_qat, lora_qat, ptq."
        )


# ---------------------------------------------------------------------------
# Logging
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
        description="Export a trained checkpoint to GGUF for llama.cpp inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported GGUF types: {", ".join(sorted(_VALID_GGUF_TYPES))}

Examples:
  # QAT checkpoint (auto-detects standard_qat or scheduled_qat)
  python -m src.utils.export \\
      --checkpoint models/checkpoints/scheduled_qat_cosine_int4/final.pt \\
      --target-bits 4 \\
      --output models/gguf/

  # LoRA-QAT adapter directory
  python -m src.utils.export \\
      --checkpoint models/checkpoints/lora_qat_int4/final_adapter \\
      --method lora_qat \\
      --target-bits 4 \\
      --output models/gguf/

  # PTQ checkpoint
  python -m src.utils.export \\
      --checkpoint models/checkpoints/ptq_int8/final.pt \\
      --method ptq \\
      --target-bits 8 \\
      --output models/gguf/

  # FP32 baseline (no checkpoint — export directly from HF Hub)
  python -m src.utils.export \\
      --method baseline \\
      --gguf-type F16 \\
      --output models/gguf/

  # Specify custom llama.cpp location
  python -m src.utils.export \\
      --checkpoint models/checkpoints/standard_qat_int4/final.pt \\
      --target-bits 4 \\
      --output models/gguf/ \\
      --llama-cpp-dir /opt/llama.cpp

  # Check which llama.cpp tools are detected
  python -m src.utils.export --detect-tools
""",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to the checkpoint .pt file or LoRA adapter directory.",
    )
    p.add_argument(
        "--method",
        default=None,
        choices=["standard_qat", "scheduled_qat", "lora_qat", "ptq", "baseline"],
        help=(
            "Quantization method. Auto-detected from checkpoint keys if omitted. "
            "Required for 'lora_qat' (directory checkpoint) and 'baseline' (no checkpoint)."
        ),
    )
    p.add_argument(
        "--target-bits",
        type=int,
        default=4,
        choices=[4, 8, 16, 32],
        help="QAT/PTQ bit-width used during training (default: 4).",
    )
    p.add_argument(
        "--gguf-type",
        default=None,
        metavar="TYPE",
        help=(
            f"GGUF quantization type. Default: Q4_K_M for 4-bit, Q8_0 for 8-bit. "
            f"Valid types: {', '.join(sorted(_VALID_GGUF_TYPES))}."
        ),
    )
    p.add_argument(
        "--output",
        default="models/gguf/",
        metavar="DIR",
        help="Output directory for the .gguf file (default: models/gguf/).",
    )
    p.add_argument(
        "--model-name",
        default=_DEFAULT_MODEL,
        help=f"HuggingFace model identifier (default: {_DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--cache-dir",
        default=_DEFAULT_CACHE_DIR,
        help=f"Local model cache directory (default: {_DEFAULT_CACHE_DIR}).",
    )
    p.add_argument(
        "--llama-cpp-dir",
        default=None,
        metavar="DIR",
        help="Path to llama.cpp checkout containing convert_hf_to_gguf.py.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help='Device for model loading (default: "cpu"; GPU not needed for export).',
    )
    p.add_argument(
        "--keep-hf-dir",
        action="store_true",
        help="Keep the intermediate HuggingFace format directory after export.",
    )
    p.add_argument(
        "--no-keep-f16",
        action="store_true",
        help="Delete the intermediate F16 GGUF after quantization (saves ~3.4 GB).",
    )
    p.add_argument(
        "--detect-tools",
        action="store_true",
        help="Print detected llama.cpp tool paths and exit without exporting.",
    )
    return p


def main() -> None:
    """CLI entry point."""
    # Make project root importable when invoked as python -m src.utils.export
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    _configure_logging()
    args = _build_arg_parser().parse_args()

    # Tool detection mode.
    if args.detect_tools:
        tools = find_llama_cpp_tools(args.llama_cpp_dir)
        print("llama.cpp tool detection:")
        print(tools.describe())
        return

    # Validate args.
    if args.method != "baseline" and args.checkpoint is None:
        print("ERROR: --checkpoint is required unless --method baseline is specified.")
        sys.exit(1)

    if args.gguf_type and args.gguf_type.upper() not in _VALID_GGUF_TYPES:
        print(f"ERROR: unknown --gguf-type {args.gguf_type!r}.")
        print(f"Valid types: {', '.join(sorted(_VALID_GGUF_TYPES))}")
        sys.exit(1)

    # Determine method from args or checkpoint structure.
    method = args.method
    if method is None and args.checkpoint:
        ckpt = Path(args.checkpoint)
        if ckpt.is_dir():
            method = "lora_qat"
            logger.info("Checkpoint is a directory → assuming method=lora_qat")
        else:
            # Peek at checkpoint keys.
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            if "sqat_controller" in raw:
                method = "scheduled_qat"
            elif "qat_controller" in raw:
                method = "standard_qat"
            else:
                method = "ptq"
                logger.warning(
                    "Could not detect method from checkpoint keys. "
                    "Assuming ptq. Use --method to override."
                )

    # Dispatch.
    keep_f16 = not args.no_keep_f16

    if method == "baseline":
        result = export_hf_model(
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            output_dir=args.output,
            gguf_type=args.gguf_type or "F16",
            llama_cpp_dir=args.llama_cpp_dir,
            device_str=args.device,
            keep_hf_dir=args.keep_hf_dir,
            keep_f16_gguf=keep_f16,
        )
    elif method == "lora_qat":
        result = export_lora_adapter(
            adapter_dir=args.checkpoint,
            output_dir=args.output,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            target_bits=args.target_bits,
            gguf_type=args.gguf_type,
            llama_cpp_dir=args.llama_cpp_dir,
            device_str=args.device,
            keep_hf_dir=args.keep_hf_dir,
            keep_f16_gguf=keep_f16,
        )
    elif method in ("standard_qat", "scheduled_qat"):
        result = export_qat_checkpoint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            target_bits=args.target_bits,
            gguf_type=args.gguf_type,
            llama_cpp_dir=args.llama_cpp_dir,
            device_str=args.device,
            keep_hf_dir=args.keep_hf_dir,
            keep_f16_gguf=keep_f16,
        )
    elif method == "ptq":
        result = export_ptq_checkpoint(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            model_name=args.model_name,
            cache_dir=args.cache_dir,
            target_bits=args.target_bits,
            gguf_type=args.gguf_type,
            llama_cpp_dir=args.llama_cpp_dir,
            device_str=args.device,
            keep_hf_dir=args.keep_hf_dir,
            keep_f16_gguf=keep_f16,
        )
    else:
        print(f"ERROR: unknown method {method!r}.")
        sys.exit(1)

    print(result.summary())


if __name__ == "__main__":
    main()
