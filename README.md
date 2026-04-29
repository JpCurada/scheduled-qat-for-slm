# Scheduled QAT for SLM

Investigating **Scheduled Quantization-Aware Training** as a way to compress
[SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) for deployment on
Android, iOS, and Raspberry Pi.

> **Hypothesis.** Standard QAT applies full quantization noise from step 1.
> Gradually reducing the simulated bit-width over training — first FP32 → 16 → 8 → 4
> — should give the weights time to adapt at each level before the next shock,
> producing lower KL divergence and lower perplexity at the same target bit-width.

## Status

| Phase | Status |
|---|---|
| Source code (configs, training loop, quantization, export) | ✅ implemented |
| Notebooks (Kaggle GPU pipeline, 01–07) | ✅ ready to run |
| FP32 baseline results | ⏳ pending — run [`notebooks/01_baseline.ipynb`](notebooks/01_baseline.ipynb) |
| PTQ / QAT / Scheduled-QAT / LoRA-QAT results | ⏳ pending |
| GGUF export & on-device benchmarks | ⏳ pending |

## What's in this repo

```
configs/        YAML experiment configs (one per method × bit-width)
src/            Pip-installable package — quantization, training, metrics, export
notebooks/      Kaggle GPU notebooks 01–07 (full pipeline; see notebooks/README.md)
examples/       Edge deployment wrappers (Android, iOS, Raspberry Pi)
SKILL.md        Long-form spec: methodology, schemas, related work, key papers
```

The four quantization strategies under comparison live in [`src/quantization/`](src/quantization/):

- **PTQ** ([`ptq.py`](src/quantization/ptq.py)) — calibrate min/max, round, ship.
- **Standard QAT** ([`standard_qat.py`](src/quantization/standard_qat.py)) — fake quantization from epoch 0, STE backward.
- **Scheduled QAT** ([`scheduled_qat.py`](src/quantization/scheduled_qat.py), [`scheduler.py`](src/quantization/scheduler.py)) — *the contribution*; precision schedule (linear / cosine / step) reduces bit-width over training.
- **LoRA-QAT** ([`lora_qat.py`](src/quantization/lora_qat.py)) — quantize base, train rank-16 adapters, merge.

## Running the pipeline

### Local (GPU machine)

```bash
git clone https://github.com/JpCurada/scheduled-qat-for-slm.git
cd scheduled-qat-for-slm
pip install -e ".[eval,viz]"

# Phase 1 — FP32 baseline (perplexity + saved logits for KL divergence)
python -m src.training.baseline --save-logits

# Phase 2 — quantization experiments (run as many as you want)
python -m src.training.trainer --config configs/ptq/ptq_int4.yaml
python -m src.training.trainer --config configs/standard_qat/qat_int4.yaml
python -m src.training.trainer --config configs/scheduled_qat/scheduled_cosine_int4.yaml
python -m src.training.trainer --config configs/lora_qat/lora_qat_int4.yaml

# Phase 4 — GGUF export for edge deployment
python -m src.utils.export \
    --checkpoint models/checkpoints/scheduled_qat_cosine_int4/final.pt \
    --method scheduled_qat --target-bits 4 \
    --output models/gguf/
```

### Kaggle (free T4 GPU)

The seven notebooks under [`notebooks/`](notebooks/) walk through the full pipeline.
See [`notebooks/README.md`](notebooks/README.md) for the run order, dataset wiring,
and time estimates. Short version:

1. Run [`01_baseline.ipynb`](notebooks/01_baseline.ipynb) → save outputs as Kaggle dataset `sqat-baseline`.
2. Run [`02_ptq`](notebooks/02_ptq.ipynb), [`03_standard_qat`](notebooks/03_standard_qat.ipynb), [`04_scheduled_qat`](notebooks/04_scheduled_qat.ipynb), [`05_lora_qat`](notebooks/05_lora_qat.ipynb) — each mounts `sqat-baseline` and writes its own outputs.
3. Run [`06_export_gguf`](notebooks/06_export_gguf.ipynb) → produces GGUF files.
4. Run [`07_benchmarks`](notebooks/07_benchmarks.ipynb) → cross-method tables and plots (CPU only).

## Base model — SmolLM2-1.7B

| Field | Value |
|---|---|
| Parameters | 1.7 B |
| Architecture | Llama 2-style, 24 layers |
| Hidden / FFN size | 2048 / 8192 |
| Tokenizer | GPT-2 BPE, vocab 49 152 |
| Context | 2048 tokens |
| Pretrained on | 11 T tokens (FineWeb-Edu, DCLM, The Stack, math & code) |
| License | Apache 2.0 |
| Paper | [arXiv 2502.02737](https://arxiv.org/abs/2502.02737) |

**Expected post-quantization sizes:**

| Precision | Size | GGUF format |
|---|---:|---|
| FP32 | ~6.5 GB | `F32` |
| FP16 | ~3.4 GB | `F16` |
| INT8 | ~1.7 GB | `Q8_0` |
| INT4 | ~850 MB | `Q4_K_M` |

No QAT results for SmolLM2-1.7B have been published — that's the research gap this project closes.

## Dataset & evaluation

- **Training & calibration:** WikiText-103 train split (103 M tokens).
- **Perplexity:** WikiText-103 test split, no overlap with training.
- **KL divergence:** vs. FP32 baseline logits saved in Phase 1. Per
  [*Accuracy is Not All You Need*](https://arxiv.org/pdf/2407.09141), KLD is the
  gold-standard quantization-quality metric — perplexity alone misses distribution
  shifts that produce different downstream behavior.
- **Cross-domain benchmarks:** MMLU, HellaSwag, ARC-Challenge, PIQA, GSM8K via
  [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness).
- **Robustness:** answer flips (correct→wrong + wrong→correct vs FP32) on MCQ benchmarks.

Single evaluation framework for everything (`lm-evaluation-harness`) so quantized
variants are comparable. Published SmolLM2 numbers use `lighteval` and may differ
slightly from our FP32 reference.

## Key results — to be filled in

This table is the headline output of [`07_benchmarks.ipynb`](notebooks/07_benchmarks.ipynb):

| Method | Bits | PPL | KLD | MMLU | HellaSwag | Size | Android tok/s | RPi tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 baseline | 32 | — | 0.0 | — | — | 6.5 GB | n/a | n/a |
| PTQ | 4 | — | — | — | — | 850 MB | — | — |
| Standard QAT | 4 | — | — | — | — | 850 MB | — | — |
| Scheduled QAT (best) | 4 | — | — | — | — | 850 MB | — | — |
| LoRA-QAT | 4 | — | — | — | — | 870 MB | — | — |

## Documentation

- [`SKILL.md`](SKILL.md) — full methodology, config schemas, related work, citations.
- [`notebooks/README.md`](notebooks/README.md) — Kaggle workflow, dataset wiring, troubleshooting.
- Per-module docstrings in [`src/`](src/) — every public function has a complete docstring with usage examples.

## License

Apache 2.0 — same as the base model.
