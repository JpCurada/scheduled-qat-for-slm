# Scheduled QAT for SLM

## Project Overview

This project investigates **Scheduled Quantization-Aware Training** as a method to
compress Small Language Models for deployment on resource-constrained edge devices
(Android phones, iOS devices, Raspberry Pi). The core hypothesis is that gradually
reducing precision during training (via a schedule) produces better quantized models
than applying quantization all at once.

**Important clarification:** QAT does NOT train a new model from scratch. We take
a pretrained SmolLM2-1.7B and run it through a training loop with fake quantization
nodes injected. The model already knows everything — QAT is "rehabilitation" that
helps it adapt to working with reduced precision. The weights get slightly adjusted
to be more robust to quantization noise, not to learn new knowledge.

## Base Model

**SmolLM2-1.7B** (HuggingFaceTB/SmolLM2-1.7B)
- 1.7B parameters, 24 Transformer layers, Llama2 architecture
- Hidden size: 2048, FFN size: 8192
- Pretrained on 11T tokens (FineWeb-Edu, DCLM, The Stack, math & code datasets)
- Tokenizer: GPT-2 BPE (vocab size: 49152)
- Sequence length: 2048
- License: Apache 2.0
- HuggingFace: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B
- Paper: https://arxiv.org/abs/2502.02737

**Expected model sizes after quantization:**

| Precision | Approximate Size |
|-----------|-----------------|
| FP32      | ~6.5 GB         |
| FP16      | ~3.4 GB         |
| INT8      | ~1.7 GB         |
| INT4      | ~850 MB         |

**Existing GGUF quantizations (PTQ only, from bartowski/Unsloth):**

| Format  | Size    | Source |
|---------|---------|--------|
| F16     | 3.42 GB | bartowski |
| Q8_0    | 1.82 GB | bartowski |
| Q5_K_M  | 1.23 GB | bartowski |
| Q4_K_M  | 1.06 GB | bartowski |

No QAT results for SmolLM2-1.7B have been published yet

## Dataset & Evaluation

### Primary Dataset: WikiText-103

Standard language modeling benchmark (103M tokens from Wikipedia Good/Featured articles).
Used for QAT training data and perplexity evaluation.
