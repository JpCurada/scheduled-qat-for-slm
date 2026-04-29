# Notebooks

Seven notebooks reproducing the full experiment pipeline on Kaggle GPUs (T4 / P100).
Each notebook is self-contained — installs the package, mounts its inputs, runs its
phase, and writes results to `/kaggle/working/results/`.

## Pipeline

| # | Notebook | Phase | Time on T4 | Inputs | Outputs |
|---|---|---|---:|---|---|
| 01 | [`01_baseline.ipynb`](01_baseline.ipynb) | FP32 ground truth | ~25 min | (none) | `fp32_logits.pt`, `baseline_results.json` |
| 02 | [`02_ptq.ipynb`](02_ptq.ipynb) | Post-Training Quantization | ~30 min | `sqat-baseline` | `ptq_int{4,8}` results |
| 03 | [`03_standard_qat.ipynb`](03_standard_qat.ipynb) | Standard QAT | ~3.5 h | `sqat-baseline` | `standard_qat_int{4,8}` checkpoints + results |
| 04 | [`04_scheduled_qat.ipynb`](04_scheduled_qat.ipynb) | **Scheduled QAT (main)** | ~5 h | `sqat-baseline` | `scheduled_qat_{linear,cosine,step}_int4` |
| 05 | [`05_lora_qat.ipynb`](05_lora_qat.ipynb) | LoRA-QAT | ~2.5 h | `sqat-baseline` | `lora_qat_int{4,8}` adapters |
| 06 | [`06_export_gguf.ipynb`](06_export_gguf.ipynb) | GGUF export | ~15 min/ckpt | all method outputs | `models/gguf/*.gguf` |
| 07 | [`07_benchmarks.ipynb`](07_benchmarks.ipynb) | Cross-method analysis | CPU only | all method outputs + `sqat-gguf` | tables + plots |

## Workflow on Kaggle

```
[01_baseline]
      │   Save Version → Add Output as Dataset → name it `sqat-baseline`
      ▼
[02_ptq] [03_standard_qat] [04_scheduled_qat] [05_lora_qat]   (each mounts sqat-baseline)
      │   Save Version → Add Output as Dataset → `sqat-ptq`, `sqat-standard-qat`, …
      ▼
[06_export_gguf]   (mounts every method dataset; produces sqat-gguf)
      │
      ▼
[07_benchmarks]    (mounts everything; CPU only, runs anywhere)
```

Each notebook has a **setup cell** that supports two install paths:

- **A. Internet on (default):** `git clone` from the GitHub URL at the top of the cell.
- **B. Internet off:** upload this repo as a Kaggle Dataset named `sqat-repo` and the
  setup cell will copy from `/kaggle/input/sqat-repo` instead.

## Kaggle-specific overrides

The published YAML configs target a multi-GPU run with `seq_length=2048`, `epochs=3`,
`batch_size=8`. None of that fits a 12-hour T4 session, so notebooks 03–05 patch
configs in-memory and write Kaggle copies to `configs_kaggle/`:

| Field | Original | Kaggle |
|---|---|---|
| `data.seq_length` | 2048 | 512 |
| `training.epochs` | 3 (LoRA: 2) | 1 |
| `training.batch_size` | 8 | 4 |
| `training.gradient_accumulation_steps` | 4 | 8 |
| `training.warmup_steps` | 500 | 100 (LoRA: 50) |
| `logging.save_every_steps` | 5000 | 999999 (final-only) |
| `logging.eval_every_steps` | 2500 | 500 |

**Effective batch size and learning rate are unchanged**, so quality numbers from
the Kaggle runs are still comparable to a full-spec rerun on better hardware.
The Scheduled-QAT bit-width transitions are linearly compressed to fit one epoch
(see notebook 04 `kaggle-overrides` cell).

## What each notebook reads / writes

```
01_baseline      ──▶  results/baseline/baseline_results.json
                       results/baseline/fp32_logits.pt        (~1.6 GB, seq_len=512)

02_ptq           ──▶  results/ptq_int{4,8}/training_results.json
                       (PTQ does not save a .pt checkpoint — see notebook 06)

03_standard_qat  ──▶  models/checkpoints/standard_qat_int{4,8}/final.pt
                       results/standard_qat_int{4,8}/training_results.json
                       results/logs/qat_int{4,8}/training_steps.jsonl

04_scheduled_qat ──▶  models/checkpoints/scheduled_qat_{linear,cosine,step}_int4/final.pt
                       results/scheduled_qat_*/training_results.json

05_lora_qat      ──▶  models/checkpoints/lora_qat_int{4,8}/final_adapter/
                       results/lora_qat_int{4,8}/training_results.json

06_export_gguf   ──▶  models/gguf/*_Q4_K_M.gguf  (also Q8_0 for INT8 sources)

07_benchmarks    ──▶  results/tables/primary_results.csv
                       results/tables/schedule_comparison.csv
                       results/plots/{ppl_vs_bits,kld_heatmap,quality_vs_size,...}.png
```

## Troubleshooting

**"Oops something went wrong"** is Kaggle's generic frontend error. It usually means
the kernel crashed (OOM or disk-full) or the browser lost the WebSocket. To get the
real error message: *Save Version → Save & Run All* runs the notebook server-side
and surfaces the actual exception in the version log. Notebook 01 has a dedicated
troubleshooting cell at the top covering the common causes.

**KL divergence is 0 or NaN.** The `fp32_logits.pt` from notebook 01 must be at the
same `seq_length` as the QAT runs. Default everywhere: 512.

**LoRA adapters fail to load on export.** Make sure `peft` is installed
(`pip install -e ".[viz]" peft`) and that you're loading from the `final_adapter/`
directory, not `final.pt`.
