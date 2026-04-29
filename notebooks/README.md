# Notebooks (Google Colab)

Seven notebooks reproducing the full experiment pipeline on Google Colab GPUs (T4).
Each notebook clones the repo to `/content/scheduled-qat-for-slm`, mounts your Google
Drive, runs its phase, and copies outputs to `/content/drive/MyDrive/sqat-<phase>/`
so the next notebook can pick them up.

## Pipeline

| # | Notebook | Phase | Time on T4 | Reads from Drive | Writes to Drive |
|---|---|---|---:|---|---|
| 01 | [`01_baseline.ipynb`](01_baseline.ipynb) | FP32 ground truth | ~25 min | (none) | `sqat-baseline/` |
| 02 | [`02_ptq.ipynb`](02_ptq.ipynb) | Post-Training Quantization | ~30 min | `sqat-baseline/` | `sqat-ptq/` |
| 03 | [`03_standard_qat.ipynb`](03_standard_qat.ipynb) | Standard QAT | ~3.5 h | `sqat-baseline/` | `sqat-standard-qat/` |
| 04 | [`04_scheduled_qat.ipynb`](04_scheduled_qat.ipynb) | **Scheduled QAT (main)** | ~5 h | `sqat-baseline/` | `sqat-scheduled-qat/` |
| 05 | [`05_lora_qat.ipynb`](05_lora_qat.ipynb) | LoRA-QAT | ~2.5 h | `sqat-baseline/` | `sqat-lora-qat/` |
| 06 | [`06_export_gguf.ipynb`](06_export_gguf.ipynb) | GGUF export | ~15 min/ckpt | `sqat-{baseline,standard-qat,scheduled-qat,lora-qat}/` | `sqat-gguf/` |
| 07 | [`07_benchmarks.ipynb`](07_benchmarks.ipynb) | Cross-method analysis | CPU only | every `sqat-*/` folder | `sqat-results/` |

## Workflow

```
[01_baseline]                           writes → /content/drive/MyDrive/sqat-baseline/
       │
       ▼
[02_ptq]  [03_standard_qat]  [04_scheduled_qat]  [05_lora_qat]
       │            │                 │                 │
       ▼            ▼                 ▼                 ▼
   sqat-ptq  sqat-standard-qat  sqat-scheduled-qat  sqat-lora-qat
       │            │                 │                 │
       └────────────┴────────┬────────┴─────────────────┘
                             ▼
                      [06_export_gguf]      writes → sqat-gguf/
                             │
                             ▼
                      [07_benchmarks]       writes → sqat-results/
```

Each notebook's setup cell:
1. `from google.colab import drive; drive.mount('/content/drive')`
2. Clones the GitHub repo to `/content/scheduled-qat-for-slm`
3. Reads inputs from `/content/drive/MyDrive/sqat-*/`

The last cell of each notebook copies its outputs to Drive.

## Independent execution

Notebooks 02, 03, 04, 05 only depend on **notebook 01**. They are independent of each
other — you can run them in **any order**, on **separate Colab sessions in parallel**,
or restart any one of them without re-running the others. The dependency only kicks
in at notebook 06 (which needs trained checkpoints) and notebook 07 (which aggregates).

## Colab-specific overrides

The published YAML configs target a multi-GPU run with `seq_length=2048`, `epochs=3`,
`batch_size=8`. None of that fits a Colab T4 session, so notebooks 03–05 patch
configs in-memory and write Colab copies to `configs_colab/`:

| Field | Original | Colab |
|---|---|---|
| `data.seq_length` | 2048 | 512 |
| `training.epochs` | 3 (LoRA: 2) | 1 |
| `training.batch_size` | 8 | 4 |
| `training.gradient_accumulation_steps` | 4 | 8 |
| `training.warmup_steps` | 500 | 100 (LoRA: 50) |
| `logging.save_every_steps` | 5000 | 999999 (final-only) |
| `logging.eval_every_steps` | 2500 | 500 |

**Effective batch size and learning rate are unchanged**, so quality numbers from
the Colab runs are still comparable to a full-spec rerun on better hardware.
The Scheduled-QAT bit-width transitions are linearly compressed to fit one epoch
(see notebook 04 `kaggle-overrides` cell).

## What each notebook produces (full list)

```
01_baseline      ──▶  sqat-baseline/results/baseline/
                          baseline_results.json
                          fp32_logits.pt          (~1.6 GB, seq_len=512)
                      sqat-baseline/models/base/  (~6.5 GB, optional)

02_ptq           ──▶  sqat-ptq/results/
                          ptq_int{4,8}/training_results.json
                          ptq_inference.json
                          ptq_summary.json

03_standard_qat  ──▶  sqat-standard-qat/
                          models/checkpoints/standard_qat_int{4,8}.pt
                          data_samples/{train,val,test}_sample.pt
                          results/metric_summary.csv
                          results/standard_qat_int{4,8}/training_results.json
                          results/standard_qat_inference.json
                          results/logs/qat_int{4,8}/per_step_loss.jsonl   (micro)
                          results/logs/qat_int{4,8}/training_steps.jsonl  (macro)

04_scheduled_qat ──▶  sqat-scheduled-qat/
                          models/checkpoints/scheduled_qat_{linear,cosine,step}_int4.pt
                          data_samples/{train,val,test}_sample.pt
                          results/metric_summary.csv
                          results/scheduled_qat_*_int4/training_results.json
                          results/scheduled_qat_inference.json
                          results/logs/scheduled_*_int4/per_step_loss.jsonl
                          results/logs/scheduled_*_int4/training_steps.jsonl

05_lora_qat      ──▶  sqat-lora-qat/
                          models/checkpoints/lora_qat_int{4,8}_adapter/
                          data_samples/{train,val,test}_sample.pt
                          results/metric_summary.csv
                          results/lora_qat_int{4,8}/training_results.json
                          results/lora_qat_inference.json
                          results/logs/lora_qat_int{4,8}/per_step_loss.jsonl
                          results/logs/lora_qat_int{4,8}/training_steps.jsonl

06_export_gguf   ──▶  sqat-gguf/models/gguf/*.gguf
                      sqat-gguf/results/gguf_inference.json

07_benchmarks    ──▶  sqat-results/results/tables/
                          primary_results.csv
                          schedule_comparison.csv
                          cross_method_inference.csv
                          conclusions.txt
                      sqat-results/results/plots/
                          ppl_vs_bits.html
                          kld_heatmap.html
                          schedule_comparison.html
                          quality_vs_size.html
```

## Two views of training loss

Each training notebook (03/04/05) produces two log files per variant:

- **`per_step_loss.jsonl`** — micro view: one entry per optimizer step, fields `{step, epoch, loss, lr}`. Use this to spot exploding gradients, schedule transitions, sudden divergences.
- **`training_steps.jsonl`** — macro view: one entry per `eval_every_steps` interval (default 500), fields `{step, epoch, loss, val_ppl, lr}`. Use this to track validation perplexity over time.

The plotly loss-curve cells in each notebook overlay both — per-step loss as a thin line in the background, val PPL as a dashed line on the right axis.

## Drive space planning

| Phase | Drive size needed |
|---|---:|
| `sqat-baseline` (logits + JSON, no model) | ~1.7 GB |
| `sqat-baseline` (with model cache) | ~8 GB |
| `sqat-standard-qat` (2 checkpoints) | ~7 GB |
| `sqat-scheduled-qat` (3 checkpoints) | ~10 GB |
| `sqat-lora-qat` (2 adapters) | ~100 MB |
| `sqat-ptq` | ~1 MB |
| `sqat-gguf` (4-8 GGUFs at varying types) | ~5-15 GB |
| `sqat-results` (tables + plots) | ~1 MB |

Free Google Drive is 15 GB total. Watch your usage with `!df -h /content/drive/MyDrive/`. If tight: skip `SAVE_MODEL_TO_DRIVE` (model auto-redownloads in each notebook), and only keep checkpoints for the methods you'll actually compare.

## Troubleshooting

**Drive mount popup hangs.** Click *Connect to Google Drive* in the popup and authorise. The cell can't continue until you do.

**"Oops something went wrong" / runtime disconnects.** Free Colab disconnects after ~90 min idle. Use *Runtime → Run all* with the tab visible, or upgrade to Colab Pro for longer sessions.

**KL divergence is 0 or NaN.** The `fp32_logits.pt` from notebook 01 must be at the same `seq_length` as the QAT runs. Default everywhere: 512.

**LoRA adapters fail to load on export.** Make sure `peft` is installed (`pip install -e ".[viz]" peft`) and that you're loading the `lora_qat_int*_adapter/` directory, not a `.pt` file.

**No training-loop logs appear.** The trainer prints progress via `print(..., flush=True)` (not Python `logging`) so output is visible regardless of notebook frontend. If you still see nothing, check the GPU is on and that the cell has actually started executing.
