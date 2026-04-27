### TO DO:

├── notebooks/                  # Research experiments & analysis
│   ├── 01_baseline.ipynb       # Load SmolLM2-1.7B, eval FP32 baselines
│   ├── 02_ptq.ipynb            # Post-training quantization experiments
│   ├── 03_standard_qat.ipynb   # Standard QAT experiments
│   ├── 04_scheduled_qat.ipynb  # Scheduled QAT experiments (main)
│   ├── 05_lora_qat.ipynb       # LoRA-QAT experiments
│   ├── 06_export_gguf.ipynb    # Convert best models to GGUF
│   └── 07_benchmarks.ipynb     # Cross-method comparison & visualization