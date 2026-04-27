### TO DO: Evaluation Benchmarks

| Benchmark | What it tests | Format | Metric | Shot |
|-----------|--------------|--------|--------|------|
| WikiText-103 | Language modeling quality | Next token prediction | Perplexity (lower = better) | N/A |
| KL Divergence | Distribution shift from FP32 | Compare output distributions | KLD (lower = better, 0 = identical) | N/A |
| MMLU | Knowledge across 57 subjects | 4-choice MCQ | Accuracy % | 5-shot |
| HellaSwag | Common sense reasoning | 4-choice sentence completion | Accuracy % | 0-shot |
| ARC-Challenge | Grade school science reasoning | 4-choice MCQ | Accuracy % | 0-shot |
| PIQA | Physical intuition | 2-choice | Accuracy % | 0-shot |
| GSM8K | Math reasoning | Open-ended word problems | Accuracy % | 5-shot |
