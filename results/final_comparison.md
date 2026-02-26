# Final Comparison: GPT-4o-mini vs Qwen (Untrained) vs Qwen (SFT)

## Overall Metrics

| Metric | GPT-4o-mini | Qwen (Untrained) | Qwen (SFT) |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | 90.9% | 70.9% | 85.5% |
| Tool Selection Rate | 92.7% | 100.0% | 100.0% |
| Execution Success Rate | 98.2% | 70.9% | 85.5% |
| Avg Latency (s) | 7.9s | 160.0s | 120.0s |

## Category Performance

| Category | GPT-4o-mini | Qwen (Untrained) | Qwen (SFT) |
|:---|:---:|:---:|:---:|
| Aggregation | 86.7% | 86.7% | 80.0% |
| Billing | 100.0% | 13.3% | 100.0% |
| Count | 80.0% | 100.0% | 40.0% |
| Load Factor | 100.0% | 100.0% | 93.3% |
| Visualization | 60.0% | 80.0% | 80.0% |

## Key Findings

- **Billing improvement:** 13.3% → 100.0% (+86.7 pts)
- **Overall improvement:** 70.9% → 85.5% (+14.6 pts)
- **⚠️ Regressions (>10pts):** count (100.0% → 40.0%)
- **Gap vs GPT-4o-mini:** 5.4 pts

## Training Efficiency

| Model | Training Data | Training Time | Billing Accuracy |
|:---|:---|:---|:---|
| GPT-4o-mini (Baseline) | N/A | N/A | 100.0% |
| Qwen (Untrained) | N/A | N/A | 13.3% |
| **Qwen (SFT)** | **15 trajectories** | **~16h CPU** | **100.0%** |
