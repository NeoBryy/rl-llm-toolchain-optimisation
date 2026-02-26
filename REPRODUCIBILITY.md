# REPRODUCIBILITY.md

## Environment
- **Hardware:** CPU-only (Intel/AMD), 16GB RAM minimum
- **OS:** Windows 11
- **Python:** 3.12 (managed via `uv`)
- **Key Packages:**
  - `transformers` 5.1.0
  - `trl` 0.8.6
  - `torch` 2.5

## Running the Project

All commands assume you are in `d:\rl-llm\rl-llm-toolchain-optimisation` and use `uv run`.

---

### Milestone 5: Baseline Evaluation

```powershell
# Run GPT-4o-mini + Qwen untrained on all 55 queries
uv run scripts\evaluate_baselines.py --model both

# Analyse baseline results
uv run scripts\analyze_baselines.py
```

---

### Milestone 6: SFT on Billing Category

#### Step 1 — Extract training data (GPT-4o-mini trajectories)
```powershell
uv run scripts\prepare_sft_data.py
# Output: data/sft_billing_train.jsonl (15 examples)
```

#### Step 2 — Verify extracted data
```powershell
uv run scripts\verify_sft_data.py
```

#### Step 3 — Fine-tune Qwen on billing trajectories (~16h CPU)
```powershell
uv run scripts\train_sft_billing.py
# Output: models/qwen_sft_billing_final/
```

#### Step 4 — Evaluate SFT model on billing queries only
```powershell
uv run scripts\evaluate_sft_billing.py
# Output: results/sft/qwen_ft_billing_results.json
```

#### Step 5 — Full 55-query evaluation of SFT model (~2.5h CPU)
```powershell
uv run scripts\evaluate_sft_full.py
# Output: results/sft/qwen_sft_full_results.json
```

#### Step 6 — Generate three-way comparison report
```powershell
uv run scripts\analyze_sft_results.py
# Output: results/final_comparison.md
```

---

## Results Directory Structure

```
results/
├── baselines/
│   ├── baseline_gpt4.json           # GPT-4o-mini on 55 queries
│   └── baseline_qwen_untrained.json # Qwen 1.5B untrained on 55 queries
├── sft/
│   ├── qwen_ft_billing_results.json # SFT model on billing queries only (15)
│   └── qwen_sft_full_results.json   # SFT model on all 55 queries
├── checkpoints/                     # Intermediate saves (every 10 queries)
└── final_comparison.md              # Three-way comparison table
```

---

## Known Issues

- **transformers 5.1.0 / trl 0.8.6 compatibility:** `Trainer.__init__` renamed `tokenizer` → `processing_class`. A monkeypatch in `train_sft_billing.py` handles this automatically.
- **CPU RAM:** Loading Qwen 1.5B in float32 requires ~8GB RAM. Close other applications before running.
- **Disk Space:** The fine-tuned model (`qwen_sft_billing_final/`) is ~4GB. Ensure 10GB+ free before training.
- **TRL API:** Avoid `optimize_cuda_cache` and certain `target_kl` parameters deprecated in trl 0.7.
