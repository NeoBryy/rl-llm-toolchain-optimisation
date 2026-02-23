# REPRODUCIBILITY.md

## Environment
- **Hardware:** CPU-only (Intel/AMD), 16GB RAM minimum
- **OS:** Windows 11 + WSL2 (Ubuntu 20.04)
- **Python:** 3.12
- **Key Packages:**
  - `transformers` 4.x
  - `trl` 0.7-0.8
  - `torch` 2.5

## Commands Used

### 1. Generating Training Data
```bash
python scripts/generate_training_queries.py
```

### 2. Running Training Smoke Test
```bash
python scripts/train_rl_agent.py
```

### 3. Baseline Comparison (Milestone 3)
```bash
# Reference command for baseline evaluation
python scripts/run_baseline_comparison.py
```

## Known Issues
- **WSL Multiprocessing:** Training can hang when using multiprocessing in WSL; it is recommended to run directly on Windows or ensure `num_workers=0` in data loaders.
- **Disk Space:** Checkpoints for 1.5B models are large (~4GB each). Ensure at least 10GB+ of free space.
- **TRL API Compatibility:** Avoid using `optimize_cuda_cache` or certain `target_kl` parameters that are deprecated or unstable in version 0.7.
