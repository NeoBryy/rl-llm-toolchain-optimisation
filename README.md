# RL Toolchain Optimisation

This project explores whether reinforcement learning (PPO) can improve the reliability of LLMs as tool-users for domain-specific energy data analysis.

## Results Summary

| Metric | Baseline (GPT-4o-mini) | RL Attempt (Epoch 1) | Status |
|--------|------------------------|----------------------|---------|
| Tool Selection Accuracy | TBD (Milestone 5) | 100% (1/1 sample) | Incomplete |
| Episode Success Rate | TBD | 100% (before collapse) | Mode collapse at Epoch 2 |
| Training Stability | N/A | Unstable on CPU | Hardware constraint identified |

### Key Findings
- **CPU Training Limitation:** PPO with 1.5B models exhibits mode collapse on CPU-only systems.
- **Hyperparameter Sensitivity:** Even conservative settings (LR=5e-6, KL=0.5) proved unstable.
- **Recommendation:** GPU acceleration is required OR a pivot to Supervised Fine-Tuning (SFT) for production deployment in hardware-constrained environments.

### Hyperparameter Tuning Journey
1. **Initial:** LR=1e-5, KL_coef=0.2 → Negative KL and instability at Epoch 2.
2. **Adjusted:** LR=5e-6, KL_coef=0.5 → Delayed instability, but still collapsed at Epoch 2.
3. **Conclusion:** Single-epoch updates or significant GPU acceleration is required for long-term stability.

## Resource Analysis
- **Smoke test (1 query × 3 epochs):** 2 hours on CPU.
- **Projected full training (55 queries × 3 epochs):** ~110 hours (4.5 days).
- **GPU alternative (estimated):** 8-12 hours on V100.
- **Conclusion:** CPU training is not viable for production-scale RL experiments.

## Documentation
- [DECISION_FRAMEWORK.md](file:///d:/rl-llm/rl-llm-toolchain-optimisation/DECISION_FRAMEWORK.md): When to use RL vs. SFT.
- [REPRODUCIBILITY.md](file:///d:/rl-llm/rl-llm-toolchain-optimisation/REPRODUCIBILITY.md): Environment setup and commands.
