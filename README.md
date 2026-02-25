# RL Toolchain Optimisation

This project explores whether reinforcement learning (PPO) can improve the reliability of LLMs as tool-users for domain-specific energy data analysis.

## Results Summary (Milestone 5)

| Metric | GPT-4o-mini (Baseline) | Qwen 2.5-1.5B (Untrained) | Status |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | **90.9%** | **70.9%** | Verified |
| Tool Selection | 92.7% | 100.0% | - |
| Execution Success | 98.2% | 70.9% | - |
| Avg Latency | 7.9s | 160.0s | CPU Bottleck |

### Category Performance
| Category | GPT-4o-mini | Qwen 1.5B |
| :--- | :--- | :--- |
| Aggregation | 86.7% | 86.7% |
| Billing | 100.0% | 13.3% |
| Count | 80.0% | 100.0% |
| Load Factor | 100.0% | 100.0% |
| Visualization | 60.0% | 80.0% |

### Key Findings
- **Qwen Competency:** The 1.5B Instruct model is surprisingly capable of handling domain-specific tools (100% selection) but lacks the reasoning depth for complex multi-step "Billing" queries without fine-tuning.
- **CPU Inference Constraint:** 160s average latency makes real-time RL training on CPU non-viable for rapid iteration.
- **RL Stability Issues:** As documented during training attempts, the model is prone to mode collapse on CPU updates, necessitating transition to SFT or GPU-accelerated RL.

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
