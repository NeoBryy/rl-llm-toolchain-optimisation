# When to Use RL vs Supervised Fine-Tuning

## Use RL (PPO) When:
- ✅ **GPU available:** Minimum 16GB VRAM for stable training of 1B+ models.
- ✅ **Clear reward signal:** When you have a reliable way to score model outputs programmatically.
- ✅ **Exploration needed:** When there are multiple valid strategies to solve a problem and you want the model to discover them.
- ✅ **Risk Tolerance:** When you can afford the potential for training instability and extensive hyperparameter tuning.

## Use Supervised Fine-Tuning (SFT) When:
- ✅ **CPU-only environment:** SFT is significantly more stable than PPO when hardware is constrained.
- ✅ **Known correct behavior:** When you have high-quality examples of the desired input-output pairs.
- ✅ **Stability preference:** When training stability is more critical than exploration.
- ✅ **Fast Iteration:** When faster development cycles are required.

## Our Case Study
**Context:** Tool-use optimization for an energy meter analyst, using a 1.5B parameter model (Qwen2.5) on a CPU-only system.

**Decision:** **Supervised Fine-Tuning (SFT) recommended.**

**Evidence:** Despite conservative hyperparameter tuning (LR=5e-6, KL=0.5), PPO training consistently resulted in mode collapse after Epoch 2. The overhead of RL training on CPU makes the iteration cycle prohibitively long (projects 4.5 days for a full run) compared to SFT.
