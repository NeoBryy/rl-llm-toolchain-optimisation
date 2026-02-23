from trl import PPOConfig
import inspect

sig = inspect.signature(PPOConfig.__init__)
info = {
    "params": list(sig.parameters.keys()),
    "batch_size": 'batch_size' in sig.parameters,
    "mini_batch_size": 'mini_batch_size' in sig.parameters,
    "minibatch_size": 'minibatch_size' in sig.parameters,
    "gradient_accumulation_steps": 'gradient_accumulation_steps' in sig.parameters,
    "ppo_epochs": 'ppo_epochs' in sig.parameters, # vs num_ppo_epochs
    "num_ppo_epochs": 'num_ppo_epochs' in sig.parameters,
    "kl_coef": 'kl_coef' in sig.parameters, # vs init_kl_coef
    "init_kl_coef": 'init_kl_coef' in sig.parameters,
}

for k, v in info.items():
    print(f"{k}: {v}")
