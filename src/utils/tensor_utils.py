"""
Tensor manipulation utilities for PPO training.
"""

import torch


def convert_to_ppo_tensors(
    tokenizer, episode_result: dict, final_reward: float
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Convert episode trajectory into PPO-compatible tensors.

    Args:
        tokenizer: Tokenizer for encoding text
        episode_result: Result dict from run_episode containing 'trajectory'
        final_reward: Scalar reward for the episode

    Returns:
        (query_tensors, response_tensors, rewards) for TRL PPOTrainer
    """
    queries = []
    responses = []
    rewards = []

    trajectory = episode_result.get("trajectory", [])

    for context, generated in trajectory:
        # Encode context (query) and generation (response)
        # Ensure no padding is added here as TRL handles it
        # return_tensors="pt" returns shape [1, seq_len], so we take [0]
        query_tensor = tokenizer.encode(context, return_tensors="pt")[0]
        response_tensor = tokenizer.encode(generated, return_tensors="pt")[0]

        queries.append(query_tensor)
        responses.append(response_tensor)

        # Broadcast final reward to all steps
        # This simplifies credit assignment by treating the whole trajectory
        # as contributing equally to the final outcome
        rewards.append(torch.tensor([final_reward], dtype=torch.float))

    return queries, responses, rewards
