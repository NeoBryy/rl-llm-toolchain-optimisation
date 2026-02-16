"""
PPO Training loop implementation.

Manages the training process using TRL's PPOTrainer, handling
rollout collection, reward calculation, and model updates.
"""

import random
from pathlib import Path
from typing import Any

from tqdm import tqdm
from trl import PPOConfig, PPOTrainer

from src import config
from src.rl.model_wrapper import ReActLlamaModel
from src.rl.reward_function import calculate_reward
from src.utils.logger import get_logger
from src.utils.tensor_utils import convert_to_ppo_tensors

logger = get_logger(__name__)


class RLTrainer:
    """
    Manages PPO training for the ReAct agent.
    """

    def __init__(self):
        """
        Initialize trainer with model and PPO config.
        """
        logger.info("Initializing PPO Trainer...")

        # Initialize policy model wrapper
        self.policy = ReActLlamaModel(config.RLConfig.MODEL_NAME)
        self.tokenizer = self.policy.tokenizer

        # TRL PPO Configuration
        self.ppo_config = PPOConfig(
            model_name=config.RLConfig.MODEL_NAME,
            learning_rate=config.RLConfig.LEARNING_RATE,
            batch_size=config.RLConfig.BATCH_SIZE,
            mini_batch_size=config.RLConfig.MINI_BATCH_SIZE,
            gradient_accumulation_steps=config.RLConfig.GRADIENT_ACCUMULATION_STEPS,
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=config.RLConfig.TARGET_KL,
            ppo_epochs=config.RLConfig.PPO_EPOCHS,
            seed=config.DataConfig.RANDOM_SEED,
            init_kl_coef=config.RLConfig.INIT_KL_COEF,
            adap_kl_ctrl=True,
        )

        # Initialize TRL Trainer
        # Note: We pass the model directly. TRL will create a reference copy automatically.
        # We need to ensure the model is in the right mode.
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.policy.model,
            ref_model=None,  # TRL creates a copy if None
            tokenizer=self.tokenizer,
        )
        logger.info("PPOTrainer initialized")

    def collect_rollout(self, query_data: dict[str, Any]) -> tuple:
        """
        Run one episode and prepare data for training.

        Args:
            query_data: Dict with 'query', 'expected_value', etc.

        Returns:
            Tuple of (queries, responses, rewards) tensors, or (None, None, None) if failed
        """
        query_text = query_data["query"]

        # Run episode using ReActLoop
        # Note: run_episode logic is inside policy wrapper
        result = self.policy.run_episode(query_text)

        # Calculate Reward
        reward = calculate_reward(query_data, result)

        # Validate trajectory exists
        if not result.get("trajectory"):
            return None, None, None

        # Convert to TRL tensors
        queries, responses, rewards = convert_to_ppo_tensors(
            self.tokenizer, result, reward
        )

        return queries, responses, rewards, reward, result["success"]

    def train(self, training_queries: list[dict[str, Any]]) -> dict[str, float]:
        """
        Main training loop.

        Args:
            training_queries: List of ground truth query dicts

        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting training with %d queries", len(training_queries))

        total_episodes = 0
        total_success = 0
        total_reward = 0.0

        batch_size = config.RLConfig.BATCH_SIZE
        batch_counter = 0

        for epoch in range(config.RLConfig.NUM_TRAIN_EPOCHS):
            logger.info("Epoch %d/%d", epoch + 1, config.RLConfig.NUM_TRAIN_EPOCHS)

            # Shuffle queries for each epoch
            random.shuffle(training_queries)

            # Batch processing
            for i in tqdm(range(0, len(training_queries), batch_size)):
                batch_queries = training_queries[i : i + batch_size]

                batch_q_tensors = []
                batch_r_tensors = []
                batch_rewards = []

                batch_stats = {"reward": [], "success": []}

                # Collection Phase (Rollout)
                for query_data in batch_queries:
                    qs, rs, rws, reward_val, success = self.collect_rollout(query_data)

                    if qs is not None:
                        batch_q_tensors.extend(qs)
                        batch_r_tensors.extend(rs)
                        batch_rewards.extend(rws)

                        batch_stats["reward"].append(reward_val)
                        batch_stats["success"].append(1 if success else 0)

                        total_episodes += 1
                        total_reward += reward_val
                        total_success += 1 if success else 0

                # Optimization Phase (PPO Step)
                if batch_q_tensors:
                    train_stats = self.ppo_trainer.step(
                        batch_q_tensors, batch_r_tensors, batch_rewards
                    )

                    batch_counter += 1

                    # Log metrics
                    if batch_counter % config.RLConfig.LOG_FREQ == 0:
                        avg_reward = (
                            sum(batch_stats["reward"]) / len(batch_stats["reward"])
                            if batch_stats["reward"]
                            else 0
                        )
                        success_rate = (
                            sum(batch_stats["success"]) / len(batch_stats["success"])
                            if batch_stats["success"]
                            else 0
                        )

                        logger.info(
                            "Step %d: Avg Reward=%.2f, KL=%.4f, Success Rate=%.2f",
                            total_episodes,
                            avg_reward,
                            train_stats["objective/kl"],
                            success_rate,
                        )

        metrics = {
            "total_episodes": total_episodes,
            "avg_reward": total_reward / total_episodes if total_episodes > 0 else 0,
            "success_rate": total_success / total_episodes if total_episodes > 0 else 0,
        }
        logger.info("Training complete. Metrics: %s", metrics)
        return metrics

    def save_model(self, path: Path) -> None:
        """
        Save the trained model and tokenizer.
        """
        path.mkdir(parents=True, exist_ok=True)
        self.ppo_trainer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Model saved to %s", path)
