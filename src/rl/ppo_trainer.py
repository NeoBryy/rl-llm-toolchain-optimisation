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
            learning_rate=config.RLConfig.LEARNING_RATE,
            batch_size=config.RLConfig.BATCH_SIZE,
            mini_batch_size=config.RLConfig.MINI_BATCH_SIZE,
            gradient_accumulation_steps=config.RLConfig.GRADIENT_ACCUMULATION_STEPS,
            ppo_epochs=config.RLConfig.PPO_EPOCHS,
            init_kl_coef=config.RLConfig.INIT_KL_COEF,
            seed=config.DataConfig.RANDOM_SEED,
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

        if result["success"] is False and not result.get("trajectory"):
             logger.warning("Episode failed with no trajectory: %s", query_text[:50])
             return None, None, None, 0.0, False
        
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

            # Buffers for dynamic batching
            buffer_q = []
            buffer_r = []
            buffer_rewards = []
            
            # Collection Phase (Rollout)
            for query_data in tqdm(training_queries):
                qs, rs, rws, reward_val, success = self.collect_rollout(query_data)

                if qs is not None:
                    buffer_q.extend(qs)
                    buffer_r.extend(rs)
                    buffer_rewards.extend(rws)

                    total_episodes += 1
                    total_reward += reward_val
                    total_success += 1 if success else 0

                # Optimization Phase (PPO Step) - Process while buffer has enough data
                while len(buffer_q) >= self.ppo_config.batch_size:
                    # Slice exact batch size
                    batch_q = buffer_q[:self.ppo_config.batch_size]
                    batch_r = buffer_r[:self.ppo_config.batch_size]
                    batch_rw = buffer_rewards[:self.ppo_config.batch_size]

                    # Remove processed items from buffer
                    buffer_q = buffer_q[self.ppo_config.batch_size:]
                    buffer_r = buffer_r[self.ppo_config.batch_size:]
                    buffer_rewards = buffer_rewards[self.ppo_config.batch_size:]

                    logger.info("Calling ppo_trainer.step()")
                    train_stats = self.ppo_trainer.step(
                        batch_q, batch_r, batch_rw
                    )
                    logger.info("ppo_trainer.step() finished")

                    batch_counter += 1

                    # Log metrics
                    if batch_counter % config.RLConfig.LOG_FREQ == 0:
                        avg_reward = total_reward / total_episodes if total_episodes > 0 else 0.0
                        success_rate = total_success / total_episodes if total_episodes > 0 else 0.0

                        logger.info(
                            "Step %d: Avg Reward=%.2f, KL=%.4f, Success Rate=%.2f",
                            batch_counter,
                            avg_reward,
                            train_stats["objective/kl"],
                            success_rate,
                        )
            
            # Drop remaining items at end of epoch (TRL requires strict batch size)
            if buffer_q:
                logger.info("Dropping %d remaining items at end of epoch", len(buffer_q))

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
