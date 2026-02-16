"""
Main entry point for PPO training.
"""
import json
from pathlib import Path

from src import config
from src.rl.ppo_trainer import RLTrainer
from src.utils.logger import get_logger

# Setup logging
logger = get_logger(__name__)


def main():
    """
    Load data and start PPO training.
    """
    logger.info("Starting RL training pipeline...")
    
    # 1. Load Training Data
    data_path = config.Paths.DATA_DIR / "training_queries.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")
        
    with open(data_path) as f:
        queries = json.load(f)
        
    logger.info("Loaded %d training queries from %s", len(queries), data_path)
    
    # 2. Initialize Trainer
    trainer = RLTrainer()
    
    # 3. Start Training
    metrics = trainer.train(queries)
    
    # 4. Save Final Model
    output_dir = config.Paths.MODELS_DIR / "ppo_llama_v1"
    trainer.save_model(output_dir)
    
    logger.info("Training complete. Model saved to %s", output_dir)
    logger.info("Final Metrics: %s", metrics)


if __name__ == "__main__":
    main()
