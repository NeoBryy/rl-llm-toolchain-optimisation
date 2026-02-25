"""
Evaluation script for Milestone 5: Baseline Performance.

Runs GPT-4o-mini and Qwen 1.5B on the 55-query training set.
Includes error handling, progress logging, and intermediate checkpoints.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src import config
from src.agents.base_agent import BaseReActAgent
from src.rl.model_wrapper import ReActLlamaModel
from src.rl.reward_function import calculate_reward
from src.utils.logger import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling Path, numpy, and datetime objects."""
    def default(self, obj):
        import numpy as np
        from datetime import datetime
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def evaluate_agent(agent: Any, queries: list[dict], agent_name: str, output_file: Path):
    """
    Run evaluation for a specific agent.
    """
    results = []
    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting evaluation for agent: {agent_name}")
    
    for i, query_data in enumerate(queries):
        logger.info(f"[{agent_name}] Query {i+1}/{len(queries)}: {query_data['query'][:50]}...")
        
        start_time = time.time()
        try:
            if hasattr(agent, "run"):
                # OpenAI BaseReActAgent
                result = agent.run(query_data["query"])
            elif hasattr(agent, "run_episode"):
                # Llama Model Wrapper
                result = agent.run_episode(query_data["query"])
            else:
                raise AttributeError("Agent does not have run() or run_episode() method")
            
            # Calculate reward (correctness)
            reward = calculate_reward(query_data, result)
            result["reward"] = reward
            result["execution_time"] = time.time() - start_time
            result["query_data"] = query_data
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Query {i+1} failed for {agent_name}: {e}", exc_info=True)
            results.append({
                "query_data": query_data,
                "error": str(e),
                "success": False,
                "execution_time": time.time() - start_time,
                "reward": -8.0  # Minimum reward for total failure
            })
        
        # Intermediate checkpoints every 10 queries
        if (i + 1) % 10 == 0:
            checkpoint_file = checkpoint_dir / f"{agent_name}_checkpoint_{i+1}.json"
            try:
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2, cls=CustomEncoder)
                logger.info(f"Saved checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint {checkpoint_file}: {e}")

    # Final save
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=CustomEncoder)
        logger.info(f"Evaluation complete for {agent_name}. Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save final results {output_file}: {e}")
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt4", "qwen", "both"], default="both")
    args = parser.parse_args()

    # 0. Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    # 1. Load data
    data_path = config.Paths.DATA_DIR / "training_queries.json"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    with open(data_path) as f:
        queries = json.load(f)
    
    logger.info(f"Loaded {len(queries)} queries.")

    # 2. GPT-4o-mini Evaluation (Fast)
    if args.model in ["gpt4", "both"]:
        logger.info("Initializing GPT-4o-mini Agent...")
        gpt4_agent = BaseReActAgent(model="gpt-4o-mini")
        evaluate_agent(gpt4_agent, queries, "gpt4_mini", Path("results/baseline_gpt4.json"))

    # 3. Qwen 1.5B Evaluation (Slow)
    if args.model in ["qwen", "both"]:
        logger.info("Initializing Qwen 1.5B (Untrained) Model...")
        qwen_model = ReActLlamaModel(config.RLConfig.MODEL_NAME)
        evaluate_agent(qwen_model, queries, "qwen_untrained", Path("results/baseline_qwen_untrained.json"))

if __name__ == "__main__":
    main()
