"""
Evaluation script for Milestone 6: Fine-tuned Qwen Model.
Re-evaluates the billing category to verify performance improvements.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

from src import config
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
    logger.info(f"Starting evaluation for agent: {agent_name}")
    
    for i, query_data in enumerate(queries):
        logger.info(f"[{agent_name}] Query {i+1}/{len(queries)}: {query_data['query'][:50]}...")
        
        start_time = time.time()
        try:
            result = agent.run_episode(query_data["query"])
            
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
                "reward": -8.0
            })

    # Save results
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, cls=CustomEncoder)
        logger.info(f"Evaluation complete. Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save final results {output_file}: {e}")
    
    # Calculate Accuracy
    success_count = sum(1 for r in results if r.get("success", False))
    accuracy = (success_count / len(results)) * 100 if results else 0
    logger.info(f"Final Accuracy: {accuracy:.1f}% ({success_count}/{len(results)})")
    
    return results

def main():
    # 1. Load data
    data_path = config.Paths.DATA_DIR / "training_queries.json"
    with open(data_path) as f:
        queries = json.load(f)
    
    # Filter for billing queries
    billing_queries = [q for q in queries if q.get("category") == "billing"]
    logger.info(f"Evaluating {len(billing_queries)} billing queries.")

    # 2. Load Fine-tuned Model
    ft_model_path = "models/qwen_sft_billing_final"
    if not Path(ft_model_path).exists():
        logger.error(f"Fine-tuned model not found at {ft_model_path}. Run training first.")
        return

    logger.info(f"Initializing Fine-tuned Qwen Model from {ft_model_path}...")
    qwen_ft = ReActLlamaModel(ft_model_path)
    
    # 3. Run Evaluation
    output_file = Path("results/sft/qwen_ft_billing_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    evaluate_agent(qwen_ft, billing_queries, "qwen_sft_billing", output_file)

if __name__ == "__main__":
    main()
