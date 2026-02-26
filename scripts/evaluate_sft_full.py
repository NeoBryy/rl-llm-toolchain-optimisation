"""
Full 55-query evaluation of the SFT-fine-tuned Qwen model.
Milestone 6: Verify no regression across all categories.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src import config
from src.rl.model_wrapper import ReActLlamaModel
from src.rl.reward_function import calculate_reward
from src.utils.logger import get_logger

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
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


def evaluate_agent(agent: Any, queries: list[dict], agent_name: str, output_file: Path, resume: bool = False):
    """Run evaluation for all queries with checkpointing every 10 queries.
    
    Args:
        resume: If True, load the latest checkpoint and skip already-completed queries.
    """
    results = []
    checkpoint_dir = Path("results/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_from = 0

    if resume:
        # Find latest checkpoint for this agent
        checkpoints = sorted(checkpoint_dir.glob(f"{agent_name}_checkpoint_*.json"))
        if checkpoints:
            latest = checkpoints[-1]
            with open(latest, encoding="utf-8") as f:
                results = json.load(f)
            start_from = len(results)
            logger.info(f"Resuming from checkpoint: {latest} ({start_from} queries already done)")
        else:
            logger.info("No checkpoint found — starting from scratch.")

    remaining_queries = queries[start_from:]
    logger.info(f"Starting evaluation: {agent_name} — {len(remaining_queries)} queries remaining (of {len(queries)} total)")

    for i, query_data in enumerate(remaining_queries, start=start_from):
        category = query_data.get("category", "unknown")
        logger.info(f"[{agent_name}] Query {i+1}/{len(queries)} [{category}]: {query_data['query'][:60]}...")

        start_time = time.time()
        try:
            result = agent.run_episode(query_data["query"])
            reward = calculate_reward(query_data, result)
            result["reward"] = reward
            result["execution_time"] = time.time() - start_time
            result["query_data"] = query_data
            results.append(result)

        except Exception as e:
            logger.error(f"Query {i+1} failed: {e}", exc_info=True)
            results.append({
                "query_data": query_data,
                "error": str(e),
                "success": False,
                "tool_calls": [],
                "execution_time": time.time() - start_time,
                "reward": -8.0,
            })

        # Checkpoint every 10 queries
        if (i + 1) % 10 == 0:
            checkpoint_file = checkpoint_dir / f"{agent_name}_checkpoint_{i+1}.json"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, cls=CustomEncoder)
            logger.info(f"Checkpoint saved: {checkpoint_file}")

            # Print running summary
            so_far = [r for r in results if "query_data" in r]
            success_count = sum(1 for r in so_far if r.get("success", False))
            logger.info(f"Running accuracy: {success_count}/{len(so_far)} ({100*success_count/len(so_far):.1f}%)")

    # Final save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=CustomEncoder)
    logger.info(f"Evaluation complete. Results saved to {output_file}")

    # Summary
    success_count = sum(1 for r in results if r.get("success", False))
    accuracy = 100 * success_count / len(results) if results else 0
    logger.info(f"Final Accuracy: {accuracy:.1f}% ({success_count}/{len(results)})")

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("query_data", {}).get("category", "unknown")
        categories.setdefault(cat, {"success": 0, "total": 0})
        categories[cat]["total"] += 1
        if r.get("success", False):
            categories[cat]["success"] += 1

    logger.info("Category breakdown:")
    for cat, stats in sorted(categories.items()):
        pct = 100 * stats["success"] / stats["total"]
        logger.info(f"  {cat}: {stats['success']}/{stats['total']} ({pct:.1f}%)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit to N queries (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # 1. Load data
    data_path = config.Paths.DATA_DIR / "training_queries.json"
    with open(data_path, encoding="utf-8") as f:
        queries = json.load(f)

    if args.limit:
        queries = queries[:args.limit]
        logger.info(f"TEST MODE: Limited to {args.limit} queries")

    logger.info(f"Loaded {len(queries)} queries.")

    # 2. Load SFT model
    ft_model_path = "models/qwen_sft_billing_final"
    if not Path(ft_model_path).exists():
        logger.error(f"Fine-tuned model not found at {ft_model_path}. Run training first.")
        return

    logger.info(f"Loading SFT model from {ft_model_path}...")
    model = ReActLlamaModel(ft_model_path)

    # 3. Run evaluation
    output_file = Path("results/sft/qwen_sft_full_results.json")
    evaluate_agent(model, queries, "qwen_sft_full", output_file, resume=args.resume)


if __name__ == "__main__":
    main()
