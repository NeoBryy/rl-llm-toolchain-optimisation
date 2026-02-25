"""
Validation script for Milestone 5.
Tests a single query on both GPT-4o-mini and Qwen 1.5B.
"""
import json
import logging
from pathlib import Path
from src import config
from src.agents.base_agent import BaseReActAgent
from src.rl.model_wrapper import ReActLlamaModel
from src.rl.reward_function import calculate_reward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load first query
    with open("data/training_queries.json") as f:
        queries = json.load(f)
    query_data = queries[0]
    logger.info(f"Testing Query: {query_data['query']}")

    # 2. Test GPT-4o-mini
    logger.info("--- Testing GPT-4o-mini ---")
    try:
        gpt4_agent = BaseReActAgent(model="gpt-4o-mini")
        result_gpt4 = gpt4_agent.run(query_data["query"])
        reward_gpt4 = calculate_reward(query_data, result_gpt4)
        logger.info(f"GPT-4 Reward: {reward_gpt4}")
        logger.info(f"GPT-4 Success: {result_gpt4['success']}")
    except Exception as e:
        logger.error(f"GPT-4 Test Failed: {e}")

    # 3. Test Qwen
    logger.info("--- Testing Qwen ---")
    try:
        qwen_model = ReActLlamaModel(config.RLConfig.MODEL_NAME)
        result_qwen = qwen_model.run_episode(query_data["query"])
        reward_qwen = calculate_reward(query_data, result_qwen)
        logger.info(f"Qwen Reward: {reward_qwen}")
        logger.info(f"Qwen Success: {result_qwen['success']}")
    except Exception as e:
        logger.error(f"Qwen Test Failed: {e}")

if __name__ == "__main__":
    main()
