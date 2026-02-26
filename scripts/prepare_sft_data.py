"""
Script to prepare SFT training data by extracting GPT-4 trajectories for billing queries.
"""

import json
import logging
from pathlib import Path
from src import config
from src.agents.base_agent import BaseReActAgent
from src.rl.reward_function import calculate_reward
from src.utils.logger import get_logger

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def prepare_sft_data():
    # 1. Load queries
    data_path = config.Paths.DATA_DIR / "training_queries.json"
    with open(data_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    
    billing_queries = [q for q in queries if q.get("category") == "billing"]
    logger.info(f"Filtered {len(billing_queries)} billing queries.")

    # 2. Initialize GPT-4 Agent
    agent = BaseReActAgent(model="gpt-4o-mini")
    
    sft_data = []
    
    # 3. Process queries
    for i, query_data in enumerate(billing_queries):
        query = query_data["query"]
        logger.info(f"Processing query {i+1}/{len(billing_queries)}: {query[:50]}...")
        
        try:
            result = agent.run(query)
            reward = calculate_reward(query_data, result)
            
            # Extract successful trajectories
            if reward > 0 and result.get("trajectory"):
                # Reconstruct full trajectory string
                # trajectory is [(c1, r1), (c2, r2), ...]
                # The full conversation is effectively c1 + r1 + (observation if exists) + r2 ...
                # But SFT instruction is the query, so the response is everything AFTER c1.
                
                full_resp = ""
                # We need to interleave the observations. 
                # result["tool_calls"] has the results in order.
                
                tool_idx = 0
                for context, response in result["trajectory"]:
                    full_resp += response
                    # If this response was an action, add the observation
                    if "Action:" in response and tool_idx < len(result["tool_calls"]):
                        obs = result["tool_calls"][tool_idx]["result"]
                        full_resp += f"\nObservation: {obs}\n"
                        tool_idx += 1
                
                sft_data.append({
                    "instruction": query,
                    "response": full_resp
                })
                logger.info(f"Captured trajectory (reward: {reward})")
            else:
                logger.warning(f"Query failed or low reward ({reward}): {query[:50]}")
                
        except Exception as e:
            logger.error(f"Failed to process query: {e}", exc_info=True)

    # 4. Save to JSONL
    output_path = Path("data/sft_billing_train.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(sft_data)} entries to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in sft_data:
            f.write(json.dumps(entry) + "\n")
            
    logger.info(f"Saved {len(sft_data)} examples to {output_path}")

    # Print first 2 examples for review
    if len(sft_data) >= 2:
        print("\n--- FIRST 2 TRAINING EXAMPLES ---\n")
        for i in range(2):
            print(f"EXAMPLE {i+1}:")
            print(f"INSTRUCTION: {sft_data[i]['instruction']}")
            # Truncate long trajectory for printing
            traj = sft_data[i]['response']
            print(f"RESPONSE:\n{traj[:500]}..." if len(traj) > 500 else f"RESPONSE:\n{traj}")
            print("-" * 40)

if __name__ == "__main__":
    prepare_sft_data()
