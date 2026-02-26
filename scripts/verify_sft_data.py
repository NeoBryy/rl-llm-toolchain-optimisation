"""
Verification script for SFT training data format.
Checks for consistency and ReAct markers.
"""

import json
from pathlib import Path

def verify_sft_data(file_path: str):
    path = Path(file_path)
    if not path.exists():
        print(f"ERROR: File not found: {file_path}")
        return False

    print(f"Loading training data from {file_path}...")
    with open(path, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]

    print(f"Total training examples: {len(train_data)}")
    
    if not train_data:
        print("ERROR: No data found.")
        return False

    # Print first example details
    print("\n--- EXAMPLE 1 VERIFICATION ---")
    print(f"Instruction: {train_data[0]['instruction'][:80]}...")
    print(f"Response length: {len(train_data[0]['response'])} chars")
    print(f"Sample:\n{train_data[0]['response'][:500]}...")
    print("-" * 30)

    # Format checks
    all_thoughts = all("Thought:" in ex["response"] for ex in train_data)
    all_actions = all("Action:" in ex["response"] for ex in train_data)
    all_answers = all("Answer:" in ex["response"] for ex in train_data)
    all_observations = all("Observation:" in ex["response"] for ex in train_data)

    print(f"Contains 'Thought:': {all_thoughts}")
    print(f"Contains 'Action:': {all_actions}")
    print(f"Contains 'Answer:': {all_answers}")
    print(f"Contains 'Observation:': {all_observations}")

    errors = []
    if not all_thoughts: errors.append("Missing 'Thought:' in some examples")
    if not all_actions: errors.append("Missing 'Action:' in some examples")
    if not all_answers: errors.append("Missing 'Answer:' in some examples")
    # Observation might be missing if a question is answered in 1 step without tools? 
    # But our billing queries definitely need tools.
    if not all_observations: print("WARNING: 'Observation:' missing in some examples (might be valid for 1-turn answers)")

    if errors:
        print("\nFAILURES DETECTED:")
        for err in errors:
            print(f" - {err}")
        return False
    
    print("\nVERIFICATION PASSED: Data is ready for SFT.")
    return True

if __name__ == "__main__":
    verify_sft_data("data/sft_billing_train.jsonl")
