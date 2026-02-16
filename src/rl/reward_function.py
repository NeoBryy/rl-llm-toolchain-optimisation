"""
Reward function for RL training in the meter analyst domain.

Evaluates agent performance based on:
1. Tool Usage (did it use tools or fabricate?)
2. Execution Success (did tools run without error?)
3. Numerical Accuracy (did the answer match ground truth?)
"""

import re
from pathlib import Path
from typing import Any

from src import config


def _extract_number(text: str) -> float | None:
    """
    Extract the most likely answer value from agent response text.

    Finds all numbers and returns the last one, as agent answers
    typically end with the calculated value.
    """
    # Look for number patterns:
    # - 123
    # - 123.45
    # - 0.123
    # Ignores currency symbols but captures the number after them
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text.replace(",", ""))

    if not matches:
        return None

    try:
        # Return last number - agent answers typically end with the value
        return float(matches[-1])
    except ValueError:
        return None


def _check_plot_created(result: dict) -> bool:
    """
    Verify if a plot file was actually created on disk.

    Checks tool call history for successful plot_generator calls
    and verifies the returned path exists.
    """
    for call in result.get("tool_calls", []):
        if call["tool"] == "plot_generator" and call["success"]:
            # plot_generator returns the absolute path as a string in 'result'
            plot_path_str = call.get("result")
            if not plot_path_str:
                continue

            try:
                plot_path = Path(plot_path_str)
                # Verify path is within allowed plots directory for safety
                if (
                    config.Paths.PLOTS_DIR in plot_path.parents
                    or plot_path.parent == config.Paths.PLOTS_DIR
                ):
                    if (
                        plot_path.exists()
                        and plot_path.is_file()
                        and plot_path.stat().st_size > 0
                    ):
                        return True
            except (OSError, ValueError):
                continue

    return False


def calculate_reward(query_data: dict[str, Any], result: dict[str, Any]) -> float:
    """
    Calculate the total reward for a single episode.

    Args:
        query_data: Dictionary containing ground truth (`expected_value`, `tolerance`, `tolerance_type`)
        result: Dictionary returned by agent.run() containing `answer`, `tool_calls`, `success`

    Returns:
        Float reward value between -8.0 and +10.0
    """
    reward = 0.0

    # 1. Tool Usage Component (+2.0 or -5.0)
    # Critical: agent MUST use tools, not fabricate answers
    tool_calls = result.get("tool_calls", [])
    if len(tool_calls) > 0:
        reward += 2.0
    else:
        # Severe penalty for hallucination/fabrication without tools
        reward -= 5.0
        # If no tools used, we stop here (likely wrong answer anyway)
        return reward

    # 2. Execution Success Component (+3.0 or -3.0)
    if result.get("success", False):
        reward += 3.0
    else:
        reward -= 3.0
        # If execution failed (error/max iterations), return early
        return reward

    # 3. Accuracy Component (+5.0 or 0.0)
    # Different logic based on tolerance_type
    tolerance_type = query_data.get("tolerance_type", "relative")
    expected_val = query_data.get("expected_value")
    tolerance = query_data.get("tolerance", 0.05)

    agent_answer_text = result.get("answer", "")

    # CASE A: Plot generation
    if tolerance_type == "plot_exists":
        if _check_plot_created(result):
            reward += 5.0
        else:
            # Agent claimed success but no plot file found/verified
            reward += 0.0

    # CASE B: Numerical checking (Exact or Relative)
    else:
        # Extract number from agent's text answer
        agent_val = _extract_number(agent_answer_text)

        if agent_val is not None and expected_val is not None:
            if tolerance_type == "exact":
                # Strict equality (allowing for small float epsilon if needed, but tolerance is usually 0.0)
                # For counts/min/max, we often want integer equality or very strict float match
                if abs(agent_val - expected_val) <= 1e-6:  # effectively exact
                    reward += 5.0
            else:
                # Relative tolerance (default 5%)
                # abs(diff) <= tolerance * expected
                # Handle zero expected value case
                if expected_val == 0:
                    if (
                        abs(agent_val) <= tolerance
                    ):  # Absolute validation if target is 0
                        reward += 5.0
                elif abs(agent_val - expected_val) <= (tolerance * abs(expected_val)):
                    reward += 5.0

    return reward
