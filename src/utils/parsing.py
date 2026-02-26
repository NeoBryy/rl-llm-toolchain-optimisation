"""
Shared parsing utilities for LLM outputs.

Handles extraction of Thought/Action/Answer blocks and JSON repair.
"""

import json
import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def extract_after_marker(text: str, marker: str) -> str:
    """
    Extract text after marker until next marker or end.

    Args:
        text: Full response text
        marker: Marker to extract after (e.g., 'Thought:')

    Returns:
        Extracted text, stripped of whitespace
    """
    parts = text.split(marker, 1)
    if len(parts) < 2:
        return ""

    content = parts[1]

    # Stop at next marker - ORDER MATTERS!
    # "Action Input:" MUST come before "Action:" to avoid substring match
    stop_markers = ["Thought:", "Action Input:", "Action:", "Answer:", "Observation:"]
    for next_marker in stop_markers:
        if next_marker in content:
            content = content.split(next_marker)[0]
            break

    return content.strip()


def clean_json_string(input_str: str) -> str:
    """
    Attempt to repair common LLM JSON formatting errors.

    Args:
        input_str: Raw JSON string from LLM output

    Returns:
        Cleaned JSON string
    """
    input_str = input_str.strip()
    
    # Strip markdown code blocks if present
    if input_str.startswith("```"):
        # Remove starting lines like ```json or ```
        input_str = re.sub(r"^```[a-z]*\n?", "", input_str)
        # Remove ending ```
        input_str = re.sub(r"```$", "", input_str).strip()

    # If there is extra text around the JSON, try to extract only the JSON part
    # Look for the first '{' and last '}'
    start_idx = input_str.find("{")
    end_idx = input_str.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        input_str = input_str[start_idx : end_idx + 1]

    input_str = re.sub(r"[\n\r\t]", " ", input_str)  # newlines inside JSON
    input_str = re.sub(r",\s*}", "}", input_str)  # trailing commas
    input_str = re.sub(r",\s*]", "]", input_str)  # trailing commas in arrays
    return input_str


def parse_response(response: str) -> dict[str, Any]:
    """
    Parse LLM response for Thought/Action/Answer markers.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed dict with type, thought, action, action_input, answer, error
    """
    result = {
        "type": None,
        "thought": None,
        "action": None,
        "action_input": None,
        "answer": None,
        "error": None,
        "raw_response": response,
    }

    # Check for Action first (prioritized over Answer if both exist in one turn)
    if "Action:" in response:
        result["type"] = "action"
        result["action"] = extract_after_marker(response, "Action:")

        # Extract Action Input (should be JSON)
        if "Action Input:" in response:
            input_str = extract_after_marker(response, "Action Input:")
            try:
                # Clean and repair common JSON errors
                input_str = clean_json_string(input_str)
                result["action_input"] = json.loads(input_str)
                logger.debug("Parsed action: %s", result["action"])
                # Even if Answer is present, we prioritize Action to ensure tool execution
                return result
            except json.JSONDecodeError as e:
                result["type"] = "error"
                result["error"] = f"Invalid JSON in Action Input: {e}"
                logger.error(
                    "JSON parse error: %s | Raw input: %s",
                    result["error"],
                    input_str[:200],
                )
                return result
        else:
            result["type"] = "error"
            result["error"] = "Action specified but no Action Input found"
            logger.error("Missing Action Input")
            return result

    # Check for Answer second (terminal condition)
    if "Answer:" in response:
        result["type"] = "answer"
        result["answer"] = extract_after_marker(response, "Answer:")
        logger.debug("Parsed final answer")
        return result

    # Check for Thought
    if "Thought:" in response:
        result["thought"] = extract_after_marker(response, "Thought:")

    if result["thought"]:
        # Has Thought but no Action/Answer - thinking phase
        result["type"] = "thought_only"
        logger.debug("Parsed thought-only response")
    else:
        # No recognizable markers
        result["type"] = "error"
        result["error"] = (
            "Could not parse response - missing Thought/Action/Answer markers"
        )
        logger.error("Unparseable response: %s", response[:100])

    return result
