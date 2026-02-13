"""
ReAct agent for meter data analysis.

Implements Reasoning + Acting pattern with OpenAI LLM and tool execution.
"""

import json
from typing import Any

from openai import OpenAI

from src import config
from src.agents import tool_registry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseReActAgent:
    """
    Baseline ReAct agent using OpenAI LLM.

    Implements thought/action/observation loop with tool execution.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize ReAct agent.

        Args:
            model: OpenAI model name (default: gpt-3.5-turbo)
        """
        self.model = model
        self.max_iterations = config.AgentConfig.MAX_ITERATIONS
        self.client = OpenAI()  # Uses OPENAI_API_KEY env var
        self.system_prompt = self._build_system_prompt()
        logger.info("Initialized BaseReActAgent with model=%s", model)

    def _build_system_prompt(self) -> str:
        """
        Load baseline.txt template and inject tool descriptions.

        Returns:
            Complete system prompt with tool descriptions

        Raises:
            FileNotFoundError: If baseline prompt template not found
        """
        template_path = config.Paths.BASELINE_PROMPT_PATH

        if not template_path.exists():
            raise FileNotFoundError(f"Baseline prompt not found at {template_path}")

        with open(template_path) as f:
            template = f.read()

        tool_descriptions = tool_registry.get_tools_description()
        return template.replace("{TOOL_DESCRIPTIONS}", tool_descriptions)

    def _call_llm(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        """
        Call OpenAI API with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            (response_text, usage_dict) where usage_dict contains token counts
        """
        logger.debug("Calling LLM with %d messages", len(messages))

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=config.AgentConfig.MAX_TOKENS_PER_CALL,
            temperature=config.AgentConfig.TEMPERATURE,
        )

        response_text = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        logger.debug("LLM response: %d tokens (%d prompt, %d completion)",
                     usage["total_tokens"], usage["prompt_tokens"], usage["completion_tokens"])

        return response_text, usage

    def _extract_after_marker(self, text: str, marker: str) -> str:
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

    def _clean_json_string(self, input_str: str) -> str:
        """
        Attempt to repair common LLM JSON formatting errors.

        Args:
            input_str: Raw JSON string from LLM output

        Returns:
            Cleaned JSON string
        """
        import re

        input_str = input_str.strip()
        input_str = re.sub(r'[\n\r\t]', ' ', input_str)   # newlines inside JSON
        input_str = re.sub(r',\s*}', '}', input_str)       # trailing commas
        input_str = re.sub(r',\s*]', ']', input_str)       # trailing commas in arrays
        return input_str

    def _parse_response(self, response: str) -> dict[str, Any]:
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

        # Check for Answer first (terminal condition)
        if "Answer:" in response:
            result["type"] = "answer"
            result["answer"] = self._extract_after_marker(response, "Answer:")
            logger.info("Parsed final answer")
            return result

        # Check for Thought
        if "Thought:" in response:
            result["thought"] = self._extract_after_marker(response, "Thought:")

        # Check for Action
        if "Action:" in response:
            result["type"] = "action"
            result["action"] = self._extract_after_marker(response, "Action:")

            # Extract Action Input (should be JSON)
            if "Action Input:" in response:
                input_str = self._extract_after_marker(response, "Action Input:")
                try:
                    # Clean and repair common JSON errors
                    input_str = self._clean_json_string(input_str)
                    result["action_input"] = json.loads(input_str)
                    logger.info("Parsed action: %s", result["action"])
                except json.JSONDecodeError as e:
                    result["type"] = "error"
                    result["error"] = f"Invalid JSON in Action Input: {e}"
                    logger.error("JSON parse error: %s | Raw input: %s", result["error"], input_str[:200])
                    return result
            else:
                result["type"] = "error"
                result["error"] = "Action specified but no Action Input found"
                logger.error("Missing Action Input")
                return result

        elif result["thought"]:
            # Has Thought but no Action - thinking phase
            result["type"] = "thought_only"
            logger.debug("Parsed thought-only response")

        else:
            # No recognizable markers
            result["type"] = "error"
            result["error"] = "Could not parse response - missing Thought/Action/Answer markers"
            logger.error("Unparseable response: %s", response[:100])

        return result

    def _execute_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        """
        Execute tool and return observation + call record.

        Args:
            tool_name: Name of tool to execute
            tool_args: Tool arguments dict

        Returns:
            (observation_string, tool_call_record)
        """
        try:
            result = tool_registry.execute_tool(tool_name, tool_args)
            observation = f"Observation: {result}"
            tool_call_record = {
                "tool": tool_name,
                "arguments": tool_args,
                "result": result,
                "success": True,
            }
            logger.info("Tool execution successful: %s", tool_name)
            return observation, tool_call_record

        except Exception as e:
            observation = f"Observation: Tool execution failed: {e}"
            tool_call_record = {
                "tool": tool_name,
                "arguments": tool_args,
                "result": None,
                "success": False,
                "error": str(e),
            }
            logger.error("Tool execution failed: %s - %s", tool_name, e)
            return observation, tool_call_record

    def _build_result(
        self,
        answer: str | None,
        iterations: int,
        tool_calls: list[dict],
        success: bool,
        error: str | None,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ) -> dict[str, Any]:
        """
        Build standardized result dict.

        Args:
            answer: Final answer text or None
            iterations: Number of iterations completed
            tool_calls: List of tool call records
            success: Whether task completed successfully
            error: Error message or None
            prompt_tokens: Total prompt tokens used
            completion_tokens: Total completion tokens used
            total_tokens: Total tokens used

        Returns:
            Standardized result dictionary
        """
        return {
            "answer": answer,
            "iterations": iterations,
            "tool_calls": tool_calls,
            "success": success,
            "error": error,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def run(self, query: str) -> dict[str, Any]:
        """
        Run ReAct loop to answer user query.

        Args:
            query: User's question about meter data

        Returns:
            Dict with answer, iterations, tool_calls, success, error, and token counts
        """
        logger.info("Starting ReAct loop for query: %s", query)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        tool_calls = []
        total_prompt_tokens = total_completion_tokens = total_tokens = 0

        for iteration in range(self.max_iterations):
            logger.info("Iteration %d/%d", iteration + 1, self.max_iterations)

            try:
                response_text, usage = self._call_llm(messages)
            except Exception as e:
                logger.error("LLM call failed: %s", e)
                return self._build_result(None, iteration + 1, tool_calls, False,
                                         f"LLM call failed: {e}",
                                         total_prompt_tokens, total_completion_tokens, total_tokens)

            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]

            messages.append({"role": "assistant", "content": response_text})
            parsed = self._parse_response(response_text)

            if parsed["type"] == "answer":
                logger.info("Task completed in %d iterations", iteration + 1)
                return self._build_result(parsed["answer"], iteration + 1, tool_calls, True, None,
                                         total_prompt_tokens, total_completion_tokens, total_tokens)

            elif parsed["type"] == "action":
                observation, tool_call_record = self._execute_tool_call(
                    parsed["action"].strip(), parsed["action_input"]
                )
                tool_calls.append(tool_call_record)
                messages.append({"role": "user", "content": observation})

            elif parsed["type"] == "thought_only":
                logger.debug("Thought-only response, continuing")
                continue

            elif parsed["type"] == "error":
                logger.error("Parse error: %s", parsed["error"])
                return self._build_result(None, iteration + 1, tool_calls, False,
                                         f"Parse error: {parsed['error']}",
                                         total_prompt_tokens, total_completion_tokens, total_tokens)

        logger.warning("Max iterations (%d) reached without answer", self.max_iterations)
        return self._build_result(None, self.max_iterations, tool_calls, False,
                                 f"Max iterations ({self.max_iterations}) reached without answer",
                                 total_prompt_tokens, total_completion_tokens, total_tokens)
