"""
Model wrapper for Llama-3.2-1B to support ReAct generation loops.

Handles token generation with custom stopping criteria to support
Reasoning + Acting patterns.
"""

from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from src import config
from src.agents.tool_registry import execute_tool, get_tools_description
from src.utils.logger import get_logger
from src.utils.parsing import parse_response

logger = get_logger(__name__)


class StringStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria to halt generation on specific substrings."""

    def __init__(self, tokenizer: PreTrainedTokenizer, stop_strings: list[str]):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Check last 20 tokens (sufficient for our stop strings)
        generated_text = self.tokenizer.decode(input_ids[0][-20:])
        for s in self.stop_strings:
            if s in generated_text:
                return True
        return False


class ReActLlamaModel:
    """
    Wrapper for Llama model to facilitate ReAct agent loop.

    Manages loading the model/tokenizer and provides methods for
    controlled generation stopping at Action/Observation boundaries.
    """

    def __init__(self, model_name: str = config.RLConfig.MODEL_NAME):
        """
        Initialize model and tokenizer.

        Args:
            model_name: HuggingFace model identifier
        """
        logger.info("Loading model: %s", model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
        )

        if config.Paths.RL_PROMPT_PATH.exists():
            self.system_prompt_template = config.Paths.RL_PROMPT_PATH.read_text()
        else:
            self.system_prompt_template = config.Paths.BASELINE_PROMPT_PATH.read_text()

        self.system_prompt = self.system_prompt_template.replace(
            "{TOOL_DESCRIPTIONS}", get_tools_description()
        )

    def _generate(self, prompt: str, stop_strings: list[str]) -> str:
        """Helper to generate text with specific stop strings."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]

        stopping_criteria = StoppingCriteriaList(
            [StringStoppingCriteria(self.tokenizer, stop_strings)]
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config.RLConfig.MAX_NEW_TOKENS,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
            )

        # Extract new tokens only using input_length slicing
        new_tokens = outputs[0][input_length:]
        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return new_text

    def generate_until_action(self, prompt: str) -> str:
        """
        Generate tokens until 'Observation:' pattern is detected.

        This captures the 'Thought', 'Action', and 'Action Input' blocks.
        """
        return self._generate(prompt, ["Observation:", "\nObservation"])

    def _build_fail_result(
        self,
        error: str,
        iterations: int,
        tool_calls: list[dict],
        trajectory: list[tuple],
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        """Helper to build standardized failure result."""
        return {
            "success": False,
            "error": error,
            "iterations": iterations,
            "tool_calls": tool_calls,
            "trajectory": trajectory,
            "answer": None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def run_episode(self, query: str) -> dict[str, Any]:
        """
        Full ReAct loop using the model methods.

        Returns:
            Dict matching BaseReActAgent.run() format, plus 'trajectory'.
        """
        prompt = f"{self.system_prompt}\n\nQuestion: {query}\n"

        tool_calls_log = []
        trajectory = []  # List of (context_str, generated_str)
        iterations = 0
        max_iters = config.AgentConfig.MAX_ITERATIONS

        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Initial token count for prompt
        total_prompt_tokens += len(self.tokenizer.encode(prompt))

        while iterations < max_iters:
            iterations += 1

            # Snapshot context before generation
            current_context = prompt

            # Generate next step (Thought/Action)
            new_text = self.generate_until_action(prompt)

            # Store trajectory step
            trajectory.append((current_context, new_text))

            current_completion_tokens = len(self.tokenizer.encode(new_text))
            total_completion_tokens += current_completion_tokens

            prompt += new_text

            # Check stopping condition and clean up string
            if new_text.endswith("Observation:"):
                search_text = new_text[: -len("Observation:")]
            elif new_text.endswith("\nObservation"):
                search_text = new_text[: -len("\nObservation")]
            else:
                search_text = new_text

            parsed = parse_response(search_text)

            # Update prompt tokens for next iteration context
            total_prompt_tokens += current_completion_tokens

            if parsed["type"] == "answer":
                return {
                    "success": True,
                    "answer": parsed["answer"],
                    "iterations": iterations,
                    "tool_calls": tool_calls_log,
                    "trajectory": trajectory,
                    "error": None,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "total_tokens": total_prompt_tokens + total_completion_tokens,
                }

            elif parsed["type"] == "action":
                tool_name = parsed["action"]
                tool_args = parsed["action_input"]

                try:
                    result_val = execute_tool(tool_name, tool_args)
                    observation = f"\nObservation: {result_val}\n"

                    tool_calls_log.append(
                        {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": result_val,
                            "success": True,
                        }
                    )
                    logger.info("Executed tool %s successfully", tool_name)

                except Exception as e:
                    observation = f"\nObservation: Error: {e}\n"
                    tool_calls_log.append(
                        {
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": str(e),
                            "success": False,
                            "error": str(e),
                        }
                    )
                    logger.error("Tool execution failed: %s", e)

                prompt += observation

            elif parsed["type"] == "error":
                return self._build_fail_result(
                    parsed["error"],
                    iterations,
                    tool_calls_log,
                    trajectory,
                    total_prompt_tokens,
                    total_completion_tokens,
                )

            elif parsed["type"] == "thought_only":
                # If model only output thought and stopped (e.g. max tokens), continue loop
                pass

        return self._build_fail_result(
            f"Max iterations ({max_iters}) reached without answer",
            iterations,
            tool_calls_log,
            trajectory,
            total_prompt_tokens,
            total_completion_tokens,
        )
