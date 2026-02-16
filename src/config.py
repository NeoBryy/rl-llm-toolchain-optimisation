"""
Configuration module for RL Meter Analyst project.

Usage in other modules:
    from src import config

    model = config.LLMConfig.BASELINE_MODEL
    tools = config.ToolConfig.AVAILABLE_TOOLS
    db_path = config.Paths.DB_PATH
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMConfig:
    """Language model configuration for baseline and RL agents."""

    # OpenAI baseline model (API-based)
    BASELINE_MODEL: str = "gpt-3.5-turbo"
    BASELINE_TEMPERATURE: float = 0.1
    BASELINE_MAX_TOKENS: int = 1500

    # Local RL model (trainable via TRL)
    # Hardware requirements: ~4GB GPU VRAM (CUDA recommended) or CPU (10-50x slower)
    RL_MODEL: str = "meta-llama/Llama-3.2-1B-Instruct"
    RL_TEMPERATURE: float = 0.1
    RL_MAX_TOKENS: int = 1500

    # Device auto-detection
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # API credentials (from environment)
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")


class ToolConfig:
    """Tool definitions and execution settings."""

    AVAILABLE_TOOLS: list[str] = [
        "sql_executor",
        "pandas_aggregator",
        "tariff_calculator",
        "plot_generator",
        "calculate_load_factor",
    ]
    TOOL_TIMEOUT_SECONDS: int = 30
    ALLOWED_AGGREGATIONS: list[str] = ["sum", "mean", "median", "count", "max", "min"]
    ALLOWED_PLOT_TYPES: list[str] = ["line", "bar", "scatter"]


class AgentConfig:
    """Agent execution and LLM settings."""

    BASELINE_MODEL: str = "gpt-4o-mini"  # OpenAI model for baseline agent
    MAX_ITERATIONS: int = 10  # Maximum ReAct loop iterations
    MAX_TOKENS_PER_CALL: int = 1500  # Maximum tokens per LLM call
    TEMPERATURE: float = 0.0  # LLM temperature (0.0 = deterministic)


class RLConfig:
    """Reinforcement Learning and PPO training settings."""

    # Model settings
    MODEL_NAME: str = "meta-llama/Llama-3.2-1B"  # HuggingFace model ID
    MAX_LENGTH: int = 512  # Max sequence length for context
    MAX_NEW_TOKENS: int = 256  # Max tokens to generate per response

    # Training hyperparameters
    LEARNING_RATE: float = 1e-5  # Conservative to prevent forgetting
    BATCH_SIZE: int = 4  # Queries per batch
    MINI_BATCH_SIZE: int = 2  # PPO minibatch size
    GRADIENT_ACCUMULATION_STEPS: int = 2  # Effective batch = 8

    # PPO-specific parameters
    PPO_EPOCHS: int = 4  # Optimization epochs per batch
    LAM: float = 0.95  # GAE lambda
    CLIPRANGE: float = 0.2  # PPO clip range
    CLIPRANGE_VALUE: float = 0.2  # Value function clip range
    VF_COEF: float = 0.1  # Value function coefficient

    # Reward function parameters
    NUMERICAL_TOLERANCE: float = 0.05  # 5% tolerance for numerical accuracy

    # KL divergence control (prevents drift from reference model)
    KL_PENALTY: str = "kl"  # KL penalty type
    TARGET_KL: float = 0.01  # Early stopping threshold
    INIT_KL_COEF: float = 0.2  # Initial KL coefficient

    # Training control
    NUM_TRAIN_EPOCHS: int = 3  # Full dataset passes
    MAX_GRAD_NORM: float = 0.5  # Gradient clipping
    SAVE_FREQ: int = 10  # Save checkpoint every N batches
    LOG_FREQ: int = 1  # Log every batch


class DataConfig:
    """Synthetic meter data generation parameters."""

    # Generation parameters
    # TODO: Increase to 1000 customers, 365 days for production dataset
    NUM_CUSTOMERS: int = 100  # Reduced for initial testing
    DATE_RANGE_DAYS: int = 30  # Reduced for initial testing
    READING_INTERVAL_MINUTES: int = 15
    RANDOM_SEED: int = 42  # For reproducible data generation

    # Database schema
    TABLE_NAMES: list[str] = ["readings", "customers", "tariffs"]

    # Entity types
    METER_TYPES: list[str] = ["smart", "standard"]
    CUSTOMER_SEGMENTS: list[str] = ["residential", "commercial"]


class Paths:
    """File and directory paths."""

    # Base directories
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    PROMPTS_DIR: Path = PROJECT_ROOT / "src" / "prompts"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    PLOTS_DIR: Path = PROJECT_ROOT / "plots"
    MODELS_DIR: Path = PROJECT_ROOT / "models"

    # Specific files
    DB_PATH: Path = DATA_DIR / "meter_data.db"
    BASELINE_PROMPT_PATH: Path = PROMPTS_DIR / "baseline.txt"
    RL_PROMPT_PATH: Path = PROMPTS_DIR / "llama_react.txt"


# === To be added in later milestones ===
# class EvalConfig: (metrics, thresholds, test set size, comparison settings)


def validate_config() -> None:
    """
    Validate configuration on import. Raises ValueError if invalid.

    Checks:
        - Required environment variables are set
        - Creates necessary directories if missing

    Raises:
        ValueError: If required configuration is missing or invalid
    """
    if not LLMConfig.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )

    # Create directories if missing
    for path in [Paths.DATA_DIR, Paths.PROMPTS_DIR, Paths.LOGS_DIR, Paths.PLOTS_DIR, Paths.MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


# Validate configuration on module import
validate_config()
