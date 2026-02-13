"""
Tool registry for ReAct agent.

Maps tool names to executable Python functions with automatic type coercion
from LLM string outputs to proper Python types (int, float, bool, str).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.tools import library
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegisteredTool:
    """
    Tool definition for agent execution.

    Attributes:
        name: Tool name matching config.ToolConfig.AVAILABLE_TOOLS
        function: Reference to actual Python function from library
        description: What the tool does (for LLM prompt)
        parameters: Parameter specs with type, required, default, description
    """

    name: str
    function: Callable
    description: str
    parameters: dict[str, dict[str, Any]]


def _get_tool_definitions() -> list[RegisteredTool]:
    """
    Get hardcoded tool definitions for all 5 tools.

    MAINTENANCE NOTE: When adding/modifying tools in src/tools/library.py,
    you MUST manually update this registry to keep them in sync. There is
    no automatic introspection - each tool must be explicitly defined here.

    Returns:
        List of RegisteredTool instances for all available tools
    """
    return [
        RegisteredTool(
            name="sql_executor",
            function=library.sql_executor,
            description="Execute read-only SQL query against meter database. Returns data as dict with columns and rows.",
            parameters={
                "query": {
                    "type": str,
                    "required": True,
                    "description": "SQL SELECT query string (no INSERT/UPDATE/DELETE/DROP)",
                },
                "row_limit": {
                    "type": int,
                    "required": False,
                    "default": 100,
                    "description": "Maximum rows to return (default 100)",
                },
            },
        ),
        RegisteredTool(
            name="pandas_aggregator",
            function=library.pandas_aggregator,
            description="Aggregate query results using pandas operations (sum, mean, median, count, max, min).",
            parameters={
                "query": {
                    "type": str,
                    "required": True,
                    "description": "SQL SELECT query to fetch data",
                },
                "operation": {
                    "type": str,
                    "required": True,
                    "description": "Aggregation type: sum, mean, median, count, max, min",
                },
                "column": {
                    "type": str,
                    "required": True,
                    "description": "Column name to aggregate",
                },
                "group_by": {
                    "type": str,
                    "required": False,
                    "default": None,
                    "description": "Optional column name to group results by",
                },
            },
        ),
        RegisteredTool(
            name="tariff_calculator",
            function=library.tariff_calculator,
            description="Calculate electricity bill with peak/off-peak pricing for customer over date range.",
            parameters={
                "customer_id": {
                    "type": int,
                    "required": True,
                    "description": "Customer identifier",
                },
                "start_date": {
                    "type": str,
                    "required": True,
                    "description": "Start date in ISO format (YYYY-MM-DD)",
                },
                "end_date": {
                    "type": str,
                    "required": True,
                    "description": "End date in ISO format (YYYY-MM-DD)",
                },
            },
        ),
        RegisteredTool(
            name="plot_generator",
            function=library.plot_generator,
            description="Generate and save plot (line, bar, scatter) from SQL query results.",
            parameters={
                "query": {
                    "type": str,
                    "required": True,
                    "description": "SQL SELECT query to fetch data",
                },
                "plot_type": {
                    "type": str,
                    "required": True,
                    "description": "Type of plot: line, bar, or scatter",
                },
                "x_column": {
                    "type": str,
                    "required": True,
                    "description": "Column name for x-axis",
                },
                "y_column": {
                    "type": str,
                    "required": True,
                    "description": "Column name for y-axis",
                },
                "title": {
                    "type": str,
                    "required": False,
                    "default": "",
                    "description": "Optional plot title",
                },
            },
        ),
        RegisteredTool(
            name="calculate_load_factor",
            function=library.calculate_load_factor,
            description="Calculate load factor (efficiency metric: average_load / peak_load) for customer.",
            parameters={
                "customer_id": {
                    "type": int,
                    "required": True,
                    "description": "Customer identifier",
                },
                "start_date": {
                    "type": str,
                    "required": True,
                    "description": "Start date in ISO format (YYYY-MM-DD)",
                },
                "end_date": {
                    "type": str,
                    "required": True,
                    "description": "End date in ISO format (YYYY-MM-DD)",
                },
            },
        ),
    ]


def get_tool_registry() -> dict[str, RegisteredTool]:
    """
    Get tool registry as dictionary for O(1) lookup.

    Returns:
        Dictionary mapping tool_name -> RegisteredTool
    """
    tools = _get_tool_definitions()
    return {tool.name: tool for tool in tools}


def get_tools_description() -> str:
    """
    Generate formatted text description of all tools for LLM prompt.

    Returns:
        Multi-line string with tool descriptions and parameters
    """
    tools = _get_tool_definitions()
    descriptions = []

    for tool in tools:
        desc = f"Tool: {tool.name}\n"
        desc += f"  Description: {tool.description}\n"
        desc += "  Parameters:\n"

        for param_name, param_spec in tool.parameters.items():
            required_str = "required" if param_spec["required"] else "optional"
            type_str = param_spec["type"].__name__
            default_str = f" (default: {param_spec.get('default')})" if not param_spec["required"] else ""
            desc += f"    - {param_name}: {type_str}, {required_str}{default_str} - {param_spec['description']}\n"

        descriptions.append(desc)

    return "\n".join(descriptions)


def _coerce_argument(value: Any, target_type: type) -> Any:
    """
    Convert LLM-provided argument to correct Python type.

    Args:
        value: Value to coerce (usually string from LLM)
        target_type: Target Python type (int, float, str, bool)

    Returns:
        Value coerced to target_type

    Raises:
        ValueError: If coercion fails
    """
    # Handle None/null
    if value is None:
        return None

    try:
        if target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is str:
            return str(value)
        elif target_type is bool:
            # Handle string representations of bool
            if isinstance(value, str):
                return value.lower() in ["true", "1", "yes"]
            return bool(value)
        else:
            # Unknown type, return as-is
            return value
    except (ValueError, TypeError) as e:
        raise ValueError(f"Cannot coerce '{value}' to type {target_type.__name__}: {e}") from e


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """
    Execute a tool by name with automatic argument type coercion.

    Args:
        tool_name: Name of tool to execute
        arguments: Dictionary of argument name -> value (may be strings from LLM)

    Returns:
        Result from tool execution (dict, str, or other)

    Raises:
        ValueError: If tool not found, required params missing, or coercion fails
        RuntimeError: If tool execution fails

    Example:
        >>> execute_tool('tariff_calculator', {
        ...     'customer_id': '1',  # String will be coerced to int
        ...     'start_date': '2026-01-01',
        ...     'end_date': '2026-01-31'
        ... })
    """
    # Look up tool
    registry = get_tool_registry()
    if tool_name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown tool '{tool_name}'. Available: {available}")

    tool = registry[tool_name]

    # Validate and coerce arguments
    coerced_args = {}

    for param_name, param_spec in tool.parameters.items():
        if param_spec["required"] and param_name not in arguments:
            raise ValueError(f"Missing required parameter '{param_name}' for tool '{tool_name}'")

        if param_name in arguments:
            # Coerce the argument
            raw_value = arguments[param_name]
            target_type = param_spec["type"]
            coerced_value = _coerce_argument(raw_value, target_type)
            coerced_args[param_name] = coerced_value
        elif "default" in param_spec:
            # Use default if not provided
            coerced_args[param_name] = param_spec["default"]

    logger.info("Executing tool '%s' with arguments: %s", tool_name, coerced_args)

    try:
        result = tool.function(**coerced_args)
        logger.info("Tool '%s' executed successfully", tool_name)
        return result
    except Exception as e:
        logger.error("Tool '%s' execution failed: %s", tool_name, e)
        raise RuntimeError(f"Tool '{tool_name}' execution failed: {e}") from e
