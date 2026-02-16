"""
Tool library for RL Meter Analyst agents.

Provides executable functions that agents can call to analyze meter data.
All tools accept primitive types only (str, int, float, bool) - no DataFrames.
"""

from typing import Any

import duckdb

from src import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def sql_executor(query: str, row_limit: int = 100) -> dict[str, Any]:
    """
    Execute read-only SQL query against meter database.

    Automatically applies row limit if not present in query to prevent
    context window flooding.

    Args:
        query: SQL SELECT query string (no INSERT/UPDATE/DELETE/DROP)
        row_limit: Maximum rows to return (default 100)

    Returns:
        dict with 'columns' (list of str) and 'rows' (list of lists)

    Raises:
        ValueError: If query contains forbidden keywords
        RuntimeError: If query execution fails

    Example:
        >>> result = sql_executor("SELECT * FROM customers WHERE segment = 'residential'")
        >>> print(result['columns'])
        ['customer_id', 'segment', 'meter_type', 'join_date']
    """
    # Validate query safety
    forbidden_keywords = [
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "CREATE",
        "ALTER",
        "TRUNCATE",
    ]
    query_upper = query.upper()

    for keyword in forbidden_keywords:
        if keyword in query_upper:
            raise ValueError(
                f"Query contains forbidden keyword '{keyword}'. "
                "Only read-only SELECT queries are allowed."
            )

    # Apply row limit if not already present
    if "LIMIT" not in query_upper:
        query = f"{query.rstrip(';')} LIMIT {row_limit}"
        logger.debug("Applied automatic LIMIT %d to query", row_limit)

    logger.info("Executing SQL query: %s", query[:100])  # Log first 100 chars

    try:
        with duckdb.connect(str(config.Paths.DB_PATH), read_only=True) as conn:
            result_df = conn.execute(query).fetchdf()

        # Convert to dict format for agent
        result = {
            "columns": result_df.columns.tolist(),
            "rows": result_df.values.tolist(),
            "row_count": len(result_df),
        }

        logger.info(
            "Query returned %d rows, %d columns",
            result["row_count"],
            len(result["columns"]),
        )

        return result

    except duckdb.Error as e:
        logger.error("SQL execution failed: %s", e)
        raise RuntimeError(f"Query execution failed: {e}") from e


def pandas_aggregator(
    query: str,
    operation: str,
    column: str,
    group_by: str | None = None,
) -> dict[str, Any]:
    """
    Aggregate data from SQL query result using pandas operations.

    Fetches data internally via sql_executor, then applies aggregation.

    Args:
        query: SQL SELECT query to fetch data
        operation: Aggregation type (sum, mean, median, count, max, min)
        column: Column name to aggregate
        group_by: Optional column name to group results by

    Returns:
        dict with operation, column, and result(s)
        - Without group_by: {"operation": "mean", "column": "x", "result": 0.5}
        - With group_by: {"operation": "mean", "column": "x", "group_by": "y",
                          "results": {"group1": 0.5, "group2": 0.7}}

    Raises:
        ValueError: If operation invalid or column not found
        RuntimeError: If query execution fails

    Example:
        >>> result = pandas_aggregator(
        ...     "SELECT * FROM readings",
        ...     "mean",
        ...     "consumption_kwh",
        ...     "segment"
        ... )
    """
    # Validate operation
    if operation not in config.ToolConfig.ALLOWED_AGGREGATIONS:
        raise ValueError(
            f"Invalid operation '{operation}'. "
            f"Allowed: {', '.join(config.ToolConfig.ALLOWED_AGGREGATIONS)}"
        )

    logger.info("Aggregating: %s(%s) group_by=%s", operation, column, group_by)

    # Fetch data using sql_executor
    result = sql_executor(query)

    # Convert to DataFrame
    import pandas as pd

    df = pd.DataFrame(result["rows"], columns=result["columns"])

    # Validate column exists
    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available: {', '.join(df.columns)}"
        )

    # Validate group_by column if provided
    if group_by and group_by not in df.columns:
        raise ValueError(
            f"Group column '{group_by}' not found. Available: {', '.join(df.columns)}"
        )

    # Apply aggregation
    if group_by:
        # Grouped aggregation
        grouped = df.groupby(group_by)[column].agg(operation)
        results_dict = grouped.to_dict()

        logger.info("Grouped aggregation returned %d groups", len(results_dict))

        return {
            "operation": operation,
            "column": column,
            "group_by": group_by,
            "results": results_dict,
        }
    else:
        # Simple aggregation
        agg_result = df[column].agg(operation)

        logger.info("Aggregation result: %s", agg_result)

        return {
            "operation": operation,
            "column": column,
            "result": float(agg_result)
            if hasattr(agg_result, "__float__")
            else agg_result,
        }


def _parse_peak_hours(peak_hours_str: str) -> list[tuple[int, int]]:
    """
    Parse peak hours string into list of (start_hour, end_hour) tuples.

    Args:
        peak_hours_str: Format "06:00-09:00,18:00-22:00"

    Returns:
        List of (start_hour, end_hour) tuples e.g. [(6, 9), (18, 22)]
    """
    ranges = []
    for time_range in peak_hours_str.split(","):
        start_str, end_str = time_range.split("-")
        start_hour = int(start_str.split(":")[0])
        end_hour = int(end_str.split(":")[0])
        ranges.append((start_hour, end_hour))
    return ranges


def _is_peak_hour(hour: int, peak_ranges: list[tuple[int, int]]) -> bool:
    """
    Check if given hour falls within any peak range.

    Args:
        hour: Hour of day (0-23)
        peak_ranges: List of (start_hour, end_hour) tuples

    Returns:
        True if hour is within any peak range
    """
    return any(start <= hour < end for start, end in peak_ranges)


def _calculate_standing_charge(
    start_date: str, end_date: str, daily_rate: float
) -> float:
    """
    Calculate standing charge for date range.

    Args:
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)
        daily_rate: Daily standing charge rate

    Returns:
        Total standing charge for period
    """
    from datetime import datetime

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    days = (end - start).days + 1  # Inclusive
    return days * daily_rate


def tariff_calculator(
    customer_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    """
    Calculate electricity bill for customer over date range with peak/off-peak pricing.

    Args:
        customer_id: Customer identifier
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)

    Returns:
        dict with total_kwh, peak_kwh, off_peak_kwh, peak_cost, off_peak_cost,
        standing_charge, total_cost, currency

    Raises:
        ValueError: If customer not found or dates invalid
        RuntimeError: If query execution fails
    """
    logger.info(
        "Calculating tariff for customer %d from %s to %s",
        customer_id,
        start_date,
        end_date,
    )

    # Fetch customer segment
    customer_query = f"SELECT segment FROM customers WHERE customer_id = {customer_id}"
    customer_result = sql_executor(customer_query, row_limit=1)

    if not customer_result["rows"]:
        raise ValueError(f"Customer {customer_id} not found")

    segment = customer_result["rows"][0][0]

    # Fetch tariff for segment
    tariff_query = f"SELECT peak_rate_per_kwh, off_peak_rate_per_kwh, standing_charge, peak_hours FROM tariffs WHERE segment = '{segment}'"
    tariff_result = sql_executor(tariff_query, row_limit=1)

    if not tariff_result["rows"]:
        raise ValueError(f"No tariff found for segment '{segment}'")

    peak_rate, off_peak_rate, daily_charge, peak_hours_str = tariff_result["rows"][0]
    peak_ranges = _parse_peak_hours(peak_hours_str)

    # Fetch readings for date range
    readings_query = f"""
        SELECT timestamp, consumption_kwh
        FROM readings
        WHERE customer_id = {customer_id}
        AND timestamp >= '{start_date}'
        AND timestamp <= '{end_date} 23:59:59'
    """
    readings_result = sql_executor(readings_query, row_limit=10000)

    if not readings_result["rows"]:
        raise ValueError(f"No readings found for customer {customer_id} in date range")

    # Calculate peak and off-peak consumption
    import pandas as pd

    readings_df = pd.DataFrame(
        readings_result["rows"], columns=readings_result["columns"]
    )

    peak_kwh = 0.0
    off_peak_kwh = 0.0

    for _, row in readings_df.iterrows():
        timestamp = pd.to_datetime(row["timestamp"])
        consumption = row["consumption_kwh"]
        hour = timestamp.hour

        if _is_peak_hour(hour, peak_ranges):
            peak_kwh += consumption
        else:
            off_peak_kwh += consumption

    # Calculate costs
    peak_cost = peak_kwh * peak_rate
    off_peak_cost = off_peak_kwh * off_peak_rate
    standing_charge = _calculate_standing_charge(start_date, end_date, daily_charge)
    total_cost = peak_cost + off_peak_cost + standing_charge

    logger.info(
        "Bill calculated: %.2f kWh (%.2f peak, %.2f off-peak) = £%.2f",
        peak_kwh + off_peak_kwh,
        peak_kwh,
        off_peak_kwh,
        total_cost,
    )

    return {
        "customer_id": customer_id,
        "start_date": start_date,
        "end_date": end_date,
        "total_kwh": round(peak_kwh + off_peak_kwh, 2),
        "peak_kwh": round(peak_kwh, 2),
        "off_peak_kwh": round(off_peak_kwh, 2),
        "peak_cost": round(peak_cost, 2),
        "off_peak_cost": round(off_peak_cost, 2),
        "standing_charge": round(standing_charge, 2),
        "total_cost": round(total_cost, 2),
        "currency": "GBP",
    }


def calculate_load_factor(
    customer_id: int,
    start_date: str,
    end_date: str,
) -> dict[str, float]:
    """
    Calculate load factor (efficiency metric) for customer over date range.

    Load factor = average_load / peak_load
    - Higher is better (more consistent usage, less strain on grid)
    - Range: 0.0 to 1.0
    - Typical: residential ~0.3-0.4, commercial ~0.5-0.7

    Args:
        customer_id: Customer identifier
        start_date: Start date (ISO format: YYYY-MM-DD)
        end_date: End date (ISO format: YYYY-MM-DD)

    Returns:
        dict with load_factor, average_load_kwh, peak_load_kwh,
        reading_count, period_days

    Raises:
        ValueError: If customer not found or no readings in range
        RuntimeError: If query execution fails
    """
    from datetime import datetime

    logger.info(
        "Calculating load factor for customer %d from %s to %s",
        customer_id,
        start_date,
        end_date,
    )

    # Fetch readings for date range
    readings_query = f"""
        SELECT consumption_kwh
        FROM readings
        WHERE customer_id = {customer_id}
        AND timestamp >= '{start_date}'
        AND timestamp <= '{end_date} 23:59:59'
    """
    readings_result = sql_executor(readings_query, row_limit=10000)

    if not readings_result["rows"]:
        raise ValueError(f"No readings found for customer {customer_id} in date range")

    # Calculate metrics
    import pandas as pd

    readings_df = pd.DataFrame(
        readings_result["rows"], columns=readings_result["columns"]
    )

    consumptions = readings_df["consumption_kwh"].values
    average_load = float(consumptions.mean())
    peak_load = float(consumptions.max())
    load_factor = average_load / peak_load if peak_load > 0 else 0.0

    # Calculate period days
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    period_days = (end_dt - start_dt).days + 1

    logger.info(
        "Load factor calculated: %.3f (avg=%.3f kWh, peak=%.3f kWh, %d readings)",
        load_factor,
        average_load,
        peak_load,
        len(consumptions),
    )

    return {
        "customer_id": customer_id,
        "start_date": start_date,
        "end_date": end_date,
        "load_factor": round(load_factor, 3),
        "average_load_kwh": round(average_load, 3),
        "peak_load_kwh": round(peak_load, 3),
        "reading_count": len(consumptions),
        "period_days": period_days,
    }


def plot_generator(
    query: str,
    plot_type: str,
    x_column: str,
    y_column: str,
    title: str = "",
) -> str:
    """
    Generate and save a plot from SQL query results.

    Auto-generates filename as: plot_{timestamp}_{plot_type}.png
    Saves to config.Paths.PLOTS_DIR

    Args:
        query: SQL SELECT query to fetch data
        plot_type: Type of plot (line, bar, scatter)
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Optional plot title

    Returns:
        Absolute path to saved plot file (as string)

    Raises:
        ValueError: If plot_type invalid or columns not found
        RuntimeError: If query execution or plotting fails
    """
    from datetime import datetime

    import matplotlib.pyplot as plt

    # Validate plot type
    if plot_type not in config.ToolConfig.ALLOWED_PLOT_TYPES:
        raise ValueError(
            f"Invalid plot type '{plot_type}'. "
            f"Allowed: {', '.join(config.ToolConfig.ALLOWED_PLOT_TYPES)}"
        )

    logger.info("Generating %s plot: %s vs %s", plot_type, x_column, y_column)

    # Fetch data
    result = sql_executor(query)

    # Convert to DataFrame
    import pandas as pd

    df = pd.DataFrame(result["rows"], columns=result["columns"])

    # Validate columns
    if x_column not in df.columns:
        raise ValueError(
            f"X column '{x_column}' not found. Available: {', '.join(df.columns)}"
        )
    if y_column not in df.columns:
        raise ValueError(
            f"Y column '{y_column}' not found. Available: {', '.join(df.columns)}"
        )

    try:
        # Create plot
        plt.figure(figsize=(10, 6))

        if plot_type == "line":
            plt.plot(df[x_column], df[y_column], marker="o")
        elif plot_type == "bar":
            plt.bar(df[x_column], df[y_column])
        elif plot_type == "scatter":
            plt.scatter(df[x_column], df[y_column])

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title if title else f"{y_column} vs {x_column}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}_{plot_type}.png"
        filepath = config.Paths.PLOTS_DIR / filename

        # Save and close
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()  # Prevent memory leak

        logger.info("Plot saved to %s", filepath)

        return str(filepath)

    except Exception as e:
        plt.close()  # Ensure cleanup even on error
        logger.error("Plot generation failed: %s", e)
        raise RuntimeError(f"Plot generation failed: {e}") from e
