"""
Generate ground truth values for training queries using verified tools.

This script runs our verified tools directly on the database to get deterministic
ground truth answers for training queries. No LLM involved.
"""

from src.tools.library import (
    pandas_aggregator,
    tariff_calculator,
    calculate_load_factor,
    sql_executor,
)

# 10 sample queries covering different tool types
SAMPLE_QUERIES = [
    {
        "query": "What is the average consumption for residential customers?",
        "category": "aggregation",
        "tool": "pandas_aggregator",
        "args": {
            "query": "SELECT * FROM readings r JOIN customers c ON r.customer_id = c.customer_id WHERE c.segment = 'residential'",
            "operation": "mean",
            "column": "consumption_kwh",
        },
    },
    {
        "query": "What is the total consumption for commercial customers?",
        "category": "aggregation",
        "tool": "pandas_aggregator",
        "args": {
            "query": "SELECT * FROM readings r JOIN customers c ON r.customer_id = c.customer_id WHERE c.segment = 'commercial'",
            "operation": "sum",
            "column": "consumption_kwh",
        },
    },
    {
        "query": "Calculate the electricity bill for customer 3 in January 2026",
        "category": "billing",
        "tool": "tariff_calculator",
        "args": {"customer_id": 3, "start_date": "2026-01-01", "end_date": "2026-01-31"},
    },
    {
        "query": "Calculate the electricity bill for customer 5 in February 2026",
        "category": "billing",
        "tool": "tariff_calculator",
        "args": {"customer_id": 5, "start_date": "2026-02-01", "end_date": "2026-02-28"},
    },
    {
        "query": "What is the load factor for customer 5 between 2026-01-01 and 2026-01-15?",
        "category": "load_factor",
        "tool": "calculate_load_factor",
        "args": {
            "customer_id": 5,
            "start_date": "2026-01-01",
            "end_date": "2026-01-15",
        },
    },
    {
        "query": "What is the load factor for customer 10 in January 2026?",
        "category": "load_factor",
        "tool": "calculate_load_factor",
        "args": {
            "customer_id": 10,
            "start_date": "2026-01-01",
            "end_date": "2026-01-31",
        },
    },
    {
        "query": "What is the maximum consumption recorded for any customer?",
        "category": "aggregation",
        "tool": "pandas_aggregator",
        "args": {
            "query": "SELECT * FROM readings",
            "operation": "max",
            "column": "consumption_kwh",
        },
    },
    {
        "query": "How many customers are in the database?",
        "category": "count",
        "tool": "sql_executor",
        "args": {"query": "SELECT COUNT(*) as customer_count FROM customers"},
    },
    {
        "query": "Calculate the electricity bill for customer 1 in January 2026",
        "category": "billing",
        "tool": "tariff_calculator",
        "args": {"customer_id": 1, "start_date": "2026-01-01", "end_date": "2026-01-31"},
    },
    {
        "query": "What is the average consumption for customer 7 in January?",
        "category": "aggregation",
        "tool": "pandas_aggregator",
        "args": {
            "query": "SELECT * FROM readings WHERE customer_id = 7 AND timestamp >= '2026-01-01' AND timestamp < '2026-02-01'",
            "operation": "mean",
            "column": "consumption_kwh",
        },
    },
]


def extract_ground_truth_value(tool_name: str, result: dict) -> float | None:
    """
    Extract the ground truth numerical value from tool result.
    
    Uses explicit key access per tool type to avoid extraction errors.
    
    Args:
        tool_name: Name of the tool that produced the result
        result: Tool execution result dictionary
        
    Returns:
        Extracted numerical value or None if extraction failed
    """
    try:
        if tool_name == "tariff_calculator":
            # Returns: {"total_cost": float, "peak_kwh": float, ...}
            return float(result["total_cost"])
        
        elif tool_name == "calculate_load_factor":
            # Returns: {"load_factor": float, "avg_load": float, ...}
            return float(result["load_factor"])
        
        elif tool_name == "pandas_aggregator":
            # Returns: {"result": float, "operation": str, ...}
            return float(result["result"])
        
        elif tool_name == "sql_executor":
            # Returns: {"columns": [...], "rows": [[value]], ...}
            if "rows" in result and len(result["rows"]) > 0:
                return float(result["rows"][0][0])
        
        return None
    
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(f"   ERROR extracting value: {e}")
        return None


def generate_ground_truth():
    """Generate ground truth for all sample queries."""
    print("=" * 80)
    print("GENERATING GROUND TRUTH FOR 10 SAMPLE QUERIES")
    print("=" * 80)
    
    results = []
    
    for i, query_data in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{i}. {query_data['query']}")
        print(f"   Category: {query_data['category']}")
        print(f"   Tool: {query_data['tool']}")
        
        try:
            # Execute the tool
            if query_data["tool"] == "pandas_aggregator":
                result = pandas_aggregator(**query_data["args"])
            elif query_data["tool"] == "tariff_calculator":
                result = tariff_calculator(**query_data["args"])
            elif query_data["tool"] == "calculate_load_factor":
                result = calculate_load_factor(**query_data["args"])
            elif query_data["tool"] == "sql_executor":
                result = sql_executor(**query_data["args"])
            else:
                raise ValueError(f"Unknown tool: {query_data['tool']}")
            
            # Extract numerical value using tool-specific logic
            expected_value = extract_ground_truth_value(query_data["tool"], result)
            
            if expected_value is None:
                print(f"   ERROR: Could not extract numerical value from result")
                print(f"   Result: {result}")
                continue
            
            print(f"   Expected value: {expected_value}")
            
            # Build training query entry
            training_entry = {
                "query": query_data["query"],
                "category": query_data["category"],
                "expected_value": expected_value,
                "tolerance": 0.05,  # 5% tolerance
            }
            
            results.append(training_entry)
            
        except Exception as e:
            print(f"   ERROR: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Generated {len(results)}/10 ground truth values")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    results = generate_ground_truth()
    
    print("\n" + "=" * 80)
    print("GROUND TRUTH RESULTS")
    print("=" * 80)
    for i, entry in enumerate(results, 1):
        print(f"{i}. {entry['query']}")
        print(f"   Expected: {entry['expected_value']:.4f}, Tolerance: {entry['tolerance']*100}%")
    
    # Save to JSON
    output_path = Path("data/ground_truth_sample.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Saved to: {output_path.absolute()}")
    print("=" * 80)
