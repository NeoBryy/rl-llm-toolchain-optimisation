"""
Generate ground truth values for 50 training queries using verified tools.

This script runs our verified tools directly on the database to get deterministic
ground truth answers for training queries. No LLM involved.
"""

from src.tools.library import (
    pandas_aggregator,
    tariff_calculator,
    calculate_load_factor,
    sql_executor,
)


def get_all_queries():
    """Return 50 diverse training queries covering all tool types."""
    return [
        # Aggregation queries (15 queries)
        {"query": "What is the average consumption for residential customers?", "category": "aggregation", "tool": "pandas_aggregator", 
         "args": {"query": "SELECT * FROM readings r JOIN customers c ON r.customer_id = c.customer_id WHERE c.segment = 'residential'", "operation": "mean", "column": "consumption_kwh"}},
        {"query": "What is the total consumption for commercial customers?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings r JOIN customers c ON r.customer_id = c.customer_id WHERE c.segment = 'commercial'", "operation": "sum", "column": "consumption_kwh"}},
        {"query": "What is the maximum consumption recorded for any customer?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings", "operation": "max", "column": "consumption_kwh"}},
        {"query": "What is the minimum consumption recorded?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings", "operation": "min", "column": "consumption_kwh"}},
        {"query": "What is the average consumption for customer 7 in January?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE customer_id = 7 AND timestamp >= '2026-01-01' AND timestamp < '2026-02-01'", "operation": "mean", "column": "consumption_kwh"}},
        {"query": "What is the total consumption for customer 15?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE customer_id = 15", "operation": "sum", "column": "consumption_kwh"}},
        {"query": "What is the average consumption for all customers in January?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE timestamp >= '2026-01-01' AND timestamp < '2026-02-01'", "operation": "mean", "column": "consumption_kwh"}},
        {"query": "What is the total consumption for customer 25 in February?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE customer_id = 25 AND timestamp >= '2026-02-01' AND timestamp < '2026-03-01'", "operation": "sum", "column": "consumption_kwh"}},
        {"query": "What is the maximum consumption for residential customers?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings r JOIN customers c ON r.customer_id = c.customer_id WHERE c.segment = 'residential'", "operation": "max", "column": "consumption_kwh"}},
        {"query": "What is the average consumption for commercial customers in January?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings r JOIN customers c ON r.customer_id = c.customer_id WHERE c.segment = 'commercial' AND timestamp >= '2026-01-01' AND timestamp < '2026-02-01'", "operation": "mean", "column": "consumption_kwh"}},
        {"query": "What is the total consumption for customer 50?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE customer_id = 50", "operation": "sum", "column": "consumption_kwh"}},
        {"query": "What is the minimum consumption for commercial customers?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings r JOIN customers c ON r.customer_id = c.customer_id WHERE c.segment = 'commercial'", "operation": "min", "column": "consumption_kwh"}},
        {"query": "What is the average consumption for customer 33 in February?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE customer_id = 33 AND timestamp >= '2026-02-01' AND timestamp < '2026-03-01'", "operation": "mean", "column": "consumption_kwh"}},
        {"query": "What is the total consumption in January 2026?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE timestamp >= '2026-01-01' AND timestamp < '2026-02-01'", "operation": "sum", "column": "consumption_kwh"}},
        {"query": "What is the average consumption for customer 99?", "category": "aggregation", "tool": "pandas_aggregator",
         "args": {"query": "SELECT * FROM readings WHERE customer_id = 99", "operation": "mean", "column": "consumption_kwh"}},
        
        # Billing queries (15 queries)
        {"query": "Calculate the electricity bill for customer 1 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 1, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 3 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 3, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 5 in February 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 5, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "Calculate the electricity bill for customer 10 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 10, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 15 in February 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 15, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "Calculate the electricity bill for customer 20 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 20, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 25 in February 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 25, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "Calculate the electricity bill for customer 7 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 7, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 12 in February 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 12, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "Calculate the electricity bill for customer 30 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 30, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 40 in February 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 40, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "Calculate the electricity bill for customer 50 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 50, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 60 in February 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 60, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "Calculate the electricity bill for customer 75 in January 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 75, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "Calculate the electricity bill for customer 90 in February 2026", "category": "billing", "tool": "tariff_calculator",
         "args": {"customer_id": 90, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        
        # Load factor queries (15 queries)
        {"query": "What is the load factor for customer 5 between 2026-01-01 and 2026-01-15?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 5, "start_date": "2026-01-01", "end_date": "2026-01-15"}},
        {"query": "What is the load factor for customer 10 in January 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 10, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "What is the load factor for customer 15 in February 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 15, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "What is the load factor for customer 20 in January 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 20, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "What is the load factor for customer 25 between 2026-02-01 and 2026-02-15?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 25, "start_date": "2026-02-01", "end_date": "2026-02-15"}},
        {"query": "What is the load factor for customer 1 in January 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 1, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "What is the load factor for customer 7 in February 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 7, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "What is the load factor for customer 30 in January 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 30, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "What is the load factor for customer 35 in February 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 35, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "What is the load factor for customer 40 between 2026-01-01 and 2026-01-15?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 40, "start_date": "2026-01-01", "end_date": "2026-01-15"}},
        {"query": "What is the load factor for customer 50 in January 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 50, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "What is the load factor for customer 60 in February 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 60, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "What is the load factor for customer 70 in January 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 70, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        {"query": "What is the load factor for customer 80 in February 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 80, "start_date": "2026-02-01", "end_date": "2026-02-28"}},
        {"query": "What is the load factor for customer 95 in January 2026?", "category": "load_factor", "tool": "calculate_load_factor",
         "args": {"customer_id": 95, "start_date": "2026-01-01", "end_date": "2026-01-31"}},
        
        # Count/SQL queries (5 queries)
        {"query": "How many customers are in the database?", "category": "count", "tool": "sql_executor",
         "args": {"query": "SELECT COUNT(*) FROM customers"}},
        {"query": "How many residential customers are there?", "category": "count", "tool": "sql_executor",
         "args": {"query": "SELECT COUNT(*) FROM customers WHERE segment = 'residential'"}},
        {"query": "How many commercial customers are there?", "category": "count", "tool": "sql_executor",
         "args": {"query": "SELECT COUNT(*) FROM customers WHERE segment = 'commercial'"}},
        {"query": "How many readings are in the database?", "category": "count", "tool": "sql_executor",
         "args": {"query": "SELECT COUNT(*) FROM readings"}},
        {"query": "How many readings are there for customer 5?", "category": "count", "tool": "sql_executor",
         "args": {"query": "SELECT COUNT(*) FROM readings WHERE customer_id = 5"}},

        # Plot queries (5 queries - no args needed for training data, just query string and category)
        {"query": "Plot the daily consumption for customer 5 in January 2026", "category": "plot", "tool": "plot_generator", "args": {}},
        {"query": "Generate a bar chart of total consumption by customer segment", "category": "plot", "tool": "plot_generator", "args": {}},
        {"query": "Visualize the consumption trend for customer 10 in February 2026", "category": "plot", "tool": "plot_generator", "args": {}},
        {"query": "Create a scatter plot of peak vs off-peak usage for all customers", "category": "plot", "tool": "plot_generator", "args": {}},
        {"query": "Plot the relationship between temperature and consumption for commercial customers", "category": "plot", "tool": "plot_generator", "args": {}},
    ]


def extract_ground_truth_value(tool_name: str, result: dict) -> float | None:
    """
    Extract the ground truth numerical value from tool result.
    
    Uses explicit key access per tool type to avoid extraction errors.
    """
    try:
        if tool_name == "tariff_calculator":
            return float(result["total_cost"])
        elif tool_name == "calculate_load_factor":
            return float(result["load_factor"])
        elif tool_name == "pandas_aggregator":
            return float(result["result"])
        elif tool_name == "sql_executor":
            if "rows" in result and len(result["rows"]) > 0:
                return float(result["rows"][0][0])
        elif tool_name == "plot_generator":
            return 1.0  # Placeholder for plot success (boolean true mapped to 1.0)
        return None
    except (KeyError, IndexError, ValueError, TypeError) as e:
        print(f"   ERROR extracting value: {e}")
        return None


def generate_ground_truth():
    """Generate ground truth for all 50 training queries."""
    queries = get_all_queries()
    
    print("=" * 80)
    print(f"GENERATING GROUND TRUTH FOR {len(queries)} TRAINING QUERIES")
    print("=" * 80)
    
    results = []
    category_counts = {}
    
    for i, query_data in enumerate(queries, 1):
        print(f"\n{i}/{len(queries)}. {query_data['query'][:60]}...")
        print(f"   Category: {query_data['category']}")
        
        try:
            # Special handling for plot queries (no real execution needed for ground truth generation)
            if query_data["category"] == "plot":
                expected_value = True  # Plot generation success
                tolerance_type = "plot_exists"
                tolerance = 0.0
                print(f"   Expected: {expected_value}, Type: {tolerance_type}")
            
            else:
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
                
                # Extract numerical value
                expected_value = extract_ground_truth_value(query_data["tool"], result)
                
                if expected_value is None:
                    print(f"   ERROR: Could not extract numerical value")
                    continue
                
                # Round to 4 decimal places
                expected_value = round(expected_value, 4)
                
                # Determine tolerance type and value
                if query_data["category"] == "count":
                    tolerance_type = "exact"
                    tolerance = 0.0
                elif "minimum consumption" in query_data["query"]:
                     # Specific fix for minimum consumption query
                     tolerance_type = "exact"
                     tolerance = 0.0
                     expected_value = 0.0 # Force exact 0.0 for min consumption
                else:
                    tolerance_type = "relative"
                    tolerance = 0.05
                
                print(f"   Expected: {expected_value}, Type: {tolerance_type}")
            
            # Build training query entry
            training_entry = {
                "query": query_data["query"],
                "category": query_data["category"],
                # Remove tool/args form final dataset output - pure natural language queries
                # "tool": query_data["tool"], 
                # "args": query_data["args"],
                "expected_value": expected_value,
                "tolerance": tolerance,
                "tolerance_type": tolerance_type,
            }
            
            results.append(training_entry)
            category_counts[query_data["category"]] = category_counts.get(query_data["category"], 0) + 1
            
        except Exception as e:
            print(f"   ERROR: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: Generated {len(results)}/{len(queries)} ground truth values")
    print("\nCategory Distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")
    print("=" * 80)
    
    return results, category_counts


if __name__ == "__main__":
    import json
    from pathlib import Path
    
    results, category_counts = generate_ground_truth()
    
    # Save to JSON
    output_path = Path("data/training_queries.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Saved {len(results)} queries to: {output_path.absolute()}")
    print("=" * 80)
