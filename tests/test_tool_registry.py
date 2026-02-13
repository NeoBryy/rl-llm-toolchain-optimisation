from src.agents import tool_registry

print("=" * 70)
print("TOOL REGISTRY TESTS")
print("=" * 70)

# Test 1: Registry lookup - verify all 5 tools registered
print("\n=== Test 1: Registry Lookup (All 5 Tools) ===")
registry = tool_registry.get_tool_registry()
print(f"Registered tools: {list(registry.keys())}")
print(f"Count: {len(registry)}")
assert len(registry) == 5, f"Expected 5 tools, got {len(registry)}"
print("OK - PASSED - All 5 tools registered")

# Test 2: Type coercion - int (customer_id as string '1')
print("\n=== Test 2: Type Coercion (String -> Int) ===")
print("Calling: execute_tool('tariff_calculator', {'customer_id': '1', 'start_date': '2026-01-01', 'end_date': '2026-01-31'})")
print("Note: customer_id is string '1', should be coerced to int 1")
result2 = tool_registry.execute_tool(
    'tariff_calculator',
    {'customer_id': '1', 'start_date': '2026-01-01', 'end_date': '2026-01-31'}
)
print(f"Result type: {type(result2)}")
print(f"Total cost: GBP{result2['total_cost']}")
print(f"Customer ID in result: {result2['customer_id']} (type: {type(result2['customer_id'])})")
assert isinstance(result2, dict), "Result should be dict"
assert 'total_cost' in result2, "Result should have total_cost"
assert isinstance(result2['customer_id'], int), "customer_id should be coerced to int"
print("OK - PASSED - String '1' correctly coerced to int 1")

# Test 3: Type coercion - float (row_limit as string)
print("\n=== Test 3: Type Coercion (String -> Int for Optional Param) ===")
print("Calling: execute_tool('sql_executor', {'query': 'SELECT * FROM customers LIMIT 5', 'row_limit': '10'})")
result3 = tool_registry.execute_tool(
    'sql_executor',
    {'query': 'SELECT * FROM customers LIMIT 5', 'row_limit': '10'}
)
print(f"Returned {result3['row_count']} rows")
assert result3['row_count'] == 5, "Should return 5 rows (query has LIMIT 5)"
print("OK - PASSED - Optional parameter coercion works")

# Test 4: Default parameter (row_limit not provided)
print("\n=== Test 4: Default Parameter (row_limit not provided) ===")
print("Calling: execute_tool('sql_executor', {'query': 'SELECT * FROM customers'})")
result4 = tool_registry.execute_tool(
    'sql_executor',
    {'query': 'SELECT * FROM customers'}
)
print(f"Returned {result4['row_count']} rows (default limit applied)")
assert result4['row_count'] <= 100, "Should apply default limit of 100"
print("OK - PASSED - Default parameter used when not provided")

# Test 5: Missing required parameter
print("\n=== Test 5: Missing Required Parameter ===")
print("Calling: execute_tool('tariff_calculator', {'customer_id': '1'})")  # Missing dates
try:
    tool_registry.execute_tool('tariff_calculator', {'customer_id': '1'})
    print("FAILED - Should have raised ValueError")
    assert False, "Should have raised ValueError for missing required params"
except ValueError as e:
    print(f"Caught expected error: {e}")
    print("OK - PASSED - Missing required parameter detected")

# Test 6: Invalid tool name
print("\n=== Test 6: Invalid Tool Name ===")
print("Calling: execute_tool('nonexistent_tool', {})")
try:
    tool_registry.execute_tool('nonexistent_tool', {})
    print("FAILED - Should have raised ValueError")
    assert False, "Should have raised ValueError for invalid tool name"
except ValueError as e:
    print(f"Caught expected error: {e}")
    print("OK - PASSED - Invalid tool name detected")

# Bonus: Tools description format
print("\n=== Bonus: Tools Description Format ===")
desc = tool_registry.get_tools_description()
print("Generated tools description (truncated):")
print(desc[:500] + "..." if len(desc) > 500 else desc)
assert "sql_executor" in desc, "Description should contain sql_executor"
assert "tariff_calculator" in desc, "Description should contain tariff_calculator"
assert len(desc.split("\n")) > 20, "Description should be multi-line with all tools"
print("OK - PASSED - Tools description properly formatted")

print("\n" + "=" * 70)
print("ALL TESTS PASSED")
print("=" * 70)

from src.agents import tool_registry

print("=" * 70)
print("TOOL REGISTRY TESTS")
print("=" * 70)

# Test 1: Registry lookup - verify all 5 tools registered
print("\n=== Test 1: Registry Lookup (All 5 Tools) ===")
registry = tool_registry.get_tool_registry()
print(f"Registered tools: {list(registry.keys())}")
print(f"Count: {len(registry)}")
assert len(registry) == 5, f"Expected 5 tools, got {len(registry)}"
print("✓ PASSED - All 5 tools registered")

# Test 2: Type coercion - int (customer_id as string '1')
print("\n=== Test 2: Type Coercion (String → Int) ===")
print("Calling: execute_tool('tariff_calculator', {'customer_id': '1', 'start_date': '2026-01-01', 'end_date': '2026-01-31'})")
print("Note: customer_id is string '1', should be coerced to int 1")
result2 = tool_registry.execute_tool(
    'tariff_calculator',
    {'customer_id': '1', 'start_date': '2026-01-01', 'end_date': '2026-01-31'}
)
print(f"Result type: {type(result2)}")
print(f"Total cost: £{result2['total_cost']}")
print(f"Customer ID in result: {result2['customer_id']} (type: {type(result2['customer_id'])})")
assert isinstance(result2, dict), "Result should be dict"
assert 'total_cost' in result2, "Result should have total_cost"
assert isinstance(result2['customer_id'], int), "customer_id should be coerced to int"
print("✓ PASSED - String '1' correctly coerced to int 1")

# Test 3: Type coercion - float (row_limit as string)
print("\n=== Test 3: Type Coercion (String → Int for Optional Param) ===")
print("Calling: execute_tool('sql_executor', {'query': 'SELECT * FROM customers LIMIT 5', 'row_limit': '10'})")
result3 = tool_registry.execute_tool(
    'sql_executor',
    {'query': 'SELECT * FROM customers LIMIT 5', 'row_limit': '10'}
)
print(f"Returned {result3['row_count']} rows")
assert result3['row_count'] == 5, "Should return 5 rows (query has LIMIT 5)"
print("✓ PASSED - Optional parameter coercion works")

# Test 4: Default parameter (row_limit not provided)
print("\n=== Test 4: Default Parameter (row_limit not provided) ===")
print("Calling: execute_tool('sql_executor', {'query': 'SELECT * FROM customers'})")
result4 = tool_registry.execute_tool(
    'sql_executor',
    {'query': 'SELECT * FROM customers'}
)
print(f"Returned {result4['row_count']} rows (default limit applied)")
assert result4['row_count'] <= 100, "Should apply default limit of 100"
print("✓ PASSED - Default parameter used when not provided")

# Test 5: Missing required parameter
print("\n=== Test 5: Missing Required Parameter ===")
print("Calling: execute_tool('tariff_calculator', {'customer_id': '1'})")  # Missing dates
try:
    tool_registry.execute_tool('tariff_calculator', {'customer_id': '1'})
    print("✗ FAILED - Should have raised ValueError")
    assert False, "Should have raised ValueError for missing required params"
except ValueError as e:
    print(f"Caught expected error: {e}")
    print("✓ PASSED - Missing required parameter detected")

# Test 6: Invalid tool name
print("\n=== Test 6: Invalid Tool Name ===")
print("Calling: execute_tool('nonexistent_tool', {})")
try:
    tool_registry.execute_tool('nonexistent_tool', {})
    print("✗ FAILED - Should have raised ValueError")
    assert False, "Should have raised ValueError for invalid tool name"
except ValueError as e:
    print(f"Caught expected error: {e}")
    print("✓ PASSED - Invalid tool name detected")

# Bonus: Tools description format
print("\n=== Bonus: Tools Description Format ===")
desc = tool_registry.get_tools_description()
print("Generated tools description (truncated):")
print(desc[:500] + "..." if len(desc) > 500 else desc)
assert "sql_executor" in desc, "Description should contain sql_executor"
assert "tariff_calculator" in desc, "Description should contain tariff_calculator"
assert len(desc.split("\n")) > 20, "Description should be multi-line with all tools"
print("✓ PASSED - Tools description properly formatted")

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
