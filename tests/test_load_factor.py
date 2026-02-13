"""Test calculate_load_factor with segment context."""
from src.tools import library

# Get customer segments for context
print("=== Customer Segments ===")
result = library.sql_executor("SELECT customer_id, segment FROM customers WHERE customer_id IN (1, 5, 50)", row_limit=10)
import pandas as pd
customers_df = pd.DataFrame(result['rows'], columns=result['columns'])
print(customers_df.to_string(index=False))

start_date = "2026-02-06"
end_date = "2026-02-13"

# Test 1: Customer 1 (commercial from earlier investigation)
print(f"\n=== Test 1: Customer 1 (commercial) - {start_date} to {end_date} ===")
lf1 = library.calculate_load_factor(1, start_date, end_date)
print(f"Load factor: {lf1['load_factor']:.3f}")
print(f"Average load: {lf1['average_load_kwh']:.3f} kWh per interval")
print(f"Peak load: {lf1['peak_load_kwh']:.3f} kWh")
print(f"Readings: {lf1['reading_count']} over {lf1['period_days']} days")
print(f"Efficiency: {'Good (consistent)' if lf1['load_factor'] > 0.5 else 'Variable (peak-heavy)'}")

# Test 2: Customer 5 (unknown segment - will discover)
print(f"\n=== Test 2: Customer 5 ({customers_df[customers_df['customer_id']==5]['segment'].values[0]}) - {start_date} to {end_date} ===")
lf2 = library.calculate_load_factor(5, start_date, end_date)
print(f"Load factor: {lf2['load_factor']:.3f}")
print(f"Average load: {lf2['average_load_kwh']:.3f} kWh per interval")
print(f"Peak load: {lf2['peak_load_kwh']:.3f} kWh")
print(f"Comparison to customer 1: {lf2['load_factor'] / lf1['load_factor']:.2f}x")

# Test 3: Invalid customer (should fail)
print("\n=== Test 3: Invalid customer (should fail) ===")
try:
    library.calculate_load_factor(9999, start_date, end_date)
    print("ERROR: Should have raised ValueError")
except ValueError as e:
    print(f"OK - Caught expected error: {e}")

# Test 4: Date range with no readings (should fail)
print("\n=== Test 4: Date range with no readings (should fail) ===")
try:
    library.calculate_load_factor(1, "2020-01-01", "2020-01-07")
    print("ERROR: Should have raised ValueError")
except ValueError as e:
    print(f"OK - Caught expected error: {e}")
