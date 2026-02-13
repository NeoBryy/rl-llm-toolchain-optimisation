"""Test sql_executor safety check."""
from src.tools import library

try:
    library.sql_executor("DELETE FROM customers")
    print("ERROR: Should have raised ValueError")
except ValueError as e:
    print("Safety test (should fail):")
    print("Caught expected error:", str(e))
