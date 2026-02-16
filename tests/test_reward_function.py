"""
Tests for the RL reward function.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.rl.reward_function import calculate_reward, _extract_number


class TestRewardFunction:
    
    def test_extract_number(self):
        """Test number extraction logic."""
        # Simple number
        assert _extract_number("The answer is 42") == 42.0
        
        # Decimal
        assert _extract_number("Value: 123.45") == 123.45
        
        # Multiple numbers - should get last one
        assert _extract_number("I found 5 items and the total is 100.50") == 100.50
        
        # Currency
        assert _extract_number("The cost is £250.00") == 250.0
        
        # No number
        assert _extract_number("No value found") is None

    def test_reward_tool_usage_and_success(self):
        """Test reward components for tool usage and success."""
        query_data = {"expected_value": 10.0}
        
        # Case 1: Success with tools (Base reward check)
        # Usage (+2) + Success (+3) + Accuracy (0 - since no answer provided in this mock)
        result = {
            "tool_calls": [{"tool": "some_tool"}],
            "success": True,
            "answer": "The answer is 0" # Wrong answer
        }
        # 2 + 3 + 0 (0 != 10) = 5.0
        # Wait, if answer is 0 and expected is 10, accuracy is 0.
        assert calculate_reward(query_data, result) == 5.0

        # Case 2: Failure with tools
        # Usage (+2) + Success (-3) = -1.0
        result_fail = {
            "tool_calls": [{"tool": "some_tool"}],
            "success": False,
            "answer": ""
        }
        assert calculate_reward(query_data, result_fail) == -1.0
        
        # Case 3: No tools used (Fabrication)
        # Usage (-5)
        result_no_tools = {
            "tool_calls": [],
            "success": True,
            "answer": "10.0"
        }
        assert calculate_reward(query_data, result_no_tools) == -5.0

    def test_reward_accuracy_numerical(self):
        """Test numerical accuracy rewards."""
        query_data = {
            "expected_value": 100.0, 
            "tolerance": 0.05, 
            "tolerance_type": "relative"
        }
        
        # Correct answer within tolerance
        result_good = {
            "tool_calls": [{"tool": "t"}], 
            "success": True, 
            "answer": "The result is 102.0" # 2% error, within 5%
        }
        # 2 + 3 + 5 = 10.0
        assert calculate_reward(query_data, result_good) == 10.0
        
        # Incorrect answer outside tolerance
        result_bad = {
            "tool_calls": [{"tool": "t"}], 
            "success": True, 
            "answer": "The result is 110.0" # 10% error
        }
        # 2 + 3 + 0 = 5.0
        assert calculate_reward(query_data, result_bad) == 5.0
        
        # Exact match required
        query_exact = {
            "expected_value": 5.0,
            "tolerance_type": "exact"
        }
        result_exact = {
            "tool_calls": [{"tool": "t"}],
            "success": True,
            "answer": "Count is 5"
        }
        assert calculate_reward(query_exact, result_exact) == 10.0
        
        result_exact_fail = {
            "tool_calls": [{"tool": "t"}],
            "success": True,
            "answer": "Count is 5.1"
        }
        assert calculate_reward(query_exact, result_exact_fail) == 5.0

    def test_reward_plot_generation(self, tmp_path):
        """Test plot generation rewards using real temp file."""
        # Create a real fake plot file
        fake_plot = tmp_path / "plot_20260101_line.png"
        fake_plot.write_bytes(b"fake image data")
        
        query_data = {
            "expected_value": True,
            "tolerance_type": "plot_exists"
        }
        
        result_plot = {
            "tool_calls": [{
                "tool": "plot_generator",
                "success": True,
                "result": str(fake_plot)
            }],
            "success": True,
            "answer": "Plot created"
        }
        
        # Patch only PLOTS_DIR to point to tmp_path
        with patch("src.rl.reward_function.config") as mock_config:
            mock_config.Paths.PLOTS_DIR = tmp_path
            assert calculate_reward(query_data, result_plot) == 10.0
            
            # Test: file missing
            fake_plot.unlink()
            assert calculate_reward(query_data, result_plot) == 5.0
