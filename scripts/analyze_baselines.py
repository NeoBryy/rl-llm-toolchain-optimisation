"""
Analysis script for Milestone 5: Baseline Performance.

Computes aggregate metrics from evaluation results and generates a comparison table.
"""

import json
from pathlib import Path
from collections import defaultdict

def analyze_results(results_file: Path, agent_name: str):
    """
    Compute metrics for a results file.
    """
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return None

    with open(results_file) as f:
        results = json.load(f)

    metrics = {
        "overall_accuracy": 0,
        "tool_selection_accuracy": 0,
        "execution_success_rate": 0,
        "avg_execution_time": 0,
        "categories": defaultdict(lambda: {"correct": 0, "total": 0})
    }

    total = len(results)
    if total == 0:
        return None

    correct_count = 0
    tools_called_count = 0
    execution_success_count = 0
    total_time = 0

    for res in results:
        query_data = res.get("query_data", {})
        category = query_data.get("category", "unknown")
        
        # Reward Threshold for Correctness: 5.0 (Accuracy component reward)
        # In calculate_reward, +5.0 is awarded for accuracy.
        # However, a perfect score can be up to 10.0 (+2 tool + 3 success + 5 accuracy).
        # We consider it "correct" if reward >= 5.0 (using tools + correct answer).
        is_correct = res.get("reward", 0) >= 5.0
        if is_correct:
            correct_count += 1
            metrics["categories"][category]["correct"] += 1
        
        metrics["categories"][category]["total"] += 1
        
        # Tool Selection Accuracy: did it call tools?
        # rewards[0] is +2.0 if tool_calls > 0
        has_tools = len(res.get("tool_calls", [])) > 0
        if has_tools:
            tools_called_count += 1
        
        # Execution Success: did it reach 'answer' type or success=True?
        if res.get("success", False):
            execution_success_count += 1
            
        total_time += res.get("execution_time", 0)

    metrics["overall_accuracy"] = (correct_count / total) * 100
    metrics["tool_selection_accuracy"] = (tools_called_count / total) * 100
    metrics["execution_success_rate"] = (execution_success_count / total) * 100
    metrics["avg_execution_time"] = total_time / total

    return metrics

def generate_markdown_table(gpt_metrics, qwen_metrics):
    """
    Generate the comparison markdown table.
    """
    table = "| Metric | GPT-4o-mini | Qwen 1.5B (Untrained) | Status |\n"
    table += "| :--- | :--- | :--- | :--- |\n"
    table += f"| Overall Accuracy | {gpt_metrics['overall_accuracy']:.1f}% | {qwen_metrics['overall_accuracy']:.1f}% | {'Verified' if gpt_metrics['overall_accuracy'] > 0 else 'TBD'} |\n"
    table += f"| Tool Selection Acc | {gpt_metrics['tool_selection_accuracy']:.1f}% | {qwen_metrics['tool_selection_accuracy']:.1f}% | - |\n"
    table += f"| Execution Success | {gpt_metrics['execution_success_rate']:.1f}% | {qwen_metrics['execution_success_rate']:.1f}% | - |\n"
    table += f"| Avg Time (s) | {gpt_metrics['avg_execution_time']:.1f}s | {qwen_metrics['avg_execution_time']:.1f}s | - |\n"
    
    table += "\n### Accuracy by Category\n\n"
    table += "| Category | GPT-4o-mini | Qwen 1.5B |\n"
    table += "| :--- | :--- | :--- |\n"
    
    all_categories = sorted(set(gpt_metrics["categories"].keys()) | set(qwen_metrics["categories"].keys()))
    for cat in all_categories:
        gpt_cat = gpt_metrics["categories"].get(cat, {"correct": 0, "total": 1})
        qwen_cat = qwen_metrics["categories"].get(cat, {"correct": 0, "total": 1})
        
        gpt_acc = (gpt_cat["correct"] / gpt_cat["total"]) * 100 if gpt_cat["total"] > 0 else 0
        qwen_acc = (qwen_cat["correct"] / qwen_cat["total"]) * 100 if qwen_cat["total"] > 0 else 0
        
        table += f"| {cat.capitalize()} | {gpt_acc:.1f}% | {qwen_acc:.1f}% |\n"
        
    return table

def main():
    gpt_file = Path("results/baseline_gpt4.json")
    qwen_file = Path("results/baseline_qwen_untrained.json")
    
    gpt_metrics = analyze_results(gpt_file, "GPT-4o-mini")
    qwen_metrics = analyze_results(qwen_file, "Qwen 1.5B")
    
    if not gpt_metrics or not qwen_metrics:
        print("Missing result files. Run evaluate_baselines.py first.")
        # Provide dummy data if only one or none exists for preview
        if not gpt_metrics: gpt_metrics = {"overall_accuracy": 0, "tool_selection_accuracy": 0, "execution_success_rate": 0, "avg_execution_time": 0, "categories": {}}
        if not qwen_metrics: qwen_metrics = {"overall_accuracy": 0, "tool_selection_accuracy": 0, "execution_success_rate": 0, "avg_execution_time": 0, "categories": {}}

    comparison_table = generate_markdown_table(gpt_metrics, qwen_metrics)
    
    with open("results/baseline_comparison.md", "w") as f:
        f.write("# Baseline Comparison Dashboard\n\n")
        f.write(comparison_table)
    
    print("Dashboard generated: results/baseline_comparison.md")

if __name__ == "__main__":
    main()
