"""
Three-way comparative analysis: GPT-4o-mini vs Qwen (untrained) vs Qwen (SFT).
Generates results/final_comparison.md
"""

import json
from pathlib import Path
from collections import defaultdict

RESULT_FILES = {
    "GPT-4o-mini": "results/baselines/baseline_gpt4.json",
    "Qwen (Untrained)": "results/baselines/baseline_qwen_untrained.json",
    "Qwen (SFT)": "results/sft/qwen_sft_full_results.json",
}

CATEGORIES = ["aggregation", "billing", "count", "load_factor", "visualization"]


def load_results(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"  WARNING: {path} not found — skipping.")
        return []
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(results: list[dict]) -> dict:
    if not results:
        return {}

    total = len(results)
    successes = [r for r in results if r.get("success", False)]
    tool_uses = [r for r in results if len(r.get("tool_calls", [])) > 0]
    exec_successes = [r for r in results if r.get("success", False) and len(r.get("tool_calls", [])) > 0]
    latencies = [r.get("execution_time", 0) for r in results if r.get("execution_time")]

    # Per-category accuracy
    cat_stats = defaultdict(lambda: {"success": 0, "total": 0})
    for r in results:
        cat = r.get("query_data", {}).get("category", "unknown")
        cat_stats[cat]["total"] += 1
        if r.get("success", False):
            cat_stats[cat]["success"] += 1

    return {
        "overall_accuracy": 100 * len(successes) / total,
        "total": total,
        "success_count": len(successes),
        "tool_selection_rate": 100 * len(tool_uses) / total,
        "execution_success_rate": 100 * len(exec_successes) / len(tool_uses) if tool_uses else 0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "categories": {cat: 100 * s["success"] / s["total"] for cat, s in cat_stats.items() if s["total"] > 0},
    }


def fmt(value, suffix="%", decimals=1) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}{suffix}"


def main():
    print("Loading results...")
    all_results = {}
    all_metrics = {}

    for model_name, path in RESULT_FILES.items():
        results = load_results(path)
        all_results[model_name] = results
        all_metrics[model_name] = compute_metrics(results)

    model_names = list(RESULT_FILES.keys())

    # Build markdown
    lines = ["# Final Comparison: GPT-4o-mini vs Qwen (Untrained) vs Qwen (SFT)", ""]
    lines += ["## Overall Metrics", ""]

    header = "| Metric | " + " | ".join(model_names) + " |"
    sep = "|:---|" + "|:---:" * len(model_names) + "|"
    lines.append(header)
    lines.append(sep)

    def row(label, key, suffix="%"):
        vals = [fmt(all_metrics[m].get(key), suffix) if all_metrics.get(m) else "N/A" for m in model_names]
        return f"| {label} | " + " | ".join(vals) + " |"

    lines.append(row("**Overall Accuracy**", "overall_accuracy"))
    lines.append(row("Tool Selection Rate", "tool_selection_rate"))
    lines.append(row("Execution Success Rate", "execution_success_rate"))
    lines.append(row("Avg Latency (s)", "avg_latency", "s"))
    lines.append("")

    # Category table
    lines += ["## Category Performance", ""]
    cat_header = "| Category | " + " | ".join(model_names) + " |"
    lines.append(cat_header)
    lines.append(sep)

    for cat in CATEGORIES:
        vals = []
        for m in model_names:
            cat_acc = all_metrics.get(m, {}).get("categories", {}).get(cat)
            vals.append(fmt(cat_acc) if cat_acc is not None else "N/A")
        lines.append(f"| {cat.replace('_', ' ').title()} | " + " | ".join(vals) + " |")
    lines.append("")

    # Key findings
    sft_metrics = all_metrics.get("Qwen (SFT)", {})
    untrained_metrics = all_metrics.get("Qwen (Untrained)", {})
    gpt_metrics = all_metrics.get("GPT-4o-mini", {})

    lines += ["## Key Findings", ""]
    if sft_metrics and untrained_metrics:
        overall_delta = sft_metrics.get("overall_accuracy", 0) - untrained_metrics.get("overall_accuracy", 0)
        billing_sft = sft_metrics.get("categories", {}).get("billing", 0)
        billing_base = untrained_metrics.get("categories", {}).get("billing", 0)
        lines.append(f"- **Billing improvement:** {fmt(billing_base)} → {fmt(billing_sft)} (+{billing_sft - billing_base:.1f} pts)")
        lines.append(f"- **Overall improvement:** {fmt(untrained_metrics.get('overall_accuracy'))} → {fmt(sft_metrics.get('overall_accuracy'))} (+{overall_delta:.1f} pts)")

        # Check for regressions
        regressions = []
        for cat in CATEGORIES:
            sft_val = sft_metrics.get("categories", {}).get(cat, 0)
            base_val = untrained_metrics.get("categories", {}).get(cat, 0)
            if base_val - sft_val > 10:
                regressions.append(f"{cat} ({fmt(base_val)} → {fmt(sft_val)})")
        if regressions:
            lines.append(f"- **⚠️ Regressions (>10pts):** {', '.join(regressions)}")
        else:
            lines.append("- **No significant regressions** in other categories (all within 10pts)")

    if sft_metrics and gpt_metrics:
        gap = gpt_metrics.get("overall_accuracy", 0) - sft_metrics.get("overall_accuracy", 0)
        lines.append(f"- **Gap vs GPT-4o-mini:** {fmt(gap)} pts")

    lines += [
        "",
        "## Training Efficiency",
        "",
        "| Model | Training Data | Training Time | Billing Accuracy |",
        "|:---|:---|:---|:---|",
        "| GPT-4o-mini (Baseline) | N/A | N/A | 100.0% |",
        f"| Qwen (Untrained) | N/A | N/A | {fmt(untrained_metrics.get('categories', {}).get('billing', 0))} |",
        f"| **Qwen (SFT)** | **15 trajectories** | **~16h CPU** | **{fmt(sft_metrics.get('categories', {}).get('billing', 0))}** |",
    ]

    output = "\n".join(lines)

    out_path = Path("results/final_comparison.md")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"\nSaved to {out_path}")
    print("\n" + output)


if __name__ == "__main__":
    main()
