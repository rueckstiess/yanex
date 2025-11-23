"""
Logging Artifacts Example

This example demonstrates how to log different types of artifacts:
- Files (CSV data)
- Text content (summary reports)
- Matplotlib figures (visualizations)
"""

import random
import time

import yanex

# Optional: Check if matplotlib is available
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: matplotlib not installed, skipping figure generation")


def analyze_data(num_samples):
    """Simulate data analysis and return results."""
    print(f"Analyzing {num_samples} data points...")
    time.sleep(0.2)  # Simulate processing

    # Generate simulated data
    data_points = [(i, random.uniform(10, 100)) for i in range(num_samples)]
    avg_value = sum(v for _, v in data_points) / len(data_points)
    max_value = max(v for _, v in data_points)
    min_value = min(v for _, v in data_points)

    return data_points, avg_value, max_value, min_value


# Get parameters
num_samples = yanex.get_param("num_samples", default=50)

# Perform analysis
data_points, avg, max_val, min_val = analyze_data(num_samples)

print(f"Analysis complete: avg={avg:.2f}, max={max_val:.2f}, min={min_val:.2f}")

# 1. Save CSV artifact directly (automatic format detection)
# Convert data to list of dicts for CSV format
csv_data = [{"index": idx, "value": value} for idx, value in data_points]
yanex.save_artifact(csv_data, "results.csv")
print("✓ Saved CSV artifact")

# 2. Log text summary as artifact
summary = f"""Data Analysis Summary
=====================
Samples analyzed: {num_samples}
Average value: {avg:.2f}
Maximum value: {max_val:.2f}
Minimum value: {min_val:.2f}
Range: {max_val - min_val:.2f}
"""

yanex.save_artifact(summary, "summary.txt")
print("✓ Saved text summary")

# 3. Log matplotlib figure as artifact (if available)
if HAS_MATPLOTLIB:
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = [idx for idx, _ in data_points]
    values = [val for _, val in data_points]

    ax.plot(indices, values, "b-", alpha=0.7, label="Data")
    ax.axhline(y=avg, color="r", linestyle="--", label=f"Average: {avg:.2f}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.set_title(f"Data Analysis Results (n={num_samples})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    yanex.save_artifact(fig, "analysis_plot.png")
    plt.close(fig)
    print("✓ Saved matplotlib figure")

# Log final metrics
yanex.log_metrics(
    {
        "num_samples": num_samples,
        "avg_value": avg,
        "max_value": max_val,
        "min_value": min_val,
    }
)

# Final status message
if yanex.is_standalone():
    print("\n✓ Script completed in standalone mode (no artifacts logged)")
else:
    print("\n✓ All artifacts logged successfully!")
