"""
Logging Artifacts Example

This example demonstrates how to log different types of artifacts:
- Files (CSV data)
- Text content (summary reports)
- Matplotlib figures (visualizations)
"""

import random
import time
from pathlib import Path

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

# 1. Log CSV file as artifact
csv_file = Path("data_results.csv")
with open(csv_file, "w") as f:
    f.write("index,value\n")
    for idx, value in data_points:
        f.write(f"{idx},{value:.2f}\n")

yanex.log_artifact("results.csv", csv_file)
csv_file.unlink()  # Clean up local file
print("✓ Logged CSV artifact")

# 2. Log text summary as artifact
summary = f"""Data Analysis Summary
=====================
Samples analyzed: {num_samples}
Average value: {avg:.2f}
Maximum value: {max_val:.2f}
Minimum value: {min_val:.2f}
Range: {max_val - min_val:.2f}
"""

yanex.log_text(summary, "summary.txt")
print("✓ Logged text summary")

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

    yanex.log_matplotlib_figure(fig, "analysis_plot.png", dpi=150)
    plt.close(fig)
    print("✓ Logged matplotlib figure")

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
