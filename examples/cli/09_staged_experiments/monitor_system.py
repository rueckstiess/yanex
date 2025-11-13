"""
Staged Experiments Example

Demonstrates preparing experiments to run later using --stage.
Useful for batch processing, overnight runs, or remote execution.
"""

import random
import time

import yanex

# Get parameters
duration = yanex.get_param("duration", default=5)
interval = yanex.get_param("interval", default=1)
metric_type = yanex.get_param("metric_type", default="cpu")

print(
    f"Monitoring {metric_type} for {duration} seconds (sampling every {interval}s)..."
)

# Collect metrics
samples = []
num_samples = int(duration / interval)

for i in range(num_samples):
    # Simulate metric collection
    if metric_type == "cpu":
        value = random.uniform(10, 90)
    elif metric_type == "memory":
        value = random.uniform(30, 80)
    elif metric_type == "disk":
        value = random.uniform(20, 70)
    else:
        value = random.uniform(0, 100)

    samples.append(value)
    print(f"  Sample {i + 1}/{num_samples}: {metric_type}={value:.1f}%")
    time.sleep(interval)

# Calculate statistics
avg_value = sum(samples) / len(samples)
max_value = max(samples)
min_value = min(samples)

print("\nMonitoring complete:")
print(f"  Average {metric_type}: {avg_value:.1f}%")
print(f"  Peak {metric_type}: {max_value:.1f}%")
print(f"  Min {metric_type}: {min_value:.1f}%")

# Log results
yanex.log_metrics(
    {
        "metric_type": metric_type,
        "duration": duration,
        "samples_collected": len(samples),
        "average": avg_value,
        "peak": max_value,
        "minimum": min_value,
    }
)
