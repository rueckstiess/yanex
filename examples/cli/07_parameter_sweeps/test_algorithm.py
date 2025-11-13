"""
Parameter Sweeps Example

Simple script to test algorithm performance with different parameters.
Focus is on sweep syntax, not the algorithm implementation.
"""

import random
import time

import yanex

# Get parameters
algorithm = yanex.get_param("algorithm", default="quicksort")
data_size = yanex.get_param("data_size", default=1000)
random_seed = yanex.get_param("random_seed", default=42)

print(f"Testing {algorithm} with {data_size} elements (seed={random_seed})...")

# Set random seed for reproducibility
random.seed(random_seed)

# Simulate algorithm execution
start_time = time.time()

# Simple simulation: execution time depends on algorithm and size
if algorithm == "quicksort":
    time.sleep(0.01 + data_size / 100000)
    comparisons = data_size * 2
elif algorithm == "mergesort":
    time.sleep(0.02 + data_size / 80000)
    comparisons = data_size * 3
elif algorithm == "heapsort":
    time.sleep(0.015 + data_size / 90000)
    comparisons = int(data_size * 2.5)
else:
    time.sleep(0.01)
    comparisons = data_size

execution_time = time.time() - start_time

# Calculate metrics
throughput = data_size / execution_time

print(f"  Execution time: {execution_time:.4f}s")
print(f"  Throughput: {throughput:.0f} elements/second")
print(f"  Comparisons: {comparisons}")

# Log results
yanex.log_metrics(
    {
        "algorithm": algorithm,
        "data_size": data_size,
        "execution_time": execution_time,
        "throughput": throughput,
        "comparisons": comparisons,
    }
)

print("✓ Test complete")
