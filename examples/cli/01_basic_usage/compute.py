"""
Basic Yanex Example: Simple Computation

This minimal example shows the core yanex API:
- get_param() to read parameters
- log_metrics() to save results

Works standalone (python compute.py) or with yanex (yanex run compute.py).
"""

import time

import yanex

# Get parameters with defaults
num_items = yanex.get_param("num_items", default=1000)
delay_ms = yanex.get_param("delay_ms", default=100)

print(f"Processing {num_items} items with {delay_ms}ms delay...")

# Simulate some work
start_time = time.time()
time.sleep(delay_ms / 1000.0)  # Convert ms to seconds
total_time = time.time() - start_time

# Log results (no-op when run standalone, saved when run with yanex)
yanex.log_metrics(
    {
        "num_items": num_items,
        "total_time_seconds": total_time,
        "items_per_second": num_items / total_time,
    }
)

print(f"Completed in {total_time:.3f} seconds")
print(f"Throughput: {num_items / total_time:.0f} items/second")
