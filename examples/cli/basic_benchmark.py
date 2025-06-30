#!/usr/bin/env python3
"""
Basic CLI example using yanex for query benchmarking.

This script demonstrates:
- Getting experiment parameters with get_params()
- Logging benchmark results with log_results()

Run with yanex:
    yanex run basic_benchmark.py --query_count 20 --query_type SELECT
"""

import random
import time

import yanex


def simulate_query(query_type="SELECT"):
    """Simulate a database query with random execution time."""
    base_time = {"SELECT": 0.01, "INSERT": 0.02, "UPDATE": 0.03}.get(query_type, 0.01)
    execution_time = base_time + random.uniform(0, 0.02)
    time.sleep(execution_time)
    return execution_time


def run_benchmark():
    """Run a simple query benchmark."""
    # Get parameters from CLI
    params = yanex.get_params()
    print(f"Parameters: {params}")

    query_count = yanex.get_param("query_count", default=10)
    query_type = yanex.get_param("query_type", default="SELECT")

    print(f"Running {query_count} {query_type} queries...")

    # Run queries and collect times
    execution_times = []
    for i in range(query_count):
        query_time = simulate_query(query_type)
        execution_times.append(query_time)
        print(f"Query {i + 1}: {query_time:.4f}s")

    # Log final results
    avg_time = sum(execution_times) / len(execution_times)
    yanex.log_results(
        {
            "total_queries": query_count,
            "query_type": query_type,
            "avg_query_time": round(avg_time, 4),
            "min_query_time": round(min(execution_times), 4),
            "max_query_time": round(max(execution_times), 4),
        }
    )

    print(f"Completed! Average time: {avg_time:.4f}s")


if __name__ == "__main__":
    run_benchmark()
