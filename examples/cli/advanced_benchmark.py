#!/usr/bin/env python3
"""
Advanced CLI example using yanex for comprehensive benchmarking with artifacts.

This script demonstrates:
- Getting experiment parameters with get_params()
- Logging benchmark results with log_results()
- Creating and logging artifacts with log_text()

Run with yanex:
    yanex run advanced_benchmark.py --config config.yaml
    yanex run advanced_benchmark.py --param workload=mixed --param duration=10
"""

import json
import random
import time

import yanex


def simulate_query(query_type="SELECT"):
    """Simulate a database query with realistic timing."""
    base_times = {"SELECT": 0.005, "INSERT": 0.015, "UPDATE": 0.020, "JOIN": 0.040}
    base_time = base_times.get(query_type, 0.010)

    # Add variance and occasional slow queries
    execution_time = base_time * random.uniform(0.8, 1.5)
    if random.random() < 0.1:  # 10% slow queries
        execution_time *= random.uniform(2, 5)

    time.sleep(execution_time)
    return execution_time


def run_advanced_benchmark():
    """Run benchmark with detailed metrics and artifacts."""
    params = yanex.get_params()
    print(f"Starting benchmark with: {params}")

    # Get configuration
    duration = yanex.get_param("duration", default=5)
    workload = yanex.get_param("workload", default="mixed")
    target_qps = yanex.get_param("target_qps", default=20)

    # Define workload mix
    workload_mix = {
        "read_heavy": [("SELECT", 0.8), ("JOIN", 0.2)],
        "write_heavy": [("INSERT", 0.6), ("UPDATE", 0.4)],
        "mixed": [("SELECT", 0.5), ("INSERT", 0.2), ("UPDATE", 0.2), ("JOIN", 0.1)],
    }

    queries = workload_mix.get(workload, workload_mix["mixed"])
    print(f"Running {workload} workload for {duration}s at {target_qps} QPS")

    # Run benchmark
    results = []
    start_time = time.time()
    query_count = 0

    while time.time() - start_time < duration:
        # Select query type based on workload
        query_type = random.choices([q[0] for q in queries], [q[1] for q in queries])[0]

        query_time = simulate_query(query_type)
        results.append(
            {
                "timestamp": time.time() - start_time,
                "query_type": query_type,
                "execution_time": query_time,
            }
        )

        query_count += 1

        # Log progress every 10 queries
        if query_count % 10 == 0:
            elapsed = time.time() - start_time
            current_qps = query_count / elapsed
            yanex.log_results(
                {
                    "queries_completed": query_count,
                    "current_qps": round(current_qps, 1),
                    "elapsed_time": round(elapsed, 1),
                },
                step=query_count,
            )

        # Rate limiting
        time.sleep(max(0, (1.0 / target_qps) - query_time))

    # Calculate final statistics
    total_time = time.time() - start_time
    all_times = [r["execution_time"] for r in results]
    avg_time = sum(all_times) / len(all_times)

    # Log final results
    yanex.log_results(
        {
            "total_queries": query_count,
            "duration": round(total_time, 2),
            "achieved_qps": round(query_count / total_time, 1),
            "avg_response_time": round(avg_time, 4),
            "p95_response_time": round(
                sorted(all_times)[int(len(all_times) * 0.95)], 4
            ),
            "workload_type": workload,
        }
    )

    # Create performance report artifact
    report = f"""Benchmark Report
===============
Workload: {workload}
Duration: {total_time:.1f}s
Total Queries: {query_count}
Achieved QPS: {query_count / total_time:.1f}
Average Response Time: {avg_time:.4f}s
95th Percentile: {sorted(all_times)[int(len(all_times) * 0.95)]:.4f}s
"""

    yanex.log_text(report, "benchmark_report.txt")

    # Log raw data as JSON artifact
    yanex.log_text(json.dumps(results, indent=2), "raw_data.json")

    print(f"Benchmark completed: {query_count} queries in {total_time:.1f}s")
    print(f"Average response time: {avg_time:.4f}s")


if __name__ == "__main__":
    run_advanced_benchmark()
