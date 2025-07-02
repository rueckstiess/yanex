#!/bin/bash
# simple_linkbench.sh - Minimal linkbench example

echo "=== LinkBench Starting ==="
echo "Experiment: $YANEX_EXPERIMENT_ID"
echo "Duration: $YANEX_PARAM_duration seconds"
echo "Threads: $YANEX_PARAM_threads"
echo "Command args: $@"

# Simulate benchmark
echo "Running benchmark..."
sleep ${YANEX_PARAM_duration:-5}

# Calculate simple results
ops=$((${YANEX_PARAM_duration:-5} * 100))
echo "Completed $ops operations"

# Save results
echo "operations,$ops" > linkbench_results.csv
echo "duration,${YANEX_PARAM_duration:-5}" >> linkbench_results.csv

echo "=== LinkBench Complete ==="