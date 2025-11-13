#!/bin/bash
# linkbench.sh - Simulated LinkBench database benchmark
#
# This script demonstrates how bash scripts can access yanex parameters
# via environment variables when executed with execute_bash_script()

echo "=== LinkBench Starting ==="

# Access yanex experiment ID (set automatically)
echo "Experiment ID: $YANEX_EXPERIMENT_ID"

# Access yanex parameters via YANEX_PARAM_* environment variables
# These are set automatically from yanex.get_params()
echo "Duration: $YANEX_PARAM_duration seconds"
echo "Threads: $YANEX_PARAM_threads"

# Script arguments (passed via command line)
echo "Script args: $@"

echo ""
echo "Running benchmark..."

# Simulate benchmark execution
duration=${YANEX_PARAM_duration:-5}
sleep $duration

# Simulate benchmark results
ops=$((duration * 100))
latency_ms=$((10 + RANDOM % 20))

echo "Completed $ops operations"
echo "Average latency: ${latency_ms}ms"

# Save results to CSV (saved in experiment directory)
cat > linkbench_results.csv << EOF
metric,value
operations,$ops
duration_seconds,$duration
threads,${YANEX_PARAM_threads:-2}
avg_latency_ms,$latency_ms
EOF

echo ""
echo "Results saved to: linkbench_results.csv"
echo "=== LinkBench Complete ==="

exit 0
