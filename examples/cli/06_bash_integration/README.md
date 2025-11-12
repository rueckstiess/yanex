# 06: Bash Script Integration

## What This Example Demonstrates

- Running bash scripts from Python with `execute_bash_script()`
- Automatic parameter passing via environment variables
- Accessing `YANEX_PARAM_*` and `YANEX_EXPERIMENT_ID` from bash
- Automatic stdout/stderr capture as artifacts
- Integrating existing bash tools/benchmarks with yanex tracking

## Files

- `run_benchmark.py` - Python wrapper that executes the bash script
- `linkbench.sh` - Bash script simulating a database benchmark (LinkBench)

## Why Bash Integration?

Many research and benchmarking tools are written in other programming languages but can be invoked via bash commands. Yanex lets you:
- **Track existing tools** without rewriting them in Python
- **Pass parameters automatically** via environment variables
- **Capture all output** as experiment artifacts
- **Log execution metrics** (exit codes, timing)
- **Preserve reproducibility** for bash-based workflows

## How It Works

### 1. Python Wrapper (`run_benchmark.py`)

```python
import yanex
from pathlib import Path

# Execute bash script
script_path = Path(__file__).parent / "linkbench.sh"
result = yanex.execute_bash_script(f"{script_path} --workload mixed")

# Log results
yanex.log_metrics({
    'exit_code': result['exit_code'],
    'execution_time': result['execution_time']
})
```

### 2. Bash Script (`linkbench.sh`)

```bash
#!/bin/bash

# Access experiment ID
echo "Experiment: $YANEX_EXPERIMENT_ID"

# Access yanex parameters
duration=$YANEX_PARAM_duration
threads=$YANEX_PARAM_threads

# Run your benchmark
./benchmark --duration $duration --threads $threads
```

## Environment Variables

Yanex automatically sets these environment variables for bash scripts:

### `YANEX_EXPERIMENT_ID`
Current experiment identifier (8-character hex)

```bash
echo "Running in experiment: $YANEX_EXPERIMENT_ID"
```

### `YANEX_PARAM_*`
All yanex parameters prefixed with `YANEX_PARAM_`

```bash
# Parameter: duration=10
echo "Duration: $YANEX_PARAM_duration"  # Outputs: 10

# Parameter: threads=4
echo "Threads: $YANEX_PARAM_threads"    # Outputs: 4
```

### Using with Defaults

Use bash parameter substitution for defaults:

```bash
# Use YANEX_PARAM_duration if set, otherwise default to 5
duration=${YANEX_PARAM_duration:-5}
threads=${YANEX_PARAM_threads:-2}
```

## How to Run

### Basic usage
```bash
yanex run run_benchmark.py
```

### With parameters
```bash
yanex run run_benchmark.py -p duration=10 -p threads=4
```

### With metadata
```bash
yanex run run_benchmark.py \
  -p duration=20 \
  -p threads=8 \
  --name "linkbench-heavy" \
  --tag benchmark --tag performance
```

## Expected Output

```
Experiment directory: /Users/you/.yanex/experiments/abc12345
Running LinkBench with duration=5s, threads=2
=== LinkBench Starting ===
Experiment ID: abc12345
Duration: 5 seconds
Threads: 2
Script args: --workload mixed --verbose

Running benchmark...
Completed 500 operations
Average latency: 15ms

Results saved to: linkbench_results.csv
=== LinkBench Complete ===

✓ Benchmark completed in 5.12s
```

## Captured Artifacts

After running, check the experiment's artifacts directory:

1. **script_stdout.txt** - Standard output from bash script (automatic)
2. **script_stderr.txt** - Standard error from bash script (automatic, if any errors)
3. **linkbench_results.csv** - Results file created by bash script

View artifacts:
```bash
yanex show <experiment_id>
yanex open <experiment_id>  # Opens experiment directory
```

## The `execute_bash_script()` Function

### Basic Usage

```python
result = yanex.execute_bash_script("./script.sh")
```

### With Command Arguments

```python
# Pass arguments to the script
result = yanex.execute_bash_script("./benchmark.sh --workload heavy --verbose")
```

### Return Value

The function returns a dictionary with:

```python
{
    'exit_code': 0,                    # Process exit code
    'stdout': '...',                   # Captured stdout
    'stderr': '...',                   # Captured stderr
    'execution_time': 5.23,            # Execution time in seconds
    'command': './script.sh',          # Command executed
    'working_directory': '/path/...'   # Directory where script ran
}
```

### Optional Parameters

```python
result = yanex.execute_bash_script(
    "./script.sh",
    timeout=300,              # Timeout in seconds (optional)
    raise_on_error=True,      # Raise exception on non-zero exit (default: False)
    stream_output=True,       # Print output in real-time (default: True)
    working_dir=Path("/tmp")  # Custom working directory (default: experiment dir)
)
```

## Passing Script Arguments vs Yanex Parameters

### Yanex Parameters (via environment variables)
Set with `--param` and accessible as `YANEX_PARAM_*`:

```bash
yanex run run_benchmark.py -p duration=10
```

```bash
# In bash script
echo $YANEX_PARAM_duration  # 10
```

### Script Arguments (via command line)
Passed directly in `execute_bash_script()`:

```python
yanex.execute_bash_script("./script.sh --workload mixed --verbose")
```

```bash
# In bash script
echo $1  # --workload
echo $2  # mixed
echo $3  # --verbose
```

**Use yanex parameters for**: Configuration that should be tracked (durations, sizes, algorithms)

**Use script arguments for**: Operational flags (--verbose, --debug, --help)

## Real-World Use Cases

### 1. Database Benchmarks
```bash
yanex run run_benchmark.py \
  -p duration=300 \
  -p threads=16 \
  -p db_size=1000000 \
  --name "linkbench-production"
```

### 2. System Performance Tests
```bash
yanex run run_sysstat.py \
  -p duration=60 \
  -p interval=5 \
  --tag system-monitoring
```

### 3. Legacy Tool Integration
```bash
# Wrap existing bash tools without modification
yanex run run_legacy_tool.py \
  -p input_file=data.txt \
  -p config=fast.conf
```

## Error Handling

### Check Exit Codes

```python
result = yanex.execute_bash_script("./script.sh")

if result['exit_code'] != 0:
    print(f"Script failed with exit code: {result['exit_code']}")
    print(f"Error output: {result['stderr']}")
```

### Raise on Error

```python
# Automatically raise exception on non-zero exit code
try:
    result = yanex.execute_bash_script("./script.sh", raise_on_error=True)
except subprocess.CalledProcessError as e:
    print(f"Script failed: {e}")
```

### Timeout Handling

```python
# Set timeout for long-running scripts
try:
    result = yanex.execute_bash_script("./long_script.sh", timeout=300)
except subprocess.TimeoutExpired:
    print("Script timed out after 5 minutes")
```

## Best Practices

### 1. Make Scripts Executable
```bash
chmod +x linkbench.sh
```

### 2. Use Absolute Paths
```python
from pathlib import Path
script_path = Path(__file__).parent / "script.sh"
```

### 3. Set Default Values in Bash
```bash
# Handle missing parameters gracefully
duration=${YANEX_PARAM_duration:-10}  # Default to 10 if not set
```

### 4. Validate Parameters
```bash
if [ -z "$YANEX_PARAM_duration" ]; then
    echo "Error: duration parameter required"
    exit 1
fi
```

### 5. Create Result Files in Script
```bash
# Scripts run in experiment directory by default
cat > results.csv << EOF
metric,value
operations,${ops}
latency_ms,${latency}
EOF
```

## Troubleshooting

### Script Not Found
```python
# Use absolute paths
script_path = Path(__file__).parent / "script.sh"
result = yanex.execute_bash_script(f"{script_path}")
```

### Parameters Not Available
```bash
# Check if running via yanex
if [ -z "$YANEX_EXPERIMENT_ID" ]; then
    echo "Warning: Not running via yanex, using defaults"
fi
```

### Permission Denied
```bash
# Make script executable
chmod +x script.sh
```

## Key Concepts

- **Automatic parameter passing**: Parameters accessible as `YANEX_PARAM_*` in bash
- **Experiment context**: Scripts get `YANEX_EXPERIMENT_ID` automatically
- **Output capture**: stdout/stderr saved as artifacts automatically
- **Working directory**: Scripts run in experiment directory by default
- **Dual usage**: Bash scripts work standalone and with yanex

## Next Steps

- Try wrapping your own bash scripts with yanex
- Use parameter sweeps with bash scripts: `yanex run ... -p "duration=5,10,20"`
- See example 07 for passing script-specific arguments
