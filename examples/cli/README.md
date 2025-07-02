# CLI Examples for Yanex

## Examples

- **`basic_benchmark.py`** - Simple parameter access and result logging
- **`advanced_benchmark.py`** - Comprehensive benchmarking with artifacts
- **`bash_script_example.py`** - Bash script integration with `execute_bash_script()`
- **`demo_linkbench.sh`** - Example bash script that uses yanex environment variables

## Bash Script Integration

### Quick Start

```bash
# Run the bash integration example
yanex run bash_script_example.py --param duration=8 --param threads=4
```

### How It Works

**Python script (`bash_script_example.py`):**
- Uses `yanex.get_experiment_dir()` to get the experiment directory
- Uses `yanex.execute_bash_script()` to run the bash script
- Parameters are automatically passed as `YANEX_PARAM_*` environment variables

**Bash script (`simple_linkbench.sh`):**
- Accesses experiment ID via `$YANEX_EXPERIMENT_ID`
- Accesses parameters via `$YANEX_PARAM_duration`, `$YANEX_PARAM_threads`, etc.
- Creates output files in the experiment directory
- yanex automatically captures stdout/stderr as artifacts

### Key Environment Variables

Your bash scripts automatically receive:
- `YANEX_EXPERIMENT_ID` - Current experiment identifier
- `YANEX_PARAM_*` - All experiment parameters (e.g., `YANEX_PARAM_duration`)

### Parameters

For the bash integration example:
- **`duration`** (default: 5) - Benchmark duration in seconds  
- **`threads`** (default: 2) - Number of concurrent threads