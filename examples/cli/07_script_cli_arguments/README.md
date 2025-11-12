# 07: Script CLI Arguments

## What This Example Demonstrates

- Using `argparse` for script-specific flags (--verbose, --format, etc.)
- Using yanex `--param` for experiment configuration (sample_size, thresholds)
- The distinction between script arguments and yanex parameters
- When to use each approach

## Files

- `validate_data.py` - Data validation tool with both argparse and yanex parameters

## Script Arguments vs Yanex Parameters

### Yanex Parameters (`--param` / `-p`)

**Use for**: Configuration that affects experiment results and should be tracked

```bash
yanex run validate_data.py \
  --param sample_size=1000 \
  --param error_threshold=0.1
```

**Characteristics:**
- Tracked in experiment metadata
- Accessible via `yanex.get_param()`
- Used in parameter sweeps
- Comparable across experiments
- Part of reproducibility tracking

**Examples:**
- `sample_size` - affects validation coverage
- `error_threshold` - affects pass/fail decision
- `learning_rate`, `batch_size` - ML hyperparameters
- `num_workers`, `timeout` - performance parameters

### Script Arguments (via `argparse`)

**Use for**: Operational flags that control script behavior but don't affect results

```bash
yanex run validate_data.py -- --verbose --format json --check-type thorough
```

**Characteristics:**
- NOT tracked in experiment metadata
- Parsed via standard Python `argparse`
- Control output formatting, verbosity, debugging
- Don't affect experiment reproducibility

**Examples:**
- `--verbose` / `-v` - show detailed output
- `--format` - output format (json, csv, table)
- `--debug` - enable debug logging
- `--quiet` - suppress output
- `--help` - show usage information

## How to Run

### Basic usage (defaults)
```bash
yanex run validate_data.py
```

### With yanex parameters
```bash
# Test with larger sample
yanex run validate_data.py -p sample_size=1000

# Stricter threshold
yanex run validate_data.py -p error_threshold=0.01

# Both parameters
yanex run validate_data.py -p sample_size=500 -p error_threshold=0.05
```

### With script arguments
```bash
# Most script arguments work without -- separator
yanex run validate_data.py --format json
yanex run validate_data.py --check-type thorough
yanex run validate_data.py --verbose

# Multiple script arguments
yanex run validate_data.py --format csv --check-type thorough --verbose

# Use -- separator only when there's a naming conflict
yanex run validate_data.py -- --help  # --help conflicts with yanex's --help
```

### Combining both
```bash
# Yanex parameters AND script arguments (no conflicts)
yanex run validate_data.py \
  -p sample_size=1000 \
  -p error_threshold=0.02 \
  --format json \
  --check-type thorough \
  --verbose

# With conflicting flags, use -- separator
yanex run validate_data.py \
  -p sample_size=1000 \
  --format json \
  -- \
  --help
#  └─ no conflict ─┘  └─ conflicts ─┘
```

**Note**: Yanex passes unrecognized arguments to your script automatically. Use `--` only when script arguments conflict with yanex flags.

## Expected Output

### Basic run
```
Running basic validation on 100 samples...

Validation Results:
  Errors found: 3/100
  Error rate: 3.0%

Validation PASSED (threshold: 5.0%)
✓ Experiment completed successfully: abc12345
```

### With --verbose
```
Running thorough validation on 1000 samples...
  Error threshold: 0.05
  Output format: table

Validation Results:
  Errors found: 23/1000
  Error rate: 2.3%

Validation PASSED (threshold: 5.0%)
✓ Experiment completed successfully: def67890
```

### With --format json
```
Running basic validation on 100 samples...
{"errors": 4, "total": 100, "rate": 0.040}

Validation PASSED (threshold: 5.0%)
✓ Experiment completed successfully: ghi11111
```

## The `--` Separator (When Needed)

Yanex automatically passes unrecognized arguments to your script. The `--` separator is **only needed** when there's a naming conflict between yanex flags and your script arguments.

```bash
yanex run script.py [yanex-flags] [script-arguments]
```

### When `--` is NOT needed

Most script-specific arguments work fine without `--`:

```bash
# These work fine - no naming conflicts
yanex run validate_data.py --format json
yanex run validate_data.py --check-type thorough
yanex run validate_data.py -p sample_size=500 --format csv
```

### When `--` IS needed

Use `--` when your script argument conflicts with a yanex flag:

```bash
# Definite conflict - use --
yanex run validate_data.py -- --help

# If you have script flags that match yanex flags, use --
yanex run script.py -- --config script-config.yaml  # If script has --config
yanex run script.py -- --name output.txt             # If script has --name

# Mixing params with conflicting args
yanex run validate_data.py -p sample_size=500 -- --help
#                          └─ yanex params ─┘ └─ script args ──┘
```

**Yanex flags that might conflict**: `--help`, `--config`, `--name`, `--tag`, `--description`, `--param`, `--stage`, `--parallel`

## When to Use Each

### Use Yanex Parameters For:

**Configuration affecting results:**
- Dataset parameters (size, split ratio, seed)
- Model hyperparameters (learning rate, layers, dropout)
- Algorithm settings (threshold, iterations, tolerance)
- Resource limits (max_workers, timeout, memory_limit)

**Why**: These affect experiment outcomes and need to be tracked for reproducibility

### Use Script Arguments For:

**Operational/presentation flags:**
- Output control (--verbose, --quiet, --format)
- Debug flags (--debug, --trace, --profile)
- Help and info (--help, --version, --list-models)
- Behavior flags (--dry-run, --force, --skip-validation)

**Why**: These control HOW results are displayed, not WHAT results are produced

## Parameter Sweeps

Only yanex parameters can be swept:

```bash
# Sweep sample_size (works - it's a yanex param)
yanex run validate_data.py -p "sample_size=100,500,1000"

# Cannot sweep --format (it's a script argument)
yanex run validate_data.py -- --format json,csv,table  # Doesn't work this way
```

If you need to sweep script arguments, convert them to yanex parameters:

```python
# Instead of argparse --format flag
output_format = yanex.get_param('output_format', default='table')
```

Then sweep:
```bash
yanex run validate_data.py -p "output_format=json,csv,table"
```

## Best Practices

### 1. Default to Yanex Parameters
When in doubt, use yanex parameters for configuration:

```python
# Good - trackable
sample_size = yanex.get_param('sample_size', default=100)

# Avoid - not tracked
parser.add_argument('--sample-size', type=int, default=100)
```

### 2. Use argparse for Pure UI/UX
Reserve argparse for flags that only affect user experience:

```python
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--quiet', action='store_true')
parser.add_argument('--format', choices=['json', 'csv'])
```

### 3. Document the Distinction
In your script's help text, clarify which are experiment params vs operational flags:

```python
parser = argparse.ArgumentParser(
    description='Data validation (use --param for experiment config)'
)
```

### 4. Standalone Mode
Scripts work standalone without yanex - argparse handles normal CLI usage:

```bash
# Works without yanex
python validate_data.py --verbose --format json

# Parameters use defaults when not run via yanex
```

## Real-World Examples

### ML Training Script
```bash
# Experiment configuration (tracked)
yanex run train.py \
  -p learning_rate=0.001 \
  -p batch_size=32 \
  -p epochs=10 \
  --save-checkpoints \
  --tensorboard \
  --verbose
#  └─ script arguments (no conflicts) ─┘
```

### Data Processing Pipeline
```bash
# Processing configuration (tracked)
yanex run process.py \
  -p chunk_size=1000 \
  -p num_workers=4 \
  --progress-bar \
  --output-format json \
  --debug
#  └─ script arguments (no conflicts) ─┘
```

### Benchmark Tool
```bash
# Benchmark parameters (tracked)
yanex run benchmark.py \
  -p duration=60 \
  -p threads=8 \
  --csv-output results.csv
#  └─ no conflict ─┘
```

## Troubleshooting

### Error: unrecognized arguments
If you get this error, the script argument conflicts with a yanex flag:

```bash
# Error - conflicts with yanex's --help flag
yanex run validate_data.py --help
# Shows: yanex help instead of script help

# Solution - use -- separator
yanex run validate_data.py -- --help
# Shows: script help
```

### Script arguments not working
Most arguments work without `--`, but if they're not being passed:
```bash
# Try adding -- separator
yanex run script.py -p param=value -- --script-arg
```

### Parameters not tracked
If you use argparse for configuration, it won't be tracked:
```python
# NOT tracked
args = parser.parse_args()
sample_size = args.sample_size

# IS tracked
sample_size = yanex.get_param('sample_size', default=100)
```

## Key Concepts

- **Yanex parameters**: Experiment configuration, tracked, reproducible, sweepable
- **Script arguments**: Operational flags, not tracked, UI/UX control
- **`--` separator**: Only needed when script args conflict with yanex flags (--verbose, --help, etc.)
- **Automatic pass-through**: Unrecognized arguments automatically passed to your script
- **When in doubt**: Use yanex parameters for anything affecting results
- **Sweeps**: Only yanex parameters can be swept, not script arguments

## Next Steps

- Try combining parameters and arguments in one command
- Convert existing argparse scripts to use yanex parameters
- See example 08 to learn about staged experiments
