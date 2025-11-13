# 07: Parameter Sweeps

## What This Example Demonstrates

- Parameter sweep syntax (comma-separated lists, range, linspace, logspace)
- Sweeps in config files vs CLI parameters
- Cartesian product (grid search) when multiple parameters are swept
- Sequential vs parallel execution of sweeps
- Each sweep combination creates a separate experiment

## Files

- `test_algorithm.py` - Algorithm performance testing script
- `config.yaml` - Standard config with single values
- `sweep-config.yaml` - Config with parameter sweeps

## Why Parameter Sweeps?

Parameter sweeps let you run the same experiment with different parameter combinations automatically:
- **Hyperparameter optimization** - Test different learning rates, batch sizes, etc.
- **Performance analysis** - Test algorithms with varying data sizes
- **Sensitivity analysis** - Understand how parameters affect results
- **A/B testing** - Compare multiple approaches systematically

## Sweep Syntax

Yanex supports four sweep syntaxes, usable in both CLI parameters and config files.

### 1. Comma-Separated Lists

The simplest syntax for discrete values:

```bash
# CLI usage
yanex run test_algorithm.py -p algorithm=quicksort,mergesort,heapsort
yanex run test_algorithm.py -p data_size=100,500,1000,5000
```

```yaml
# Config file usage
algorithm: "quicksort,mergesort,heapsort"
data_size: "100,500,1000,5000"
```

**Equivalent to**: `list(quicksort,mergesort,heapsort)` (old syntax, still supported)

**Note**: Quotes are optional in both CLI and YAML config files. Examples below use quotes for consistency.

### 2. Range Syntax

Generate integer sequences like Python's `range()`:

```bash
# range(stop) - generates 0, 1, 2, ..., stop-1
yanex run test_algorithm.py -p "data_size=range(5)"  # 0, 1, 2, 3, 4

# range(start, stop) - generates start, start+1, ..., stop-1
yanex run test_algorithm.py -p "data_size=range(1000, 5000)"  # 1000, 1001, ..., 4999

# range(start, stop, step) - generates start, start+step, ..., < stop
yanex run test_algorithm.py -p "data_size=range(1000, 5000, 1000)"  # 1000, 2000, 3000, 4000
```

```yaml
# Config file
data_size: "range(1000, 5000, 1000)"
```

**Like Python's range()**: Supports 1, 2, or 3 arguments. Generates integers only.

### 3. Linspace Syntax

Generate evenly spaced values (for continuous parameters):

```bash
# linspace(start, stop, num) - num evenly spaced values from start to stop
yanex run test_algorithm.py -p "threshold=linspace(0.1, 0.9, 5)"
# Generates: 0.1, 0.3, 0.5, 0.7, 0.9
```

```yaml
# Config file
threshold: "linspace(0.1, 0.9, 5)"
```

**Use for**: Continuous parameters like learning rates, thresholds, probabilities

### 4. Logspace Syntax

Generate logarithmically spaced values:

```bash
# logspace(start_exp, stop_exp, num) - num values from 10^start to 10^stop
yanex run test_algorithm.py -p "learning_rate=logspace(-4, -1, 4)"
# Generates: 0.0001, 0.001, 0.01, 0.1 (i.e., 10^-4, 10^-3, 10^-2, 10^-1)
```

```yaml
# Config file
learning_rate: "logspace(-4, -1, 4)"
```

**Use for**: Parameters that vary over orders of magnitude (learning rates, regularization)

## How to Run

### Single experiment (no sweep)
```bash
yanex run test_algorithm.py --config config.yaml
```

### CLI parameter sweep
```bash
# Test multiple algorithms (3 experiments)
yanex run test_algorithm.py -p "algorithm=quicksort,mergesort,heapsort"

# Test multiple data sizes (4 experiments)
yanex run test_algorithm.py -p "data_size=range(1000, 5000, 1000)"

# Using linspace for thresholds (5 experiments)
yanex run test_algorithm.py -p "threshold=linspace(0.1, 0.9, 5)"
```

### Config file sweep
```bash
# Sequential execution (one after another)
yanex run test_algorithm.py --config sweep-config.yaml

# Parallel execution with 4 workers
yanex run test_algorithm.py --config sweep-config.yaml --parallel 4

# Auto-detect CPU count
yanex run test_algorithm.py --config sweep-config.yaml -j 0
```

### Grid search (cartesian product)
```bash
# 3 algorithms × 4 sizes = 12 experiments
yanex run test_algorithm.py \
  -p "algorithm=quicksort,mergesort,heapsort" \
  -p "data_size=range(1000, 5000, 1000)"

# Run in parallel
yanex run test_algorithm.py \
  -p "algorithm=quicksort,mergesort,heapsort" \
  -p "data_size=range(1000, 5000, 1000)" \
  --parallel 4
```

## Understanding Cartesian Products

When **multiple parameters** have sweeps, yanex creates experiments for **all combinations** (cartesian product):

### Example 1: Simple Grid
```bash
yanex run test_algorithm.py \
  -p "algorithm=quicksort,mergesort" \
  -p "data_size=1000,5000"
```

**Creates 4 experiments** (2 × 2):
1. algorithm=quicksort, data_size=1000
2. algorithm=quicksort, data_size=5000
3. algorithm=mergesort, data_size=1000
4. algorithm=mergesort, data_size=5000

### Example 2: Three-Way Grid
```bash
yanex run test_algorithm.py \
  -p "algorithm=quicksort,mergesort,heapsort" \
  -p "data_size=1000,5000" \
  -p "random_seed=42,123"
```

**Creates 12 experiments** (3 × 2 × 2):
- 3 algorithms × 2 sizes × 2 seeds = 12 combinations

### Config File Grid Search

```yaml
# sweep-config.yaml creates 3 × 4 = 12 experiments
algorithm: "quicksort,mergesort,heapsort"  # 3 values
data_size: "range(1000, 5000, 1000)"       # 4 values
random_seed: 42                            # 1 value (not swept)
```

## Sequential vs Parallel Execution

### Sequential (default)
Experiments run one after another:

```bash
yanex run test_algorithm.py --config sweep-config.yaml
# Runs 12 experiments sequentially
# Total time: ~12 × experiment_duration
```

**Use when**:
- Limited resources (CPU, memory)
- Experiments can affect each other
- Experiments are very fast

### Parallel Execution
Experiments run simultaneously using multiple workers:

```bash
# Specify number of workers
yanex run test_algorithm.py --config sweep-config.yaml --parallel 4

# Short flag
yanex run test_algorithm.py --config sweep-config.yaml -j 4

# Auto-detect CPU count
yanex run test_algorithm.py --config sweep-config.yaml -j 0
```

**Use when**:
- Multi-core system available
- Experiments are independent
- Time-sensitive analysis
- Large sweeps (many combinations)

**Benefits**:
- ⚡ Much faster for large sweeps
- 🎯 Each experiment in isolated process
- 📊 Progress tracking shows completion status

## Expected Output

### Single Experiment
```
Testing quicksort with 1000 elements (seed=42)...
  Execution time: 0.0123s
  Throughput: 81301 elements/second
  Comparisons: 2000
✓ Test complete
✓ Experiment completed successfully: abc12345
```

### Sweep (Sequential)
```
Testing quicksort with 1000 elements (seed=42)...
✓ Experiment completed successfully: abc12345

Testing quicksort with 2000 elements (seed=42)...
✓ Experiment completed successfully: def67890

Testing quicksort with 3000 elements (seed=42)...
✓ Experiment completed successfully: ghi11111
...
```

### Sweep (Parallel with -j 4)
```
Running 12 experiments with 4 parallel workers...
[Progress indicators showing completion]
✓ Completed 12/12 experiments
```

## Analyzing Sweep Results

After running a sweep, compare results:

```bash
# View all experiments from the sweep
yanex list

# Compare experiments side-by-side
yanex compare

# Show only parameters and metrics that differ
yanex compare --only-different

# Filter comparison by tag
yanex compare --tag sweep
```

## Mixing Sweeps and Single Values

You can combine swept and fixed parameters:

```bash
# Sweep algorithms, fix data size
yanex run test_algorithm.py \
  -p "algorithm=quicksort,mergesort,heapsort" \
  -p "data_size=5000"  # Fixed value

# Creates 3 experiments
```

```yaml
# Config file
algorithm: "quicksort,mergesort,heapsort"   # Swept
data_size: 5000                             # Fixed
random_seed: 42                             # Fixed
```

## Sweep Syntax Comparison

| Syntax | Use Case | Example | Output |
|--------|----------|---------|--------|
| **Comma-separated** | Discrete values | `"a,b,c"` | a, b, c |
| **list()** | Discrete values (old) | `"list(a,b,c)"` | a, b, c |
| **range()** | Integer sequences | `"range(0,10,2)"` | 0, 2, 4, 6, 8 |
| **linspace()** | Even spacing | `"linspace(0,1,5)"` | 0.0, 0.25, 0.5, 0.75, 1.0 |
| **logspace()** | Log spacing | `"logspace(-2,0,3)"` | 0.01, 0.1, 1.0 |

## Best Practices

### Start Small
```bash
# Test with small sweep first
yanex run test_algorithm.py -p "data_size=100,500"

# Then scale up
yanex run test_algorithm.py -p "data_size=range(100, 10000, 100)"
```

### Use Tags for Organization
```bash
# Tag sweep experiments
yanex run test_algorithm.py \
  --config sweep-config.yaml \
  --tag algorithm-comparison \
  --tag sweep
```

### Preview Sweep Size
Before running large sweeps, calculate combinations:
- 3 algorithms × 5 sizes × 4 seeds = **60 experiments**
- Consider parallel execution for large sweeps

### Name Your Sweeps
```bash
yanex run test_algorithm.py \
  --config sweep-config.yaml \
  --name "algorithm-benchmark-v1" \
  --description "Comparing sorting algorithms with data_size 1000-5000"
```

## Key Concepts

- **Comma-separated syntax**: Simplest way to define discrete value sweeps
- **Multiple sweep syntaxes**: Choose the right one for your parameter type
- **Cartesian product**: Multiple swept parameters create all combinations
- **Each combination = one experiment**: Fully tracked and reproducible
- **Parallel execution**: Use `-j N` to run sweeps faster on multi-core systems
- **Config file sweeps**: Define sweeps in YAML for repeatability

## Next Steps

- Run a sweep and compare results: `yanex compare`
- Try different parallel worker counts: `-j 2`, `-j 4`, `-j 0`
- See example 09 to learn about staged experiments with parallel execution
