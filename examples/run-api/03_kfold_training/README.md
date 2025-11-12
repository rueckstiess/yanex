# 03: K-Fold Cross-Validation (Orchestrator/Executor Pattern)

## What This Example Demonstrates

- **Orchestrator/Executor pattern** - Single script acts as both orchestrator and executor
- **Sentinel parameter** - Using `_fold_idx` to detect mode
- **Accessing CLI args** with `get_cli_args()` for parallel execution control
- **K-fold cross-validation** - Common ML workflow with metric aggregation
- **Self-spawning experiments** - Script creates and runs child experiments
- **Results API usage** - Reading metrics from completed experiments for aggregation
- **Two-level hierarchy** - Top-level experiment logs aggregated metrics, folds log individual metrics

## Files

- `train_kfold.py` - K-fold training script (both orchestrator and executor)

## The Orchestrator/Executor Pattern

This advanced pattern allows a single script to:
1. **Orchestrate** - Spawn multiple child experiments
2. **Execute** - Run as a single experiment when spawned

**Mode detection** uses a sentinel parameter (`_fold_idx`):
- `_fold_idx = None` → **Orchestrator mode** (spawns folds)
- `_fold_idx = 0, 1, 2, ...` → **Executor mode** (runs single fold)

## How to Run

### Run Directly (Sequential)

```bash
python train_kfold.py
```

This runs the orchestrator, which spawns 5 fold experiments sequentially.

### Run with yanex CLI (Parallel)

```bash
yanex run train_kfold.py --parallel 3
```

The orchestrator detects `--parallel` from CLI args and uses it for spawning folds.

**How it works:**
1. `yanex run` creates experiment context
2. Script runs in orchestrator mode (no `_fold_idx`)
3. `get_cli_args()` retrieves `parallel=3`
4. Orchestrator spawns 5 folds with 3 parallel workers
5. Each fold runs in executor mode (`_fold_idx` set)

## Expected Output (Direct Run)

```
============================================================
ORCHESTRATOR MODE: Spawning 5-fold cross-validation
Learning rate: 0.001
Parallel workers: sequential
============================================================

Executing 5 folds...

============================================================
K-FOLD RESULTS SUMMARY
============================================================

Completed: 5/5
Failed: 0/5

Experiment IDs:
  ✓ kfold-0: abc12345 (1.42s)
  ✓ kfold-1: def67890 (1.41s)
  ✓ kfold-2: ghi11111 (1.43s)
  ✓ kfold-3: jkl22222 (1.42s)
  ✓ kfold-4: mno33333 (1.41s)

Aggregating metrics from completed folds...

Aggregated Results (n=5 folds):
  Average train loss: 0.1234
  Average val loss: 0.1678
  Average val accuracy: 0.8821 ± 0.0156

To view and compare individual fold results:
  yanex list --tag kfold
  yanex compare --tag kfold
```

**Note:** When run directly with `python`, only the 5 fold experiments are created (no top-level experiment).

## Expected Output (With --parallel)

```bash
$ yanex run train_kfold.py --parallel 3
```

```
Using --parallel=3 from CLI args

============================================================
ORCHESTRATOR MODE: Spawning 5-fold cross-validation
Learning rate: 0.001
Parallel workers: 3
============================================================

Executing 5 folds...

============================================================
K-FOLD RESULTS SUMMARY
============================================================

Completed: 5/5
Failed: 0/5

Experiment IDs:
  ✓ kfold-0: abc12345 (1.12s)
  ✓ kfold-1: def67890 (1.11s)
  ✓ kfold-2: ghi11111 (1.13s)
  ✓ kfold-3: jkl22222 (1.12s)
  ✓ kfold-4: mno33333 (1.11s)

Aggregating metrics from completed folds...

Aggregated Results (n=5 folds):
  Average train loss: 0.1234
  Average val loss: 0.1678
  Average val accuracy: 0.8821 ± 0.0156

✓ Logged aggregated metrics to top-level experiment
  Top-level experiment: xyz99999

To view and compare individual fold results:
  yanex list --tag kfold
  yanex compare --tag kfold

✓ Experiment completed successfully: xyz99999
  Directory: /Users/you/.yanex/experiments/xyz99999
```

**Note:** When run with `yanex run`, a **top-level experiment** (xyz99999) is created for the orchestrator. This experiment tracks the orchestration process, while each fold (abc12345, etc.) has its own experiment with metrics.

## Key Concepts

### Two-Level Experiment Structure

This example demonstrates a hierarchical pattern:

**When run with `python train_kfold.py`:**
- Creates **5 fold experiments** (one per fold)
- No top-level experiment (orchestrator runs standalone)

**When run with `yanex run train_kfold.py`:**
- Creates **1 top-level experiment** (orchestrator)
- Creates **5 fold experiments** (children spawned by orchestrator)
- Total: **6 experiments**

**Experiment hierarchy:**
```
xyz99999 (top-level orchestrator)
  ├── Tracks orchestration process
  ├── tags: [] (user-provided tags from CLI)
  └── Spawned these child experiments:
      ├── abc12345 (kfold-0)
      │   └── metrics: {fold_idx: 0, val_accuracy: 0.89, ...}
      ├── def67890 (kfold-1)
      │   └── metrics: {fold_idx: 1, val_accuracy: 0.91, ...}
      ├── ghi11111 (kfold-2)
      │   └── metrics: {fold_idx: 2, val_accuracy: 0.88, ...}
      ├── jkl22222 (kfold-3)
      │   └── metrics: {fold_idx: 3, val_accuracy: 0.90, ...}
      └── mno33333 (kfold-4)
          └── metrics: {fold_idx: 4, val_accuracy: 0.89, ...}
```

**Benefits:**

1. **Individual fold analysis** - Each fold tracked separately
2. **Orchestration tracking** - Top-level shows when/how folds were run
3. **Complete provenance** - Can trace back from fold to orchestrator
4. **Tag inheritance** - Tags applied to orchestrator help organize all related folds

**Viewing results:**

```bash
# View orchestrator experiment
yanex show xyz99999

# View individual fold
yanex show abc12345

# Compare all folds
yanex compare --tag kfold

# List all k-fold experiments
yanex list --tag kfold
```

**Note:** This example demonstrates programmatic metric aggregation using the Results API. See the "Results API for Metric Aggregation" section below for details.

### ExperimentResult Objects

The `run_multiple()` function returns a list of `ExperimentResult` objects with metadata about each experiment:

```python
results = yanex.run_multiple(experiments, parallel=3)

for result in results:
    print(f"ID: {result.experiment_id}")          # "abc12345"
    print(f"Name: {result.name}")                  # "kfold-0"
    print(f"Status: {result.status}")              # "completed", "failed", "cancelled"
    print(f"Duration: {result.duration}")          # 1.42 (seconds)
    print(f"Error: {result.error_message}")        # None (or error message if failed)
```

**What `ExperimentResult` contains:**
- `experiment_id` - Unique 8-character hex ID
- `name` - Experiment name (if provided)
- `status` - Final status string
- `duration` - Execution time in seconds
- `error_message` - Error details (if failed)

**What it does NOT contain:**
- Logged metrics (`log_metrics()` data)
- Logged artifacts
- Configuration parameters
- Git information

**To access metrics and results:** Use the Results API (see section below) or CLI commands:
```bash
yanex show abc12345        # View all experiment data
yanex compare --tag kfold  # Compare metrics across experiments
```

### Sentinel Parameter Pattern

Use a special parameter to detect mode:

```python
# Detect mode
fold_idx = yanex.get_param('_fold_idx', default=None)

if fold_idx is None:
    # ORCHESTRATION MODE
    experiments = [
        yanex.ExperimentSpec(
            script_path=Path(__file__),
            config={'_fold_idx': i, ...},  # Set sentinel
            name=f'fold-{i}'
        )
        for i in range(n_folds)
    ]
    results = yanex.run_multiple(experiments)

else:
    # EXECUTION MODE
    train_single_fold(fold_idx)
```

**Why use `_fold_idx` as the name?**
- Leading underscore indicates "internal/meta" parameter
- Clearly separates orchestration logic from model hyperparameters
- Won't conflict with actual training parameters

### `yanex.get_cli_args()`

Access CLI flags used to run the script:

```python
cli_args = yanex.get_cli_args()
# Returns: {'script': 'train_kfold.py', 'parallel': 3, 'param': [], 'tag': [], ...}

# Get specific flag
parallel_workers = cli_args.get('parallel')
tags = cli_args.get('tag', [])
```

**Available keys:**
- `script` - Script path
- `config` - Config file path (if provided)
- `param` - List of parameter strings
- `name` - Experiment name (if provided)
- `tag` - List of tags
- `description` - Description (if provided)
- `parallel` - Parallel worker count (if provided)
- `stage` - Whether experiment is staged
- `staged` - Whether running staged experiments

**Returns empty dict in standalone mode:**

```python
# When run directly: python script.py
cli_args = yanex.get_cli_args()  # {}

# When run via CLI: yanex run script.py --parallel 3
cli_args = yanex.get_cli_args()  # {'parallel': 3, ...}
```

### Why get_cli_args() is Useful

**Problem:** Orchestrator scripts need to know CLI flags provided by user:

```python
# User runs: yanex run orchestrator.py --parallel 5

# Without get_cli_args():
# Orchestrator can't access --parallel flag
# Has to hard-code parallel workers or use argparse

# With get_cli_args():
cli_args = yanex.get_cli_args()
parallel = cli_args.get('parallel', 1)  # Gets 5!
results = yanex.run_multiple(experiments, parallel=parallel)
```

**Use cases:**
- Pass through `--parallel` to child experiments
- Respect user's tags (`--tag`) in orchestration
- Access custom metadata (`--name`, `--description`)
- Conditional logic based on CLI flags

### Results API for Metric Aggregation

This example demonstrates using the Results API to read metrics from completed experiments and calculate aggregated statistics.

**Why we need the Results API:**

`ExperimentResult` objects returned by `run_multiple()` contain only metadata (id, name, status, duration, error). To access logged metrics, we need the Results API.

**Pattern used in this example:**

```python
import yanex.results as yr

# After run_multiple() completes
results = yanex.run_multiple(experiments, parallel=3)
completed = [r for r in results if r.status == "completed"]

# Read metrics from each experiment
fold_metrics = []
for result in completed:
    # Get experiment object
    exp = yr.get_experiment(result.experiment_id)

    # Get specific metrics
    train_loss = exp.get_metric("train_loss")
    val_loss = exp.get_metric("val_loss")
    val_accuracy = exp.get_metric("val_accuracy")

    fold_metrics.append({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    })

# Calculate aggregated statistics
avg_val_accuracy = sum(m["val_accuracy"] for m in fold_metrics) / len(fold_metrics)
std_val_accuracy = (
    sum((m["val_accuracy"] - avg_val_accuracy) ** 2 for m in fold_metrics) / len(fold_metrics)
) ** 0.5

print(f"Average val accuracy: {avg_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
```

**Results API methods:**
- `yanex.results.get_experiment(experiment_id)` - Get experiment object
- `experiment.get_metric(metric_name)` - Get metric value(s) for that metric

**Note:** The Results API is introduced here to demonstrate a complete k-fold workflow. Full Results API documentation will be covered in future `examples/results-api`.

**When to use Results API:**
- Reading metrics from completed batch experiments
- Aggregating results across multiple experiments
- Building custom analysis pipelines
- Creating summary reports

**Alternative:** Use CLI commands for quick analysis:
```bash
yanex compare --tag kfold  # Compare metrics across experiments
yanex show abc12345        # View detailed experiment data
```

## How the Pattern Works

### 1. Direct Execution (Orchestrator)

```bash
python train_kfold.py
```

**Flow:**
1. Script starts, no experiment context
2. `get_param('_fold_idx')` returns `None` (not set)
3. Enters **orchestrator mode**
4. Creates 5 `ExperimentSpec` objects with `_fold_idx=0,1,2,3,4`
5. Calls `run_multiple()` which spawns 5 child processes
6. Each child runs `train_kfold.py` with experiment context

### 2. Child Execution (Executor)

**Flow (for each child):**
1. Child process starts with experiment context
2. `get_param('_fold_idx')` returns `0` (or 1, 2, 3, 4)
3. Enters **executor mode**
4. Trains single fold
5. Logs metrics
6. Exits

### 3. CLI Execution with --parallel

```bash
yanex run train_kfold.py --parallel 3
```

**Flow:**
1. `yanex run` creates experiment context for orchestrator
2. Script runs in orchestrator mode
3. `get_cli_args()` retrieves `{'parallel': 3, ...}`
4. Passes `parallel=3` to `run_multiple()`
5. Spawns 5 folds with 3 parallel workers

## Why This Pattern is Powerful

### Single Script for Everything

```python
# One script handles both:
# 1. Orchestration logic (create and spawn child experiments)
# 2. Execution logic (train single fold)
```

**Benefits:**
- **No separate orchestrator** - Everything in one file
- **Easy to modify** - Change training logic in one place
- **Portable** - Single script contains complete workflow
- **Testable** - Can test individual folds directly
- **Hierarchical tracking** - Top-level + child experiments for complete provenance
- **Flexible execution** - Run with/without top-level experiment tracking

### Flexible Invocation

```bash
# Quick test - sequential
python train_kfold.py

# Production - parallel
yanex run train_kfold.py --parallel 5

# With custom settings
yanex run train_kfold.py --parallel 5 --tag experiment-v2 --name kfold-final
```

### Parameter Passthrough

Orchestrator can pass user-provided parameters to child experiments:

```python
# User provides: yanex run train_kfold.py --param learning_rate=0.01 --parallel 3

def orchestrate_kfold(n_folds):
    # Get user's learning rate
    learning_rate = yanex.get_param('learning_rate', 0.001)

    # Get user's parallel flag
    cli_args = yanex.get_cli_args()
    parallel = cli_args.get('parallel', 1)

    # Pass to children
    experiments = [
        yanex.ExperimentSpec(
            script_path=Path(__file__),
            config={'_fold_idx': i, 'learning_rate': learning_rate},
            name=f'fold-{i}'
        )
        for i in range(n_folds)
    ]

    results = yanex.run_multiple(experiments, parallel=parallel)
```

## Advanced: Testing Individual Folds

You can test a single fold directly by setting the sentinel parameter:

```bash
# Run just fold 2
yanex run train_kfold.py --param _fold_idx=2 --param n_folds=5
```

This enters executor mode directly, useful for debugging individual folds.

## Common Variations

### Variation 1: Nested Orchestration

Orchestrators can spawn other orchestrators:

```python
# Top-level: Grid search over hyperparameters
for lr in [0.001, 0.01, 0.1]:
    # Each spawns k-fold CV
    yanex.ExperimentSpec(
        script_path=Path("kfold_orchestrator.py"),
        config={'learning_rate': lr, '_is_orchestrator': True}
    )
```

### Variation 2: Conditional Orchestration

Only orchestrate for certain parameters:

```python
fold_idx = yanex.get_param('_fold_idx', default=None)
enable_cv = yanex.get_param('enable_cv', default=True)

if fold_idx is None and enable_cv:
    # Run k-fold
    orchestrate_kfold()
elif fold_idx is not None:
    # Execute single fold
    execute_single_fold(fold_idx)
else:
    # Just train once, no CV
    train_single_model()
```

### Variation 3: Different Sentinel Names

Use different sentinels for different orchestration types:

```python
fold_idx = yanex.get_param('_fold_idx', default=None)
ensemble_idx = yanex.get_param('_ensemble_idx', default=None)

if fold_idx is not None:
    # K-fold mode
    train_single_fold(fold_idx)
elif ensemble_idx is not None:
    # Ensemble mode
    train_ensemble_member(ensemble_idx)
else:
    # Orchestrator mode
    orchestrate()
```

## When to Use This Pattern

**Use orchestrator/executor pattern when:**
- Running k-fold cross-validation
- Training ensembles with different seeds
- Running ablation studies (enable/disable features)
- Processing multiple datasets with same pipeline
- Any workflow where you spawn N similar experiments

**Don't use when:**
- Simple grid search (use example 02 pattern)
- Each experiment needs different scripts
- Orchestration logic is complex (separate script may be clearer)

## Troubleshooting

### Infinite Recursion

**Problem:** Orchestrator spawns itself recursively.

**Cause:** Sentinel parameter not working correctly.

**Solution:** Make sure executor mode sets `_fold_idx`:

```python
# ✗ Wrong - will recurse forever
experiments = [
    yanex.ExperimentSpec(
        script_path=Path(__file__),
        config={'learning_rate': lr}  # Missing _fold_idx!
    )
]

# ✓ Correct
experiments = [
    yanex.ExperimentSpec(
        script_path=Path(__file__),
        config={'_fold_idx': i, 'learning_rate': lr}  # Has _fold_idx
    )
]
```

### get_cli_args() Returns Empty Dict

**Problem:** `get_cli_args()` returns `{}`.

**Cause:** Script not run via `yanex run`.

**Solution:** This is expected when running directly (`python script.py`). Use default values:

```python
cli_args = yanex.get_cli_args()
parallel = cli_args.get('parallel', 1)  # Default to 1 if not provided
```

### ExperimentContextError

**Problem:** `ExperimentContextError: Cannot use run_multiple() when script is run via 'yanex run'`.

**Cause:** Calling `run_multiple()` from executor mode (when `_fold_idx` is set).

**Solution:** Make sure executor mode doesn't call `run_multiple()`:

```python
fold_idx = yanex.get_param('_fold_idx', default=None)

if fold_idx is None:
    # Only orchestrator calls run_multiple()
    results = yanex.run_multiple(experiments)
else:
    # Executor never calls run_multiple()
    train_single_fold(fold_idx)
```

## Next Steps

- Try modifying `n_folds` and `learning_rate`
- Add additional hyperparameters to sweep over
- Extend aggregation to include more statistics (min, max, percentiles)
- Create nested orchestration (grid search + k-fold)
- Use `yanex compare --tag kfold` to view and compare fold results
