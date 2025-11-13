# 02: Batch Execution with run_multiple()

## What This Example Demonstrates

- Creating multiple experiment specifications with `ExperimentSpec`
- Running experiments in parallel with `run_multiple()`
- Handling experiment results with `ExperimentResult`
- Grid search pattern for hyperparameter tuning
- Sequential vs parallel execution

## Files

- `train.py` - Simple training script executed by grid search
- `grid_search.py` - Grid search orchestrator using `run_multiple()`

## Why Batch Execution?

The `run_multiple()` API enables:

- **Grid search** - Test multiple hyperparameter combinations
- **Ensemble training** - Train multiple models with different seeds
- **Cross-validation** - Run k-fold validation (see example 03)
- **Parallel execution** - Utilize multiple CPU cores
- **Batch processing** - Process multiple datasets simultaneously

## How to Run

**Important:** Run the grid search script directly with Python (NOT with `yanex run`):

```bash
# Run grid search
python grid_search.py
```

This will:
1. Create 6 experiment specifications (3 learning rates × 2 batch sizes)
2. Execute them in parallel with 3 workers
3. Display results summary
4. Show commands to view detailed results

**Note:** The code is wrapped in `if __name__ == "__main__":` which is required for parallel execution on macOS/Windows (which use the `spawn` method for multiprocessing).

## Expected Output

```
Grid Search Configuration
  Learning rates: [0.001, 0.01, 0.1]
  Batch sizes: [16, 32]
  Epochs: 10
  Total experiments: 6

Created 6 experiment specifications

Running experiments with 3 parallel workers...

============================================================
GRID SEARCH RESULTS
============================================================

Total: 6
Completed: 6
Failed: 0

Completed experiments:
  ✓ grid-lr0_001-bs16: abc12345 (2.15s)
  ✓ grid-lr0_001-bs32: def67890 (2.14s)
  ✓ grid-lr0_01-bs16: ghi11111 (2.16s)
  ✓ grid-lr0_01-bs32: jkl22222 (2.15s)
  ✓ grid-lr0_1-bs16: mno33333 (2.17s)
  ✓ grid-lr0_1-bs32: pqr44444 (2.14s)

To view experiment details:
  yanex show abc12345
  yanex list --tag grid-search
  yanex compare --tag grid-search
```

## Key Concepts

### `yanex.ExperimentSpec`

Specification for a single experiment:

```python
spec = yanex.ExperimentSpec(
    script_path=Path("train.py"),           # Script to execute
    config={"learning_rate": 0.01},          # Parameters
    script_args=["--verbose"],               # Optional script arguments
    name="experiment-1",                      # Optional name
    tags=["ml", "training"],                  # Optional tags
    description="Training run 1"              # Optional description
)
```

**Parameters:**
- `script_path` (Path, required): Path to Python script to execute
- `config` (dict): Configuration parameters accessible via `yanex.get_params()`
- `script_args` (list[str]): Arguments passed to script via `sys.argv`
- `name` (str): Human-readable experiment name
- `tags` (list[str]): Tags for organization and filtering
- `description` (str): Experiment description

### `yanex.run_multiple()`

Execute multiple experiments programmatically:

```python
results = yanex.run_multiple(
    experiments=[spec1, spec2, spec3],
    parallel=4,      # Number of workers (None=sequential, 0=auto)
    verbose=False    # Show detailed output
)
```

**Parameters:**
- `experiments` (list[ExperimentSpec]): List of experiments to run
- `parallel` (int | None):
  - `None`: Sequential execution (default)
  - `0`: Auto-detect CPU count
  - `N > 0`: Use N parallel workers
- `verbose` (bool): Show detailed execution output (default: False)

**Returns:**
- `list[ExperimentResult]`: Results for all experiments (both successful and failed)

**Behavior:**
- Blocks until all experiments complete
- Individual failures don't abort the batch
- Each experiment runs in isolated subprocess
- Uncommitted git changes automatically captured as patches

**Important:** Do NOT call `run_multiple()` from within a `yanex run` context. Use this API when running orchestrator scripts directly: `python grid_search.py`

### `yanex.ExperimentResult`

Result of running a single experiment:

```python
result = results[0]

# Check status
if result.status == "completed":
    print(f"Success: {result.experiment_id}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Name: {result.name}")
elif result.status == "failed":
    print(f"Failed: {result.error_message}")
```

**Attributes:**
- `experiment_id` (str): Unique 8-character hex experiment ID
- `status` (str): Status - `"completed"`, `"failed"`, or `"cancelled"`
- `error_message` (str | None): Error message if experiment failed
- `duration` (float | None): Execution duration in seconds
- `name` (str | None): Experiment name if provided

## Grid Search Pattern

The grid search pattern creates experiments for all parameter combinations:

```python
# Define parameter grid
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]
dropout = [0.1, 0.2]

# Create experiment specs for cartesian product
experiments = []
for lr in learning_rates:
    for bs in batch_sizes:
        for dp in dropout:
            experiments.append(
                yanex.ExperimentSpec(
                    script_path=Path("train.py"),
                    config={
                        "learning_rate": lr,
                        "batch_size": bs,
                        "dropout": dp,
                    },
                    name=f"grid-lr{lr}-bs{bs}-dp{dp}",
                    tags=["grid-search"]
                )
            )

# Run in parallel
results = yanex.run_multiple(experiments, parallel=8)
```

Total experiments: 3 × 3 × 2 = 18

## Sequential vs Parallel Execution

### Sequential Execution (Default)

```python
# Run one at a time
results = yanex.run_multiple(experiments, parallel=None)
```

**Use when:**
- Limited memory/CPU resources
- Debugging experiments
- Resource-intensive experiments (GPU, memory)

### Parallel Execution

```python
# Run with 4 workers
results = yanex.run_multiple(experiments, parallel=4)

# Auto-detect CPU count
results = yanex.run_multiple(experiments, parallel=0)
```

**Use when:**
- Many lightweight experiments
- Multi-core CPU available
- Independent experiments (no shared state)

**Performance:**
- Each worker runs in separate process
- True parallelism (bypasses Python GIL)
- Memory overhead: N × experiment memory

## Error Handling

Individual experiment failures don't abort the batch:

```python
results = yanex.run_multiple(experiments, parallel=4)

# Filter results by status
completed = [r for r in results if r.status == "completed"]
failed = [r for r in results if r.status == "failed"]

print(f"Completed: {len(completed)}/{len(experiments)}")
print(f"Failed: {len(failed)}/{len(experiments)}")

# Investigate failures
for result in failed:
    print(f"Failed: {result.name}")
    print(f"  Error: {result.error_message}")
    print(f"  Experiment ID: {result.experiment_id}")
```

Failed experiments still create experiment directories with metadata, so you can investigate what went wrong.

## Common Patterns

### Pattern 1: Random Search

Sample random configurations instead of grid:

```python
import random

experiments = []
for i in range(20):  # 20 random samples
    lr = 10 ** random.uniform(-4, -1)  # Log scale
    bs = random.choice([16, 32, 64, 128])
    dropout = random.uniform(0.1, 0.5)

    experiments.append(
        yanex.ExperimentSpec(
            script_path=Path("train.py"),
            config={"learning_rate": lr, "batch_size": bs, "dropout": dropout},
            name=f"random-{i}",
            tags=["random-search"]
        )
    )

results = yanex.run_multiple(experiments, parallel=5)
```

### Pattern 2: Ensemble Training

Train multiple models with different random seeds:

```python
experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"learning_rate": 0.001, "random_seed": seed},
        name=f"ensemble-{seed}",
        tags=["ensemble"]
    )
    for seed in range(10)
]

results = yanex.run_multiple(experiments, parallel=10)
```

### Pattern 3: Multi-Dataset Batch

Process multiple datasets with same model:

```python
datasets = ["dataset_A", "dataset_B", "dataset_C"]

experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"learning_rate": 0.001},
        script_args=["--dataset", dataset],
        name=f"train-{dataset}",
        tags=["multi-dataset"]
    )
    for dataset in datasets
]

results = yanex.run_multiple(experiments, parallel=3)
```

### Pattern 4: Resuming Failed Experiments

Re-run only failed experiments:

```python
# First run
results = yanex.run_multiple(experiments, parallel=4)

# Filter failures
failed = [r for r in results if r.status == "failed"]
print(f"Failed: {len(failed)} experiments")

# Create specs for failed experiments (need to recreate them)
retry_experiments = [
    # Recreate ExperimentSpec with same config
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config=original_configs[result.name],
        name=f"{result.name}-retry"
    )
    for result in failed
]

# Retry
retry_results = yanex.run_multiple(retry_experiments, parallel=4)
```

## Viewing Results

After batch execution, use yanex CLI to analyze results:

```bash
# List all experiments from grid search
yanex list --tag grid-search

# Compare results
yanex compare --tag grid-search

# View specific experiment
yanex show abc12345

# View best result (highest accuracy)
yanex list --tag grid-search --status completed | head -n 1
```

## Important: Multiprocessing Requirements

When using `run_multiple()` with `parallel > 0`, you **must** wrap your code in `if __name__ == "__main__":`:

```python
# ✓ Correct
def main():
    experiments = [...]
    results = yanex.run_multiple(experiments, parallel=4)

if __name__ == "__main__":
    main()
```

```python
# ✗ Wrong - will crash on macOS/Windows
experiments = [...]
results = yanex.run_multiple(experiments, parallel=4)
```

**Why?** On macOS and Windows, Python uses the `spawn` method for multiprocessing, which imports the main module in child processes. Without the `if __name__ == "__main__":` guard, the code runs again in each child, causing infinite recursion.

**Error message if missing:**
```
RuntimeError: An attempt has been made to start a new process before the
current process has finished its bootstrapping phase.
```

**Solution:** Always wrap batch execution code in a main function with the guard.

## Next Steps

- Try different parameter grids (more learning rates, add dropout, etc.)
- Implement random search instead of grid search
- Use `yanex compare` to find best hyperparameters
- See example 03 for k-fold cross-validation pattern (advanced)
