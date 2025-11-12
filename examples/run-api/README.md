# Yanex Run API Examples

This directory contains examples demonstrating the Run API for programmatic experiment creation and batch execution.

## Examples

### [01: Creating Experiments](01_creating_experiments/README.md)
Create experiments programmatically with `create_experiment()` context manager. Access experiment information and directories.

**Concepts**: `create_experiment()`, `get_experiment_id()`, `get_experiment_dir()`, experiment lifecycle

---

### [02: Batch Execution](02_batch_execution/README.md)
Run multiple experiments in parallel with `run_multiple()`. Grid search pattern for hyperparameter tuning.

**Concepts**: `ExperimentSpec`, `run_multiple()`, `ExperimentResult`, parallel execution, grid search

---

### [03: K-Fold Cross-Validation](03_kfold_training/README.md)
Advanced orchestrator/executor pattern for k-fold CV. Single script acts as both orchestrator and executor. Demonstrates `get_cli_args()`.

**Concepts**: Orchestrator/executor pattern, sentinel parameters, `get_cli_args()`, k-fold cross-validation

---

## Learning Path

**New to the Run API?** Start in order:
1. **01**: Learn programmatic experiment creation
2. **02**: Learn batch execution and parallel processing
3. **03**: Master advanced orchestration patterns

**Quick references:**
- **Jupyter notebooks**: Example 01
- **Grid search/hyperparameter tuning**: Example 02
- **K-fold CV/ensemble training**: Example 03

## Running Examples

**Important:** Run API examples should be run **directly with Python**, NOT with `yanex run`:

```bash
# ✓ Correct
python 01_creating_experiments/process_data.py
python 02_batch_execution/grid_search.py
python 03_kfold_training/train_kfold.py

# ✗ Wrong - will cause errors
yanex run 01_creating_experiments/process_data.py  # Error!
```

**Exception:** Example 03 can optionally be run with `yanex run` to pass CLI flags:

```bash
# Also works for example 03 (accesses --parallel via get_cli_args)
yanex run 03_kfold_training/train_kfold.py --parallel 3
```

## API vs CLI

**Run API** (these examples):
- Programmatic experiment creation
- Batch execution with `run_multiple()`
- For notebooks, grid search, k-fold CV
- Run directly: `python script.py`

**CLI** (examples/cli):
- Dual-mode scripts (standalone + tracked)
- Run with: `yanex run script.py`
- Primary recommended pattern for most users

See [CLI examples](../cli/README.md) for the primary usage pattern.

## Key API Functions

### Experiment Creation
- `yanex.create_experiment()` - Create experiment with context manager
- `yanex.get_experiment_id()` - Get current experiment ID
- `yanex.get_experiment_dir()` - Get experiment directory path

### Batch Execution
- `yanex.ExperimentSpec` - Specification for single experiment
- `yanex.run_multiple()` - Execute multiple experiments
- `yanex.ExperimentResult` - Result of experiment execution

### CLI Integration
- `yanex.get_cli_args()` - Access CLI flags from orchestrator scripts

See [Run API documentation](../../docs/run-api.md) for complete reference.
