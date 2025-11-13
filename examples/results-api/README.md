# Yanex Results API Examples

This directory contains Jupyter notebooks demonstrating the Results API for querying, filtering, and analyzing completed experiments.

## What is the Results API?

The Results API provides programmatic access to experiment data stored by Yanex. Use it to:
- Query and filter experiments by name, tags, status, dates
- Read metrics, parameters, and metadata from completed experiments
- Compare experiments side-by-side using pandas DataFrames
- Build custom analysis pipelines and visualizations
- Aggregate results across multiple experiments

## Examples

### [01: Reading Experiments](01_reading_experiments.ipynb)
Learn the basics of reading experiment data programmatically.

**Concepts:**
- `get_experiment(id)` - Get single experiment by ID
- `get_experiments()` - Query multiple experiments
- `get_best()` - Find optimal experiments
- `Experiment` object - Access metrics, parameters, metadata
- Navigating experiment attributes

**Prerequisites:** None

---

### [02: Filtering and Comparing](02_filtering_comparing.ipynb)
Advanced querying and side-by-side experiment comparison with pandas DataFrames.

**Concepts:**
- `compare()` method - Create pandas DataFrame
- Multi-level DataFrame structure
- Flattening columns for simpler access
- Accessing parameters and metrics in DataFrames
- Finding optimal hyperparameters
- Grouping and aggregating experiment results

**Prerequisites:** Notebook 01

---

## Prerequisites

**Important:** These notebooks analyze existing experiment data. Before starting, you need to create sample experiments.

### Creating Sample Experiments

Both notebooks use the same 12 training experiments. Run this command from the `examples/cli/05_multi_step_metrics/` directory:

```bash
cd examples/cli/05_multi_step_metrics
yanex run train_model.py \
  --param "epochs=10,20,30" \
  --param "learning_rate=logspace(-4, -1, 4)" \
  --param "batch_size=32" \
  --tag results-demo \
  --parallel 0
```

This creates **12 experiments** (3 epochs × 4 learning rates) with the tag `results-demo` that all notebook examples use.

**What this trains:**
- A simple neural network training simulation
- Logs metrics at each epoch: `train_loss`, `train_accuracy`
- Occasionally logs validation metrics: `val_loss`, `val_accuracy`
- Parameters: `epochs`, `learning_rate`, `batch_size`

### Install Jupyter and Dependencies

```bash
# If using pip
pip install jupyter pandas matplotlib

# If using uv (recommended)
uv pip install jupyter pandas matplotlib
```

### Running the Notebooks

```bash
# Navigate to examples directory
cd examples/results-api

# Launch Jupyter
jupyter notebook
```

This will open Jupyter in your browser. Click on any notebook to open it.

### Running in Order

For the best learning experience, **run the notebooks in order (01 → 02)**. Notebook 02 builds on concepts from notebook 01.

## Relationship to Other Examples

**Run API examples** (`examples/run-api/`):
- Focus on **creating and running** experiments programmatically
- Use `create_experiment()`, `run_multiple()`, `ExperimentSpec`

**Results API examples** (`examples/results-api/`):
- Focus on **reading and analyzing** completed experiments
- Use `get_experiment()`, `get_experiments()`, `compare()`

**CLI examples** (`examples/cli/`):
- Focus on **dual-mode scripts** that work standalone or tracked
- Primary pattern for most users

**Typical workflow:**
1. Run experiments (CLI or Run API) → generates data
2. Analyze results (Results API) → explore and visualize
3. Iterate and improve

## Key Results API Functions

### Querying Experiments

```python
import yanex.results as yr

# Get single experiment
exp = yr.get_experiment("abc12345")

# Get all experiments
all_exps = yr.get_experiments()

# Filter by name pattern
exps = yr.get_experiments(name="training-run-*")

# Filter by tags
exps = yr.get_experiments(tags=["ml", "production"])

# Filter by status
exps = yr.get_experiments(status="completed")

# Combine filters
exps = yr.get_experiments(
    tags=["experiment-v2"],
    status="completed",
    started_after="2025-01-01"
)
```

### Experiment Object

```python
# Metadata
exp.id                  # Experiment ID
exp.name                # Experiment name
exp.status              # Status (completed, failed, etc.)
exp.started_at          # Start timestamp
exp.completed_at        # End timestamp
exp.duration            # Duration timedelta
exp.tags                # List of tags

# Data access
exp.get_params()        # Dict of all parameters
exp.get_param("lr")     # Single parameter value
exp.get_metrics()       # List of all metric dictionaries
exp.get_metric("loss")  # Single metric value(s)
exp.get_artifacts()     # List of artifact paths
```

### Comparison and Analysis

```python
# Create comparison DataFrame
df = yr.compare(tags=["grid-search"])

# DataFrame has multi-level columns
df[("param", "learning_rate")]   # Access parameter column
df[("metric", "accuracy")]       # Access metric column
df[("meta", "name")]             # Access metadata column

# Pandas operations
best_idx = df[("metric", "accuracy")].idxmax()
best_exp = df.loc[best_idx]

grouped = df.groupby(("param", "batch_size"))
summary = grouped[[("metric", "loss")]].mean()
```

## Tips and Best Practices

### Performance
- Use filters to limit results: `get_experiments(tags=["specific-tag"])` is faster than getting all
- The `compare()` method loads all data into memory - filter first for large result sets
- Consider using `status="completed"` filter to exclude failed/cancelled experiments

### Working with Metrics
- `get_metric(name)` returns a single value for single-step metrics
- For multi-step metrics (training curves), it returns a list of values (one per step)
- Use `get_metrics()` to get all metrics with timestamps and step numbers
- `compare()` returns the **final/most recent value** for each metric in the DataFrame

### Cleanup
- Use `delete_experiments(tags=["demo"])` to clean up test experiments
- **Warning:** Deletion is permanent - verify your filter first with `get_experiments()`

### Integration with Run API
- The k-fold example in `run-api/03_kfold_training/` demonstrates Results API usage
- Results API is great for post-experiment analysis after batch runs
- Combine `run_multiple()` results with Results API for comprehensive workflows

## Next Steps

1. **Complete the notebooks** in order (01 → 02)
2. **Try with your own experiments** - run some experiments and analyze them
3. **Explore the Run API** - See `examples/run-api/` for creating experiments programmatically
4. **Check the CLI** - See `examples/cli/` for the primary usage pattern
5. **Read the docs** - Full Results API documentation in project docs

## Troubleshooting

**No experiments found:**
- Make sure you've run some experiments first
- Check your filter criteria - try `get_experiments()` without filters
- Verify experiment storage location (`~/.yanex/experiments/` by default)

**Import errors:**
- Install Jupyter and dependencies: `pip install jupyter pandas matplotlib`
- Ensure yanex is installed: `pip install -e .` from project root

**Notebook won't run:**
- Make sure Jupyter kernel matches your Python environment
- Restart kernel if you've updated yanex code: Kernel → Restart

**DataFrame column access issues:**
- Remember the multi-level column structure: `df[("category", "column_name")]`
- Use `df.columns` to see all available columns
