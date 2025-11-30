# Results API Reference

The Results API provides programmatic access to experiment data for analysis and comparison.

## Quick Reference

```python
import yanex.results as yr

# Access individual experiments
exp = yr.get_experiment("abc12345")
print(f"{exp.name}: {exp.status}")

# Find and compare experiments
experiments = yr.get_experiments(status="completed", tags=["training"])
df = yr.compare(params=["learning_rate"], metrics=["accuracy"])

# Get best experiment
best = yr.get_best("accuracy", maximize=True, status="completed")
```

---

## Core Functions

### Individual Experiment Access

#### `yanex.results.get_experiment(experiment_id)`

Get a single experiment by ID.

```python
import yanex.results as yr

exp = yr.get_experiment("abc12345")
print(f"{exp.name}: {exp.status}")
print(f"Accuracy: {exp.get_metrics()}")
```

**Parameters:**
- `experiment_id` (str): The experiment ID to retrieve

**Returns:**
- `Experiment`: Experiment instance

**Raises:**
- `ExperimentNotFoundError`: If experiment doesn't exist

#### `yanex.results.get_latest(**filters)`

Get the most recently created experiment matching filters.

```python
# Get latest training experiment
latest = yr.get_latest(tags=["training"])

# Get latest completed experiment matching name pattern
latest = yr.get_latest(status="completed", name="model_*")
```

**Parameters:**
- `**filters`: Filter arguments (status, tags, name patterns, etc.)

**Returns:**
- `Experiment` or `None`: Most recent experiment or None if no matches

#### `yanex.results.get_best(metric, maximize=True, **filters)`

Get the experiment with the best value for a specific metric.

```python
# Get best accuracy
best = yr.get_best("accuracy", maximize=True, status="completed")

# Get lowest loss from training experiments
best = yr.get_best("loss", maximize=False, tags=["training"])
```

**Parameters:**
- `metric` (str): Metric name to optimize
- `maximize` (bool): True to find maximum value, False for minimum
- `**filters`: Filter arguments

**Returns:**
- `Experiment` or `None`: Best experiment or None if no matches

### Multiple Experiment Access

#### `yanex.results.get_experiments(**filters)`

Find experiments matching filter criteria.

```python
# Find by status and tags
experiments = yr.get_experiments(status="completed", tags=["training"])

# Complex filtering with time range
experiments = yr.get_experiments(
    status=["completed", "failed"],
    tags=["training", "cnn"],
    started_after="2024-01-01",
    limit=10
)

# Find by IDs
experiments = yr.get_experiments(ids=["abc123", "def456"])
```

**Supported Filters:**
- `ids`: list[str] - Match any of these IDs (OR logic)
- `status`: str | list[str] - Match any of these statuses (OR logic)
- `name`: str - Glob pattern matching
- `tags`: list[str] - Must have ALL these tags (AND logic)
- `started_after`: str | datetime - Started >= this time
- `started_before`: str | datetime - Started <= this time
- `ended_after`: str | datetime - Ended >= this time
- `ended_before`: str | datetime - Ended <= this time
- `archived`: bool - True/False/None (both)
- `limit`: int - Maximum number of results

**Returns:**
- `list[Experiment]`: List of Experiment instances


### Comparison and DataFrames

#### `yanex.results.compare(params=None, metrics=None, only_different=False, **filters)`

Compare experiments and return pandas DataFrame.

```python
# Compare specific experiments
df = yr.compare(
    ids=["abc123", "def456", "ghi789"],
    params=["learning_rate", "epochs"],
    metrics=["accuracy", "loss"]
)

# Compare by filter criteria
df = yr.compare(
    status="completed",
    tags=["training"],
    params=["learning_rate", "batch_size"],
    metrics=["accuracy", "f1_score"]
)

# Access data
print(df[("param", "learning_rate")])  # Parameter column
print(df[("metric", "accuracy")].max())  # Best accuracy
params_df = df.xs("param", axis=1, level=0)  # All parameters
```

**Parameters:**
- `params` (list[str], optional): Parameter names to include (auto-discovered if None)
- `metrics` (list[str], optional): Metric names to include (auto-discovered if None)
- `only_different` (bool): Only show columns where values differ
- `**filters`: Filter arguments to select experiments

**Returns:**
- `pandas.DataFrame`: DataFrame with hierarchical columns for comparison

**Raises:**
- `ImportError`: If pandas is not available

#### `yanex.results.get_metrics(metrics=None, include_params='auto', as_dataframe=True, **filters)`

Get time-series metrics from multiple experiments in long (tidy) format, optimized for visualization with matplotlib and pandas.

```python
# Get all metrics for visualization
df = yr.get_metrics(tags=["training"])
print(df.columns)  # ['experiment_id', 'step', 'metric_name', 'value', 'learning_rate', ...]

# Filter specific metrics
df = yr.get_metrics(tags=["training"], metrics=["train_loss", "val_loss"])

# Plot with matplotlib groupby
import matplotlib.pyplot as plt
for lr, group in df[df.metric_name == "train_loss"].groupby("learning_rate"):
    plt.plot(group.step, group.value, label=f"lr={lr}")
plt.legend()
plt.show()

# Control parameter inclusion
df = yr.get_metrics(tags=["training"], include_params="all")     # All params
df = yr.get_metrics(tags=["training"], include_params="none")    # No params
df = yr.get_metrics(tags=["training"], include_params=["lr"])    # Specific params
```

**Parameters:**
- `metrics` (str | list[str], optional): Metric name(s) to include. If None, includes all metrics.
- `include_params` (str | list[str], default='auto'):
  - `'auto'`: Include only parameters that vary across experiments (default)
  - `'all'`: Include all parameters
  - `'none'`: Include no parameters
  - `list[str]`: Include specific parameter names
- `as_dataframe` (bool, default=True): Return as pandas DataFrame (long format) or dict
- `**filters`: Filter arguments to select experiments

**Returns:**
- `pandas.DataFrame` (default): Long-format DataFrame with columns:
  - `experiment_id`: Experiment ID
  - `step`: Metric step/iteration number
  - `metric_name`: Name of the metric
  - `value`: Metric value
  - `<param_cols>`: Parameter columns (based on `include_params`)
- `dict[str, list[dict]]` (if `as_dataframe=False`): Experiment ID → metrics list

**Use Cases:**
- **Time-series visualization**: Plot metric progression across experiments
- **Hyperparameter comparison**: Group by parameters to compare training curves
- **Statistical analysis**: Calculate aggregates (mean, std) across runs
- **Grid search analysis**: Visualize all parameter combinations

**Example - Compare learning rates:**
```python
df = yr.get_metrics(tags=["sweep"], metrics="train_loss")
for lr, group in df.groupby("learning_rate"):
    plt.plot(group.step, group.value, label=f"lr={lr}")
```

**Example - Final accuracy by epoch length:**
```python
df = yr.get_metrics(tags=["sweep"], metrics="accuracy")
final = df.loc[df.groupby("experiment_id")["step"].idxmax()]
plt.boxplot([final[final.epochs == e]["value"] for e in [10, 20, 30]])
```

**Raises:**
- `ImportError`: If pandas is not available and `as_dataframe=True`

### Utility Functions

#### `yanex.results.get_experiment_count(**filters)`

Get count of experiments matching filters.

```python
total = yr.get_experiment_count()
completed = yr.get_experiment_count(status="completed")
recent_training = yr.get_experiment_count(
    tags=["training"],
    started_after="1 week ago"
)
```

#### `yanex.results.experiment_exists(experiment_id, include_archived=True)`

Check if an experiment exists.

```python
if yr.experiment_exists("abc12345"):
    exp = yr.get_experiment("abc12345")
    print(f"Found: {exp.name}")
```

---

## Experiment Class

The `Experiment` class provides convenient access to all experiment data.

### Properties

```python
exp = yr.get_experiment("abc12345")

# Basic info
print(exp.id)           # "abc12345"
print(exp.name)         # "my-experiment" or None
print(exp.description)  # "Description text" or None
print(exp.status)       # "completed", "failed", etc.
print(exp.tags)         # ["training", "cnn"]

# Timing
print(exp.started_at)   # datetime object or None
print(exp.completed_at) # datetime object or None
print(exp.duration)     # timedelta object or None

# File info
print(exp.script_path)  # Path object or None
print(exp.archived)     # True/False
print(exp.experiment_dir) # Path to experiment directory
print(exp.artifacts_dir)  # Path to artifacts directory

# Dependencies
print(exp.has_dependencies) # True/False
print(exp.dependency_ids)   # ["abc123", "def456"]
```

### Dependency Properties

#### `has_dependencies`

Boolean indicating if experiment has any dependencies.

```python
if exp.has_dependencies:
    print("This experiment depends on other experiments")
```

#### `dependency_ids`

List of experiment IDs that this experiment depends on.

```python
dep_ids = exp.dependency_ids
print(f"Depends on: {dep_ids}")  # ["abc123", "def456"]
```

#### `get_dependencies(transitive=False, include_self=False)`

Get dependency experiments as `Experiment` objects.

```python
# Get direct dependencies
deps = exp.get_dependencies()
for dep in deps:
    print(f"{dep.id}: {dep.name} ({dep.status})")

# Get all transitive dependencies
all_deps = exp.get_dependencies(transitive=True)

# Include current experiment in results
pipeline = exp.get_dependencies(transitive=True, include_self=True)
```

**Parameters:**
- `transitive` (bool): If True, include transitive dependencies recursively (default: False)
- `include_self` (bool): If True, include current experiment in results (default: False)

**Returns:**
- `list[Experiment]`: List of dependency Experiment objects in topological order

**Example Use Cases:**

```python
# Load artifacts from dependencies
deps = exp.get_dependencies()
if deps:
    preprocessed_data = deps[0].load_artifact("data.pkl")

# List artifacts across all dependencies
for dep in exp.get_dependencies(transitive=True):
    print(f"{dep.id} artifacts: {dep.list_artifacts()}")

# Analyze full pipeline
pipeline = exp.get_dependencies(transitive=True, include_self=True)
for i, stage in enumerate(pipeline):
    metrics = stage.get_metrics()
    print(f"Stage {i}: {stage.name} - {metrics}")
```

**See Also:** [Dependencies Guide](dependencies.md) for complete dependency patterns.

### Data Access Methods

#### `get_params()`

Get all experiment parameters.

```python
params = exp.get_params()
lr = params.get("learning_rate", 0.001)
```

#### `get_param(key, default=None)`

Get a specific parameter with support for dot notation.

```python
lr = exp.get_param("learning_rate", 0.001)
layers = exp.get_param("model.layers", 12)  # Dot notation for nested params
```

#### `get_metrics(step=None)`

Get experiment metrics.

```python
# Get all metrics
all_metrics = exp.get_metrics()  # List of dicts with step info

# Get metrics for specific step
step_5_metrics = exp.get_metrics(step=5)  # Dict of metrics for step 5
```

#### `get_metric(name)`

Get a specific metric by name.

```python
# Get specific metric - single value if one step, list if multiple steps
accuracy = exp.get_metric("accuracy")  # 0.95 or [0.8, 0.85, 0.9]
loss = exp.get_metric("loss")          # 0.05 or [0.2, 0.15, 0.1]

# Returns None if metric doesn't exist
precision = exp.get_metric("precision")  # None if not logged
```

#### `get_artifacts()`

Get list of artifact paths.

```python
artifacts = exp.get_artifacts()  # List of Path objects
for artifact in artifacts:
    print(f"Artifact: {artifact.name}")

# Access artifacts directory directly via property
artifacts_dir = exp.artifacts_dir
model_path = artifacts_dir / "model.pkl"
if model_path.exists():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
```

**Note:** Use the `artifacts_dir` property for direct access to the artifacts directory path. This is equivalent to `exp.experiment_dir / "artifacts"`.

### Metadata Update Methods

#### `set_name(name)`, `set_description(description)`

Update experiment metadata.

```python
exp.set_name("improved-model")
exp.set_description("Updated with better hyperparameters")
```

#### `add_tags(tags)`, `remove_tags(tags)`

Manage experiment tags.

```python
exp.add_tags(["production", "v2"])
exp.remove_tags(["debug"])
```

#### `set_status(status)`

Update experiment status.

```python
exp.set_status("completed")  # Valid: created, running, completed, failed, cancelled
```

### Utility Methods

#### `to_dict()`

Get complete experiment data as dictionary.

```python
data = exp.to_dict()
# Contains: id, name, description, status, tags, timing, params, metrics, artifacts, etc.
```

#### `refresh()`

Refresh cached data by reloading from storage.

```python
exp.refresh()  # Reload all data from disk
```

---

## Bulk Operations

### `yanex.results.archive_experiments(**filters)`

Archive experiments matching filters.

```python
# Archive failed experiments older than 1 month
count = yr.archive_experiments(
    status="failed",
    ended_before="1 month ago"
)
print(f"Archived {count} experiments")

# Archive specific experiments
count = yr.archive_experiments(ids=["abc123", "def456"])
```

**Parameters:**
- `**filters`: Filter arguments to select experiments

**Returns:**
- `int`: Number of experiments successfully archived

### `yanex.results.export_experiments(path, format="json", **filters)`

Export experiments matching filters to a file.

```python
# Export training results to JSON
yr.export_experiments(
    "training_results.json",
    format="json",
    tags=["training"],
    status="completed"
)

# Export comparison data to CSV
yr.export_experiments(
    "comparison.csv",
    format="csv",
    ids=["abc123", "def456", "ghi789"]
)
```

**Parameters:**
- `path` (str): Output file path
- `format` (str): Export format ("json", "csv", "yaml")
- `**filters`: Filter arguments to select experiments

**Raises:**
- `ValueError`: If format is not supported
- `IOError`: If file cannot be written

---

**Related:**
- [Python API Overview](python-api.md) - Overview of both APIs
- [Run API](run-api.md) - Experiment execution API
- [CLI Commands](cli-commands.md) - Command-line interface
- [Configuration Guide](configuration.md) - Parameter management
- [Best Practices](best-practices.md) - Usage patterns and tips