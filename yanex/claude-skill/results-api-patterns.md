# Results API Patterns

Use the Results API for programmatic analysis in notebooks or temporary scripts.

## Import

```python
import yanex.results as yr
```

## Querying Experiments

### Get Single Experiment
```python
exp = yr.get_experiment("abc12345")
print(f"{exp.name}: {exp.status}")
print(f"Params: {exp.get_params()}")
print(f"Accuracy: {exp.get_metric('accuracy')}")
```

### Get Multiple Experiments
```python
# By filters
exps = yr.get_experiments(status="completed")
exps = yr.get_experiments(name="yelp-2-*")
exps = yr.get_experiments(tags=["training", "sweep"])
exps = yr.get_experiments(started_after="1 week ago")

# Combined filters
exps = yr.get_experiments(
    name="yelp-*",
    status="completed",
    tags=["training"],
    started_after="2025-01-01"
)
```

### Find Best Experiment
```python
best = yr.get_best("accuracy", maximize=True, status="completed")
best = yr.get_best("loss", maximize=False, tags=["training"])

print(f"Best: {best.name} ({best.id})")
print(f"Accuracy: {best.get_metric('accuracy'):.4f}")
```

### Get Latest Experiment
```python
latest = yr.get_latest(tags=["training"])
latest = yr.get_latest(name="yelp-3-*", status="completed")
```

## Experiment Object

```python
exp = yr.get_experiment("abc12345")

# Metadata
exp.id                  # Experiment ID
exp.name                # Experiment name
exp.status              # Status (completed, failed, etc.)
exp.started_at          # Start timestamp
exp.completed_at        # End timestamp
exp.duration            # Duration timedelta
exp.tags                # List of tags
exp.script_path         # Path to script

# Parameters
exp.get_params()        # Dict of all parameters
exp.get_param("lr")     # Single parameter value
exp.get_param("model.hidden_size")  # Nested param

# Metrics
exp.get_metrics()       # List of all metric dicts (with step)
exp.get_metric("loss")  # Single metric value(s)

# Artifacts
exp.list_artifacts()    # List of artifact filenames
exp.load_artifact("model.pt")  # Load artifact
```

## Comparison DataFrame

```python
# Compare experiments as pandas DataFrame
df = yr.compare(tags=["sweep"])
df = yr.compare(name="yelp-2-*", status="completed")
df = yr.compare(
    tags=["training"],
    params=["learning_rate", "batch_size"],
    metrics=["accuracy", "loss"]
)

# Access columns (multi-level)
df[("param", "learning_rate")]   # Parameter column
df[("metric", "accuracy")]       # Metric column
df[("meta", "name")]             # Metadata column

# Find best
best_idx = df[("metric", "accuracy")].idxmax()
best_row = df.loc[best_idx]

# Group by parameter
grouped = df.groupby(("param", "batch_size"))
summary = grouped[[("metric", "loss")]].mean()
```

## Time-Series Metrics (for Plotting)

```python
# Get metrics in long format for visualization
df = yr.get_metrics(tags=["sweep"])
df = yr.get_metrics(name="yelp-2-*", metrics="train_loss")
df = yr.get_metrics(
    tags=["training"],
    metrics=["train_loss", "val_loss"],
    include_params="auto"  # Include varying params
)

# DataFrame columns: experiment_id, step, metric_name, value, <params...>
```

### Plotting with Matplotlib

```python
import yanex.results as yr
import matplotlib.pyplot as plt

df = yr.get_metrics(name="yelp-2-hpo*", metrics="train_loss")

# Group by learning rate
for lr, group in df.groupby('learning_rate'):
    plt.plot(group.step, group.value, label=f'lr={lr}')

plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Training Loss by Learning Rate')
plt.show()
```

### Multiple Metrics with Subplots

```python
df = yr.get_metrics(tags=["training"])
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot loss
df_loss = df[df.metric_name == 'train_loss']
for lr, group in df_loss.groupby('learning_rate'):
    ax1.plot(group.step, group.value, label=f'lr={lr}')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot accuracy
df_acc = df[df.metric_name == 'train_accuracy']
for lr, group in df_acc.groupby('learning_rate'):
    ax2.plot(group.step, group.value, label=f'lr={lr}')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()
```

## Bulk Operations

```python
# Archive old experiments
count = yr.archive_experiments(
    status="failed",
    ended_before="1 month ago"
)

# Delete experiments
count = yr.delete_experiments(ids=["abc123", "def456"])

# Export to file
yr.export_experiments(
    "results.json",
    format="json",
    tags=["training"],
    status="completed"
)
```

## Quick Analysis Script Template

```python
#!/usr/bin/env python
"""Quick analysis of experiment results."""
import yanex.results as yr

# Query experiments
exps = yr.get_experiments(name="yelp-*", status="completed")
print(f"Found {len(exps)} experiments")

# Find best
best = yr.get_best("test_accuracy", maximize=True, name="yelp-*")
if best:
    print(f"\nBest experiment: {best.name} ({best.id})")
    print(f"  Accuracy: {best.get_metric('test_accuracy'):.4f}")
    print(f"  Params: {best.get_params()}")

# Compare as DataFrame
df = yr.compare(name="yelp-*", status="completed")
print(f"\nComparison DataFrame shape: {df.shape}")
print(df.head())
```

## Filter Reference

All query functions accept these filters:
- `ids`: list[str] - Match any of these IDs
- `status`: str | list[str] - Match status(es)
- `name`: str - Glob pattern matching
- `tags`: list[str] - Must have ALL tags (AND logic)
- `started_after`: str | datetime - Started >= time
- `started_before`: str | datetime - Started <= time
- `ended_after`: str | datetime - Ended >= time
- `ended_before`: str | datetime - Ended <= time
- `archived`: bool - Include archived experiments
- `limit`: int - Max number of results
