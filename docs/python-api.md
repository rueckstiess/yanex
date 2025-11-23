# Python API Overview

Yanex provides two Python APIs for different use cases:

1. **[Run API](run-api.md)** (`yanex.*`) - For running experiments and logging data during execution
2. **[Results API](results-api.md)** (`yanex.results.*`) - For accessing and analyzing completed experiment data

## Quick Reference

### Run API (for experiment execution)

```python
import yanex

# PRIMARY PATTERN - CLI-driven (Recommended)
params = yanex.get_params()
lr = params.get('learning_rate', 0.001)

# Your training code
accuracy = train_model(lr=lr)

# Logging works in both standalone and yanex contexts
yanex.log_metrics({"accuracy": accuracy})
```

```bash
# Run with yanex CLI for full tracking
yanex run script.py --param learning_rate=0.01
```

```python
# ADVANCED PATTERN - Explicit control  
with yanex.create_experiment(
    script_path=Path(__file__),
    name="my-experiment"
):
    yanex.log_metrics({"accuracy": 0.95})
```

### Results API (for data analysis)

```python
import yanex.results as yr

# Access individual experiments
exp = yr.get_experiment("abc12345")
print(f"{exp.name}: {exp.status}")

# Find and compare experiments
experiments = yr.find(status="completed", tags=["training"])
df = yr.compare(params=["learning_rate"], metrics=["accuracy"])

# Get best experiment
best = yr.get_best("accuracy", maximize=True, status="completed")
```

---

## When to Use Each API

### Use the Run API when:
- **Running experiments** - Logging metrics, parameters, and artifacts during execution
- **CLI workflows** - Running scripts with `yanex run` for automatic tracking
- **Notebook experiments** - Creating experiments with explicit control
- **Parameter management** - Getting configuration values and defaults

### Use the Results API when:
- **Analyzing results** - Exploring completed experiment data
- **Comparing experiments** - Creating comparison tables and DataFrames
- **Finding experiments** - Searching by status, tags, metrics, or time ranges
- **Building dashboards** - Programmatic access to experiment metadata
- **Data science workflows** - Integrating with pandas for analysis

---

## Key Differences

| Aspect | Run API | Results API |
|--------|---------|-------------|
| **Purpose** | Execute and track experiments | Access and analyze data |
| **Import** | `import yanex` | `import yanex.results as yr` |
| **Context** | Active experiment required | Works with any experiment |
| **Main Use Cases** | Logging, parameter access, execution | Querying, comparison, analysis |
| **Data Flow** | Write data (metrics, artifacts) | Read data (params, metrics, metadata) |

---

## Common Workflows

### 1. CLI-Driven Experiment with Post-Analysis

```python
# train.py - Run with: yanex run train.py --param lr=0.01
import yanex

# Get parameters and train
params = yanex.get_params()
accuracy = train_model(lr=params.get('lr', 0.001))
yanex.log_metrics({"accuracy": accuracy})
```

```python
# analysis.py - Analyze results afterwards
import yanex.results as yr

# Find the latest training run
latest = yr.get_latest(tags=["training"])
print(f"Best accuracy: {latest.get_metrics()}")

# Compare multiple runs
df = yr.compare(status="completed", metrics=["accuracy", "loss"])
```

### 2. Parameter Sweep with Analysis

```python
# sweep.py - Create multiple experiments
import yanex
from pathlib import Path

learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    with yanex.create_experiment(
        script_path=Path(__file__),
        config={"learning_rate": lr},
        tags=["sweep"]
    ):
        accuracy = train_model(lr=lr)
        yanex.log_metrics({"accuracy": accuracy})
```

```python
# find_best.py - Find best configuration
import yanex.results as yr

# Get best performing experiment from sweep
best = yr.get_best("accuracy", maximize=True, tags=["sweep"])
print(f"Best LR: {best.get_param('learning_rate')}")
print(f"Best accuracy: {best.get_metrics()}")
```

### 3. Production Model Training and Monitoring

```python
# production_train.py
import yanex

if yanex.has_context():
    # Running via yanex - full tracking
    params = yanex.get_params()
    model = train_production_model(params)
    yanex.log_metrics({"final_accuracy": model.accuracy})
    yanex.copy_artifact(model.save_path, "model.pkl")
else:
    # Standalone mode - basic logging
    model = train_production_model()
    print(f"Accuracy: {model.accuracy}")
```

```python
# monitoring.py - Check recent production runs
import yanex.results as yr

# Get recent production experiments
recent = yr.find(
    tags=["production"],
    started_after="1 week ago",
    status="completed"
)

for exp_data in recent:
    exp = yr.get_experiment(exp_data["id"])
    metrics = exp.get_metrics()
    print(f"{exp.name}: {metrics}")
```

---

## Detailed Documentation

- **[Run API Reference](run-api.md)** - Complete Run API documentation
- **[Results API Reference](results-api.md)** - Complete Results API documentation

## Related Documentation

- **[CLI Commands](cli-commands.md)** - Command-line interface
- **[Configuration Guide](configuration.md)** - Parameter management  
- **[Experiment Structure](experiment-structure.md)** - File organization
- **[Best Practices](best-practices.md)** - Usage patterns and tips