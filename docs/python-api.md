# Python API Reference

Complete reference for Yanex's Python API.

## Quick Reference

```python
import yanex

# PRIMARY PATTERN - CLI-driven (Recommended)
params = yanex.get_params()
lr = params.get('learning_rate', 0.001)

# Your training code
accuracy = train_model(lr=lr)

# Logging works in both standalone and yanex contexts
yanex.log_results({"accuracy": accuracy})
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
) as exp:
    exp.log_results({"accuracy": 0.95})
```

---

## Two Usage Patterns

### CLI-Driven Pattern (Primary)

**Use this for:** Most experiments, production workflows, team collaboration

Your script works both standalone and with yanex tracking:

```python
# train.py
import yanex

# Get parameters (empty dict in standalone mode)
params = yanex.get_params()
lr = params.get('learning_rate', 0.001)
epochs = params.get('epochs', 10)

# Training code
for epoch in range(epochs):
    loss = train_epoch(lr)
    # Logs to yanex when run via CLI, no-op when standalone
    yanex.log_results({"epoch": epoch, "loss": loss})

final_accuracy = evaluate_model()
yanex.log_results({"final_accuracy": final_accuracy})
```

```bash
# Standalone execution (no tracking)
python train.py

# Yanex execution (full tracking)
yanex run train.py --param learning_rate=0.01 --param epochs=50
```

### Explicit Creation Pattern (Advanced)

**Use this for:** Notebooks, parameter sweeps, when you need fine control

```python
import yanex
from pathlib import Path

# Explicit experiment creation
with yanex.create_experiment(
    script_path=Path(__file__),
    name="hyperparameter-sweep",
    config={"learning_rate": 0.01, "batch_size": 32},
    tags=["sweep", "optimization"],
    description="Grid search over learning rates"
) as exp:
    
    # Your experiment code
    accuracy = train_model()
    
    # Log results using context manager methods
    exp.log_results({"accuracy": accuracy})
    exp.log_artifact("model.pth", "path/to/model.pth")
```

> **Important:** Don't mix patterns! If you use `yanex.create_experiment()` in a script, don't run it with `yanex run` - this will raise an error.

---

## Core Functions

### Parameter Access

#### `yanex.get_params()`

Get experiment parameters. Returns empty dict in standalone mode.

```python
params = yanex.get_params()

# Safe access with defaults
lr = params.get('learning_rate', 0.001)
epochs = params.get('epochs', 10)

# Nested parameter access
model_config = params.get('model', {})
layers = model_config.get('layers', 12)
```

**Returns:**
- `dict`: Configuration parameters (CLI overrides → env vars → config file → defaults)

#### `yanex.get_param(key, default=None)`

Get a specific parameter with default value.

```python
# Get individual parameters
lr = yanex.get_param('learning_rate', 0.001)
batch_size = yanex.get_param('batch_size', 32)

# Shows warning if parameter not found
dropout = yanex.get_param('dropout')  # Warning if missing
```

**Parameters:**
- `key` (str): Parameter key to retrieve
- `default` (any): Default value if key not found

**Returns:**
- Parameter value or default

### Context Detection

#### `yanex.is_standalone()`

Check if running in standalone mode (no experiment tracking).

```python
if yanex.is_standalone():
    print("Running without experiment tracking")
    # Maybe use different logging or configuration
else:
    print(f"Experiment ID: {yanex.get_experiment_id()}")
```

**Returns:**
- `bool`: True if no active experiment context

#### `yanex.has_context()`

Check if there is an active experiment context.

```python
if yanex.has_context():
    # We're tracking this experiment
    yanex.log_results({"setup_complete": True})
```

**Returns:**
- `bool`: True if there is an active experiment context

### Result Logging

#### `yanex.log_results(data, step=None)`

Log experiment results. No-op in standalone mode.

```python
# Log simple metrics
yanex.log_results({"accuracy": 0.95, "loss": 0.05})

# Log with explicit step number
yanex.log_results({"epoch_loss": 0.3}, step=10)

# Log complex data structures
yanex.log_results({
    "model_config": {"layers": 12, "dropout": 0.1},
    "training_time": 3600,
    "gpu_memory_used": "8GB"
})
```

**Parameters:**
- `data` (dict): Results data to log
- `step` (int, optional): Step number (auto-incremented if None)

#### `yanex.log_artifact(name, file_path)`

Log file artifact. No-op in standalone mode.

```python
from pathlib import Path

# Log model files
yanex.log_artifact("model.pth", Path("./model.pth"))

# Log any file type
yanex.log_artifact("config.json", Path("config.json"))
yanex.log_artifact("results.csv", Path("outputs/results.csv"))
```

**Parameters:**
- `name` (str): Name for the artifact
- `file_path` (Path): Path to source file

#### `yanex.log_text(content, filename)`

Save text content as artifact. No-op in standalone mode.

```python
# Log training summary
summary = f"Training completed. Final accuracy: {accuracy}"
yanex.log_text(summary, "training_summary.txt")

# Log configuration as text
import json
config_text = json.dumps(params, indent=2)
yanex.log_text(config_text, "config.json")
```

**Parameters:**
- `content` (str): Text content to save
- `filename` (str): Name for the artifact file

#### `yanex.log_matplotlib_figure(fig, filename, **kwargs)`

Save matplotlib figure as artifact. No-op in standalone mode.

```python
import matplotlib.pyplot as plt

# Create and save plot
fig, ax = plt.subplots()
ax.plot(losses)
ax.set_title("Training Loss")
yanex.log_matplotlib_figure(fig, "loss_curve.png", dpi=300)

# Additional savefig arguments
yanex.log_matplotlib_figure(
    fig, 
    "detailed_plot.pdf", 
    bbox_inches='tight',
    transparent=True
)
```

**Parameters:**
- `fig`: Matplotlib figure object
- `filename` (str): Name for the artifact file
- `**kwargs`: Additional arguments passed to `fig.savefig()`

### Experiment Information

#### `yanex.get_experiment_id()`

Get current experiment ID. Returns None in standalone mode.

```python
exp_id = yanex.get_experiment_id()
if exp_id:
    print(f"Experiment ID: {exp_id}")
```

**Returns:**
- `str` or `None`: Current experiment ID

#### `yanex.get_status()`

Get current experiment status. Returns None in standalone mode.

```python
status = yanex.get_status()
print(f"Status: {status}")  # "created", "running", "completed", etc.
```

**Returns:**
- `str` or `None`: Current experiment status

#### `yanex.get_metadata()`

Get complete experiment metadata. Returns empty dict in standalone mode.

```python
metadata = yanex.get_metadata()
print(f"Name: {metadata.get('name')}")
print(f"Tags: {metadata.get('tags')}")
print(f"Description: {metadata.get('description')}")
```

**Returns:**
- `dict`: Complete experiment metadata

---

## Advanced API

### Explicit Experiment Creation

#### `yanex.create_experiment(script_path, name=None, config=None, tags=None, description=None)`

Create a new experiment with explicit control.

```python
from pathlib import Path

# Create experiment
with yanex.create_experiment(
    script_path=Path(__file__),
    name="parameter-sweep",
    config={"learning_rate": 0.01, "batch_size": 64},
    tags=["sweep", "optimization"],
    description="Grid search experiment"
) as exp:
    
    # Experiment code here
    accuracy = train_model()
    exp.log_results({"accuracy": accuracy})
```

**Parameters:**
- `script_path` (Path): Path to the experiment script
- `name` (str, optional): Experiment name
- `config` (dict, optional): Experiment configuration
- `tags` (list, optional): List of tags
- `description` (str, optional): Experiment description

**Returns:**
- `ExperimentContext`: Context manager for the experiment

**Raises:**
- `ExperimentContextError`: If called when running via `yanex run`

#### `yanex.create_context(experiment_id)`

Create context for an existing yanex.

```python
# Resume an existing experiment
with yanex.create_context("abc12345") as exp:
    # Continue experiment
    exp.log_results({"additional_metric": 0.88})
```

**Parameters:**
- `experiment_id` (str): ID of existing experiment

**Returns:**
- `ExperimentContext`: Context manager for the experiment

### Manual Experiment Control

#### `yanex.completed()`

Manually mark experiment as completed and exit context.

```python
with yanex.create_experiment(Path(__file__)) as exp:
    try:
        result = risky_computation()
        exp.log_results({"result": result})
        
        # Manually complete experiment
        yanex.completed()
        
    except Exception:
        # This won't be reached due to completed() above
        pass
```

**Raises:**
- `ExperimentContextError`: If no active experiment context

#### `yanex.fail(message)`

Mark experiment as failed with message and exit context.

```python
with yanex.create_experiment(Path(__file__)) as exp:
    if not validate_data():
        yanex.fail("Data validation failed")
        # Context exits here
    
    # This code won't run if validation failed
    result = train_model()
```

**Parameters:**
- `message` (str): Error message describing the failure

**Raises:**
- `ExperimentContextError`: If no active experiment context

#### `yanex.cancel(message)`

Mark experiment as cancelled with message and exit context.

```python
with yanex.create_experiment(Path(__file__)) as exp:
    try:
        for i in range(1000):
            # Long computation
            time.sleep(1)
            
    except KeyboardInterrupt:
        yanex.cancel("User interrupted computation")
```

**Parameters:**
- `message` (str): Cancellation reason

**Raises:**
- `ExperimentContextError`: If no active experiment context

---

## Usage Examples

### Basic Training Script

```python
# train.py
import yanex

def main():
    # Get parameters (works in both modes)
    params = yanex.get_params()
    lr = params.get('learning_rate', 0.001)
    epochs = params.get('epochs', 10)
    
    print(f"Training with lr={lr}, epochs={epochs}")
    
    # Training loop
    for epoch in range(epochs):
        loss = train_epoch(lr)
        yanex.log_results({"epoch": epoch, "loss": loss})
    
    # Final evaluation
    accuracy = evaluate_model()
    yanex.log_results({"final_accuracy": accuracy})
    
    # Save model
    save_model("model.pth")
    yanex.log_artifact("model.pth", Path("model.pth"))

if __name__ == "__main__":
    main()
```

```bash
# Run standalone
python train.py

# Run with yanex
yanex run train.py --param learning_rate=0.01 --param epochs=50
```

### Parameter Sweep with Explicit Creation

```python
# sweep.py
import yanex
from pathlib import Path

def run_single_experiment(lr, batch_size):
    """Run one experiment configuration."""
    with yanex.create_experiment(
        script_path=Path(__file__),
        name=f"sweep-lr{lr}-bs{batch_size}",
        config={"learning_rate": lr, "batch_size": batch_size},
        tags=["parameter-sweep"]
    ) as exp:
        
        # Train model with these parameters
        accuracy = train_model(lr=lr, batch_size=batch_size)
        
        exp.log_results({
            "learning_rate": lr,
            "batch_size": batch_size,
            "accuracy": accuracy
        })
        
        return accuracy

def main():
    # Define search space
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    
    results = []
    for lr in learning_rates:
        for bs in batch_sizes:
            accuracy = run_single_experiment(lr, bs)
            results.append((lr, bs, accuracy))
    
    # Find best configuration
    best_lr, best_bs, best_acc = max(results, key=lambda x: x[2])
    print(f"Best: lr={best_lr}, bs={best_bs}, acc={best_acc}")

if __name__ == "__main__":
    main()
```

### Dual-Mode Script

```python
# flexible.py - Works in both contexts
import yanex

def main():
    # Detect mode
    if yanex.is_standalone():
        print("Running in standalone mode")
    else:
        print(f"Running as experiment: {yanex.get_experiment_id()}")
    
    # Get parameters (adaptive to context)
    params = yanex.get_params()
    lr = params.get('learning_rate', 0.001)
    
    # Your code here
    accuracy = train_model(lr)
    
    # Logging (works in both modes)
    yanex.log_results({"accuracy": accuracy})
    
    # Conditional behavior based on context
    if yanex.has_context():
        # Additional logging when tracked
        yanex.log_text(f"Experiment completed with accuracy: {accuracy}", "summary.txt")

if __name__ == "__main__":
    main()
```

---

## Error Handling

### Common Errors

**Mixing patterns:**
```python
# DON'T DO THIS - Will raise ExperimentContextError
with yanex.create_experiment(Path(__file__)) as exp:
    pass
```
```bash
yanex run script.py  # Error: Cannot use create_experiment with yanex run
```

**Missing experiment context:**
```python
# These functions require active experiment context:
yanex.completed()  # ExperimentContextError if no context
yanex.fail("error")  # ExperimentContextError if no context
yanex.cancel("cancelled")  # ExperimentContextError if no context
```

### Best Practices

1. **Prefer CLI-first pattern** for most use cases
2. **Use explicit creation** only when you need fine control
3. **Don't mix patterns** in the same script
4. **Handle both modes gracefully** when writing dual-mode scripts
5. **Use safe parameter access** with `.get()` and defaults

---

**Related:**
- [CLI Commands](cli-commands.md) - Command-line interface
- [Configuration Guide](configuration.md) - Parameter management
- [Best Practices](best-practices.md) - Usage patterns and tips