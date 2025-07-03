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
):
    yanex.log_results({"accuracy": 0.95})
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
):
    
    # Your experiment code
    accuracy = train_model()
    
    # Log results using context manager methods
    yanex.log_results({"accuracy": accuracy})
    yanex.log_artifact("model.pth", "path/to/model.pth")
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

#### `yanex.get_experiment_dir()`

Get absolute path to current experiment directory. Returns None in standalone mode.

```python
exp_dir = yanex.get_experiment_dir()
if exp_dir:
    print(f"Experiment directory: {exp_dir}")
    
    # Save files directly to experiment directory
    output_file = exp_dir / "custom_results.txt"
    output_file.write_text("Custom output data")
```

**Returns:**
- `Path` or `None`: Absolute path to experiment directory

#### `yanex.execute_bash_script(command, timeout=None, raise_on_error=False, stream_output=True, working_dir=None)`

Execute bash script or shell command within experiment context. No-op in standalone mode.

```python
# Execute a script with automatic parameter passing
result = yanex.execute_bash_script("./linkbench.sh")

# Execute with command line arguments
result = yanex.execute_bash_script("./benchmark.sh --workload mixed --verbose")

# Execute with timeout and error handling
result = yanex.execute_bash_script(
    "./long_script.sh",
    timeout=300,  # 5 minutes
    raise_on_error=True,
    stream_output=False  # Capture output silently
)

print(f"Exit code: {result['exit_code']}")
print(f"Execution time: {result['execution_time']:.2f}s")
print(f"Output: {result['stdout']}")
```

**Features:**
- **Automatic parameter passing** - Experiment parameters available as `YANEX_PARAM_*` environment variables
- **Experiment context** - Script receives `YANEX_EXPERIMENT_ID` environment variable
- **Output capture** - stdout/stderr automatically saved as artifacts
- **Working directory** - Defaults to experiment directory
- **Comprehensive logging** - Execution details logged as experiment results

**Parameters:**
- `command` (str): Shell command or script to execute
- `timeout` (float, optional): Timeout in seconds
- `raise_on_error` (bool): Raise exception on non-zero exit code (default: False)
- `stream_output` (bool): Print output in real-time (default: True)
- `working_dir` (Path, optional): Working directory (defaults to experiment directory)

**Returns:**
- `dict`: Execution result with keys:
  - `exit_code` (int): Process exit code
  - `stdout` (str): Standard output
  - `stderr` (str): Standard error output
  - `execution_time` (float): Execution time in seconds
  - `command` (str): The executed command
  - `working_directory` (str): Working directory used

**Raises:**
- `ExperimentContextError`: If no active experiment context
- `subprocess.TimeoutExpired`: If command times out
- `subprocess.CalledProcessError`: If `raise_on_error=True` and command fails

**Environment Variables (available to scripts):**
- `YANEX_EXPERIMENT_ID`: Current experiment identifier
- `YANEX_PARAM_*`: All experiment parameters (e.g., `YANEX_PARAM_learning_rate`)

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

---

**Related:**
- [CLI Commands](cli-commands.md) - Command-line interface
- [Configuration Guide](configuration.md) - Parameter management
- [Best Practices](best-practices.md) - Usage patterns and tips