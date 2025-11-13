# Run API Reference

The Run API is used for executing experiments and logging data during execution.

## Quick Reference

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
    yanex.log_metrics({"epoch": epoch, "loss": loss})

final_accuracy = evaluate_model()
yanex.log_metrics({"final_accuracy": final_accuracy})
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
    yanex.log_metrics({"accuracy": accuracy})
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

#### `yanex.get_cli_args()`

Get parsed CLI arguments used to run the experiment via `yanex run`.

Returns a dictionary with yanex CLI flags for easy access. This is useful for
orchestrator scripts that spawn child experiments and need to pass through CLI
flags (like `--parallel`) that were provided by the user.

```python
# Example: Orchestrator script that respects --parallel flag
import yanex

# Get parsed CLI args as dictionary
cli_args = yanex.get_cli_args()
# e.g., {'script': 'train.py', 'parallel': 3, 'param': ['lr=0.01'], 'tag': []}

# Clean access with defaults
parallel_workers = cli_args.get('parallel', 1)
tags = cli_args.get('tag', [])

# Use when spawning child experiments
results = yanex.run_multiple(experiments, parallel=parallel_workers)
```

**Returns:**
- `dict[str, Any]`: Dictionary with parsed CLI flags (empty dict in standalone mode)
  - Keys: `script`, `config`, `clone_from`, `param`, `name`, `tag`, `description`,
    `dry_run`, `stage`, `staged`, `parallel`
  - Note: `script_args` are NOT included - they're passed separately to your script
  - Note: `ignore_dirty` is deprecated and excluded (uncommitted changes are automatically captured as patches)

**Usage:**
```bash
# When run this way:
yanex run orchestrator.py --parallel 3 --tag ml --param lr=0.01

# Inside orchestrator.py, get_cli_args() returns:
# {
#     'script': 'orchestrator.py',
#     'parallel': 3,
#     'tag': ['ml'],
#     'param': ['lr=0.01'],
#     ...
# }
```

**Before/After:**
```python
# OLD (manual parsing - no longer needed):
if '--parallel' in cli_args:
    idx = cli_args.index('--parallel')
    parallel = int(cli_args[idx + 1])

# NEW (clean dict access):
parallel = cli_args.get('parallel', 1)
```

See [examples/api/kfold_training.py](../examples/api/kfold_training.py) for a complete example.

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
    yanex.log_metrics({"setup_complete": True})
```

**Returns:**
- `bool`: True if there is an active experiment context

### Result Logging

#### `yanex.log_metrics(data, step=None)`

Log experiment metrics. No-op in standalone mode.

```python
# Log simple metrics
yanex.log_metrics({"accuracy": 0.95, "loss": 0.05})

# Log with explicit step number
yanex.log_metrics({"epoch_loss": 0.3}, step=10)

# Multiple calls to same step merge metrics
yanex.log_metrics({"accuracy": 0.90}, step=5)
yanex.log_metrics({"loss": 0.15}, step=5)    # Merges with step 5
yanex.log_metrics({"accuracy": 0.95}, step=5) # Updates accuracy, keeps loss

# Log complex data structures
yanex.log_metrics({
    "model_config": {"layers": 12, "dropout": 0.1},
    "training_time": 3600,
    "gpu_memory_used": "8GB"
})
```

**Step Behavior:**
- If `step` is None: Auto-increments to next available step
- If `step` already exists: **Merges** new metrics with existing ones
- Conflicting metric keys: New values overwrite existing ones
- Original timestamp preserved; `last_updated` field tracks latest modification

**Parameters:**
- `data` (dict): Metrics data to log
- `step` (int, optional): Step number (auto-incremented if None)

**Storage:** Metrics are stored in `metrics.json` within the experiment directory. See [Experiment Structure](experiment-structure.md) for details.

#### `yanex.log_results(data, step=None)` ⚠️ Deprecated

> **Deprecated:** This function is deprecated. Use `log_metrics()` instead.

Log experiment results. No-op in standalone mode. This function now shows a deprecation warning and will be removed in a future version.

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
- **Comprehensive logging** - Execution details logged to `script_runs.json`

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

**Storage:** Execution details are logged to `script_runs.json` and stdout/stderr are saved as artifacts. See [Experiment Structure](experiment-structure.md) for details.

**Raises:**
- `ExperimentContextError`: If no active experiment context
- `subprocess.TimeoutExpired`: If command times out
- `subprocess.CalledProcessError`: If `raise_on_error=True` and command fails

**Environment Variables (available to scripts):**
- `YANEX_EXPERIMENT_ID`: Current experiment identifier
- `YANEX_PARAM_*`: All experiment parameters (e.g., `YANEX_PARAM_learning_rate`)

---

## Advanced API

### Batch Experiment Execution

Execute multiple experiments programmatically, either sequentially or in parallel.

#### `yanex.run_multiple(experiments, parallel=None, verbose=False)`

Run multiple experiments from within a Python script. Useful for k-fold cross-validation, grid search, ensemble training, and batch processing.

```python
from pathlib import Path
import yanex

# Create experiment specifications
experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"learning_rate": lr, "batch_size": bs},
        script_args=["--data-exp", "abc123"],
        name=f"grid-lr{lr}-bs{bs}",
        tags=["grid-search"]
    )
    for lr in [0.001, 0.01, 0.1]
    for bs in [16, 32, 64]
]

# Execute in parallel with 4 workers
results = yanex.run_multiple(experiments, parallel=4)

# Check results (blocks until all experiments complete)
completed = [r for r in results if r.status == "completed"]
failed = [r for r in results if r.status == "failed"]
print(f"Completed: {len(completed)}/{len(experiments)}")
```

**Parameters:**
- `experiments` (list[ExperimentSpec]): List of experiments to run
- `parallel` (int, optional): Number of parallel workers
  - `None`: Sequential execution (default)
  - `0`: Auto-detect number of CPU cores
  - `N > 0`: Use N parallel workers
- `verbose` (bool): Show detailed execution output (default: False)

**Note:** Uncommitted git changes are automatically captured as patch files, so no special flag is needed.

**Returns:**
- `list[ExperimentResult]`: Results for all experiments (both successful and failed)

**Behavior:**
- **Blocks** until all experiments complete
- Individual experiment failures don't abort the batch
- Each experiment runs in an isolated subprocess with full experiment tracking
- Results include experiment IDs, status, error messages, and execution duration

**Raises:**
- `ValueError`: If experiments list is empty or contains invalid specs
- `ExperimentContextError`: If called from within `yanex run` context (use this API when running scripts directly: `python script.py`)

#### `yanex.ExperimentSpec`

Specification for a single experiment to run.

```python
from pathlib import Path

spec = yanex.ExperimentSpec(
    script_path=Path("train.py"),              # Required: script to execute
    config={"learning_rate": 0.01},            # Optional: parameters
    script_args=["--data-exp", "abc123"],      # Optional: script arguments
    name="experiment-1",                        # Optional: experiment name
    tags=["ml", "training"],                    # Optional: tags
    description="Training run 1"                # Optional: description
)
```

**Attributes:**
- `script_path` (Path): Path to Python script to execute
- `config` (dict): Configuration parameters (accessible via `yanex.get_params()`)
- `script_args` (list[str]): Arguments passed to script via `sys.argv`
- `name` (str, optional): Experiment name
- `tags` (list[str]): List of tags for organization
- `description` (str, optional): Experiment description
- `function` (Callable, optional): **Not yet supported** - reserved for future inline function execution

**Validation:**
- Must specify `script_path` (function execution not yet supported)
- Script path must exist and be a Python file

#### `yanex.ExperimentResult`

Result of running a single experiment.

```python
result = results[0]

# Check status
if result.status == "completed":
    print(f"Success: {result.experiment_id}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Name: {result.name}")
elif result.status == "failed":
    print(f"Failed: {result.name}")
    print(f"Error: {result.error_message}")
```

**Attributes:**
- `experiment_id` (str): Unique 8-character hex experiment ID
- `status` (str): Experiment status - `"completed"`, `"failed"`, or `"cancelled"`
- `error_message` (str, optional): Error message if experiment failed
- `duration` (float, optional): Execution duration in seconds
- `name` (str, optional): Experiment name if provided

#### K-Fold Cross-Validation Pattern

Use a sentinel parameter to detect orchestration vs execution mode:

```python
# train.py - Acts as both orchestrator and executor
import yanex
from pathlib import Path

# Detect mode using sentinel parameter
fold_idx = yanex.get_param('_fold_idx', default=None)

if fold_idx is None:
    # ORCHESTRATION MODE: Spawn experiments for each fold
    print("Spawning 5-fold cross-validation...")

    experiments = [
        yanex.ExperimentSpec(
            script_path=Path(__file__),
            config={'_fold_idx': i, 'learning_rate': 0.01},
            name=f'fold-{i}',
            tags=['kfold']
        )
        for i in range(5)
    ]

    # Execute all folds in parallel
    results = yanex.run_multiple(experiments, parallel=5)

    # Aggregate results
    completed = [r for r in results if r.status == "completed"]
    print(f"Completed {len(completed)}/5 folds")

else:
    # EXECUTION MODE: Train single fold
    print(f"Training fold {fold_idx}...")

    # Your training code here
    train_data, val_data = load_fold(fold_idx)
    model = train_model(train_data, val_data)
    accuracy = evaluate(model, val_data)

    # Log results
    yanex.log_metrics({'fold': fold_idx, 'accuracy': accuracy})
```

Run the orchestration script directly:
```bash
# Script detects no _fold_idx and spawns 5 experiments
python train.py
```

**Why this works:**
- Script runs without yanex tracking when called directly
- Orchestrator creates experiments with `_fold_idx` set
- Each spawned experiment runs with full yanex tracking
- Context prevention ensures no nested `run_multiple()` calls

**Examples:**
- `examples/api/kfold_training.py` - Complete k-fold CV example
- `examples/api/batch_execution.py` - Grid search and ensemble training

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
    exp.log_metrics({"accuracy": accuracy})
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
    exp.log_metrics({"additional_metric": 0.88})
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
        exp.log_metrics({"result": result})
        
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

**Related:**
- [Python API Overview](python-api.md) - Overview of both APIs
- [Results API](results-api.md) - Data access and analysis API
- [CLI Commands](cli-commands.md) - Command-line interface
- [Configuration Guide](configuration.md) - Parameter management
- [Best Practices](best-practices.md) - Usage patterns and tips