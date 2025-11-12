# 01: Creating Experiments Programmatically

## What This Example Demonstrates

- Creating experiments using `yanex.create_experiment()` context manager
- Passing configuration parameters programmatically
- Accessing experiment information with `get_experiment_id()` and `get_experiment_dir()`
- Logging metrics during execution
- Automatic experiment lifecycle management (status transitions)

## Files

- `process_data.py` - Data processing script that creates experiment programmatically

## Why Programmatic Experiment Creation?

The `create_experiment()` API is useful when you need to:

- **Create experiments from Python code** - No CLI needed
- **Integrate with notebooks** - Jupyter/IPython workflows
- **Build custom workflows** - Scripts that create multiple experiments
- **Control experiment metadata** - Set name, tags, description in code

## How to Run

**Important:** This example uses `create_experiment()`, so run it directly with Python (NOT with `yanex run`):

```bash
# Run directly with Python
python process_data.py
```

> **Note:** Do NOT use `yanex run process_data.py` - this will raise an error because the script already creates its own experiment context.

## Expected Output

```
Started experiment: abc12345
Experiment directory: /Users/you/.yanex/experiments/abc12345

Processing 1000 items in chunks of 200...
  Chunk 1/5: processed 200 items
  Chunk 2/5: processed 400 items
  Chunk 3/5: processed 600 items
  Chunk 4/5: processed 800 items
  Chunk 5/5: processed 1000 items

Processing complete!
Total items processed: 1000

Experiment results saved to: /Users/you/.yanex/experiments/abc12345
✓ Experiment completed successfully: abc12345
  Directory: /Users/you/.yanex/experiments/abc12345
```

## Key Concepts

### The `create_experiment()` Context Manager

The context manager handles the complete experiment lifecycle:

```python
with yanex.create_experiment(
    script_path=Path(__file__),
    name="my-experiment",
    config={"param1": value1, "param2": value2},
    tags=["tag1", "tag2"],
    description="Description of the experiment"
):
    # Your experiment code here
    yanex.log_metrics({"accuracy": 0.95})
```

**Lifecycle:**
1. **Enter context**: Creates experiment, sets status to `running`
2. **Execute code**: Your experiment code runs
3. **Exit context**:
   - Normal exit → status set to `completed`
   - Exception → status set to `failed`
   - KeyboardInterrupt → status set to `cancelled`

### Parameters

All parameters are **required** except `name`, `tags`, and `description`:

- `script_path` (Path): Path to the script file (use `Path(__file__)`)
- `name` (str, optional): Human-readable experiment name
- `config` (dict, optional): Configuration parameters
- `tags` (list, optional): List of tags for organization
- `description` (str, optional): Experiment description

### Accessing Experiment Information

```python
# Get experiment ID (8-character hex string)
exp_id = yanex.get_experiment_id()  # e.g., "abc12345"

# Get experiment directory path
exp_dir = yanex.get_experiment_dir()  # Path object
```

### Why `get_experiment_dir()` is Useful

The experiment directory contains all experiment artifacts:

```python
exp_dir = yanex.get_experiment_dir()

# Save custom files directly to experiment directory
output_file = exp_dir / "custom_results.txt"
output_file.write_text("Custom output data")

# Read experiment metadata
metadata_file = exp_dir / "metadata.json"
metadata = json.loads(metadata_file.read_text())
```

**Directory structure:**
```
~/.yanex/experiments/abc12345/
├── metadata.json       # Experiment metadata
├── config.json         # Parameters
├── metrics.json        # Logged metrics
├── artifacts/          # Logged artifacts
└── git_patch.diff      # Git changes (if any)
```

## Comparison with CLI Usage

| Aspect | CLI Usage (`yanex run`) | API Usage (`create_experiment()`) |
|--------|-------------------------|-----------------------------------|
| **When to use** | Most experiments | Notebooks, custom workflows |
| **Invocation** | `yanex run script.py` | `python script.py` |
| **Experiment creation** | Automatic by CLI | Explicit in code |
| **Parameters** | CLI flags | Python dict |
| **Standalone mode** | Script works standalone too | Script requires yanex |

**Example - CLI usage:**
```python
# script.py - Works both standalone and with yanex
params = yanex.get_params()
lr = params.get('learning_rate', 0.001)
yanex.log_metrics({"accuracy": 0.95})
```

```bash
python script.py                           # Standalone (no tracking)
yanex run script.py -p learning_rate=0.01  # Tracked
```

**Example - API usage (this example):**
```python
# script.py - Requires yanex, always creates experiment
with yanex.create_experiment(
    script_path=Path(__file__),
    config={"learning_rate": 0.01}
):
    yanex.log_metrics({"accuracy": 0.95})
```

```bash
python script.py  # Always creates experiment
```

## Common Use Cases

### 1. Jupyter Notebooks

```python
# In a Jupyter cell
with yanex.create_experiment(
    script_path=Path("notebook.ipynb"),  # Or any path
    name="notebook-experiment",
    config={"param1": value1}
):
    # Your analysis code
    results = run_analysis()
    yanex.log_metrics(results)
```

### 2. Custom Orchestration

```python
# Run multiple experiments with different configs
for lr in [0.001, 0.01, 0.1]:
    with yanex.create_experiment(
        script_path=Path(__file__),
        name=f"sweep-lr-{lr}",
        config={"learning_rate": lr},
        tags=["sweep"]
    ):
        accuracy = train_model(lr)
        yanex.log_metrics({"accuracy": accuracy})
```

(See example 04 for a better way to do this with `run_multiple()`)

### 3. Conditional Experiment Creation

```python
# Create experiment only for production runs
if args.mode == "production":
    with yanex.create_experiment(...):
        run_production_job()
else:
    # Run without tracking
    run_development_job()
```

## Important Notes

### Don't Mix CLI and API Patterns

**This will raise an error:**
```bash
# ✗ DON'T DO THIS
yanex run process_data.py
```

**Error message:**
```
ExperimentContextError: Cannot use yanex.create_experiment() when script
is run via 'yanex run'. Either:
  - Run directly: python script.py
  - Or remove yanex.create_experiment() and use: yanex run script.py
```

Choose one pattern:
- **CLI pattern**: Use `yanex run` and let CLI create experiments
- **API pattern**: Use `create_experiment()` and run with `python`

### Experiment Context is Required

All yanex functions (`log_metrics`, `get_experiment_id`, etc.) require an active experiment context:

```python
# ✗ This won't work (no experiment context)
yanex.log_metrics({"accuracy": 0.95})

# ✓ This works (inside experiment context)
with yanex.create_experiment(Path(__file__)):
    yanex.log_metrics({"accuracy": 0.95})
```

## Next Steps

- Try changing the config parameters and running again
- Explore the experiment directory contents
- See example 02 to learn about batch execution with `run_multiple()`
- See example 03 for k-fold cross-validation (advanced orchestrator/executor pattern)
