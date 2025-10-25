# `yanex run` - Execute Tracked Experiments

Run Python scripts as tracked experiments with automatic parameter management, Git integration, and result logging.

## Quick Start

```bash
# Basic experiment
yanex run my_script.py

# With configuration file
yanex run my_script.py --config config.yaml

# With parameter overrides
yanex run my_script.py --param learning_rate=0.01 --param epochs=100

# With metadata
yanex run my_script.py --name "baseline-model" --tag production
```

## Overview

The `yanex run` command is the core of Yanex - it executes your Python scripts while automatically tracking:

- **Parameters**: From config files and CLI overrides
- **Git State**: Current commit, branch, and working directory status
- **Results**: Via `experiment.log_metrics()` calls in your script
- **Artifacts**: Files saved during execution (e.g. model checkpoints, logs)
- **Metadata**: Experiment name, description, tags, timing

For each experiment run, Yanex assigns a unique ID and creates a dedicated directory to store all related data (defaults to `~/.yanex/experiments/<id>`). 

Yanex logs stdout and stderr outputs, all logged artifact files, logged results, the config file with all parameters used (including overrides), and metadata like the start time, end time and duration of your experiments, the git state of your repository at the time of execution, name, description, and tags.

This ensures reproducibility and easy comparison of results.


## Command Options

### Required Arguments

- `script_path`: Path to Python script to execute

### Optional Arguments

#### Parameter Overrides
- `--param KEY=VALUE`: Override configuration parameter
- `--config PATH`: Use specific configuration file

#### Metadata
- `--name NAME`: Set experiment name
- `--description DESC`: Set experiment description  
- `--tag TAG`: Add tag (can be used multiple times)

#### Git Options
- `--ignore-dirty`: Allow execution with uncommitted changes

#### Staging
- `--stage`: Stage the experiment for later execution
- `--staged`: Run all staged experiments

#### Parallel Execution (v0.5.0+)
- `--parallel N` / `-j N`: Run experiments in parallel with N workers (0=auto-detect CPUs)

#### General Options

- `--dry-run`: Validate parameters without executing the script
- `--help`: Show help message and exit



## Basic Usage

### Simple Execution

```bash
yanex run script.py
```

Runs `script.py` with default configuration (loads `./config.yaml` if present).

### With Parameters

```bash
# Override single parameter
yanex run script.py --param learning_rate=0.01

# Override multiple parameters
yanex run script.py --param learning_rate=0.01 --param batch_size=64 --param epochs=200
```

### With Metadata

```bash
yanex run script.py \
  --name "baseline-experiment" \
  --description "Initial training run with default hyperparameters" \
  --tag baseline \
  --tag production
```

## Staging Experiments

Yanex allows you to stage experiments before running them, which is useful for preparing batches of experiments or reviewing configurations before execution.

```bash
# Stage a single experiment
yanex run script.py --stage

# Stage parameter sweeps
yanex run script.py --param "lr=list(0.01, 0.001)" --stage
```

### Run Staged Experiments

```bash
# Run all staged experiments sequentially
yanex run --staged

# Run all staged experiments in parallel (v0.5.0+)
yanex run --staged --parallel 4
```


## Parameter Management

### Configuration Files

Create `config.yaml` for default parameters:

```yaml
# config.yaml
model:
  learning_rate: 0.001
  batch_size: 32
  num_layers: 12

training:
  epochs: 100
  optimizer: "adam"
  
data:
  dataset: "cifar10"
  validation_split: 0.2
```

### Parameter Hierarchy

Parameters are resolved in order of priority (highest first):

1. **CLI overrides** (`--param key=value`)
2. **Environment variables** (`YANEX_PARAM_key=value`)
3. **Configuration file** (`config.yaml`)
4. **Script defaults** (hardcoded in your script)

```bash
# CLI override takes precedence
yanex run train.py --param model.learning_rate=0.005  # Uses 0.005, not config.yaml value
```

### Custom Config Files

```bash
# Use specific config file
yanex run script.py --config custom_config.yaml

# Combine custom config with overrides
yanex run script.py --config production.yaml --param batch_size=128
```

## Parameter Sweeps

Yanex supports parameter sweeps to run multiple experiments with different configurations. As of v0.6.0, you can execute sweeps immediately or stage them for later execution.

### Direct Sweep Execution (v0.6.0+)

Run parameter sweeps immediately without staging:

```bash
# Run sweep sequentially (one after another)
yanex run script.py --param "workload_size=list(50, 100, 200)"

# Run sweep in parallel with 4 workers
yanex run script.py --param "workload_size=list(50, 100, 200)" --parallel 4

# Auto-detect CPU count for parallel execution
yanex run script.py --param "learning_rate=logspace(-4, -1, 10)" --parallel 0
```

### Sweep Syntax

The following sweep syntax is supported in both CLI parameters and config files:

- `list(value1, value2, ...)` - Enumerates multiple values
- `range(start, end, step)` - Generates a range of values (Python range syntax)
- `linspace(start, end, num)` - Generates evenly spaced values
- `logspace(start_exp, end_exp, num)` - Generates logarithmically spaced values

**CLI Examples:**

```bash
# List of specific values
--param "batch_size=list(16, 32, 64, 128)"

# Range: generates [4, 5, 6, 7]
--param "workload_size=range(4, 8, 1)"

# Linspace: generates 5 evenly spaced values from 10 to 100
--param "n_nodes=linspace(10, 100, 5)"

# Logspace: generates [0.0001, 0.001, 0.01, 0.1]
--param "learning_rate=logspace(-4, -1, 4)"
```

**Config File Examples:**

```yaml
# config.yaml
# Define sweeps directly in configuration files
learning_rate: "list(0.001, 0.01, 0.1)"
batch_size: "list(32, 64, 128)"
dropout: "linspace(0.1, 0.5, 5)"
warmup_steps: "range(0, 1000, 200)"
```

```bash
# Run config file sweep
yanex run script.py --config config.yaml

# Run in parallel with 4 workers
yanex run script.py --config config.yaml --parallel 4
```

**Note:** Whitespace in sweep syntax is flexible (e.g., `list(1,2,3)` equals `list(1, 2, 3)`)

### Grid Search (Multi-Parameter Sweeps)

If multiple parameter sweeps are defined (via CLI or config file), Yanex performs a grid search across all combinations:

**CLI Grid Search:**

```bash
# Sequential execution of 9 experiments (3 × 3)
yanex run script.py \
  --param "learning_rate=list(0.001, 0.01, 0.1)" \
  --param "batch_size=list(32, 64, 128)"

# Parallel execution with 4 workers
yanex run script.py \
  --param "learning_rate=list(0.001, 0.01, 0.1)" \
  --param "batch_size=list(32, 64, 128)" \
  --parallel 4
```

**Config File Grid Search:**

```yaml
# config.yaml
# Creates 9 experiments (3 × 3)
learning_rate: "list(0.001, 0.01, 0.1)"
batch_size: "list(32, 64, 128)"
epochs: 100  # Regular parameter (not swept)
```

```bash
# Run grid search from config
yanex run script.py --config config.yaml --parallel 4
```

This creates and runs 9 experiments with all combinations of the two swept parameters.

### Staged Sweep Execution (Original Workflow)

You can still stage experiments for later execution:

```bash
# Stage parameter sweep
yanex run script.py --param "workload_size=list(50, 100, 200)" --stage

# Run all staged experiments sequentially
yanex run --staged

# Run all staged experiments in parallel with 4 workers (v0.5.0+)
yanex run --staged --parallel 4
```

**When to use staging:**
- Preparing multiple experiment batches before execution
- Running experiments at a later time (e.g., overnight)
- Reviewing experiment configurations before execution

### Parallel Execution (v0.5.0+)

The `--parallel` flag enables parallel execution using multiple workers:

```bash
# Specify worker count
yanex run script.py --param "lr=logspace(-4, -1, 10)" --parallel 4

# Auto-detect CPU count (uses all available cores)
yanex run script.py --param "lr=logspace(-4, -1, 10)" --parallel 0

# Short flag syntax (like make -j)
yanex run script.py --param "lr=logspace(-4, -1, 10)" -j 4
```

**Benefits:**
- True parallelism using separate processes (bypasses Python GIL)
- Each experiment runs in isolation with separate storage
- Progress tracking with completion summary
- Ideal for multi-core systems and hyperparameter tuning
- Independent `yanex run` commands from different shells can run concurrently

**Usage Notes:**
- `--parallel` flag is for throttling managed execution (sweeps and staged experiments)
- Cannot be combined with `--stage` (stage first, then run with `--parallel`)
- For running multiple independent experiments concurrently, simply run `yanex run` from different shells - no special flags needed


## Git Integration

### Automatic Tracking

Yanex automatically records:
- Current Git commit hash
- Branch name
- Working directory status (clean/dirty)
- Remote repository URL

### Clean State Enforcement

By default, Yanex requires a clean Git state (no uncommitted changes) to ensure reproducibility. If your working directory is dirty, it will raise an error.

You can override this with the `--ignore-dirty` flag, but this is not recommended for production runs.

```bash
# Requires clean Git state (no uncommitted changes)
yanex run script.py

# Allow dirty state (not recommended for production)
yanex run script.py --ignore-dirty
```

### Best Practices

```bash
# 1. Commit your changes first
git add .
git commit -m "Update model architecture"

# 2. Run experiment
yanex run train.py --tag "new-architecture"

# 3. Experiment is now fully reproducible
```


---

**Related:**
- [Python API Reference](../python-api.md)
- [Configuration Guide](../configuration.md)
- [Git Integration](../git-integration.md)