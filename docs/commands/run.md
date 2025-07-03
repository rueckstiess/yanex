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
- **Results**: Via `experiment.log_results()` calls in your script
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

Yanex allows you to stage experiments before running them, which is useful for preparing complex configurations or parameter sweeps.

```bash
# Stage an experiment
yanex stage script.py --stage
```

### Run Staged Experiments

```bash
# Run all staged experiments
yanex run --staged
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

You can perform parameter sweeps by staging multiple experiments with different configurations with a single command.

```bash
# Explicit list of parameters
yanex run script.py --param "workload_size=list(50, 100, 200)" --stage
```

Then run all staged experiments:

```bash
yanex run --staged
```

The following sweep syntax is supported:

- `list(value1, value2, ...)` - Enumerates multiple values
- `range(start, end, step)` - Generates a range of values
- `linspace(start, end, num)` - Generates evenly spaced values
- `logspace(start, end, num)` - Generates logarithmically spaced values

### Grid Search

If multiple parameter sweeps are defined, Yanex will perform a grid search across all combinations.

```bash
# Example: Sweep across two parameters
yanex run script.py --param "learning_rate=list(0.001, 0.01, 0.1)" --param "batch_size=list(32, 64, 128)" --stage
```

This will stage 9 experiments (3 learning rates x 3 batch sizes).

Note that parameter sweeps must use the `--stage` flag to prepare them for execution. You can then run all staged experiments with `yanex run --staged`.


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