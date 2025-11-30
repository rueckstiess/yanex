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
- `--config PATH` / `-c PATH`: Use configuration file (can be repeated for multiple files)
- `--clone-from EXP_ID`: Clone configuration from another experiment

#### Dependencies
- `--depends-on EXP_ID` / `-D EXP_ID`: Specify dependency experiment(s) (comma-separated for multiple)

#### Metadata
- `--name NAME`: Set experiment name
- `--description DESC`: Set experiment description  
- `--tag TAG`: Add tag (can be used multiple times)

#### Git Options
- `--ignore-dirty`: *(Deprecated)* This flag is no longer needed and will be removed in a future version

#### Staging
- `--stage`: Stage the experiment for later execution
- `--staged`: Run all staged experiments

#### Parallel Execution
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

# Run all staged experiments in parallel
yanex run --staged --parallel 4
```

## Dependencies

Create multi-stage experiment pipelines by declaring dependencies between experiments.

### Single Dependency

```bash
# Step 1: Run preprocessing
yanex run prepare_data.py --name "data-prep-v1"
# Returns: abc12345

# Step 2: Run training with dependency on preprocessing
yanex run train.py -D abc12345 --name "training-v1"
# Returns: def67890

# Step 3: Run evaluation with dependency on training
yanex run evaluate.py -D def67890 --name "eval-v1"
```

### Multiple Dependencies

```bash
# Train multiple models
yanex run train.py -p model=resnet18 --name "resnet-v1"  # Returns: aaa111
yanex run train.py -p model=resnet50 --name "resnet-v2"  # Returns: bbb222
yanex run train.py -p model=vgg16 --name "vgg-v1"        # Returns: ccc333

# Ensemble evaluation depends on all three
yanex run ensemble.py -D aaa111,bbb222,ccc333 --name "ensemble-eval"
```

### Accessing Dependencies in Scripts

```python
# train.py
import yanex

# Assert required dependency exists
yanex.assert_dependency("prepare_data.py")

# Access dependencies
deps = yanex.get_dependencies()
if deps:
    # Load artifact from dependency
    data = yanex.load_artifact("processed_data.pkl", from_experiment=deps[0].id)
```

**See Also:** [Dependencies Guide](../dependencies.md) for complete usage patterns and examples.

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
3. **Configuration files** (merged in order, later files override earlier ones)
4. **Script defaults** (hardcoded in your script)

```bash
# CLI override takes precedence
yanex run train.py --param model.learning_rate=0.005  # Uses 0.005, not config.yaml value

# With multiple config files, later files override earlier ones
yanex run train.py --config base.yaml --config prod.yaml  # prod.yaml values override base.yaml
```

### Custom Config Files

```bash
# Use specific config file
yanex run script.py --config custom_config.yaml

# Combine custom config with overrides
yanex run script.py --config production.yaml --param batch_size=128
```

### Multiple Config Files

Yanex supports loading multiple configuration files that are merged together. Later config files override values from earlier ones:

```bash
# Merge multiple config files
yanex run script.py --config base.yaml --config model.yaml --config data.yaml

# Short form (-c)
yanex run script.py -c base.yaml -c model.yaml -c data.yaml

# Practical example: environment-specific configs
yanex run script.py --config shared.yaml --config production.yaml
```

**Example:**

```yaml
# base.yaml
learning_rate: 0.01
batch_size: 32
epochs: 100
```

```yaml
# production.yaml
learning_rate: 0.001  # Overrides base.yaml
workers: 8            # New parameter
```

```bash
# Result: learning_rate=0.001, batch_size=32, epochs=100, workers=8
yanex run train.py --config base.yaml --config production.yaml
```

This is particularly useful for:
- **Modular configs:** Separate data, model, and training configurations
- **Environment-specific settings:** Share a base config, override with dev/staging/prod configs
- **Reusability:** Swap individual config files without duplicating shared settings

## Parameter Sweeps

Yanex supports parameter sweeps to run multiple experiments with different configurations. You can execute sweeps immediately or stage them for later execution.

### Direct Sweep Execution

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

# Run all staged experiments in parallel with 4 workers
yanex run --staged --parallel 4
```

**When to use staging:**
- Preparing multiple experiment batches before execution
- Running experiments at a later time (e.g., overnight)
- Reviewing experiment configurations before execution

### Parallel Execution

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
- **Uncommitted changes patch** (automatically captured when present)

### Handling Uncommitted Changes

Yanex no longer enforces a clean git state. Instead, it automatically captures and stores any uncommitted changes as a git patch file, ensuring full reproducibility even when working with uncommitted code.

**What happens when you have uncommitted changes:**
1. Yanex detects uncommitted changes (staged or unstaged)
2. Automatically generates a patch file: `git diff HEAD`
3. **Scans the patch for potential secrets** (API keys, tokens, credentials)
4. **Validates patch size** (warns if larger than 1MB)
5. Saves the patch as `artifacts/git_diff.patch` in the experiment directory
6. Stores metadata flags: `has_uncommitted_changes` and `patch_file` location

```bash
# Works seamlessly with uncommitted changes
yanex run script.py

# Patch is automatically captured and stored
# Check experiment metadata to see if a patch was saved
yanex show <experiment_id>
```

**Patch Contents:**
- Includes both staged and unstaged changes
- Only tracks files already in the repository (excludes untracked files)
- Can be applied later to reproduce the exact code state

**Security and Performance Checks:**

Yanex automatically scans git patches for potential security issues:

- **Secret Detection**: Uses the `detect-secrets` library to scan for API keys, tokens, credentials, and other sensitive data
- **Patch Size Validation**: Warns if patches exceed 1MB to prevent performance issues
- **Automatic Warnings**: Security findings are logged during experiment creation

```bash
# Example warning output when secrets are detected
⚠️ Potential secrets detected in git patch! Found 2 potential secret(s).
  - Base64 High Entropy String in config.py at line 42
  - Secret Keyword in credentials.txt at line 156
```

**Note**: Line numbers refer to the actual line numbers in your source files (after applying the uncommitted changes), making it easy to locate and review the flagged content.

**Security Metadata:**

The following security information is stored in experiment metadata:
- `patch_has_secrets`: Boolean indicating if potential secrets were detected
- `patch_secret_count`: Number of potential secrets found
- `patch_size_bytes`: Patch size in bytes
- `patch_size_mb`: Patch size in megabytes

```bash
# View security information
yanex show <experiment_id>
```

**Important Security Notes:**
- Secret detection may produce false positives - review findings carefully
- If secrets are detected, review your patch before sharing experiments
- Consider committing sensitive changes separately or using environment variables
- The `detect-secrets` library is automatically installed with yanex

### Best Practices

While Yanex now handles uncommitted changes gracefully, committing your code before experiments is still recommended for:
- Clean version history
- Better collaboration
- Simplified code reviews

```bash
# Recommended: Commit changes before experiments
git add .
git commit -m "Update model architecture"
yanex run train.py --tag "new-architecture"

# Also supported: Run with uncommitted changes
# Yanex automatically captures the changes as a patch
yanex run train.py --tag "experimental-changes"
```

### Cloning Experiment Configurations

Use `--clone-from` to re-run an experiment with the same configuration:

```bash
# Clone configuration from a previous experiment
yanex run train.py --clone-from abc123

# Clone and override specific parameters
yanex run train.py --clone-from abc123 --param learning_rate=0.01

# Clone configuration but use current script version
yanex run train.py --clone-from abc123
```

This is useful for:
- Re-running experiments with slight variations
- Testing bug fixes with same parameters
- Comparing different script versions with identical configs

### Reproducing Experiments with Patches

If an experiment was run with uncommitted changes, you can reproduce it by:
1. Checking out the recorded commit: `git checkout <commit_hash>`
2. Applying the saved patch: `git apply <experiment_dir>/artifacts/git_diff.patch`
3. Running the experiment again



---

**Related:**
- [Python API Reference](../python-api.md)
- [Configuration Guide](../configuration.md)
- [Best Practices - Git Workflow](../best-practices.md#git-workflow-integration)