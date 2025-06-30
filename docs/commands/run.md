# `yanex run` - Execute Tracked Experiments

Run Python scripts as tracked experiments with automatic parameter management, Git integration, and result logging.

## Quick Start

```bash
# Basic experiment
yanex run my_script.py

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
- **Artifacts**: Files saved during execution
- **Metadata**: Experiment name, description, tags, timing

## Basic Usage

### Simple Execution

```bash
yanex run script.py
```

Runs `script.py` with default configuration (loads `config.yaml` if present).

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
yanex run train.py --param learning_rate=0.01  # Uses 0.01, not config.yaml value
```

### Nested Parameters

For nested configuration structures:

```bash
# Override nested parameters with dot notation
yanex run train.py --param model.learning_rate=0.01
yanex run train.py --param training.optimizer=sgd
```

### Custom Config Files

```bash
# Use specific config file
yanex run script.py --config custom_config.yaml

# Combine custom config with overrides
yanex run script.py --config production.yaml --param batch_size=128
```

## Git Integration

### Automatic Tracking

Yanex automatically records:
- Current Git commit hash
- Branch name
- Working directory status (clean/dirty)
- Remote repository URL

### Clean State Enforcement

```bash
# Requires clean Git state (no uncommitted changes)
yanex run script.py

# Allow dirty state (not recommended for production)
yanex run script.py --force-dirty
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

## Script Integration

Your Python script can make use of the Yanex API to access parameters, log results, and manage artifacts.

```python
# train.py
import yanex

# Load parameters
params = experiment.get_params()

# Your training code
model = create_model(params['model'])
accuracy = train_model(model, params['training'])

# Log results
yanex.log_results({
    "final_accuracy": accuracy,
    "epochs_trained": params['training']['epochs']
})

# Save artifacts
yanex.log_artifact("model.pth", "path/to/saved/model.pth")
```

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
- `--force-dirty`: Allow execution with uncommitted changes

#### Execution Options
- `--no-capture`: Don't capture stdout/stderr
- `--verbose`: Enable verbose output

## Examples

### Basic Training Script

```python
# train.py
import yanex
import torch

def main():
    params = yanex.get_params()
    
    # Model setup
    model = create_model(
        lr=params.get('learning_rate', 0.001),
        layers=params.get('num_layers', 3)
    )
    
    # Training loop
    for epoch in range(params.get('epochs', 10)):
        loss = train_epoch(model)
        yanex.log_results({"epoch": epoch, "loss": loss})
    
    # Final results
    accuracy = evaluate_model(model)
    yanex.log_results({"final_accuracy": accuracy})
    
    # Save model
    torch.save(model.state_dict(), "model.pth")
    yanex.log_artifact("model.pth", "model.pth")

if __name__ == "__main__":
    main()
```

```bash
# Run with different configurations
yanex run train.py --param learning_rate=0.01 --param epochs=50
yanex run train.py --param learning_rate=0.001 --param epochs=100 --tag "long-training"
```

### Parameter Sweep

```bash
# Hyperparameter search
for lr in 0.001 0.01 0.1; do
    for bs in 16 32 64; do
        yanex run train.py \
            --param learning_rate=$lr \
            --param batch_size=$bs \
            --tag "hp-sweep" \
            --name "lr${lr}-bs${bs}"
    done
done
```

### Production Pipeline

```bash
# Production training run
yanex run train.py \
    --config production.yaml \
    --name "production-model-v1.2" \
    --description "Production model with updated architecture and larger dataset" \
    --tag production \
    --tag validated
```

### Data Processing

```python
# process_data.py
import yanex 
import pandas as pd

def main():
    params = yanex.get_params()
    
    # Load and process data
    df = pd.read_csv(params['input_file'])
    
    # Processing steps
    df_clean = clean_data(df)
    df_features = extract_features(df_clean)
    
    # Log statistics
    yanex.log_results({
        "input_rows": len(df),
        "output_rows": len(df_features),
        "features_created": len(df_features.columns)
    })
    
    # Save processed data
    df_features.to_csv("processed_data.csv", index=False)
    yanex.log_artifact("processed_data.csv", "processed_data.csv")

if __name__ == "__main__":
    main()
```

```bash
yanex run process_data.py \
    --param input_file=raw_data.csv \
    --tag data-processing \
    --name "data-prep-v2"
```

---

**Related:**
- [Python API Reference](../python-api.md)
- [Configuration Guide](../configuration.md)
- [Git Integration](../git-integration.md)