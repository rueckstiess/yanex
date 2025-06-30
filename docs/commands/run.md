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

Your Python script should use the Yanex API:

```python
# train.py
from yanex import experiment

# Load parameters
params = experiment.get_params()

# Run experiment
with experiment.run() as exp:
    # Your training code
    model = create_model(params['model'])
    accuracy = train_model(model, params['training'])
    
    # Log results
    exp.log_results({
        "final_accuracy": accuracy,
        "epochs_trained": params['training']['epochs']
    })
    
    # Save artifacts
    exp.log_artifact("model.pth", "path/to/saved/model.pth")
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
from yanex import experiment
import torch

def main():
    params = experiment.get_params()
    
    with experiment.run() as exp:
        # Model setup
        model = create_model(
            lr=params.get('learning_rate', 0.001),
            layers=params.get('num_layers', 3)
        )
        
        # Training loop
        for epoch in range(params.get('epochs', 10)):
            loss = train_epoch(model)
            exp.log_results({"epoch": epoch, "loss": loss})
        
        # Final results
        accuracy = evaluate_model(model)
        exp.log_results({"final_accuracy": accuracy})
        
        # Save model
        torch.save(model.state_dict(), "model.pth")
        exp.log_artifact("model.pth", "model.pth")

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
from yanex import experiment
import pandas as pd

def main():
    params = experiment.get_params()
    
    with experiment.run() as exp:
        # Load and process data
        df = pd.read_csv(params['input_file'])
        
        # Processing steps
        df_clean = clean_data(df)
        df_features = extract_features(df_clean)
        
        # Log statistics
        exp.log_results({
            "input_rows": len(df),
            "output_rows": len(df_features),
            "features_created": len(df_features.columns)
        })
        
        # Save processed data
        df_features.to_csv("processed_data.csv", index=False)
        exp.log_artifact("processed_data.csv", "processed_data.csv")

if __name__ == "__main__":
    main()
```

```bash
yanex run process_data.py \
    --param input_file=raw_data.csv \
    --tag data-processing \
    --name "data-prep-v2"
```

## Advanced Usage

### Environment Variables

```bash
# Set parameters via environment
export YANEX_PARAM_learning_rate=0.005
export YANEX_PARAM_batch_size=128

yanex run train.py  # Uses environment parameters
```

### Multiple Configs

```yaml
# base_config.yaml
model:
  architecture: "resnet"
  layers: 18

training:
  epochs: 100
```

```yaml
# experiment_config.yaml  
model:
  layers: 50  # Override base config

training:
  learning_rate: 0.01  # Add new parameter
```

```bash
# Layer configs (experiment_config overrides base_config)
yanex run train.py --config base_config.yaml --config experiment_config.yaml
```

### Script Arguments

You can pass additional arguments to your script:

```bash
# Pass arguments after --
yanex run script.py --param lr=0.01 -- --verbose --debug
```

```python
# script.py
import sys
from yanex import experiment

# Get yanex parameters
params = experiment.get_params()

# Get script arguments
script_args = sys.argv[1:]  # Arguments after --
verbose = "--verbose" in script_args
```

### Complex Parameter Types

```yaml
# config.yaml
model:
  layers: [64, 128, 256, 512]  # List
  dropout_rates: [0.1, 0.2, 0.3, 0.4]  # List
  use_batch_norm: true  # Boolean
  activation: "relu"  # String

optimizer:
  type: "adam"
  params:
    lr: 0.001
    weight_decay: 0.0001
```

```python
# Access in script
params = experiment.get_params()

layers = params['model']['layers']  # [64, 128, 256, 512]
dropout = params['model']['dropout_rates'][0]  # 0.1
use_bn = params['model']['use_batch_norm']  # True
```

## Error Handling

### Experiment Failures

```python
# robust_experiment.py
from yanex import experiment

def main():
    params = experiment.get_params()
    
    with experiment.run() as exp:
        try:
            # Your experiment code
            result = train_model(params)
            exp.log_results({"accuracy": result.accuracy})
            exp.add_tag("completed")
            
        except Exception as e:
            # Log error information
            exp.log_results({
                "error": str(e),
                "error_type": type(e).__name__
            })
            exp.add_tag("failed")
            
            # Re-raise to maintain normal error handling
            raise

if __name__ == "__main__":
    main()
```

### Git State Issues

```bash
# Check Git status
git status

# Clean working directory
git add .
git commit -m "Clean up before experiment"

# Or force dirty state (not recommended)
yanex run script.py --force-dirty
```

## Best Practices

### 1. Reproducible Experiments

```bash
# Always commit changes before important experiments
git add .
git commit -m "Implement new feature"
yanex run experiment.py --tag "feature-test"
```

### 2. Meaningful Names and Tags

```bash
yanex run train.py \
    --name "resnet50-imagenet-baseline" \
    --description "ResNet-50 baseline on ImageNet with standard augmentation" \
    --tag baseline \
    --tag resnet \
    --tag imagenet
```

### 3. Parameter Documentation

```yaml
# config.yaml - Document your parameters
# Model architecture parameters
model:
  learning_rate: 0.001  # Initial learning rate for Adam optimizer
  batch_size: 32       # Training batch size
  num_layers: 12       # Number of transformer layers

# Training parameters  
training:
  epochs: 100          # Maximum training epochs
  early_stopping: 10   # Stop if no improvement for N epochs
```

### 4. Organized Experiments

```bash
# Use consistent tagging scheme
yanex run train.py --tag "phase1" --tag "baseline"
yanex run train.py --tag "phase1" --tag "ablation" --tag "no-dropout"
yanex run train.py --tag "phase2" --tag "improved-arch"
```

---

**Related:**
- [Python API Reference](../python-api.md)
- [Configuration Guide](../configuration.md)
- [Git Integration](../git-integration.md)