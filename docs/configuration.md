# Configuration Guide

Complete guide to parameter management and configuration in Yanex.

## Quick Start

```yaml
# config.yaml
learning_rate: 0.001
batch_size: 32
epochs: 100
model_type: "transformer"
```

```bash
# Override parameters at runtime
yanex run train.py --config config.yaml --param learning_rate=0.01 --param epochs=50
```

## Configuration Files

### Basic Structure

Yanex uses YAML configuration files for default parameters:

```yaml
# config.yaml
# Model parameters
model:
  learning_rate: 0.001
  batch_size: 32
  num_layers: 12
  dropout: 0.1

# Training parameters
training:
  epochs: 100
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping: 10

# Data parameters
data:
  dataset: "cifar10"
  validation_split: 0.2
  augmentation: true
```

### Multiple Configuration Files

Yanex supports loading multiple configuration files that are merged together, allowing you to organize your configuration into modular, reusable pieces. When multiple config files are provided, they are merged in order from left to right, with later files taking precedence over earlier ones.

**Example Setup:**

```yaml
# data-config.yaml
data:
  filename: "my-data.jsonl"
  split: 0.2
```

```yaml
# model-config.yaml
model:
  epochs: 1000
  learning_rate: 0.01
```

```yaml
# scripts-config.yaml
yanex:
  scripts:
    - name: "prepare_data.py"
    - name: "train_model.py"
      dependencies:
        data: "prepare_data.py"
```

**Usage:**

```bash
# Load and merge multiple config files
yanex run train.py --config data-config.yaml --config model-config.yaml --config scripts-config.yaml

# All configs are merged into a single configuration
# The final config contains: data, model, and yanex sections
```

**Benefits:**

- **Modularity:** Separate data, model, and infrastructure configurations
- **Reusability:** Swap out individual config files while keeping others the same
- **Environment-specific:** Easily switch between dev/staging/production configs
- **Merge Order:** Later configs override earlier ones for conflicting keys

**Merge Example:**

```yaml
# base-config.yaml
learning_rate: 0.01
batch_size: 32
epochs: 100
```

```yaml
# override-config.yaml
learning_rate: 0.001  # Overrides base-config value
optimizer: "adam"     # New parameter
```

```bash
yanex run train.py --config base-config.yaml --config override-config.yaml
# Result: learning_rate=0.001, batch_size=32, epochs=100, optimizer="adam"
```

### Accessing Configuration

```python
import yanex

# Load configuration as a dictionary
params = yanex.get_params()

# Load individual parameters (with defaults)
lr = yanex.get_param('learning_rate', default=0.001)
batch_size = yanex.get_param('batch_size', default=32)
```

## Parameter Hierarchy

Experiment parameters are resolved in order of priority (highest first):

1. **CLI overrides** (`--param key=value`)
2. **Environment variables** (`YANEX_PARAM_key=value`)
3. **Configuration file** (`config.yaml`)
4. **Default values** (in your code)

> **Note:** CLI argument defaults (from the `yanex` section) follow a separate hierarchy where CLI arguments override config defaults. See [CLI Defaults from Config](#cli-defaults-from-config) for details.

### Example

```yaml
# config.yaml
learning_rate: 0.001
batch_size: 32
```

```bash
# Environment variable
export YANEX_PARAM_learning_rate=0.005

# CLI override
yanex run train.py --param learning_rate=0.01
```

**Result:** `learning_rate=0.01` (CLI wins), `batch_size=32` (from config)

## CLI Parameter Overrides

### Basic Syntax

```bash
# Single parameter
yanex run script.py --param learning_rate=0.01

# Multiple parameters
yanex run script.py --param learning_rate=0.01 --param epochs=200 --param batch_size=64
```

### Nested Parameters

```bash
# Nested structure access with dot notation
yanex run script.py --param model.learning_rate=0.01
yanex run script.py --param training.optimizer=sgd
yanex run script.py --param data.validation_split=0.3
```

### Parameter Types

```bash
# Numbers
--param learning_rate=0.001      # Float
--param epochs=100               # Integer

# Strings
--param model_type=transformer   # String
--param optimizer=adam           # String

# Booleans
--param use_dropout=true         # Boolean
--param debug=false              # Boolean

# Lists (JSON format)
--param "layers=[64,128,256]"      # List of integers
--param "tags=[\"exp\",\"test\"]"  # List of strings (note: requires escaping)
```

### Parameter Sweeps

Parameter sweeps can be executed immediately with support for both sequential and parallel execution. Parameter sweeps can be defined via CLI parameters or directly in configuration files.

#### CLI Parameter Sweeps

```bash
# Comma-separated lists (preferred syntax)
--param "batch_size=32,64,128"
--param "optimizer=adam,sgd,rmsprop"

# Range syntax (supports 1, 2, or 3 parameters like Python's range)
--param "n_epochs=range(10)"        # Generates [0, 1, 2, ..., 9]
--param "n_epochs=range(5,10)"      # Generates [5, 6, 7, 8, 9]
--param "workload_size=range(4,8,2)" # Generates [4, 6]

# Linspace and logspace for numeric sweeps
--param "n_nodes=linspace(10,100,5)"        # Generates [10, 30, 50, 70, 100]
--param "learning_rate=logspace(-4,-1,4)"   # Generates [0.0001, 0.001, 0.01, 0.1]

# Explicit list() syntax (also supported for backwards compatibility)
--param "batch_size=list(32,64,128)"
```

#### Config File Sweeps

Parameter sweeps can also be defined directly in YAML configuration files:

```yaml
# config.yaml
# Single parameter sweep (comma-separated syntax)
learning_rate: "0.001,0.01,0.1"
batch_size: 32
epochs: 100

# Multiple parameter sweeps (creates grid search)
model:
  dropout: "linspace(0.1,0.5,5)"
  hidden_size: "128,256,512"

training:
  lr_schedule: "constant,linear,cosine"
  warmup_steps: "range(0,1000,200)"
```

**Whitespace and trailing commas:** Whitespace and trailing commas are handled gracefully:
```yaml
# All of these are equivalent
epochs: "10,20,30"
epochs: "10, 20, 30"
epochs: "10, 20, 30,"
```

**Execution:**

```bash
# Run config file sweep sequentially
yanex run script.py --config config.yaml

# Run config file sweep in parallel with 4 workers
yanex run script.py --config config.yaml --parallel 4

# Override config sweep with CLI parameter (CLI takes precedence)
yanex run script.py --config config.yaml --param "learning_rate=0.001"
```

#### Executing Parameter Sweeps

```bash
# Run sweep sequentially
yanex run script.py --param "lr=0.001,0.01,0.1"

# Run sweep in parallel with 4 workers
yanex run script.py --param "lr=0.001,0.01,0.1" --parallel 4

# Auto-detect CPU count
yanex run script.py --param "lr=logspace(-4,-1,10)" --parallel 0
```

#### Staged Execution

```bash
# Stage for later execution
yanex run script.py --param "lr=0.001,0.01,0.1" --stage

# Execute staged experiments
yanex run --staged --parallel 4
```

See [`yanex run`](commands/run.md) for complete documentation on parameter sweeps and parallel execution

## CLI Defaults from Config

You can set default values for `yanex run` CLI arguments directly in your configuration file using a special `yanex` section. This allows you to avoid repeating common CLI options while still allowing them to be overridden when needed.

### Setting CLI Defaults

```yaml
# config.yaml
# Your experiment parameters
learning_rate: 0.001
batch_size: 32
epochs: 100

# CLI defaults for yanex run command
yanex:
  name: "transformer-experiment"
  tag: ["ml", "transformer", "dev"]
  description: "Training transformer model on text data"
  dry_run: false
  stage: false
```

### CLI Override Precedence

CLI arguments still override config defaults, maintaining expected precedence:

```bash
# Uses name from config ("transformer-experiment")
yanex run train.py --config config.yaml

# Overrides name from config
yanex run train.py --config config.yaml --name "custom-experiment"

# Uses tags from config but overrides name
yanex run train.py --config config.yaml --name "override" --tag production
```

### Supported CLI Parameters

The `yanex` section supports these `yanex run` parameters:

- `name`: Experiment name
- `tag`: Tags (single string or list)
- `description`: Experiment description
- `dry_run`: Dry run mode (boolean)
- `stage`: Stage changes before run (boolean)

**Note:** This feature only works with the `yanex run` command - other commands like `list`, `show`, etc. are not affected.

### Git Integration

Yanex automatically captures uncommitted changes as patch files, so there's no need to worry about git state. Each experiment stores:
- The current commit hash
- Branch name
- Uncommitted changes (saved as `artifacts/git_diff.patch`)

See [Best Practices - Git Workflow](best-practices.md#git-workflow-integration) for details.

## Environment Variables

Set parameters using environment variables with `YANEX_PARAM_` prefix:

```bash
# Set individual parameters
export YANEX_PARAM_learning_rate=0.005
export YANEX_PARAM_batch_size=128
export YANEX_PARAM_model_type=resnet

# Run experiment (uses environment parameters)
yanex run train.py
```

### Nested Parameters

```bash
# For nested config structures, use double underscores
export YANEX_PARAM_model__learning_rate=0.01
export YANEX_PARAM_training__epochs=200
```


### Environment-Specific Configs

```yaml
# base_config.yaml
model:
  architecture: "resnet"
  layers: 18

training:
  epochs: 100
  optimizer: "adam"
```

```yaml
# production.yaml
model:
  layers: 50  # Override base config

training:
  epochs: 200
  batch_size: 128
```

```bash
# Production run
yanex run train.py --config base_config.yaml --config production.yaml
```

## Advanced Configuration

### Complex Data Types

```yaml
# config.yaml
model:
  # Lists
  layer_sizes: [64, 128, 256, 512]
  dropout_rates: [0.1, 0.2, 0.3, 0.4]
  
  # Nested objects
  optimizer:
    type: "adam"
    params:
      lr: 0.001
      weight_decay: 0.0001
      betas: [0.9, 0.999]
  
  # Mixed types
  features:
    - name: "conv1"
      filters: 64
      kernel_size: 3
    - name: "conv2"  
      filters: 128
      kernel_size: 3

# Boolean and null values
training:
  use_gpu: true
  checkpoint_path: null  # Explicitly null
  mixed_precision: false
```

### Accessing Complex Types

```python
params = yanex.get_params()

# Lists
layer_sizes = params['model']['layer_sizes']  # [64, 128, 256, 512]
first_layer = layer_sizes[0]  # 64

# Nested objects
optimizer_lr = params['model']['optimizer']['params']['lr']  # 0.001

# Mixed structures
conv_layers = params['model']['features']
first_conv_filters = conv_layers[0]['filters']  # 64
```

## Configuration Validation

### Parameter Validation in Scripts

```python
import yanex

def validate_params(params):
    """Validate experiment parameters."""
    required_keys = ['learning_rate', 'batch_size', 'epochs']
    
    for key in required_keys:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
    if params['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")
    
    if params['batch_size'] not in [16, 32, 64, 128]:
        raise ValueError("Batch size must be one of: 16, 32, 64, 128")

def main():
    params = yanex.get_params()
    validate_params(params)
    
    # Your experiment code
      
```

### Default Values Pattern

```python
def get_model_config(params):
    """Get model configuration with sensible defaults."""
    model_params = params.get('model', {})
    
    return {
        'learning_rate': model_params.get('learning_rate', 0.001),
        'batch_size': model_params.get('batch_size', 32),
        'num_layers': model_params.get('num_layers', 6),
        'dropout': model_params.get('dropout', 0.1),
        'activation': model_params.get('activation', 'relu')
    }
```

## Best Practices

### 1. Organize Configuration Hierarchically

```yaml
# Good: Organized by component
model:
  architecture: "transformer"
  num_layers: 12
  hidden_size: 768

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

data:
  dataset: "wikitext"
  seq_length: 512
  vocab_size: 50000
```

```yaml
# Avoid: Flat structure for complex configs
learning_rate: 0.001
batch_size: 32
num_layers: 12
hidden_size: 768
dataset: "wikitext"
seq_length: 512
# ... becomes hard to manage
```

### 2. Document Parameters

```yaml
# config.yaml with documentation
model:
  learning_rate: 0.001    # Initial learning rate for Adam optimizer
  batch_size: 32         # Training batch size (memory dependent)
  num_layers: 12         # Number of transformer layers
  dropout: 0.1           # Dropout probability for regularization

training:
  epochs: 100            # Maximum training epochs
  early_stopping: 10     # Stop if no improvement for N epochs
  gradient_clip: 1.0     # Gradient clipping threshold
```

### 3. Use Environment-Specific Configs

```bash
# Directory structure
configs/
├── base.yaml           # Common parameters
├── development.yaml    # Dev-specific overrides
├── staging.yaml        # Staging overrides
└── production.yaml     # Production overrides
```

```bash
# Environment-specific runs
yanex run train.py --config configs/base.yaml --config configs/development.yaml
yanex run train.py --config configs/base.yaml --config configs/production.yaml
```

### 4. Validate Critical Parameters

```python
def main():
    params = experiment.get_params()
    
    # Validate critical parameters early
    assert params['learning_rate'] > 0, "Learning rate must be positive"
    assert params['batch_size'] > 0, "Batch size must be positive"
    assert params['epochs'] > 0, "Epochs must be positive"
    
    # Your experiment code
```


## Troubleshooting

### Common Issues

**Parameter not found:**
```python
# Problem: KeyError when accessing nested parameter
lr = params['model']['learning_rate']  # KeyError if 'model' doesn't exist

# Solution: Use safe access
lr = params.get('model', {}).get('learning_rate', 0.001)
```

**Type mismatches:**
```bash
# Problem: String instead of number
--param learning_rate="0.01"  # String "0.01"

# Solution: Proper numeric syntax
--param learning_rate=0.01    # Float 0.01
```

**Complex parameter override:**
```bash
# Problem: Can't override nested list/dict from CLI
--param model.layers=[64,128]  # Doesn't work as expected

# Solution: Use JSON format with quotes
--param "model.layers=[64,128,256]"
```

---

**Related:**
- [`yanex run`](commands/run.md) - Running experiments with parameters
- [Python API](python-api.md) - Using parameters in code
- [Best Practices](best-practices.md) - Configuration organization tips