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
yanex run train.py --param learning_rate=0.01 --param epochs=50
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

### Accessing Configuration

```python
import yanex

# Load configuration
params = yanex.get_params()

# Access parameters
lr = params['model']['learning_rate']  # 0.001
epochs = params['training']['epochs']   # 100

# Safe access with defaults
dropout = params.get('model', {}).get('dropout', 0.0)
```

## Parameter Hierarchy

Parameters are resolved in order of priority (highest first):

1. **CLI overrides** (`--param key=value`)
2. **Environment variables** (`YANEX_PARAM_key=value`)
3. **Configuration file** (`config.yaml`)
4. **Default values** (in your code)

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
--param "layers=[64,128,256]"    # List of integers
--param "tags=[\"exp\",\"test\"]"    # List of strings
```

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

## Multiple Configuration Files

### Custom Config Files

```bash
# Use specific config file
yanex run script.py --config custom_config.yaml

# Chain multiple configs (later files override earlier ones)
yanex run script.py --config base.yaml --config experiment.yaml
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