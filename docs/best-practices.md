# Best Practices Guide

Tips and patterns for effective experiment tracking with Yanex.

## General Principles

### 1. Commit Before Experiments

Always commit your code changes before running important experiments:

```bash
# Good workflow
git add .
git commit -m "Implement attention mechanism"
yanex run train.py --tag "attention-experiment"

# Avoid: Running with uncommitted changes
# yanex run train.py --force-dirty  # Not recommended
```

**Why:** Ensures full reproducibility and tracks exactly what code produced your results.

### 2. Use Meaningful Names and Tags

```bash
# Good: Descriptive names and organized tags
yanex run train.py \
    --name "resnet50-imagenet-baseline" \
    --tag baseline \
    --tag resnet \
    --tag imagenet

# Avoid: Generic names
yanex run train.py --name "experiment1"
```

### 3. Document Your Experiments

```yaml
# config.yaml - Include comments
model:
  learning_rate: 0.001    # Tuned for this dataset size
  batch_size: 32         # Limited by GPU memory
  dropout: 0.1           # Prevents overfitting on small dataset
```

```python
# In your script (vision_transformer.py)
import yanex

params = yanex.get_params()
# Your training code
accuracy = train_model(params)
yanex.log_results({"accuracy": accuracy})
```

## Experiment Organization

### Tagging Strategy

Use a consistent tagging hierarchy:

```bash
# Project phase
--tag "phase1" --tag "phase2" --tag "final"

# Model family
--tag "resnet" --tag "transformer" --tag "cnn"

# Experiment type
--tag "baseline" --tag "ablation" --tag "hyperparameter-search"

# Dataset
--tag "cifar10" --tag "imagenet" --tag "custom-data"

# Status
--tag "production" --tag "development" --tag "debug"
```

### Naming Conventions

```bash
# Pattern: {model}-{dataset}-{key-feature}
yanex run train.py --name "resnet50-cifar10-baseline"
yanex run train.py --name "transformer-wikitext-attention-v2"
yanex run train.py --name "cnn-custom-augmented"

# For parameter sweeps: include key parameters
yanex run train.py --name "resnet-lr0.01-bs128-wd0.0001"
```

## Parameter Management

### Configuration Organization

```yaml
# Good: Hierarchical organization
# config.yaml
model:
  architecture: "resnet50"
  pretrained: true
  num_classes: 1000

training:
  learning_rate: 0.001
  batch_size: 256
  epochs: 90
  optimizer: "sgd"
  momentum: 0.9
  weight_decay: 0.0001

data:
  dataset: "imagenet"
  image_size: 224
  augmentation:
    random_crop: true
    horizontal_flip: true
    normalize: true

hardware:
  num_gpus: 4
  mixed_precision: true
  dataloader_workers: 8
```

### Parameter Validation

```python
def validate_config(params):
    """Validate experiment parameters before training."""
    # Required parameters
    required = ['learning_rate', 'batch_size', 'epochs']
    for key in required:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")
    
    # Value constraints
    if params['learning_rate'] <= 0:
        raise ValueError("Learning rate must be positive")
        
    if params['batch_size'] not in [16, 32, 64, 128, 256]:
        raise ValueError("Batch size must be power of 2 between 16-256")
    
    # Logical constraints
    if params.get('use_dropout', False) and params.get('dropout', 0) == 0:
        raise ValueError("Dropout enabled but dropout rate is 0")

def main():
    params = yanex.get_params()
    validate_config(params)
    
    # Training code
    accuracy = train_model(params)
    yanex.log_results({"accuracy": accuracy})
```

## Result Logging

### Comprehensive Logging

```python
# Log configuration for reference
yanex.log_results({"config": params})

# Training loop with detailed logging
for epoch in range(params['epochs']):
    # Training metrics
    train_loss, train_acc = train_epoch(model, train_loader)
    
    # Validation metrics
    val_loss, val_acc = validate(model, val_loader)
    
    # Log per-epoch metrics
    yanex.log_results({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        yanex.log_artifact("best_model.pth", "best_model.pth")

# Final summary
yanex.log_results({
    "final_train_accuracy": train_acc,
    "final_val_accuracy": val_acc,
    "best_val_accuracy": best_acc,
    "total_epochs": epoch + 1,
    "converged": val_acc > 0.95
})
```

---

## Common Anti-Patterns to Avoid

### ❌ Don't Do This

```bash
# Vague naming
yanex run train.py --name "test1"

# No tags or organization  
yanex run train.py

# Running with dirty git state
yanex run train.py --force-dirty

# No error handling
# (script crashes without logging error info)

# Hardcoded parameters
# (no config file, no CLI overrides)
```

### ✅ Do This Instead

```bash
# Descriptive naming
yanex run train.py --name "resnet50-cifar10-baseline"

# Organized with tags
yanex run train.py --tag baseline --tag resnet --tag cifar10

# Clean git state
git commit -m "Fix preprocessing bug"
yanex run train.py

# Proper error handling
# (log errors, tag failed experiments)

# Flexible parameters
yanex run train.py --config base.yaml --param learning_rate=0.01
```

---

**Related:**
- [Configuration Guide](configuration.md) - Parameter management
- [Python API](python-api.md) - Using Yanex in your code  
- [CLI Commands](cli-commands.md) - Command reference