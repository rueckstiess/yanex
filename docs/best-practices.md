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
from yanex import experiment

params = experiment.get_params()
# Your training code
accuracy = train_model(params)
experiment.log_results({"accuracy": accuracy})
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
    params = experiment.get_params()
    validate_config(params)
    
    # Training code
    accuracy = train_model(params)
    experiment.log_results({"accuracy": accuracy})
```

## Result Logging

### Comprehensive Logging

```python
# Log configuration for reference
experiment.log_results({"config": params})

# Training loop with detailed logging
for epoch in range(params['epochs']):
    # Training metrics
    train_loss, train_acc = train_epoch(model, train_loader)
    
    # Validation metrics
    val_loss, val_acc = validate(model, val_loader)
    
    # Log per-epoch metrics
    experiment.log_results({
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
        experiment.log_artifact("best_model.pth", "best_model.pth")

# Final summary
experiment.log_results({
    "final_train_accuracy": train_acc,
    "final_val_accuracy": val_acc,
    "best_val_accuracy": best_acc,
    "total_epochs": epoch + 1,
    "converged": val_acc > 0.95
})
```

### Error Handling

```python
with experiment.run() as exp:
    try:
        # Training code
        result = train_model()
        exp.log_results({"status": "completed", "accuracy": result.accuracy})
        exp.add_tag("completed")
        
    except KeyboardInterrupt:
        exp.log_results({"status": "interrupted", "reason": "user_cancelled"})
        exp.add_tag("interrupted")
        raise
        
    except Exception as e:
        exp.log_results({
            "status": "failed",
            "error": str(e),
            "error_type": type(e).__name__
        })
        exp.add_tag("failed")
        raise
```

## Research Workflows

### Systematic Experimentation

```bash
# 1. Establish baseline
yanex run train.py --name "baseline-resnet50" --tag baseline

# 2. Ablation studies (remove components)
yanex run train.py --param use_dropout=false --name "no-dropout" --tag ablation
yanex run train.py --param use_batch_norm=false --name "no-bn" --tag ablation

# 3. Hyperparameter search
for lr in 0.001 0.01 0.1; do
    yanex run train.py --param learning_rate=$lr --name "hp-lr$lr" --tag hp-search
done

# 4. Architecture variations
yanex run train.py --param model.layers=34 --name "resnet34" --tag architecture
yanex run train.py --param model.layers=101 --name "resnet101" --tag architecture

# 5. Compare all results
yanex compare --tag baseline --tag ablation --tag hp-search --tag architecture
```

### Parameter Sweeps

```python
# systematic_search.py
from yanex import experiment
from itertools import product

def main():
    # Define search space
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    dropouts = [0.1, 0.3, 0.5]
    
    best_accuracy = 0
    best_config = None
    
    for lr, bs, dropout in product(learning_rates, batch_sizes, dropouts):
        with experiment.run(
            name=f"sweep-lr{lr}-bs{bs}-drop{dropout}",
            tags=["hyperparameter-sweep"]
        ) as exp:
            
            config = {
                'learning_rate': lr,
                'batch_size': bs,
                'dropout': dropout
            }
            
            accuracy = train_model(config)
            
            exp.log_results({
                **config,
                'accuracy': accuracy
            })
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = config
                exp.add_tag("best-so-far")
    
    print(f"Best config: {best_config} (accuracy: {best_accuracy})")

if __name__ == "__main__":
    main()
```

### A/B Testing

```python
# ab_test.py
from yanex import experiment
import random

def main():
    # Test two different approaches
    approaches = ["method_a", "method_b"]
    
    for approach in approaches:
        # Run multiple seeds for statistical significance
        for seed in range(5):
            with experiment.run(
                name=f"{approach}-seed{seed}",
                tags=["ab-test", approach, f"seed-{seed}"]
            ) as exp:
                
                random.seed(seed)
                # ... set all other random seeds ...
                
                if approach == "method_a":
                    result = train_with_method_a()
                else:
                    result = train_with_method_b()
                
                exp.log_results({
                    "method": approach,
                    "seed": seed,
                    "accuracy": result.accuracy,
                    "training_time": result.time
                })

if __name__ == "__main__":
    main()
```

```bash
# Analyze A/B test results
yanex compare --tag ab-test --only-different --export ab_test_results.csv

# Statistical analysis
python analyze_results.py ab_test_results.csv
```

## Production Workflows

### Production Checklist

Before deploying models to production:

```bash
# 1. Validate on test set
yanex run evaluate.py --config production.yaml --tag validation

# 2. Performance benchmarking
yanex run benchmark.py --config production.yaml --tag performance

# 3. Final production training
yanex run train.py \
    --config production.yaml \
    --name "production-model-v1.2" \
    --description "Final model for production deployment" \
    --tag production \
    --tag validated

# 4. Export for deployment
yanex show production-model-v1.2 --files  # Get model artifacts
```

### Model Versioning

```python
# production_training.py
from yanex import experiment
import datetime

def main():
    params = experiment.get_params()
    
    # Generate version number
    version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with experiment.run(
        name=f"production-model-{version}",
        description=f"Production model version {version}",
        tags=["production", f"v{version}"]
    ) as exp:
        
        # Training with production config
        model = train_production_model(params)
        
        # Comprehensive evaluation
        test_accuracy = evaluate_on_test_set(model)
        benchmark_time = benchmark_inference_speed(model)
        
        exp.log_results({
            "version": version,
            "test_accuracy": test_accuracy,
            "inference_time_ms": benchmark_time,
            "model_size_mb": get_model_size(model),
            "production_ready": test_accuracy > 0.95
        })
        
        # Save production artifacts
        save_model_for_deployment(model, f"model-{version}")
        exp.log_artifact(f"model-{version}.pth", f"model-{version}.pth")
        exp.log_artifact(f"model-{version}.onnx", f"model-{version}.onnx")
        
        if test_accuracy > 0.95:
            exp.add_tag("production-ready")
```

## Collaboration

### Team Conventions

```bash
# Personal experiments: use initials
yanex run train.py --name "js-attention-experiment" --tag "jane-smith"

# Shared experiments: use descriptive names
yanex run train.py --name "team-baseline-v2" --tag "team" --tag "baseline"

# Code review experiments
yanex run train.py --name "pr-123-feature-test" --tag "code-review"
```

### Shared Infrastructure

```yaml
# team_config.yaml
# Shared team configuration
shared:
  data_path: "/shared/datasets"
  output_path: "/shared/experiments"
  
  # Standard model configs
  models:
    small: {layers: 6, hidden_size: 256}
    medium: {layers: 12, hidden_size: 512}
    large: {layers: 24, hidden_size: 1024}

# Personal overrides
personal:
  gpu_id: 0  # Override per user
  debug: false
```

## Debugging and Development

### Development Experiments

```bash
# Quick debug runs
yanex run train.py \
    --param epochs=1 \
    --param batch_size=8 \
    --name "debug-quick-test" \
    --tag debug

# Integration tests
yanex run full_pipeline.py \
    --param use_small_dataset=true \
    --name "integration-test" \
    --tag testing
```

### Performance Profiling

```python
# profile_experiment.py
from yanex import experiment
import time
import psutil
import torch

def main():
    with experiment.run(name="performance-profile", tags=["profiling"]) as exp:
        
        # System info
        exp.log_results({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            "cpu_count": psutil.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / 1e9
        })
        
        # Training with profiling
        start_time = time.time()
        peak_memory = 0
        
        for epoch in range(10):
            train_start = time.time()
            
            # Training step
            loss = train_epoch()
            
            # Memory tracking
            current_memory = torch.cuda.memory_allocated() / 1e9
            peak_memory = max(peak_memory, current_memory)
            
            epoch_time = time.time() - train_start
            
            exp.log_results({
                "epoch": epoch,
                "loss": loss,
                "epoch_time_sec": epoch_time,
                "gpu_memory_gb": current_memory
            })
        
        total_time = time.time() - start_time
        exp.log_results({
            "total_training_time": total_time,
            "peak_gpu_memory_gb": peak_memory,
            "avg_epoch_time": total_time / 10
        })
```

## Data Management

### Dataset Versioning

```python
# data_pipeline.py
from yanex import experiment
import hashlib

def main():
    with experiment.run(name="data-preprocessing-v2", tags=["data"]) as exp:
        
        # Load and process data
        raw_data = load_raw_data()
        processed_data = preprocess_data(raw_data)
        
        # Data validation
        validate_data_quality(processed_data)
        
        # Dataset hash for versioning
        data_hash = hashlib.md5(str(processed_data).encode()).hexdigest()[:8]
        
        exp.log_results({
            "raw_samples": len(raw_data),
            "processed_samples": len(processed_data),
            "data_hash": data_hash,
            "train_split": 0.8,
            "val_split": 0.1,
            "test_split": 0.1
        })
        
        # Save processed data
        save_data(processed_data, f"dataset-{data_hash}")
        exp.log_artifact(f"dataset-{data_hash}.pkl", f"dataset-{data_hash}.pkl")
        
        exp.add_tag(f"data-{data_hash}")
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