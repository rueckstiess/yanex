# Best Practices

This guide provides recommended patterns and workflows for effective experiment tracking with Yanex.

## Table of Contents

- [Organizing Experiments](#organizing-experiments)
- [Configuration Strategies](#configuration-strategies)
- [When to Use What](#when-to-use-what)
- [Metric Logging Strategies](#metric-logging-strategies)
- [Parameter Sweep Strategies](#parameter-sweep-strategies)
- [Git Workflow Integration](#git-workflow-integration)
- [Debugging Failed Experiments](#debugging-failed-experiments)
- [Performance Tips](#performance-tips)

## Organizing Experiments

### Use Tags Strategically

Tags help categorize and filter experiments:

**By Purpose:**
```bash
yanex run train.py --tag baseline
yanex run train.py --tag experiment
yanex run train.py --tag production
```

**By Model Type:**
```bash
yanex run train.py --tag cnn --tag resnet
yanex run train.py --tag transformer --tag bert
```

**By Dataset:**
```bash
yanex run train.py --tag cifar10
yanex run train.py --tag imagenet
```

**Multiple Tags:**
```bash
# Combine tags for fine-grained organization
yanex run train.py --tag ml --tag hyperparameter-sweep --tag resnet --tag v2
```

**Example**: [04_metadata_and_tags](../examples/cli/04_metadata_and_tags/README.md)

### Naming Conventions

Use descriptive names that indicate purpose:

```bash
# Good: Descriptive and searchable
yanex run train.py --name "resnet50-baseline-v1"
yanex run train.py --name "bert-finetune-squad"
yanex run train.py --name "ablation-no-dropout"

# Avoid: Too generic
yanex run train.py --name "test"
yanex run train.py --name "run1"
```

**Naming patterns:**
- `{model}-{dataset}-{variation}`: "resnet50-cifar10-augmented"
- `{task}-{version}`: "sentiment-analysis-v2"
- `{experiment-type}-{detail}`: "ablation-batch-norm"

### Add Descriptions for Important Runs

Document significant experiments:

```bash
yanex run train.py \
  --name "production-v1" \
  --tag production \
  --description "Final model for production deployment. Trained on full dataset with optimal hyperparameters from sweep."
```

### Archive Old Experiments

Keep your workspace clean:

```bash
# Archive experiments older than 3 months
yanex archive --started-before "3 months ago"

# Archive failed experiments
yanex archive -s failed --started-before "1 month ago"

# Archive by tag
yanex archive -t experiment -t test
```

Archived experiments are moved out of active storage but remain accessible.

## Configuration Strategies

### YAML Config Hierarchy

Organize configs by specificity:

**Base config** (`config.yaml`):
```yaml
# Shared defaults
model:
  architecture: "resnet50"

training:
  optimizer: "adam"
  batch_size: 32

data:
  dataset: "cifar10"
```

**Experiment-specific overrides**:
```bash
# Override specific parameters
yanex run train.py --config config.yaml --param training.learning_rate=0.001
```

**Example**: [02_config_files](../examples/cli/02_config_files/)

### CLI Defaults in Config

Set common CLI flags in your config:

```yaml
# config.yaml
yanex:
  name: "resnet-experiment"
  tag:
    - ml
    - experiment
  description: "ResNet training experiments"

model:
  learning_rate: 0.001
```

CLI arguments still override config defaults:

```bash
# Uses defaults from config
yanex run train.py --config config.yaml

# Overrides name from config
yanex run train.py --config config.yaml --name "custom-run"
```

See [Configuration Guide](configuration.md#cli-defaults) for details.

### Parameter Organization

Structure parameters logically:

```yaml
# Good: Organized by component
model:
  architecture: "transformer"
  num_layers: 12
  hidden_size: 768

training:
  learning_rate: 0.0001
  batch_size: 32
  epochs: 100

data:
  dataset: "wikitext"
  max_length: 512
```

Access with dot notation:

```python
import yanex

arch = yanex.get_param('model.architecture')
lr = yanex.get_param('training.learning_rate')
```

## When to Use What

### CLI (Primary Usage)

**Use for**: Daily experiment tracking, parameter sweeps, reproducible runs

**Pattern**: Dual-mode scripts that work standalone or with tracking

```python
# script.py
import yanex

params = yanex.get_params()
lr = params.get('learning_rate', 0.001)

# Training code...
accuracy = train_model(lr=lr)

yanex.log_metrics({"accuracy": accuracy})
```

```bash
# Standalone (no tracking)
python script.py

# With tracking
yanex run script.py --param learning_rate=0.01
```

**Examples**: [CLI examples directory](../examples/cli/)

### Run API (Advanced Patterns)

**Use for**: K-fold cross-validation, ensemble training, batch processing, orchestration

**Pattern**: Programmatic experiment creation for complex workflows

```python
# orchestrator.py
import yanex

experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"fold": i, "learning_rate": 0.01},
        name=f"fold-{i}"
    )
    for i in range(5)
]

results = yanex.run_multiple(experiments, parallel=5)
```

**Examples**: [Run API examples directory](../examples/run-api/)

**Use cases**:
- K-fold cross-validation: [03_kfold_training](../examples/run-api/03_kfold_training/)
- Grid search: [02_batch_execution](../examples/run-api/02_batch_execution/)
- Orchestrator patterns: Scripts that spawn child experiments

### Results API (Analysis)

**Use for**: Querying completed experiments, analysis, visualization

**Pattern**: Post-experiment analysis and reporting

```python
import yanex.results as yr
import pandas as pd

# Query experiments
experiments = yr.get_experiments(tags=["hyperparameter-sweep"], status="completed")

# Create DataFrame for analysis
df = yr.compare(tags=["hyperparameter-sweep"])

# Find best configuration
best = yr.get_best("accuracy", maximize=True, tags=["hyperparameter-sweep"])
```

**Examples**: [Results API Jupyter notebooks](../examples/results-api/)

**Use cases**:
- Analyzing hyperparameter sweeps
- Finding optimal configurations
- Creating reports and visualizations
- Comparing experiment results

## Metric Logging Strategies

### Incremental Metric Building

**Key feature**: Metrics logged to the same step are **merged**, not overwritten.

This allows you to build metrics incrementally, which is especially useful when some metrics are only available at certain intervals:

```python
for epoch in range(1, epochs + 1):
    # Always log training metrics
    yanex.log_metrics({
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
    }, step=epoch)

    # Every N epochs, add validation metrics
    if epoch % validation_frequency == 0:
        val_loss, val_accuracy = validate_model()
        # Merges with existing metrics for this epoch
        yanex.log_metrics({
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, step=epoch)
```

**Result**: Epochs 0, N, 2N, ... will have both training AND validation metrics. Other epochs only have training metrics.

**Benefits**:
- Avoid expensive validation at every step
- Keep all metrics for the same epoch together
- Flexible conditional logging without complex logic

**Important**: The `step` parameter must be separate, not inside the metrics dictionary:

```python
# ✓ Correct
yanex.log_metrics({'train_loss': loss}, step=epoch)

# ✗ Wrong
yanex.log_metrics({'step': epoch, 'train_loss': loss})
```

**Example**: [05_multi_step_metrics](../examples/cli/05_multi_step_metrics/README.md)

## Parameter Sweep Strategies

### Start Small, Scale Up

```bash
# Test with small sweep first
yanex run train.py --param "lr=0.001,0.01,0.1" --parallel 3

# Scale to full sweep after validation
yanex run train.py --param "lr=logspace(-4, -1, 20)" --parallel 8
```

### Use Appropriate Sweep Types

**Discrete values** (comma-separated):
```bash
--param "batch_size=16,32,64,128"
```

**Linear spacing**:
```bash
--param "dropout=linspace(0.1, 0.5, 5)"  # [0.1, 0.2, 0.3, 0.4, 0.5]
```

**Logarithmic spacing** (for learning rates):
```bash
--param "lr=logspace(-4, -1, 10)"  # 10 values from 0.0001 to 0.1
```

**Range** (step-based):
```bash
--param "epochs=range(10, 100, 10)"  # [10, 20, 30, ..., 90]
```

### Multi-Parameter Sweeps

Create cross-product sweeps:

```bash
# 3 × 4 × 2 = 24 experiments
yanex run train.py \
  --param "lr=0.001,0.01,0.1" \
  --param "batch_size=16,32,64,128" \
  --param "dropout=0.1,0.5" \
  --parallel 8 \
  --tag hyperparameter-sweep
```

### Parallel Execution

Choose parallelism based on your hardware:

```bash
# Auto-detect CPU count
--parallel 0

# Specific worker count
--parallel 4

# Sequential (for debugging)
(omit --parallel flag)
```

**Rule of thumb**: Use `parallel=0` for automatic detection, or set to number of CPU cores.

**Example**: [07_parameter_sweeps](../examples/cli/07_parameter_sweeps/)

### Analyze Sweep Results

Use Results API to find optimal configurations:

```python
import yanex.results as yr

# Compare all sweep experiments
df = yr.compare(tags=["hyperparameter-sweep"], status="completed")

# Find best configuration
best = df.loc[df[("metric", "accuracy")].idxmax()]

print(f"Best LR: {best[('param', 'learning_rate')]}")
print(f"Best accuracy: {best[('metric', 'accuracy')]:.4f}")
```

## Git Workflow Integration

### Automatic Change Capture

Yanex automatically captures uncommitted changes:

**No clean git state required**:
```bash
# Works even with uncommitted changes
yanex run train.py

# Uncommitted changes saved as artifacts/git_diff.patch
```

**Metadata tracking**:
- `has_uncommitted_changes`: Boolean flag
- `patch_file`: Path to patch file (or null)
- Patches only include tracked files (excludes untracked files)

### When to Commit

**Commit before important runs**:
```bash
# Good practice for production or baseline experiments
git add .
git commit -m "Add baseline model"
yanex run train.py --tag baseline
```

**Experimentation without commits**:
```bash
# Quick iterations - changes automatically captured
yanex run train.py --param lr=0.001
yanex run train.py --param lr=0.01
yanex run train.py --param lr=0.1

# Commit once you find what works
git add .
git commit -m "Optimal learning rate experiments"
```

### Reproducibility

Each experiment captures:
- Git commit hash
- Git branch name
- Uncommitted changes (as patch file)
- Repository state (clean or dirty)

To reproduce an experiment:

```bash
# 1. Check out the commit
git checkout <commit_hash>

# 2. Apply patch if needed
git apply experiments/<exp_id>/artifacts/git_diff.patch

# 3. Re-run with same config
yanex run script.py --config experiments/<exp_id>/params.yaml
```

Or use `--clone-from`:
```bash
yanex run script.py --clone-from abc123
```

## Debugging Failed Experiments

### Check Logs

```bash
# View experiment details
yanex show abc123

# Open directory to inspect logs
yanex open abc123
```

Check `stdout.txt` and `stderr.txt` for error messages.

### Common Issues

**Import errors**:
```bash
# Check environment.txt for installed packages
yanex open abc123
# View environment.txt
```

**Parameter errors**:
```bash
# Verify parameters in params.yaml
yanex show abc123
```

**Script errors**:
```bash
# Check stderr.txt for Python tracebacks
yanex open abc123
```

### Re-run with Debugging

Clone the failed experiment and add debug flags:

```bash
yanex run script.py --clone-from abc123 --name "debug-run"
```

### Failed Sweep Recovery

If a sweep partially fails:

```bash
# List failed experiments
yanex list -s failed -t hyperparameter-sweep

# Inspect specific failures
yanex show <failed_exp_id>

# Re-run failed configurations
yanex run script.py --param lr=0.001  # The specific config that failed
```

## Performance Tips

### Parallel Execution

Maximize throughput for independent experiments:

```bash
# Use all CPU cores
yanex run train.py --param "lr=logspace(-4, -1, 10)" --parallel 0

# Or specify workers
yanex run train.py --param "lr=logspace(-4, -1, 10)" --parallel 8
```

### Filter Before Analysis

Reduce data loading time:

```python
# Good: Filter upfront
df = yr.compare(tags=["sweep"], status="completed")

# Avoid: Loading all experiments then filtering
df_all = yr.compare()  # Slow for large databases
df_filtered = df_all[df_all[("meta", "tags")].str.contains("sweep")]
```

### Archive Old Experiments

Keep active workspace manageable:

```bash
# Regular archival
yanex archive --started-before "3 months ago"

# Archive by status
yanex archive -s failed -s cancelled
```

Archived experiments don't appear in default listings, improving performance.

### Batch Processing

Use Run API for efficient batch operations:

```python
import yanex

# Create all specs first
experiments = [
    yanex.ExperimentSpec(
        script_path=Path("train.py"),
        config={"lr": lr},
        tags=["batch"]
    )
    for lr in [0.001, 0.01, 0.1]
]

# Execute in parallel
results = yanex.run_multiple(experiments, parallel=3)
```

More efficient than sequential CLI calls.

## Summary

**Organizing**:
- Use tags for categories
- Descriptive names for searchability
- Archive old experiments regularly

**Configuration**:
- YAML configs for defaults
- CLI overrides for variations
- Logical parameter organization

**Workflow**:
- CLI for daily tracking
- Run API for advanced patterns
- Results API for analysis

**Sweeps**:
- Start small, scale up
- Use appropriate sweep types
- Analyze with Results API

**Git**:
- Uncommitted changes captured automatically
- Commit for important milestones
- Full reproducibility with patches

**Debugging**:
- Check logs and metrics
- Use `open` to inspect files
- Clone and re-run with debugging

**Performance**:
- Parallel execution for sweeps
- Filter before analysis
- Archive old experiments

## See Also

- [CLI Commands](cli-commands.md) - Command reference
- [Configuration Guide](configuration.md) - Parameter management
- [Run API](run-api.md) - Advanced patterns
- [Results API](results-api.md) - Analysis and querying
- [Examples](../examples/) - Practical demonstrations
