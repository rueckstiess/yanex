# Experiment Dependencies

Yanex supports experiment dependencies, enabling you to build multi-stage pipelines where later experiments depend on outputs from earlier ones.

## Quick Reference

```bash
# Run experiment with dependency (auto-named as "dep1")
yanex run train.py -D abc12345

# Run with named dependency slot
yanex run train.py -D data=abc12345

# Run with multiple named dependencies
yanex run evaluate.py -D data=prep_id -D model=train_id

# Sweep over dependencies (comma-separated)
yanex run evaluate.py -D model=train1,train2,train3

# Check dependencies in script
import yanex

# Assert required dependency exists
yanex.assert_dependency("prepare_data.py")

# Get specific dependency by slot name
data_exp = yanex.get_dependency("data")
model = data_exp.load_artifact("model.pkl")

# Get all direct dependencies (returns dict[slot, Experiment])
deps = yanex.get_dependencies()
model = deps["model"].load_artifact("model.pkl")
```

---

## Overview

Dependencies allow you to:
- **Build pipelines**: Data preprocessing → Training → Evaluation
- **Reuse results**: Load artifacts and metrics from previous experiments
- **Track lineage**: See the full dependency chain of any experiment
- **Ensure correctness**: Assert that required dependencies exist before running

## Basic Usage

### Creating Dependencies

Use the `-D` / `--depends-on` flag when running experiments:

```bash
# 1. Run preprocessing
yanex run prepare_data.py --name "data-prep-v1"
# Returns: abc12345

# 2. Run training with dependency on preprocessing (auto-named as "dep1")
yanex run train.py -D abc12345 --name "training-v1"
# Returns: def67890

# 3. Run evaluation with dependency on training
yanex run evaluate.py -D def67890 --name "eval-v1"
```

### Named Dependency Slots

Use the `slot=id` syntax to give dependencies meaningful names:

```bash
# Name the dependency slot "data"
yanex run train.py -D data=abc12345 --name "training-v1"

# Multiple named dependencies
yanex run evaluate.py -D model=train_id -D data=prep_id --name "eval-v1"
```

In your script, access by slot name:

```python
import yanex

# Get specific dependency by slot name
data_exp = yanex.get_dependency("data")
model_exp = yanex.get_dependency("model")

# Or get all as dict
deps = yanex.get_dependencies()  # Returns {"data": Experiment, "model": Experiment}
```

### Multiple Dependencies

Experiments can depend on multiple parent experiments:

```bash
# Train multiple models
yanex run train.py -p model=resnet18 --name "resnet-v1"    # Returns: aaa111
yanex run train.py -p model=resnet50 --name "resnet-v2"    # Returns: bbb222
yanex run train.py -p model=vgg16 --name "vgg-v1"          # Returns: ccc333

# Ensemble evaluation depends on all three (as separate slot-named dependencies)
yanex run ensemble.py -D model1=aaa111 -D model2=bbb222 -D model3=ccc333 --name "ensemble-eval"
```

### Dependency Sweeps vs Combined Dependencies

There are two ways to specify multiple dependencies, with different meanings:

| Syntax | Slot Name | Result |
|--------|-----------|--------|
| `-D exp1` | `dep1` (auto) | 1 experiment with 1 dependency |
| `-D train=exp1` | `train` | 1 experiment with 1 named dependency |
| `-D exp1,exp2` | `dep1` (sweep) | 2 experiments, each with 1 dependency |
| `-D train=exp1,exp2` | `train` (sweep) | 2 experiments, each with 1 named dependency |
| `-D exp1 -D exp2` | `dep1`, `dep2` | 1 experiment with both dependencies |
| `-D data=exp1 -D model=exp2` | `data`, `model` | 1 experiment with named dependencies |
| `-D data=exp1,exp2 -D model=exp3,exp4` | `data`, `model` (cross-product) | 4 experiments |

**Comma-separated (sweep):** Creates multiple experiments, each using a different dependency:

```bash
# Run experiment once for each model (3 separate runs)
yanex run evaluate.py -D model=aaa111,bbb222,ccc333
# Creates 3 experiments:
#   model-aaa111: depends on aaa111 (slot "model")
#   model-bbb222: depends on bbb222 (slot "model")
#   model-ccc333: depends on ccc333 (slot "model")
```

**Multiple -D flags (combined):** Creates one experiment that depends on all:

```bash
# Run ensemble that needs ALL three models
yanex run ensemble.py -D model1=aaa111 -D model2=bbb222 -D model3=ccc333
# Creates 1 experiment with 3 named dependencies
```

**Cross-product:** Combine both for more complex scenarios:

```bash
# Two data preprocessing runs
yanex run preprocess.py --name "data-v1"  # Returns: data1
yanex run preprocess.py --name "data-v2"  # Returns: data2

# Two model architectures
yanex run train.py --name "resnet"  # Returns: model1
yanex run train.py --name "vgg"     # Returns: model2

# Evaluate all combinations: 2 datasets × 2 models = 4 experiments
yanex run evaluate.py -D data=data1,data2 -D model=model1,model2
# Creates 4 experiments with names including dependency info:
#   data-data1-model-model1
#   data-data1-model-model2
#   data-data2-model-model1
#   data-data2-model-model2
```

## Accessing Dependencies in Scripts

### API Summary

| Function | Parameters | Returns |
|----------|------------|---------|
| `yanex.get_dependency(slot)` | `slot: str` | `Experiment \| None` |
| `yanex.get_dependencies()` | `transitive: bool = False` | `dict[str, Experiment]` (direct) or `list[Experiment]` (transitive) |
| `yanex.assert_dependency(script, slot=None)` | `script: str`, `slot: str \| None` | `None` (raises on failure) |

### Get Dependencies by Slot Name

Use `yanex.get_dependency(slot)` to access a specific dependency by its slot name:

```python
# train.py
import yanex

# Get specific dependency by slot name
data_exp = yanex.get_dependency("data")

if data_exp:
    print(f"Using data from: {data_exp.id}")
    print(f"Data config: {data_exp.get_params()}")

    # Load artifact directly from dependency
    data = data_exp.load_artifact("processed_data.pkl")
else:
    print("No 'data' dependency - using default data")
```

### Get All Dependencies

Use `yanex.get_dependencies()` to access all dependency experiments:

```python
# train.py
import yanex

# Get all direct dependencies (returns dict: slot name -> Experiment)
deps = yanex.get_dependencies()

if deps:
    for slot, exp in deps.items():
        print(f"Slot '{slot}': experiment {exp.id}")
        print(f"  Config: {exp.get_params()}")
else:
    print("No dependencies - using defaults")

# Access specific dependency from dict
if "data" in deps:
    data = deps["data"].load_artifact("processed_data.pkl")
```

### Assert Required Dependencies

Use `yanex.assert_dependency()` to validate dependencies at script start:

```python
# train.py
import yanex

# Fail fast if dependency from prepare_data.py is missing
yanex.assert_dependency("prepare_data.py")

# Or check a specific slot
yanex.assert_dependency("prepare_data.py", slot="data")

# If we get here, dependency exists
data_exp = yanex.get_dependency("data") or list(yanex.get_dependencies().values())[0]
data = data_exp.load_artifact("processed_data.pkl")

# Rest of training code...
```

**Behavior:**
- **With dependency**: Script continues normally
- **Without dependency**: Prints error message and fails experiment cleanly
- **Standalone mode**: No-op (allows script to run without yanex tracking)

### Transitive Dependencies

Access the full dependency chain using `transitive=True`:

```python
import yanex

# Get only direct dependencies (returns dict[slot, Experiment])
direct = yanex.get_dependencies()
print(f"Direct deps: {list(direct.keys())}")  # e.g., ["data", "model"]

# Get all dependencies recursively (returns list[Experiment])
all_deps = yanex.get_dependencies(transitive=True)
print(f"All deps: {[d.id for d in all_deps]}")

# Example pipeline: preprocess → train → evaluate
# evaluate.py depends on train.py (which depends on preprocess.py)
# all_deps will include both train.py and preprocess.py experiments
```

**Note:** When `transitive=True`, the return type changes from `dict[str, Experiment]` to `list[Experiment]` since transitive dependencies don't have slot names.

## Loading Artifacts from Dependencies

### Automatic Search

`load_artifact()` automatically searches dependencies if artifact not found locally:

```python
import yanex

# Searches: current experiment → dependencies → transitive dependencies
model = yanex.load_artifact("model.pkl")
```

### Explicit Dependency

Load from a specific dependency using the `Experiment.load_artifact()` method:

```python
import yanex

# Get dependencies as dict (slot name -> Experiment)
deps = yanex.get_dependencies()

# Load from a specific slot
if "model" in deps:
    model = deps["model"].load_artifact("model.pkl")

# Or get a single dependency by slot name
data_exp = yanex.get_dependency("data")
if data_exp:
    preprocessed_data = data_exp.load_artifact("data.pkl")
```

### Access All Artifacts

List artifacts across dependencies:

```python
import yanex

deps = yanex.get_dependencies(transitive=True)

# Collect all model checkpoints
all_models = []
for dep in deps:
    for artifact_name in dep.list_artifacts():
        if "model" in artifact_name:
            model = dep.load_artifact(artifact_name)
            all_models.append((dep.id, artifact_name, model))
```

## Common Patterns

### Data Preprocessing Pipeline

```bash
# Step 1: Prepare data
yanex run prepare_data.py \
  --param dataset=cifar10 \
  --param split=0.8 \
  --name "data-prep"
# Returns: prep_abc

# Step 2: Train model (depends on preprocessing)
yanex run train.py \
  -D prep_abc \
  --param learning_rate=0.01 \
  --name "training"
# Returns: train_def

# Step 3: Evaluate (depends on training, transitively on preprocessing)
yanex run evaluate.py \
  -D train_def \
  --name "evaluation"
```

```python
# train.py
import yanex

yanex.assert_dependency("prepare_data.py")

# Load preprocessed data from dependency (auto-searches dependencies)
data = yanex.load_artifact("processed_data.pkl")

# Or explicitly load from a specific dependency slot
data_exp = yanex.get_dependency("dep1")  # or "data" if using named slots
if data_exp:
    data = data_exp.load_artifact("processed_data.pkl")

# Train model
model = train_model(data)

# Save for next stage
yanex.save_artifact(model, "trained_model.pkl")
yanex.log_metrics({"accuracy": 0.95})
```

### Hyperparameter Sweeps with Evaluation

```bash
# Train multiple models with different hyperparameters
yanex run train.py -p "lr=range(0.001, 0.1, 0.01)" --stage
# Creates 10 staged experiments

# Run all trainings in parallel
yanex run --staged --parallel 4
# Returns IDs: model_1, model_2, ..., model_10

# Evaluate all models
yanex run evaluate.py -D model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9,model_10
```

### Ensemble Models

```python
# ensemble.py
import yanex

# Get all model dependencies (dict: slot name -> Experiment)
deps = yanex.get_dependencies()

# Load all models
models = []
for slot, dep in deps.items():
    model = dep.load_artifact("model.pkl")
    models.append(model)
    print(f"Loaded model from {slot} ({dep.id}): accuracy={dep.get_metric('accuracy')}")

# Create ensemble
ensemble = create_ensemble(models)

# Evaluate ensemble
ensemble_accuracy = evaluate(ensemble)
yanex.log_metrics({"ensemble_accuracy": ensemble_accuracy})
```

### Comparing Configurations

```python
# compare_configs.py
import yanex

# Get dependencies to compare (dict: slot name -> Experiment)
deps = yanex.get_dependencies()

# Compare parameters and results
for slot, dep in deps.items():
    params = dep.get_params()
    metrics = dep.get_metrics()

    print(f"Experiment {dep.id} (slot: {slot}):")
    print(f"  Config: {params}")
    print(f"  Results: {metrics}")
    print()

# Find best configuration
best = max(deps.values(), key=lambda d: d.get_metric('accuracy'))
print(f"Best experiment: {best.id} (accuracy={best.get_metric('accuracy')})")
```

## Results API

Query and analyze dependencies programmatically:

```python
import yanex.results as yr

# Get experiment
exp = yr.get_experiment("abc12345")

# Check for dependencies
if exp.has_dependencies:
    # Get dependencies dict (slot -> experiment_id)
    print(f"Dependencies: {exp.dependencies}")  # {"data": "prep123", "model": "train456"}

    # Get specific dependency by slot
    data_dep = exp.get_dependency("data")
    if data_dep:
        print(f"Data from: {data_dep.id} ({data_dep.name})")

    # Get all as Experiment objects (dict)
    deps = exp.get_dependencies()
    for slot, dep in deps.items():
        print(f"  {slot}: {dep.id} - {dep.name} ({dep.status})")

# Get transitive dependencies (returns list)
all_deps = exp.get_dependencies(transitive=True)
print(f"Full pipeline has {len(all_deps)} experiments")
```

## Best Practices

### ✅ Do

- **Assert dependencies early**: Use `yanex.assert_dependency()` at the start of scripts
- **Handle standalone mode**: Check if `deps` is empty before accessing
- **Use meaningful names**: Name experiments clearly to identify pipeline stages
- **Tag pipeline stages**: Use tags like `preprocessing`, `training`, `evaluation`
- **Document requirements**: Add comments explaining what dependencies are needed

### ❌ Avoid

- **Hardcoding experiment IDs**: Use `yanex run -D` to pass dependencies dynamically
- **Assuming dependency order**: Dependencies are unordered - use script name to identify
- **Ignoring missing artifacts**: Always check if `load_artifact()` returns None
- **Deep dependency chains**: Keep pipelines shallow (2-4 stages) for clarity

## Troubleshooting

### Dependency Not Found

```
Error: No dependency from 'prepare_data.py' found
```

**Solution**: Ensure you specified the dependency with `-D`:
```bash
yanex run train.py -D <prep-experiment-id>
```

### Artifact Not Found

```python
# Returns None if artifact doesn't exist
data = yanex.load_artifact("data.pkl")
if data is None:
    print("Artifact not found in current experiment or dependencies")
```

**Solution**: Check the artifact name and ensure the dependency experiment saved it.

### Wrong Dependency Script

```
Error: No dependency from 'prepare_data.py' found
Current dependencies are from: other_script.py
```

**Solution**: You specified a dependency from the wrong script. Use the correct experiment ID.

## See Also

- [Run API Reference](run-api.md#dependency-tracking) - `get_dependencies()`, `assert_dependency()`
- [Results API Reference](results-api.md#dependency-properties) - Experiment dependency properties
- [CLI: `yanex run`](commands/run.md#dependencies) - `-D` / `--depends-on` parameter
- [Examples: Dependency Pipeline](../examples/cli/10_dependencies/) - Complete working example
