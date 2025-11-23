# 10: Dependencies and Multi-Stage Pipelines

## What This Example Demonstrates

- Creating multi-stage ML pipelines with experiment dependencies
- Linear pipelines: preprocessing → training → evaluation
- Dependency sweeps: running experiments with multiple dependencies
- Cartesian products: dependencies × parameters
- Incremental pipeline staging: stage entire workflows step-by-step
- Using `-D` / `--depends-on` to link experiments

## Files

- `prepare_data.py` - Data preprocessing (simulated)
- `train_model.py` - Model training (depends on preprocessing)
- `evaluate_model.py` - Model evaluation (depends on training)

## Why Use Dependencies?

Dependencies let you build **multi-stage workflows** where experiments build on previous results:

- **ML Pipelines**: Preprocess → Train → Evaluate
- **Data Pipelines**: Extract → Transform → Load → Analyze
- **Reproducibility**: Track complete lineage of results
- **Efficiency**: Reuse preprocessing across multiple training runs
- **Artifact Sharing**: Load artifacts from dependencies automatically

## How Dependencies Work

When an experiment depends on another:
1. **Validation**: Yanex ensures the dependency completed successfully
2. **Tracking**: Full dependency lineage is recorded
3. **Artifact Access**: Child can load artifacts from parent via `yanex.load_artifact()`
4. **Metadata**: Dependency relationships visible in `yanex show`

## Basic Usage: Linear Pipeline

### Step 1: Run preprocessing

```bash
# No dependencies - this is the first step
yanex run prepare_data.py -p dataset=mnist -p samples=1000
# Output: ✓ Experiment completed successfully: abc12345
```

### Step 2: Train model using preprocessed data

```bash
# Depends on preprocessing (use the experiment ID from step 1)
yanex run train_model.py -D abc12345 -p learning_rate=0.01
# Output: ✓ Experiment completed successfully: def67890
```

### Step 3: Evaluate the trained model

```bash
# Depends on training (use the experiment ID from step 2)
yanex run evaluate_model.py -D def67890
# Output: ✓ Experiment completed successfully: ghi11111
```

**Result**: Complete pipeline tracked with full lineage!

## Dependency Sweeps: Multiple Parents

Run the same experiment with **multiple dependencies**:

```bash
# Step 1: Create multiple preprocessing runs
yanex run prepare_data.py -p "dataset=mnist,cifar10"
# Creates 2 experiments: abc12345 (mnist), def67890 (cifar10)

# Step 2: Train model on BOTH datasets (dependency sweep)
yanex run train_model.py -D abc12345,def67890 -p learning_rate=0.01
# ✓ Sweep detected: running 2 experiments
# Creates:
#   - Experiment 1: depends on abc12345 (mnist)
#   - Experiment 2: depends on def67890 (cifar10)
```

**Use cases**:
- Compare preprocessing methods
- Test model across multiple datasets
- Ensemble predictions from multiple models

## Cartesian Products: Dependencies × Parameters

Combine dependency sweeps with parameter sweeps for **maximum flexibility**:

```bash
# Step 1: Create 2 preprocessing variants
yanex run prepare_data.py -p "dataset=mnist,cifar10"
# Creates: abc12345 (mnist), def67890 (cifar10)

# Step 2: Train with 2 deps × 3 learning rates = 6 experiments
yanex run train_model.py \
  -D abc12345,def67890 \
  -p "learning_rate=0.001,0.01,0.1"
# ✓ Sweep detected: running 6 experiments
# Creates all combinations:
#   mnist + lr=0.001, mnist + lr=0.01, mnist + lr=0.1
#   cifar10 + lr=0.001, cifar10 + lr=0.01, cifar10 + lr=0.1
```

## Incremental Pipeline Staging

For complex workflows, **stage each step** before execution:

```bash
# Step 1: Stage preprocessing (2 datasets)
yanex run prepare_data.py \
  -p "dataset=mnist,cifar10" \
  --stage \
  --name preprocessing
# ✓ Staged 2 sweep experiments

# Step 2: Get IDs of staged preprocessing experiments
yanex list --status staged
# Shows: abc12345 (mnist), def67890 (cifar10)

# Step 3: Stage training depending on STAGED preprocessing
# This is the "incremental staging" pattern
yanex run train_model.py \
  -D abc12345,def67890 \
  -p "learning_rate=0.001,0.01,0.1" \
  --stage \
  --name training
# ✓ Sweep detected: expanding into 6 experiments
# ✓ Staged 6 sweep experiments (2 deps × 3 lrs)

# Step 4: Execute ALL staged experiments in parallel
yanex run --staged --parallel 8
# Runs preprocessing first, then training (respects dependencies)
```

**Benefits**:
- Review entire pipeline before execution
- Modify staged experiments before running
- Run complex workflows with one command
- Automatic dependency resolution

## Parallel Execution

Dependency sweeps support parallel execution:

```bash
# Create 4 preprocessing variants
yanex run prepare_data.py -p "samples=100,500,1000,5000"

# Train on all 4 in parallel with 4 workers
yanex run train_model.py -D <ids> --parallel 4
# Each training run executes independently
```

## Using Short IDs

Yanex resolves ID prefixes automatically:

```bash
# Full ID
yanex run train_model.py -D abc12345

# Short ID (first 4+ characters, if unique)
yanex run train_model.py -D abc1

# Multiple short IDs
yanex run train_model.py -D abc1,def6,ghi1
```

## Accessing Dependency Artifacts

Child experiments can **automatically load** artifacts from dependencies:

```python
# train_model.py
import yanex

# Get dependencies (returns list of Experiment objects)
deps = yanex.get_dependencies()
prep = deps[0]

# Load artifact from dependency
data = prep.load_artifact("processed_data.pkl")

# Or use yanex.load_artifact() which searches dependencies automatically
data = yanex.load_artifact("processed_data.pkl")
```

## Viewing Pipeline Lineage

```bash
# Show experiment with full dependency information
yanex show ghi11111

# Output includes:
# Dependencies:
#   - def67890: training run (completed)
#   - abc12345: preprocessing (completed) [transitive]
```

## Example Workflows

### Simple: Preprocessing + Training

```bash
# 1. Preprocess
prep_id=$(yanex run prepare_data.py -p dataset=mnist | grep -o '[a-f0-9]\{8\}')

# 2. Train
train_id=$(yanex run train_model.py -D $prep_id -p lr=0.01 | grep -o '[a-f0-9]\{8\}')

# 3. Evaluate
yanex run evaluate_model.py -D $train_id
```

### Hyperparameter Search

```bash
# 1. Preprocess once
yanex run prepare_data.py -p dataset=mnist -p samples=5000
# Output: abc12345

# 2. Grid search over learning rates (3 experiments, all use same preprocessing)
yanex run train_model.py \
  -D abc12345 \
  -p "learning_rate=0.001,0.01,0.1" \
  --parallel 3

# 3. Compare results
yanex compare --tag training
```

### Multi-Dataset Comparison

```bash
# 1. Prepare multiple datasets
yanex run prepare_data.py -p "dataset=mnist,cifar10,fashion"
# Creates 3 experiments

# 2. Train same model on each (dependency sweep)
yanex run train_model.py -D <ids-from-step-1> -p lr=0.01
# Creates 3 training experiments

# 3. Evaluate all models
yanex run evaluate_model.py -D <ids-from-step-2>
# Creates 3 evaluation experiments
```

### Full Pipeline with Staging

```bash
# Stage preprocessing
yanex run prepare_data.py --stage -p "dataset=mnist,cifar10"
# ✓ Staged 2 sweep experiments: prep1, prep2

# Stage training (depends on staged preprocessing)
yanex run train_model.py --stage -D prep1,prep2 -p "lr=0.01,0.1"
# ✓ Staged 4 sweep experiments (2 deps × 2 lrs): train1-4

# Stage evaluation (depends on staged training)
yanex run evaluate_model.py --stage -D train1,train2,train3,train4
# ✓ Staged 4 experiments: eval1-4

# Execute entire pipeline
yanex run --staged --parallel 4
# Runs 10 experiments total (2 + 4 + 4)
# Dependencies executed in correct order automatically
```

## Expected Output

### Linear Pipeline
```
$ yanex run prepare_data.py -p dataset=mnist
📊 Preprocessing dataset: mnist (1000 samples)
  Features extracted: 784
  Train/test split: 800/200
✓ Experiment completed successfully: abc12345

$ yanex run train_model.py -D abc12345 -p learning_rate=0.01
📦 Loading preprocessed data from dependency abc12345
🤖 Training model with lr=0.01
  Epoch 1/10: loss=0.5234
  Epoch 5/10: loss=0.2341
  Epoch 10/10: loss=0.1123
✓ Experiment completed successfully: def67890

$ yanex run evaluate_model.py -D def67890
📦 Loading trained model from dependency def67890
📈 Evaluating model...
  Test accuracy: 92.5%
  Test loss: 0.1534
✓ Experiment completed successfully: ghi11111
```

### Dependency Sweep
```
$ yanex run train_model.py -D abc12345,def67890 -p learning_rate=0.01
✓ Sweep detected: running 2 experiments

Experiment 1/2 (dependency: abc12345)...
✓ Experiment completed successfully: train001

Experiment 2/2 (dependency: def67890)...
✓ Experiment completed successfully: train002

✓ Sweep execution completed
  Total: 2
  Completed: 2
```

### Cartesian Product
```
$ yanex run train_model.py -D abc1,def6 -p "learning_rate=0.001,0.01,0.1"
✓ Sweep detected: running 6 experiments
Running 6 experiments with 4 parallel workers...
✓ Completed 6/6 experiments
```

## Common Patterns

### Pattern 1: Reuse Preprocessing
```bash
# Preprocess once
prep=$(yanex run prepare_data.py ...)

# Run multiple training experiments with same data
yanex run train_model.py -D $prep -p "lr=0.001,0.01,0.1"
```

### Pattern 2: Compare Preprocessing Methods
```bash
# Try different preprocessing
yanex run prepare_data.py -p "method=standard,minmax,robust"
# Get IDs: prep1, prep2, prep3

# Train same model on each
yanex run train_model.py -D prep1,prep2,prep3 -p lr=0.01
```

### Pattern 3: Multi-Stage Grid Search
```bash
# Preprocessing grid
yanex run prepare_data.py -p "samples=1000,5000" --stage

# Training grid (depends on preprocessing)
yanex run train_model.py -D <prep-ids> \
  -p "lr=0.001,0.01,0.1" \
  -p "batch_size=32,64,128" \
  --stage

# Execute all
yanex run --staged --parallel 8
```

## Key Concepts

- **`-D` / `--depends-on`**: Link experiment to dependencies
- **Dependency sweeps**: Multiple dependencies create multiple experiments
- **Cartesian products**: Dependencies × Parameters = all combinations
- **Incremental staging**: Stage each pipeline step before execution
- **Artifact loading**: Automatic access to dependency artifacts
- **Full lineage**: Complete pipeline history tracked automatically

## Best Practices

### Use Descriptive Names
```bash
yanex run prepare_data.py --name "mnist-preprocessing" ...
yanex run train_model.py -D <id> --name "resnet-training" ...
```

### Tag Pipeline Stages
```bash
yanex run prepare_data.py --tag preprocessing ...
yanex run train_model.py -D <id> --tag training ...
yanex run evaluate_model.py -D <id> --tag evaluation ...
```

### Validate Dependencies Before Scaling
```bash
# Test pipeline with small data first
yanex run prepare_data.py -p samples=100
yanex run train_model.py -D <id> -p epochs=1
yanex run evaluate_model.py -D <id>

# Then scale up
yanex run prepare_data.py -p "samples=1000,5000,10000" --stage
```

### Use Parallel Execution for Independent Branches
```bash
# These can run in parallel (different preprocessing)
yanex run train_model.py -D prep1,prep2,prep3 --parallel 3
```

## Troubleshooting

### "Dependency not found"
Ensure the experiment ID exists and hasn't been deleted:
```bash
yanex list  # Check if dependency exists
yanex show <dep-id>  # Verify dependency status
```

### "Dependency has invalid status"
Dependencies must be `completed` before use:
```bash
yanex show <dep-id>  # Check status
# Wait for experiment to complete if still running
```

### Staging with staged dependencies
Use `--stage` flag - yanex allows depending on staged experiments when creating staged experiments:
```bash
yanex run prepare_data.py --stage  # Creates staged experiment
yanex run train_model.py -D <staged-id> --stage  # OK! Incremental staging
```

## Next Steps

- Run a full pipeline and examine with `yanex show <final-id>`
- Try `yanex compare` to compare different dependency branches
- Explore `yanex.get_dependencies()` API for programmatic access
- See example 07 for parameter sweeps and example 09 for staged execution
