# 09: Multi-Step Metrics Logging

## What This Example Demonstrates

- Using the `step` parameter to track metrics across iterations
- Auto-incrementing steps when `step` is not specified
- Merging metrics for the same step (incremental metric building)
- Logging validation metrics only every N steps
- Multiple calls to `log_metrics()` during execution
- Use case: ML training with epoch-wise metrics

## Files

- `train_model.py` - Model training simulation with per-epoch logging

## Why Step-Based Logging?

Many experiments have iterative processes where you want to track metrics at each step:

- **ML Training**: Log loss/accuracy per epoch
- **Optimization**: Track objective function per iteration
- **Simulations**: Record state at each time step
- **Progress Tracking**: Monitor convergence and detect issues early

## How to Run

### Basic training run
```bash
yanex run train_model.py
```

### With parameters
```bash
# More epochs
yanex run train_model.py -p epochs=20

# Different validation frequency
yanex run train_model.py -p val_frequency=5

# Multiple parameters
yanex run train_model.py \
  -p epochs=15 \
  -p learning_rate=0.005 \
  -p val_frequency=3
```

### Compare different runs
```bash
# Run with different learning rates
yanex run train_model.py -p learning_rate=0.001 --name "lr-0.001"
yanex run train_model.py -p learning_rate=0.01 --name "lr-0.01"
yanex run train_model.py -p learning_rate=0.1 --name "lr-0.1"

# Compare results
yanex compare
```

## Expected Output

```
Training model for 10 epochs (lr=0.01, batch_size=32)...
Validation every 2 epoch(s)

Epoch 1/10:
  Train Loss: 1.8234, Train Acc: 52.34%

Epoch 2/10:
  Train Loss: 1.6012, Train Acc: 58.23%
  Val Loss:   1.7234, Val Acc:   54.56%

Epoch 3/10:
  Train Loss: 1.3845, Train Acc: 64.78%

Epoch 4/10:
  Train Loss: 1.2123, Train Acc: 72.45%
  Val Loss:   1.3234, Val Acc:   68.23%

...

Training complete!
Final train accuracy: 92.45%
✓ Experiment completed successfully: abc12345
```

## The `step` Parameter

### Basic Usage

The `log_metrics()` function accepts an optional `step` parameter:

```python
yanex.log_metrics(metrics, step=epoch)
```

**Two modes:**

1. **With `step` parameter**: Metrics are tagged with that specific step number
2. **Without `step` parameter**: Step is auto-incremented (0, 1, 2, ...)

### Auto-Incrementing Steps

When you don't specify `step`, yanex automatically increments:

```python
yanex.log_metrics({'loss': 1.5})  # step=0
yanex.log_metrics({'loss': 1.2})  # step=1
yanex.log_metrics({'loss': 0.9})  # step=2
```

### Explicit Steps

Use explicit steps to control numbering:

```python
for epoch in range(1, epochs + 1):
    train_loss = train()

    # Explicitly set step=epoch (starts at 1, not 0)
    yanex.log_metrics({'train_loss': train_loss}, step=epoch)
```

### Incremental Metric Building

**Key feature**: Metrics for the same step are **merged**, not overwritten.

This allows you to build metrics incrementally:

```python
for step in range(max_steps):
    # Always log training loss
    yanex.log_metrics({
        'train_loss': loss,
    }, step=step)

    # Every 10 steps, also log validation metrics
    if step % 10 == 0:
        val_acc = validate()
        # Merges with existing metrics for this step
        yanex.log_metrics({'val_accuracy': val_acc}, step=step)
```

**Result**: Step 0, 10, 20, ... will have both `train_loss` AND `val_accuracy`. Other steps only have `train_loss`.

### Example from train_model.py

```python
for epoch in range(1, epochs + 1):
    # Log training metrics
    yanex.log_metrics({
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
    }, step=epoch)

    # Every N epochs, add validation metrics
    if epoch % val_frequency == 0:
        yanex.log_metrics({
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, step=epoch)  # Merged with training metrics for this epoch
```

## Important: Don't Include `step` in Metrics Dict

**Wrong:**
```python
yanex.log_metrics({
    'step': epoch,      # ✗ Don't do this
    'train_loss': loss,
})
```

**Right:**
```python
yanex.log_metrics({'train_loss': loss}, step=epoch)  # ✓ Use step parameter
```

The `step` is a separate parameter, not a metric field.

## Metrics Storage

All metrics are stored in:
```
~/.yanex/experiments/<experiment_id>/metrics.json
```

Format with explicit steps:
```json
[
  {
    "step": 1,
    "train_loss": 1.8234,
    "train_accuracy": 0.5234,
    "learning_rate": 0.01,
    "timestamp": "2024-01-15T10:30:01.234Z"
  },
  {
    "step": 2,
    "train_loss": 1.6012,
    "train_accuracy": 0.5823,
    "learning_rate": 0.01,
    "val_loss": 1.7234,
    "val_accuracy": 0.5456,
    "timestamp": "2024-01-15T10:30:02.456Z"
  }
]
```

Notice step 2 has both training AND validation metrics (merged).

## Viewing Metrics

### Quick View with `yanex show`

The easiest way to view metrics is with `yanex show`:

```bash
yanex show <experiment_id>
```

This displays:
- **Last 10 steps** of metrics in a table format
- All metric values for each step
- Timestamps for each step
- Makes it easy to see training progression at a glance

**Example output:**
```
╭────────────────────── Results (showing last 10 of 11) ───────────────────────╮
│   Step   Timestamp    train_loss   train_acc   val_loss   val_acc            │
│  ───────────────────────────────────────────────────────────────────────     │
│      2   2025-...     1.73         0.62        1.89       0.53               │
│      3   2025-...     1.58         0.68        -          -                  │
│      4   2025-...     1.50         0.75        1.65       0.69               │
│      5   2025-...     1.25         0.81        -          -                  │
│     ...                                                                      │
│     11   2025-...     -            -           -          -                  │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Notice:
- Steps with validation (2, 4, 6, 8, 10) show both training and validation metrics
- Steps without validation (3, 5, 7, 9) show only training metrics
- This clearly demonstrates the incremental metric building pattern

### View All Metrics

For complete metrics beyond the last 10 steps:

```bash
# View full metrics file
cat ~/.yanex/experiments/<experiment_id>/metrics.json | python3 -m json.tool

# Extract specific metrics with jq
cat ~/.yanex/experiments/<id>/metrics.json | jq '.[] | {step, train_loss, val_loss}'
```

## Common Patterns

### 1. Epoch-Based Training (Explicit Steps)

```python
for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch()

    yanex.log_metrics({
        'train_loss': train_loss,
    }, step=epoch)
```

### 2. Iteration-Based with Periodic Validation

```python
for iteration in range(max_iterations):
    # Always log training metrics
    yanex.log_metrics({
        'train_loss': loss,
    }, step=iteration)

    # Validate every 100 iterations
    if iteration % 100 == 0:
        val_loss = validate()
        yanex.log_metrics({
            'val_loss': val_loss,
        }, step=iteration)
```

### 3. Time-Based (Auto-Increment)

```python
t = 0
while t < simulation_time:
    state = simulate_step(state, dt)
    t += dt

    # Auto-increment (step 0, 1, 2, ...)
    yanex.log_metrics({
        'time': t,
        'energy': compute_energy(state),
    })
```

### 4. Checkpoints with Explicit Steps

```python
checkpoints = [10, 20, 50, 100, 200]
for checkpoint in checkpoints:
    metrics = evaluate_model(checkpoint)

    # Use checkpoint number as step
    yanex.log_metrics({
        'accuracy': metrics['accuracy'],
    }, step=checkpoint)
```

### 5. Mixed: Final Summary Without Step

```python
# During training (with steps)
for epoch in range(epochs):
    yanex.log_metrics({'loss': loss}, step=epoch)

# After training (auto-increment continues)
yanex.log_metrics({
    'final_accuracy': final_acc,
    'total_time': total_time,
})  # Gets next auto-incremented step
```

## Best Practices

### 1. Be Consistent with Steps

```python
# Good - consistent explicit steps
for epoch in range(epochs):
    yanex.log_metrics({'loss': loss}, step=epoch)

# Confusing - mixing explicit and auto-increment
yanex.log_metrics({'loss': 1.5}, step=0)
yanex.log_metrics({'loss': 1.2})  # Auto-increment might conflict
```

### 2. Use Explicit Steps for Meaningful Numbers

```python
# Good - epoch numbers are meaningful
yanex.log_metrics({'loss': loss}, step=epoch)

# Less clear - what does step 0, 1, 2 mean?
yanex.log_metrics({'loss': loss})  # Auto-increment
```

### 3. Don't Over-Log

Balance detail vs performance:

```python
# Too frequent
for i in range(10000):
    yanex.log_metrics({'loss': loss}, step=i)  # 10,000 log calls!

# Reasonable
for i in range(10000):
    if i % 100 == 0:
        yanex.log_metrics({'loss': loss}, step=i)  # 100 log calls
```

### 4. Leverage Incremental Metrics

```python
# Efficient - log expensive metrics less frequently
for step in range(1000):
    # Always log cheap metrics
    yanex.log_metrics({'train_loss': loss}, step=step)

    # Expensive validation only every 50 steps
    if step % 50 == 0:
        yanex.log_metrics({'val_accuracy': validate()}, step=step)
```


## Troubleshooting

### Metrics not showing up
```python
# Make sure you're calling log_metrics
yanex.log_metrics({'loss': loss}, step=epoch)  # ✓

# Not just printing
print(f"Loss: {loss}")  # ✗
```

### Wrong step numbers
```python
# Check: are you mixing explicit and auto-increment?
yanex.log_metrics({'loss': 1.0}, step=0)
yanex.log_metrics({'loss': 0.9})  # Auto-increment might be 1, not 0

# Fix: be consistent
yanex.log_metrics({'loss': 1.0}, step=0)
yanex.log_metrics({'loss': 0.9}, step=1)
```

### Metrics file too large
```python
# Log less frequently
if step % 10 == 0:
    yanex.log_metrics({...}, step=step)
```

### Duplicate metrics not merging
```python
# This overwrites timestamp, doesn't truly merge
yanex.log_metrics({'loss': 1.0}, step=5)
yanex.log_metrics({'loss': 0.9}, step=5)  # Only last loss value kept

# This correctly merges different metrics
yanex.log_metrics({'train_loss': 1.0}, step=5)
yanex.log_metrics({'val_loss': 0.9}, step=5)  # Both metrics present
```

## Key Concepts

- **`step` parameter**: Optional parameter to `log_metrics()` for explicit step tracking
- **Auto-increment**: Without `step`, yanex auto-increments (0, 1, 2, ...)
- **Metric merging**: Multiple `log_metrics()` calls with same `step` are merged
- **Incremental building**: Log expensive metrics less frequently using merging
- **Separate parameter**: `step` is a parameter, not a key in the metrics dict
- **Timestamps**: Automatically added to each log entry

## Next Steps

- Try logging metrics with different validation frequencies
- Extract and plot training curves from metrics.json
- Implement early stopping based on validation metrics
- Use metric merging for efficient periodic validation
