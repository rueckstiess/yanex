# 02: Configuration Files

## What This Example Demonstrates

- YAML configuration files for default parameters
- Nested parameter structure (model/training/data)
- Accessing nested params with dot notation: `get_param('model.learning_rate')`
- Parameter hierarchy: CLI overrides > config file > code defaults
- Loading configs with `--config`
- **Parameter tracking**: Yanex automatically tracks which parameters your script actually uses

## Files

- `train.py` - Simulated ML training script
- `train-config.yaml` - Configuration file with nested parameters

## How to Run

### With config file (uses defaults from train-config.yaml)
```bash
yanex run train.py --config train-config.yaml
```

### Override specific parameters from CLI
```bash
# Override a single nested parameter
yanex run train.py --config train-config.yaml --param model.learning_rate=0.01

# Override multiple parameters
yanex run train.py --config train-config.yaml \
  -p model.learning_rate=0.005 \
  -p training.epochs=20 \
  -p training.batch_size=64
```

### Without config (uses code defaults)
```bash
yanex run train.py
```

### Standalone mode
```bash
python train.py
```

## Expected Output

```
Starting training...
Dataset: mnist
Training with lr=0.001, epochs=10, batch_size=32
  Epoch 1/10: loss=100.0000
  Epoch 2/10: loss=92.0000
  ...
  Epoch 10/10: loss=43.4712

Training complete!
Final loss: 43.4712
Final accuracy: 0.5000
```

## Configuration Structure

The `train-config.yaml` file organizes parameters into logical groups:

```yaml
model:                    # Model architecture parameters
  learning_rate: 0.001    # ✅ Used by script
  hidden_size: 128        # ❌ NOT used - won't be tracked
  dropout: 0.1            # ❌ NOT used - won't be tracked

training:                 # Training process parameters
  epochs: 10              # ✅ Used by script
  batch_size: 32          # ✅ Used by script
  optimizer: adam         # ❌ NOT used - won't be tracked

data:                     # Dataset parameters
  dataset: mnist          # ✅ Used by script
  train_split: 0.8        # ❌ NOT used - won't be tracked
  validation_split: 0.2   # ❌ NOT used - won't be tracked
```

## Parameter Hierarchy

Parameters are resolved in this order (highest priority first):

1. **CLI overrides** (`--param model.learning_rate=0.01`)
2. **Config file** (`train-config.yaml`)
3. **Code defaults** (`get_param(..., default=0.001)`)

Example:
```bash
# train-config.yaml has learning_rate: 0.001
# CLI provides --param model.learning_rate=0.01
# Result: Uses 0.01 (CLI wins)
```

## Accessing Nested Parameters

Use dot notation to access nested parameters:

```python
# Access nested parameter with default
lr = yanex.get_param('model.learning_rate', default=0.001)
epochs = yanex.get_param('training.epochs', default=10)
dataset = yanex.get_param('data.dataset', default='mnist')

# Or get the full dict
params = yanex.get_params()
lr = params['model']['learning_rate']
```

## Parameter Tracking (New!)

**Yanex tracks which parameters your script actually accesses** and only saves those to the experiment record. This has several benefits:

1. **Clearer experiment records**: Only parameters that affected the experiment are saved
2. **Reduced storage**: No bloat from unused shared config parameters
3. **Better introspection**: Easy to see which parameters actually mattered

**In this example:**

The script accesses only 4 parameters:
- `model.learning_rate` ✅
- `training.epochs` ✅
- `training.batch_size` ✅
- `data.dataset` ✅

The following parameters are defined in `train-config.yaml` but **NOT accessed** by the script, so they won't be tracked:
- `model.hidden_size` ❌ (defined but never read)
- `model.dropout` ❌ (defined but never read)
- `training.optimizer` ❌ (defined but never read)
- `data.train_split` ❌ (defined but never read)
- `data.validation_split` ❌ (defined but never read)

After running the experiment, check `~/.yanex/experiments/<id>/params.yaml` and you'll see it contains only the 4 parameters that were actually accessed!

## What to Look For

After running:
- **View the config used**: `yanex show <id>` shows the tracked parameters (only the ones your script accessed)
- **Check stored params**: `~/.yanex/experiments/<id>/params.yaml` contains only the 4 parameters that were accessed (model.learning_rate, training.epochs, training.batch_size, data.dataset)
- **Notice what's missing**: Parameters like `model.hidden_size` and `training.optimizer` won't be in `params.yaml` because they weren't accessed
- **Compare experiments**: `yanex compare` shows parameter differences (only tracked parameters)

## Key Concepts

- **Config files avoid repetition**: Set defaults once, override when needed
- **Organized parameters**: Nested structure keeps configs readable
- **CLI flexibility**: Quick parameter tweaks without editing files
- **Reproducibility**: Only accessed parameters are saved with each experiment
- **Automatic tracking**: No code changes needed - yanex tracks parameter access automatically
- **Shared configs work great**: Define many parameters in config files, only the ones you use are tracked

## Next Steps

- Try different learning rates and compare results: `yanex compare`
- Create environment-specific configs (dev.yaml, prod.yaml)
- See example 03 to learn about logging artifacts
