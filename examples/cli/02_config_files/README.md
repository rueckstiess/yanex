# 02: Configuration Files

## What This Example Demonstrates

- YAML configuration files for default parameters
- Nested parameter structure (model/training/data)
- Accessing nested params with dot notation: `get_param('model.learning_rate')`
- Parameter hierarchy: CLI overrides > config file > code defaults
- Loading configs with `--config`

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
  learning_rate: 0.001
  hidden_size: 128
  dropout: 0.1

training:                 # Training process parameters
  epochs: 10
  batch_size: 32
  optimizer: adam

data:                     # Dataset parameters
  dataset: mnist
  train_split: 0.8
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

## What to Look For

After running:
- **View the config used**: `yanex show <id>` shows all parameters (including overrides)
- **Check stored config**: `~/.yanex/experiments/<id>/config.json` has the resolved parameters (e.g. CLI parameter overrides)
- **Compare experiments**: `yanex compare` shows parameter differences

## Key Concepts

- **Config files avoid repetition**: Set defaults once, override when needed
- **Organized parameters**: Nested structure keeps configs readable
- **CLI flexibility**: Quick parameter tweaks without editing files
- **Reproducibility**: Config is saved with each experiment

## Next Steps

- Try different learning rates and compare results: `yanex compare`
- Create environment-specific configs (dev.yaml, prod.yaml)
- See example 03 to learn about logging artifacts
