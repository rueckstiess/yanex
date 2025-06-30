# Yanex - Yet Another Experiment Tracker

A lightweight, Git-aware experiment tracking system for Python that makes reproducible research effortless.

## Why Yanex?

**Stop losing track of your experiments.** Yanex automatically tracks parameters, results, and code state so you can focus on what matters - your research.

```python
import yanex

# read parameters from config file or CLI arguments
lr = yanex.get_param('lr', default=0.001)
epochs = yanex.get_param('epochs', default=10)

# access nested parameters with dot notation
model_lr = yanex.get_param('model.learning_rate', default=0.001)
optimizer_type = yanex.get_param('model.optimizer.type', default='adam')

# your experiment code
# ...

# log results, artifacts and figures
yanex.log_results({"step": epoch, "loss", loss, "accuracy": accuracy})
yanex.log_artifact("model.pth", model_path)
yanex.log_matplotlib_figure(fig, "loss_curve.png")
```

Run from the command line:

```bash
# Run with yanex CLI for automatic tracking
yanex run train.py --name "my-experiment" --tag testing --param lr=0.001 --param epochs=10
```

That's it. Yanex tracks the experiment, saves the logged results and files, stdout and stderr outptus, Python environment
information, and even the Git state of your code repository. You can then compare results, search experiments, and reproduce them with ease.

## Key Features

- 🔒 **Reproducible**: Automatic Git state tracking ensures every experiment is reproducible
- 📊 **Interactive Comparison**: Compare experiments side-by-side with an interactive table
- ⚙️ **Flexible Parameters**: YAML configs with CLI overrides for easy experimentation and syntax for parameter sweeps
- 📈 **Rich Logging**: Track metrics, artifacts, and figures
- 🔍 **Powerful Search**: Find experiments by status, parameters, tags, or time ranges
- 📦 **Zero Dependencies**: No external services required - works offline

## Quick Start

### Install
```bash
pip install yanex
```

### 1. Run Your First Experiment

```python
# experiment.py
import yanex

params = yanex.get_params()
print(f"Learning rate: {params.get('learning_rate', 0.001)}")

# Simulate training
accuracy = 0.85 + (params.get('learning_rate', 0.001) * 10)

yanex.log_results({
    "accuracy": accuracy,
    "loss": 1 - accuracy
})
```

```bash
# Run with default parameters
yanex run experiment.py

# Override parameters
yanex run experiment.py --param learning_rate=0.01 --param epochs=50

# Add tags for organization
yanex run experiment.py --tag baseline --tag "quick-test"
```

### 2. Compare Results

```bash
# Interactive comparison table
yanex compare

# Compare specific experiments
yanex compare exp1 exp2 exp3

# Filter and compare
yanex compare --status completed --tag baseline
```

### 3. Track Everything

List, search, and manage your experiments:

```bash
# List recent experiments
yanex list

# Find experiments by criteria
yanex list --status completed --tag production
yanex list --started-after "1 week ago"

# Show detailed experiment info
yanex show exp_id

# Archive old experiments
yanex archive --started-before "1 month ago"
```

## Two Ways to Use Yanex

Yanex supports two usage patterns:

### 1. CLI-First (Recommended)
Write scripts that work both standalone and with yanex tracking:

```python
# train.py - Works both ways!
import yanex

params = yanex.get_params()  # Gets parameters or defaults
lr = params.get('learning_rate', 0.001)

# Your training code
accuracy = train_model(lr=lr)

# Logging works in both contexts
yanex.log_results({"accuracy": accuracy})
```

```bash
# Run standalone (no tracking)
python train.py

# Run with yanex (full tracking)
yanex run train.py --param learning_rate=0.01
```

### 2. Explicit Experiment Creation (Advanced)
For Jupyter notebook usage, or when you need fine control:

```python
import yanex
from pathlib import Path

with yanex.create_experiment(
    script_path=Path(__file__),
    name="my-experiment",
    config={"learning_rate": 0.01}
) as exp:
    # Your code here
    exp.log_results({"accuracy": 0.95})
```

> **Note:** Don't mix both patterns! Use CLI-first for most cases, explicit creation for advanced scenarios.


## Configuration Files

Create `config.yaml` for default parameters:

```yaml
# config.yaml
model:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100

data:
  dataset: "cifar10"
  augmentation: true

training:
  optimizer: "adam"
  scheduler: "cosine"
```


## Documentation

📚 **[Complete Documentation](docs/README.md)** - Detailed guides and API reference

**Quick Links:**
- [CLI Commands](docs/cli-commands.md) - All yanex commands with examples
- [Python API](docs/python-api.md) - Complete Python API reference  
- [Configuration](docs/configuration.md) - Parameter management and config files
- [Comparison Tool](docs/compare.md) - Interactive experiment comparison
- [Best Practices](docs/best-practices.md) - Tips for effective experiment tracking


## Contributing

Yanex is open source and welcomes contributions! See our [contributing guidelines](CONTRIBUTING.md) for details.

**Built with assistance from [Claude](https://claude.ai).**

## License

MIT License - see [LICENSE](LICENSE) for details.

