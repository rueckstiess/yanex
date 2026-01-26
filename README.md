# Yanex - Yet Another Experiment Tracker

[![PyPI version](https://img.shields.io/pypi/v/yanex)](https://pypi.org/project/yanex/)
[![Python versions](https://img.shields.io/pypi/pyversions/yanex)](https://pypi.org/project/yanex/)
[![License](https://img.shields.io/pypi/l/yanex)](https://github.com/rueckstiess/yanex/blob/main/LICENSE)

A lightweight, Git-aware experiment tracking system for Python that makes reproducible research effortless.

Yanex is inspired by the now-discontinued [Guild AI](https://guild.ai) ([more info](https://github.com/guildai/guildai/discussions/523)).

## Why Yanex?

**Stop losing track of your experiments.** Yanex automatically tracks parameters, results, and code state so you can focus on what matters - your research.

```python
import yanex

# read parameters from config file or CLI arguments
lr = yanex.get_param('lr', default=0.001)
epochs = yanex.get_param('epochs', default=10)

# your experiment code
# ...

# log results, artifacts and figures
yanex.log_metrics({"loss": loss, "accuracy": accuracy}, step=epoch)
yanex.copy_artifact(model_path, "model.pth")
yanex.save_artifact(fig, "loss_curve.png")
```

Run from the command line:

```bash
# Run with yanex CLI for automatic tracking
yanex run train.py --param lr=10e-3 --param epochs=10
```

That's it. Yanex creates a separate directory for each experiment, saves the logged results and files, stdout and stderr outptus, Python environment information, and even the Git state of your code repository. You can then compare results, search experiments, and reproduce them with ease.

## Key Features

- 🔒 **Reproducible**: Automatic Git state tracking ensures every experiment is reproducible
- 📊 **Interactive Comparison**: Compare experiments side-by-side with an interactive table
- ⚙️ **Flexible Parameters**: YAML configs with CLI overrides and syntax for parameter sweeps
- ⚡ **Parallel Execution**: Run multiple experiments simultaneously on multi-core systems
- 📈 **Rich Logging**: Track metrics, artifacts, and figures
- 🔍 **Powerful Search**: Find experiments by status, parameters, tags, or time ranges
- 🤖 **AI-Friendly**: Machine-readable output formats, Claude Code skill, and `yanex get` for seamless AI assistant integration

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

yanex.log_metrics({
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
yanex compare -s completed -t baseline
```

### 3. Track Everything

List, search, and manage your experiments:

```bash
# List recent experiments
yanex list

# Find experiments by criteria
yanex list -s completed -t production
yanex list --started-after "1 week ago"

# Show detailed experiment info
yanex show exp_id

# Archive old experiments
yanex archive --started-before "1 month ago"
```

## Programmatic Access

Yanex provides two APIs for working with experiments:

- **[Run API](docs/run-api.md)**: Create and execute experiments programmatically, ideal for k-fold cross-validation, ensemble training, and batch processing
- **[Results API](docs/results-api.md)**: Query, filter, and analyze completed experiments with pandas integration for advanced analysis

See the [examples directory](examples/) for practical demonstrations of both APIs.


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

## Parameter Sweeps & Parallel Execution

Run parameter sweeps with automatic parallelization:

```bash
# Run sweep in parallel with 4 workers
yanex run train.py --param "lr=range(0.01, 0.1, 0.01)" --parallel 4

# Auto-detect CPU count
yanex run train.py --param "lr=logspace(-4, -1, 10)" --parallel 0
```

**Sweep Syntax:**
```bash
# List of values
--param "batch_size=16, 32, 64, 128"

# Range: start, stop, step
--param "lr=range(0.01, 0.1, 0.01)"

# Linspace: start, stop, num_points
--param "lr=linspace(0.001, 0.1, 10)"

# Logspace: start_exp, stop_exp, num_points (uses powers of 10)
--param "lr=logspace(-4, -1, 10)"


# Multi-parameter sweep (cartesian product)
yanex run train.py \
  --param "lr=range(0.01, 0.1, 0.01)" \
  --param "batch_size=32, 64" \
  --parallel 4
```

See [Configuration Guide](docs/configuration.md#parameter-sweeps) for complete sweep syntax details.


## Documentation

📚 **[Complete Documentation](docs/README.md)** - Detailed guides and API reference

**Quick Links:**
- [CLI Commands](docs/cli-commands.md) - All yanex commands with examples
- [Experiment Structure](docs/experiment-structure.md) - Directory layout and file organization
- [Configuration](docs/configuration.md) - Parameter management and config files
- [Run API](docs/run-api.md) - Programmatic experiment execution
- [Results API](docs/results-api.md) - Querying and analyzing experiment results
- [AI & Automation](docs/ai-usage.md) - Machine-readable output and Claude Code skill

## Examples

- **[CLI Examples](examples/cli/)** - Main use case: Dual-mode scripts that work standalone or with yanex tracking
- **[Run API Examples](examples/run-api/)** - Programmatic experiment creation for advanced patterns like k-fold cross-validation and batch processing
- **[Results API Examples](examples/results-api/)** - Querying and analyzing completed experiments with pandas integration

## AI & Automation

Yanex is designed for seamless integration with AI assistants and automation scripts.

**Machine-readable output** - Most commands support `--format json`, `--format csv`, and `--format markdown`:
```bash
yanex list -s completed -F json | jq '.[] | .id'
yanex compare -t sweep -F csv > results.csv
```

**Quick field extraction** - The `yanex get` command extracts individual values for scripting:
```bash
yanex get status abc12345           # → "completed"
yanex get params.lr abc12345        # → "0.001"
yanex get stdout abc12345 --tail 20 # Last 20 lines of output
```

**[→ Complete AI & Automation Guide](docs/ai-usage.md)**

### Claude Code Skill

Yanex includes a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill for AI-assisted experiment management:

```bash
# Find your yanex installation
YANEX_PATH=$(python -c "import yanex; print(yanex.__path__[0])")

# Install globally or per-project
ln -s "$YANEX_PATH/claude-skill" ~/.claude/skills/tracking-yanex-experiments
# or: ln -s "$YANEX_PATH/claude-skill" .claude/skills/tracking-yanex-experiments
```

The skill enables Claude to run experiments, query results, monitor progress, and analyze data using yanex commands and APIs.

## Contributing

Yanex is open source and welcomes contributions! See our [contributing guidelines](CONTRIBUTING.md) for details.

**Built with assistance from [Claude](https://claude.ai).**

#### Contributors

The Yanex web UI (`yanex ui`) is being developed by Leon Lei ([lytlei](https://github.com/lytlei)) as part of his
Honours Thesis at the [University of Sydney](https://www.sydney.edu.au). 


## License

MIT License - see [LICENSE](LICENSE) for details.



