# Yanex - Yet Another Experiment Tracker

A lightweight experiment tracking harness for Python that focuses on reproducibility and organization.

## Features

- **Reproducible Experiments**: Automatic git commit tracking and clean state validation
- **Parameter Management**: YAML configuration with CLI override support
- **Result Logging**: Step-based result tracking with automatic timestamping
- **Artifact Storage**: Automatic file and matplotlib figure management
- **Rich CLI**: Interactive experiment listing, comparison, and management
- **Context Manager API**: Clean Python API using `with` statements

## Quick Start

### Installation

```bash
pip install yanex
```

### Basic Usage

1. Create a configuration file `config.yaml`:
```yaml
n_docs: 1000
batch_size: 32
learning_rate: 0.001
```

2. Write your experiment script:
```python
from yanex import experiment

params = experiment.get_params()

with experiment.run():
    # Your experiment code here
    result = {"accuracy": 0.95, "loss": 0.05}
    experiment.log_results(result)
    experiment.log_artifact("model.pkl", "path/to/model.pkl")
```

3. Run your experiment:
```bash
yanex run my_experiment.py --param learning_rate=0.01
```

## CLI Commands

- `yanex run <script>` - Run an experiment
- `yanex list` - List experiments with filtering
- `yanex rerun <id>` - Re-run an experiment
- `yanex compare <id1> <id2>` - Compare experiment results
- `yanex archive <id>` - Archive an experiment

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT License