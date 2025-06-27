# Yanex - Yet Another Experiment Tracker

## Overview
Yanex is a lightweight experiment tracking harness for Python that focuses on reproducibility and organization. It's designed for non-ML experiments that need parameter management, result logging, and artifact storage.

## Core Requirements

### Experiment Properties
- **ID**: Random 8-character hex string (e.g., `a1b2c3d4`)
- **Name**: Optional user-defined string, must be unique if provided
- **Git commit**: Automatically captured, experiments only run on clean git state
- **Script**: Python script path to execute
- **Configuration**: YAML file with parameters
- **Tags**: Optional list of strings for categorization
- **Description**: Optional text description
- **Status**: `running`, `completed`, `failed`, `cancelled`

### Experiment Data Storage
```
./experiments/
├── a1b2c3d4/
│   ├── metadata.json       # ID, name, git commit, timestamps, status, environment
│   ├── config.yaml         # Final resolved parameters
│   ├── results.json        # Experiment results/metrics with steps
│   ├── artifacts/          # Files, plots, etc.
│   ├── stdout.log          # Script output
│   └── stderr.log          # Script errors
```

### Environment Capture
- Python version
- Git commit hash
- Requirements/dependencies
- System info

## CLI Interface

### Commands
1. `yanex run <script> [--name NAME] [--config CONFIG] [--param key=value] [--tag TAG] [--desc "description"]`
2. `yanex list [--status STATUS] [--name PATTERN] [--tag TAG] [--commit COMMIT] [--started TIMESPEC] [--ended TIMESPEC]`
3. `yanex rerun <id_or_name> [--param key=value]`
4. `yanex archive <id_or_name>`
5. `yanex compare <id_or_name1> <id_or_name2> [...]`

### Features
- Parameter overrides via `--param key=value`
- Filtering by multiple criteria in `list`
- Interactive comparison table with sorting
- Targeting experiments by ID or name
- Parallel execution prevention

## Python API

### Import
```python
from yanex import experiment
```

### Core Usage Pattern
```python
params = experiment.get_params()

with experiment.run():
    # Status automatically set to 'running'
    
    # Log results with automatic step incrementing
    experiment.log_results({"accuracy": 0.85})  # step=0
    experiment.log_results({"accuracy": 0.89})  # step=1
    experiment.log_results({"accuracy": 0.91, "step": 5})  # explicit step=5
    
    # Handle artifacts
    experiment.log_artifact("data.csv", "/path/to/data.csv")
    experiment.log_matplotlib_figure(fig, "training_curve.png")
    experiment.log_text("Summary: good results", "summary.txt")
    
    # Status automatically set to 'completed' on normal exit
```

### API Functions

#### Parameter Management
- `experiment.get_params() -> dict` - Returns merged parameters from YAML + CLI overrides
- `experiment.get_param(key, default=None) -> any` - Get single parameter with default

#### Context Management
- `experiment.run() -> context_manager` - Main experiment context

#### Status Control (inside context)
- `experiment.get_status() -> str` - Returns current status
- `experiment.completed()` - Explicit completion + exit context
- `experiment.fail(message)` - Set failed status + exit context
- `experiment.cancel(message)` - Set cancelled status + exit context

#### Logging (inside context)
- `experiment.log_results(dict, step=None)` - Log metrics with auto-increment or explicit step
- `experiment.log_artifact(name, file_path)` - Copy file to artifacts/ folder
- `experiment.log_matplotlib_figure(fig, filename, **kwargs)` - Save matplotlib figure
- `experiment.log_text(content, filename)` - Save text content as artifact

### Context Manager Behavior
- **Enter**: Create experiment directory, set status='running'
- **Normal exit**: Set status='completed' automatically
- **Exception**: Set status='failed', log exception details
- **Explicit exit**: Via `completed()`, `fail()`, `cancel()`

### Results Format
Results are stored as a list of step dictionaries with automatic timestamping:
```json
[
  {"step": 0, "timestamp": "2024-01-01T12:00:00", "accuracy": 0.85},
  {"step": 1, "timestamp": "2024-01-01T12:01:00", "accuracy": 0.89},
  {"step": 5, "timestamp": "2024-01-01T12:05:00", "accuracy": 0.91}
]
```

### Step Handling
- Auto-increment when step=None
- Replace existing step data when explicit step provided
- Warning issued when replacing existing steps
- Steps can be any numeric value

## Key Design Decisions

1. **Git Integration**: Experiments only run on clean git state for reproducibility
2. **Step Replacement**: Allow overwriting previous step results with warning
3. **Automatic Completion**: Context manager sets 'completed' status on normal exit
4. **Artifact Copying**: Files copied into experiment directory, not referenced
5. **Single Execution**: Prevent parallel experiment execution
6. **Flexible Targeting**: Use either experiment ID or name in CLI commands

## Future Considerations
- Nested experiments for complex workflows
- Progress tracking for long-running experiments
- Experiment templates for common patterns
- Export/import capabilities