# Experiment Directory Structure

Yanex creates a structured directory layout for each experiment, organizing all experiment-related files and data. This document explains the organization and purpose of each file and directory.

## Overview

Each experiment is stored in its own directory under the `experiments/` folder (or your configured experiments directory). The experiment directory is named using the 8-character hexadecimal experiment ID.

```
experiments/
├── a1b2c3d4/           # Experiment directory (ID: a1b2c3d4)
│   ├── metadata.json   # Experiment metadata and status
│   ├── config.json     # Configuration parameters
│   ├── metrics.json    # User-logged metrics from yanex.log_metrics()
│   ├── script_runs.json # Bash script execution logs (if any)
│   └── artifacts/      # Directory for experiment artifacts
│       ├── script_stdout.txt
│       ├── script_stderr.txt
│       ├── git_diff.patch    # Uncommitted changes (if any)
│       ├── model.pth
│       └── plots.png
└── e5f6g7h8/          # Another experiment directory
    └── ...
```

## File Descriptions

### Core Files

#### `metadata.json`
Contains experiment metadata, status information, timing, and environment details.

```json
{
  "id": "a1b2c3d4",
  "name": "baseline-model",
  "status": "completed",
  "created_at": "2023-12-01T10:00:00Z",
  "started_at": "2023-12-01T10:00:01Z", 
  "completed_at": "2023-12-01T10:05:30Z",
  "duration": 329.5,
  "script_path": "/path/to/train.py",
  "tags": ["baseline", "production"],
  "description": "Initial baseline model training",
  "git_commit": "abc123...",
  "git_branch": "main",
  "has_uncommitted_changes": false,
  "patch_file": null,
  "python_version": "3.11.5",
  "platform": "darwin"
}
```

#### `config.json` 
Contains all configuration parameters used for the experiment, including CLI overrides, environment variables, and config file values.

```json
{
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 100,
  "model_type": "transformer",
  "data_path": "/data/train.csv"
}
```

#### `metrics.json`
Contains user-logged metrics from calls to `yanex.log_metrics()`. Each entry represents a step in the experiment with associated metrics.

```json
[
  {
    "step": 0,
    "accuracy": 0.85,
    "loss": 0.45,
    "learning_rate": 0.001,
    "timestamp": "2023-12-01T10:01:00Z"
  },
  {
    "step": 1,
    "accuracy": 0.87,
    "loss": 0.42,
    "timestamp": "2023-12-01T10:02:00Z",
    "last_updated": "2023-12-01T10:02:15Z"
  }
]
```

#### `script_runs.json` *(Optional)*
Contains execution logs from `yanex.execute_bash_script()` calls. Each entry represents a script execution with metadata.

```json
[
  {
    "command": "python preprocess.py",
    "exit_code": 0,
    "execution_time": 45.2,
    "stdout_lines": 10,
    "stderr_lines": 0,
    "working_directory": "/path/to/experiment/dir",
    "timestamp": "2023-12-01T10:00:30Z",
    "recorded_at": "2023-12-01T10:01:15Z"
  }
]
```

### Artifacts Directory

#### `artifacts/`
Contains all files created during experiment execution, including:

- **Git patches**: `git_diff.patch` - Automatically captured uncommitted changes (when present)
- **Script outputs**: `script_stdout.txt`, `script_stderr.txt` (from `execute_bash_script()`)
- **Models**: Saved model files (`.pth`, `.pkl`, `.h5`, etc.)
- **Plots**: Visualization files (`.png`, `.pdf`, `.svg`)
- **Data**: Generated datasets, processed files
- **Logs**: Custom log files created by your scripts
- **Any file**: Added via `yanex.copy_artifact()` or `yanex.save_artifact()`

**Git Patch Capture:**
When you run an experiment with uncommitted changes, Yanex automatically generates and saves a git patch file (`git_diff.patch`) containing all tracked file changes. This ensures full reproducibility even when working with uncommitted code. The metadata fields `has_uncommitted_changes` and `patch_file` indicate whether a patch was captured.

## Directory Lifecycle

### Creation
Experiment directories are created when:
- Running `yanex run script.py`
- Calling `yanex.create_experiment()` in Python

### Active vs Archived
- **Active experiments**: Located in `experiments/` directory
- **Archived experiments**: Located in `experiments_archive/` directory (moved via `yanex archive`)

### Cleanup
Experiment directories are:
- **Archived**: Moved to archive directory with `yanex archive`
- **Deleted**: Permanently removed with `yanex delete`
- **Restored**: Moved back from archive with `yanex unarchive`

## File Format Notes

- **JSON files**: All structured data files use JSON format with 2-space indentation
- **UTF-8 encoding**: All text files use UTF-8 encoding
- **Timestamps**: ISO 8601 format in UTC (`YYYY-MM-DDTHH:MM:SSZ`)
- **Paths**: Stored as absolute paths when possible

## Backward Compatibility

Yanex automatically handles legacy experiment directories:
- **Legacy `results.json`**: Automatically migrated to `metrics.json` when accessed
- **Missing files**: Gracefully handled with appropriate defaults
- **Version upgrades**: Transparent compatibility across yanex versions

## Related Documentation

- [Python API Reference](python-api.md) - Methods that create these files
- [`yanex show`](commands/show.md) - View experiment details
- [`yanex archive`](commands/archive.md) - Archive experiment directories