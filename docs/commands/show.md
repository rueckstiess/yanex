# `yanex show` - Display Experiment Details

Show comprehensive information about a specific experiment including configuration, results, artifacts, and environment details.

## Quick Start

```bash
# Show experiment by ID
yanex show a1b2c3d4

# Show experiment by name
yanex show "baseline-model"

# Show only specific metrics
yanex show a1b2c3d4 --show-metric "accuracy,loss"
```

## Overview

The `yanex show` command displays detailed information about a single experiment, including:

- **Metadata**: Name, status, timestamps, duration, tags, description
- **Configuration**: All parameters used for the experiment
- **Results**: Logged metrics and values from the experiment execution
- **Artifacts**: Files created during the experiment (models, logs, plots)
- **Environment**: Git state, Python version, platform information
- **Error Details**: Failure or cancellation reasons (if applicable)

## Command Options

### Required Arguments

- `experiment_identifier`: Experiment ID (8-character hex) or experiment name

### Optional Arguments

- `--show-metric METRICS`: Comma-separated list of specific metrics to display (e.g., "accuracy,loss,f1_score")
- `--archived`: Include archived experiments in search
- `--format`, `-F FORMAT`: Output format (default, json, csv, markdown)
- `--help`: Show help message and exit

## Usage

### Show by Experiment ID

```bash
yanex show a1b2c3d4
```

Shows complete details for the experiment with ID `a1b2c3d4`. This is the most reliable way to reference a specific experiment.

### Show by Experiment Name

```bash
yanex show "baseline-model"
```

Shows details for the experiment named "baseline-model". If multiple experiments have the same name, a list will be displayed and you'll need to use the specific experiment ID instead.

### Filter Specific Metrics

```bash
yanex show a1b2c3d4 --show-metric "accuracy,loss,learning_rate"
```

Shows only the specified metrics in the results table, which is useful when experiments log many metrics and you want to focus on specific ones.

### Include Archived Experiments

```bash
yanex show old_experiment --archived
```

Searches for experiments in both active and archived directories. By default, only active experiments are searched.

## Output Sections

The show command displays information in several organized sections:

1. **Header**: Experiment name, ID, status, directory path, and timing information
2. **Experiment Info**: Tags and description (if present)
3. **Configuration**: All parameters and their values
4. **Results**: Logged metrics in a table format (last 10 entries)
5. **Artifacts**: Files created during experiment execution
6. **Environment**: Git state, Python version, platform details
7. **Error Information**: Failure details (for failed experiments only)

## Output Format

Control output format with `--format` or `-F`:

```bash
# Default: rich terminal output
yanex show a1b2c3d4

# JSON for scripting
yanex show a1b2c3d4 -F json

# CSV for data export
yanex show a1b2c3d4 -F csv

# Markdown for documentation
yanex show a1b2c3d4 -F markdown
```

---

**Related:**
- [`yanex list`](list.md) - List and filter experiments
- [`yanex compare`](compare.md) - Compare multiple experiments
- [`yanex get`](get.md) - Get specific field values
- [Python API Reference](../python-api.md)