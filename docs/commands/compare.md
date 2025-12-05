# `yanex compare` - Interactive Experiment Comparison

Compare experiments side-by-side in an interactive terminal table, similar to Guild Compare.

## Quick Start

```bash
# Compare all experiments in interactive table
yanex compare

# Compare specific experiments
yanex compare exp1 exp2 exp3

# Compare with filters
yanex compare -s completed -t production
```

## Overview

The `yanex compare` command provides a powerful way to analyze and compare experiments:

- **Interactive Table**: Navigate with keyboard, sort by any column
- **Flexible Filtering**: Compare subsets of experiments using various criteria (with convenient short aliases)
- **Column Selection**: Focus on specific parameters and metrics
- **Export Options**: Save comparison data to CSV for external analysis

## Basic Usage

### Interactive Mode (Default)

```bash
yanex compare
```

Launches an interactive table with all experiments. Use keyboard shortcuts to navigate and analyze:

- **Navigation**: Arrow keys or `hjkl` to move around
- **Sorting**: `s/S` for ascending/descending, `1/2` for numeric sorting
- **Help**: Press `?` to see all keyboard shortcuts
- **Export**: Press `e` to export current view to CSV
- **Quit**: Press `q` or `Ctrl+C` to exit

### Static Table Mode

```bash
yanex compare --no-interactive
```

Displays a static table in the terminal without interactive features.

### CSV Export

```bash
yanex compare --export results.csv
```

Exports comparison data directly to CSV without showing the interactive table.

## Filtering Experiments

### By Experiment IDs

```bash
# Compare specific experiments (positional arguments)
yanex compare abc1234 def5678 ghi9012

# Mix IDs and names (if unique)
yanex compare baseline-model experiment-v2 abc1234

# Using --ids filter (comma-separated)
yanex compare --ids abc1234,def5678,ghi9012
yanex compare -i a1,b2,c3

# Useful for piping from other commands
yanex compare --ids $(yanex get upstream d7742130 -F sweep)
```

### By Status

Filter using `--status` or `-s`:

```bash
yanex compare -s completed
yanex compare -s failed
yanex compare -s running
```

### By Tags

Filter using `--tag` or `-t`:

```bash
# Experiments with specific tag
yanex compare -t production

# Multiple tags (experiments must have ALL tags)
yanex compare -t baseline -t validated
```

### By Name Pattern

Filter using `--name` or `-n`:

```bash
# Wildcard matching
yanex compare -n "*baseline*"
yanex compare -n "model-v*"
```

### By Time Range

```bash
# Recent experiments
yanex compare --started-after "1 week ago"
yanex compare --started-after "2023-12-01"

# Date range
yanex compare --started-after "2023-12-01" --started-before "2023-12-31"

# Experiments that finished recently
yanex compare --ended-after "yesterday"
```

### Include Archived

Use `--archived` or `-a`:

```bash
# Include archived experiments in comparison
yanex compare -a
```

## Column Selection

### Parameters and Metrics

```bash
# Show only specific parameters
yanex compare --params learning_rate,batch_size,epochs

# Show only specific metrics  
yanex compare --metrics accuracy,loss,f1_score

# Combine both
yanex compare --params learning_rate,epochs --metrics accuracy,loss
```

### Only Different Values

```bash
# Hide columns where all experiments have identical values
yanex compare --only-different

# Useful for parameter sweeps
yanex compare --tag "lr-sweep" --only-different
```

## Interactive Controls

When in interactive mode, use these keyboard shortcuts:

### Navigation
- `↑/↓` or `j/k` - Move up/down rows
- `←/→` or `h/l` - Move left/right columns
- `Home/End` - Jump to first/last row
- `Page Up/Down` - Navigate by page

### Sorting
- `s` - Sort current column ascending
- `S` - Sort current column descending  
- `1` - Numeric sort ascending
- `2` - Numeric sort descending
- `r` - Reset to default sort (by start time)
- `R` - Reverse current sort order

### Other Controls
- `e` - Export current view to CSV
- `?` - Show help with all shortcuts
- `q` or `Ctrl+C` - Quit

### Visual Indicators
- **Sort arrows**: `↑/↓` next to column headers show current sort
- **Parameter columns**: Prefixed with 📊 icon
- **Metric columns**: Prefixed with 📈 icon
- **Missing values**: Shown as `-`

## Output Options

### Output Format

Control output format with `--format` or `-F`:

```bash
# Default: interactive table
yanex compare

# JSON for scripting
yanex compare -F json

# CSV for spreadsheets
yanex compare -F csv

# Markdown for documentation
yanex compare -F markdown
```

### Limit Results

```bash
# Show only first N experiments
yanex compare --max-rows 10
```

### Export to CSV

```bash
# Export to file
yanex compare --export comparison.csv

# Export filtered results
yanex compare --tag production --only-different --export prod_results.csv
```

## Examples

### Compare Recent Successful Experiments

```bash
yanex compare -s completed --started-after "1 week ago"
```

### Parameter Sweep Analysis

```bash
# After running a learning rate sweep
yanex compare -t "lr-sweep" --only-different
```

### Model Comparison

```bash
# Compare different model architectures
yanex compare -n "*resnet*" --only-different
```

### Production Model Analysis

```bash
# Compare all production models, export for reporting
yanex compare -t production --export production_models.csv
```

### Failed Experiment Analysis

```bash
# Analyze what went wrong with failed experiments
yanex compare -s failed --started-after "yesterday"
```

## Short Aliases

All options have convenient short aliases:

- `-i` for `--ids`
- `-s` for `--status`
- `-n` for `--name`
- `-t` for `--tag`
- `-a` for `--archived`
- `-F` for `--format`

These aliases match the `run` and `list` commands for consistency.

## Column Types and Formatting

### Fixed Columns

Always shown (left side of table):
- **ID**: Experiment identifier
- **Name**: Human-readable experiment name
- **Started**: Experiment start time (YYYY-MM-DD HH:MM:SS)
- **Duration**: Experiment runtime (HH:MM:SS format)
- **Status**: Current status (completed, failed, running, etc.)
- **Tags**: Experiment tags (comma-separated)

### Parameter Columns

Prefixed with 📊, derived from `params.yaml`:
- Automatically discovered from all experiment configs
- Formatted based on data type (numbers, strings, booleans)
- Missing values shown as `-`

### Metric Columns  

Prefixed with 📈, derived from logged results:
- Automatically discovered from `experiment.log_metrics()` calls
- For list-based results, shows the latest/last value
- Numeric values formatted with appropriate precision

---

**Related:**
- [`yanex get`](get.md) - Get specific field values
- [CLI Commands Overview](../cli-commands.md) - See filtering patterns section
- [Python API](../python-api.md)