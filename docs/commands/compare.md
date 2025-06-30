# `yanex compare` - Interactive Experiment Comparison

Compare experiments side-by-side in an interactive terminal table, similar to Guild Compare.

## Quick Start

```bash
# Compare all experiments in interactive table
yanex compare

# Compare specific experiments
yanex compare exp1 exp2 exp3

# Compare with filters
yanex compare --status completed --tag production
```

## Overview

The `yanex compare` command provides a powerful way to analyze and compare experiments:

- **Interactive Table**: Navigate with keyboard, sort by any column
- **Flexible Filtering**: Compare subsets of experiments using various criteria
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
# Compare specific experiments
yanex compare abc1234 def5678 ghi9012

# Mix IDs and names (if unique)
yanex compare baseline-model experiment-v2 abc1234
```

### By Status

```bash
yanex compare --status completed
yanex compare --status failed
yanex compare --status running
```

### By Tags

```bash
# Experiments with specific tag
yanex compare --tag production

# Multiple tags (experiments must have ALL tags)
yanex compare --tag baseline --tag validated
```

### By Name Pattern

```bash
# Wildcard matching
yanex compare --name "*baseline*"
yanex compare --name "model-v*"
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

```bash
# Include archived experiments in comparison
yanex compare --archived
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
yanex compare --status completed --started-after "1 week ago"
```

### Parameter Sweep Analysis

```bash
# After running a learning rate sweep
yanex compare --tag "lr-sweep" --only-different
```

### Model Comparison

```bash
# Compare different model architectures
yanex compare --name "*resnet*" --name "*transformer*" --only-different
```

### Production Model Analysis

```bash
# Compare all production models, export for reporting
yanex compare --tag production --export production_models.csv
```

### Failed Experiment Analysis

```bash
# Analyze what went wrong with failed experiments
yanex compare --status failed --started-after "yesterday"
```

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

Prefixed with 📊, derived from `config.yaml`:
- Automatically discovered from all experiment configs
- Formatted based on data type (numbers, strings, booleans)
- Missing values shown as `-`

### Metric Columns  

Prefixed with 📈, derived from logged results:
- Automatically discovered from `experiment.log_results()` calls
- For list-based results, shows the latest/last value
- Numeric values formatted with appropriate precision

## Advanced Usage

### Complex Filtering

```bash
# Multi-criteria filtering
yanex compare \
  --status completed \
  --tag baseline \
  --started-after "2023-12-01" \
  --params learning_rate,epochs \
  --only-different
```

### Integration with Shell

```bash
# Get experiment IDs from list, then compare
EXPERIMENTS=$(yanex list --status completed --tag production | grep -o '^[a-z0-9]*')
yanex compare $EXPERIMENTS --export production_comparison.csv
```

### Automated Reporting

```bash
#!/bin/bash
# Weekly experiment report
DATE=$(date +%Y-%m-%d)
yanex compare \
  --started-after "1 week ago" \
  --status completed \
  --export "weekly_report_${DATE}.csv"
echo "Weekly report saved to weekly_report_${DATE}.csv"
```

## Common Workflows

### Hyperparameter Optimization

```bash
# 1. Run parameter sweep
for lr in 0.001 0.01 0.1; do
  for bs in 16 32 64; do
    yanex run train.py --param learning_rate=$lr --param batch_size=$bs --tag hp-sweep
  done
done

# 2. Analyze results
yanex compare --tag hp-sweep --only-different

# 3. Export best results
yanex compare --tag hp-sweep --export hp_sweep_results.csv
```

### Model Development

```bash
# Compare different model versions
yanex compare --name "*model-v*" --only-different

# Focus on key metrics
yanex compare --name "*model-v*" --metrics accuracy,f1_score,inference_time
```

### Production Monitoring

```bash
# Monitor recent production experiments
yanex compare --tag production --started-after "1 day ago"

# Compare against baseline
yanex compare baseline-model --tag production --started-after "1 week ago"
```

## Troubleshooting

### No Experiments Found

```bash
# Check available experiments
yanex list

# Check archived experiments
yanex list --archived

# Verify filter criteria
yanex compare --status completed  # Remove filters one by one
```

### Empty Columns

- **No parameters**: Experiments may not have `config.yaml` files
- **No metrics**: Experiments may not have called `experiment.log_results()`
- **Missing values**: Some experiments may be missing specific parameters/metrics

### Performance with Many Experiments

```bash
# Limit results for better performance
yanex compare --max-rows 100

# Use more specific filters
yanex compare --started-after "1 month ago"
```

---

**Related:**
- [CLI Commands Overview](../cli-commands.md)
- [Python API](../python-api.md)
- [Filtering Guide](../filtering.md)