# CLI Commands

This guide provides an overview of all Yanex CLI commands with common usage patterns and links to detailed documentation.

## Command Overview

| Command | Purpose | Common Use |
|---------|---------|------------|
| [`run`](commands/run.md) | Execute tracked experiments | `yanex run script.py --param lr=0.01` |
| [`list`](commands/list.md) | List and filter experiments | `yanex list -s completed -t production` |
| [`show`](commands/show.md) | Display experiment details | `yanex show abc123` |
| [`get`](commands/get.md) | Get specific field values | `yanex get status abc123` |
| [`compare`](commands/compare.md) | Interactive comparison table | `yanex compare exp1 exp2 exp3` |
| [`archive`](commands/archive.md) | Archive old experiments | `yanex archive --started-before "1 month ago"` |
| [`unarchive`](commands/unarchive.md) | Restore archived experiments | `yanex unarchive abc123` |
| [`delete`](commands/delete.md) | Permanently delete experiments | `yanex delete --status failed` |
| [`update`](commands/update.md) | Update experiment metadata | `yanex update abc123 --name "production-v2"` |
| [`ui`](commands/ui.md) | Launch web interface | `yanex ui` |
| [`open`](commands/open.md) | Open experiment directory | `yanex open abc123` |

## Core Workflow Commands

### run - Execute Experiments

Run Python scripts with automatic tracking:

```bash
# Basic execution
yanex run train.py

# With parameters
yanex run train.py --param lr=0.01 --param epochs=100

# With config file
yanex run train.py --config config.yaml

# Parameter sweep
yanex run train.py --param "lr=logspace(-4, -1, 10)" --parallel 4

# Add metadata
yanex run train.py --name "baseline-v1" --tag production --description "Initial baseline"
```

See [run command documentation](commands/run.md) for complete details.

### list - Find Experiments

List and filter experiments by various criteria:

```bash
# Recent experiments
yanex list

# Filter by status
yanex list -s completed

# Filter by tags
yanex list -t production -t baseline

# Filter by time
yanex list --started-after "1 week ago"

# Combine filters
yanex list -s completed -t ml --started-after "2025-01-01"
```

See [list command documentation](commands/list.md) for all filtering options.

### show - Inspect Details

Display comprehensive information about an experiment:

```bash
# Show by ID (full or prefix)
yanex show abc123

# Show latest experiment
yanex show --latest

# Show with specific tag
yanex show --latest -t baseline
```

See [show command documentation](commands/show.md) for more options.

### get - Extract Field Values

Retrieve specific values for scripting and automation:

```bash
# Get experiment status
yanex get status abc123

# Get parameter value
yanex get params.lr abc123

# Get last logged metric
yanex get metrics.accuracy abc123

# Get IDs of completed experiments (for bash substitution)
yanex get id -s completed -F sweep

# Build dynamic dependencies
yanex run train.py -D data=$(yanex get id -n "*-prep-*" -F sweep)
```

See [get command documentation](commands/get.md) for all available fields and formats.

### compare - Analyze Results

Interactive side-by-side comparison:

```bash
# Compare specific experiments
yanex compare abc123 def456 ghi789

# Compare by filter
yanex compare -s completed -t hyperparameter-sweep

# Compare with custom columns
yanex compare --param lr --param batch_size --metric accuracy --metric loss
```

See [compare command documentation](commands/compare.md) for advanced usage.

## Management Commands

### archive / unarchive

Move experiments out of active storage:

```bash
# Archive old experiments
yanex archive --started-before "3 months ago"

# Archive by status
yanex archive -s failed

# Restore from archive
yanex unarchive abc123
```

See [archive](commands/archive.md) and [unarchive](commands/unarchive.md) documentation.

### delete

Permanently remove experiments:

```bash
# Delete specific experiments
yanex delete abc123 def456

# Delete by filter (use with caution!)
yanex delete -s failed --started-before "6 months ago"
```

⚠️ **Warning**: Deletion is permanent. Always verify with `list` first.

See [delete command documentation](commands/delete.md) for safety guidelines.

### update

Modify experiment metadata:

```bash
# Update name
yanex update abc123 --name "production-final"

# Add tags
yanex update abc123 --add-tag verified --add-tag deployed

# Update description
yanex update abc123 --description "Final production model"
```

See [update command documentation](commands/update.md) for all options.

## Utility Commands

### ui - Web Interface

Launch the interactive web UI:

```bash
# Start web server (default port 8000)
yanex ui

# Custom port
yanex ui --port 8080
```

See [ui command documentation](commands/ui.md) for configuration options.

### open - File Explorer

Open experiment directory in your file manager:

```bash
# Open by ID
yanex open abc123

# Open latest
yanex open --latest
```

See [open command documentation](commands/open.md) for platform details.

## Common Patterns

### Filtering Experiments

All list-based commands (`list`, `compare`, `archive`, `delete`, `update`) support consistent filtering:

**By Experiment IDs:**
```bash
--ids abc123           # or: -i abc123
--ids abc123,def456    # Multiple IDs (comma-separated)
--ids $(yanex get upstream d7742130 -F sweep)  # Dynamic from other commands
```

**By Status:**
```bash
--status completed  # or: -s completed
--status failed
--status cancelled
--status running
```

**By Tags:**
```bash
--tag production    # or: -t production
--tag ml --tag baseline  # Multiple tags (AND logic)
```

**By Name Pattern:**
```bash
--name "train-*"    # or: -n "train-*"
--name "*baseline*"
--name ""           # Match unnamed experiments
```

**By Time:**
```bash
--started-after "2025-01-01"
--started-after "1 week ago"
--started-before "2024-12-31"
--ended-after "2025-01-15"
--ended-before "yesterday"
```

**Combining Filters:**
```bash
yanex list -s completed -t production --started-after "1 month ago"
```

### Experiment Selection

Commands that operate on specific experiments (`show`, `update`, `open`, `archive`, `unarchive`, `delete`) support:

**ID Prefix Matching:**
```bash
# Full ID: abc12345
yanex show abc     # Matches if unique
yanex show abc1    # More specific if needed
```

**Latest Experiment:**
```bash
yanex show --latest
yanex show --latest -t baseline  # Latest with specific tag
```

**Multiple Experiments:**
```bash
yanex delete abc123 def456 ghi789
yanex compare exp1 exp2 exp3
```

### Parameter Sweeps

Use sweep syntax for hyperparameter tuning:

```bash
# Range sweep
yanex run train.py --param "lr=range(0.01, 0.1, 0.01)" --parallel 4

# List of values (comma-separated)
yanex run train.py --param "lr=0.001,0.01,0.1" --parallel 4

# Logarithmic sweep
yanex run train.py --param "lr=logspace(-4, -1, 10)" --parallel 4

# Multi-parameter (cross-product)
yanex run train.py \
  --param "lr=0.001,0.01,0.1" \
  --param "batch_size=16,32,64" \
  --parallel 8
```

See [Configuration Guide](configuration.md) for sweep syntax details.

## Getting Help

Every command has built-in help:

```bash
yanex --help              # Overview of all commands
yanex run --help          # Detailed help for run command
yanex list --help         # Detailed help for list command
```

## Next Steps

- **Detailed Command Docs**: See [commands/](commands/) directory for comprehensive documentation
- **Configuration**: Learn about [config files and parameter management](configuration.md)
- **Examples**: Explore [CLI examples](../examples/cli/) for practical demonstrations
- **Best Practices**: Review [recommended patterns and workflows](best-practices.md)
