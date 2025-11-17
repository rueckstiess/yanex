# `yanex id` - Get Experiment IDs

Get experiment IDs matching filter criteria for use in bash substitution and automation.

## Quick Start

```bash
# Get IDs as comma-separated list (default)
yanex id --tag training

# Use in bash substitution
yanex run evaluate.py --depends-on model=$(yanex id --tag training)

# Get IDs one per line
yanex id --status completed --format newline
```

## Basic Usage

```bash
# Default CSV format with double quotes
yanex id
# Output: "abc12345,def67890,ghi13579"

# JSON array format
yanex id --format json
# Output: ["abc12345", "def67890", "ghi13579"]

# Newline-separated format
yanex id --format newline
# Output:
# abc12345
# def67890
# ghi13579
```

## Output Formats

### CSV Format (Default)

The default format returns a comma-separated list wrapped in double quotes, perfect for bash substitution:

```bash
yanex id --tag training
# Output: "abc12345,def67890"

# Use directly in commands
yanex run evaluate.py --depends-on model=$(yanex id --tag training)
```

### Newline Format

One ID per line, useful for processing in scripts:

```bash
yanex id --format newline -s completed
# Output:
# abc12345
# def67890
# ghi13579

# Use in bash loops
for id in $(yanex id --format newline -t production); do
  yanex show $id
done
```

### JSON Format

JSON array format for programmatic processing:

```bash
yanex id --format json -s failed
# Output: ["abc12345", "def67890"]

# Use with jq
yanex id --format json -s failed | jq '.[0]'
```

## Filtering Options

The `id` command supports all the same filtering options as [`list`](list.md):

### By Status

```bash
# Completed experiments
yanex id -s completed

# Failed experiments
yanex id -s failed

# Staged experiments
yanex id -s staged

# Running experiments
yanex id -s running
```

### By Tags

```bash
# Single tag
yanex id -t production

# Multiple tags (experiments must have ALL tags)
yanex id -t training -t baseline
```

### By Name Pattern

```bash
# Exact match
yanex id -n "baseline-model"

# Wildcard patterns
yanex id -n "*training*"
yanex id -n "model-v*"
```

### By Script Pattern

```bash
# Specific script
yanex id --script train.py

# Pattern matching
yanex id --script "*prep*"
```

### By Time Range

```bash
# Started after date
yanex id --started-after "2025-01-01"
yanex id --started-after "1 week ago"

# Started before date
yanex id --started-before "2025-12-31"

# Date range
yanex id --started-after "1 month ago" --started-before "yesterday"

# Ended after/before
yanex id --ended-after "yesterday"
yanex id --ended-before "1 week ago"
```

### By Archive Status

```bash
# Get archived experiment IDs
yanex id --archived
yanex id -a
```

### Limit Results

```bash
# Get last 5 experiment IDs
yanex id -l 5

# Get last 10 completed experiments
yanex id -s completed -l 10
```

## Bash Substitution Examples

The `id` command is designed for use in bash substitution, particularly for dependency tracking:

### Dependency Tracking

```bash
# Run evaluation depending on training experiments
yanex run evaluate.py --depends-on model=$(yanex id --tag training)

# Run training depending on staged data prep
yanex run train.py -D dataprep=$(yanex id --status staged --script dataprep.py)

# Multiple dependencies
yanex run ensemble.py \
  --depends-on model1=$(yanex id --name "model-v1") \
  --depends-on model2=$(yanex id --name "model-v2")
```

### Batch Operations

```bash
# Archive all failed experiments
for id in $(yanex id --format newline -s failed); do
  yanex archive $id
done

# Show details of all production experiments
for id in $(yanex id --format newline -t production); do
  yanex show $id
done

# Delete old failed experiments
yanex delete $(yanex id -s failed --ended-before "1 month ago")
```

### Complex Workflows

```bash
# Get latest completed training run
TRAINING_ID=$(yanex id -s completed -t training -l 1)
echo "Latest training: $TRAINING_ID"

# Use in downstream tasks
yanex run evaluate.py --depends-on model=$TRAINING_ID

# Run sweep depending on all data prep experiments
DATAPREP_IDS=$(yanex id --script dataprep.py -s completed)
yanex run train.py --depends-on data=$DATAPREP_IDS --param "lr=0.001,0.01,0.1" --stage
```

## Format Option

Use `--format` or `-f` to specify output format:

```bash
yanex id --format csv       # Default: "id1,id2,id3"
yanex id --format newline   # One per line
yanex id --format json      # ["id1", "id2", "id3"]

# Short form
yanex id -f csv
yanex id -f newline
yanex id -f json
```

## Short Aliases

All filtering options have convenient short aliases:

- `-s` for `--status`
- `-n` for `--name`
- `-t` for `--tag`
- `-c` for `--script`
- `-a` for `--archived`
- `-l` for `--limit`
- `-f` for `--format`

## Examples

### Daily Workflow

```bash
# Get today's experiment IDs
yanex id --started-after today

# Get recent failures
yanex id -s failed --started-after "3 days ago"

# Get production experiments from last week
yanex id -t production --started-after "1 week ago"
```

### Project Automation

```bash
# Get all training experiments for evaluation
MODELS=$(yanex id -t training -s completed)
yanex run evaluate.py --depends-on models=$MODELS

# Get staged data prep experiments
DATAPREP=$(yanex id --script dataprep.py -s staged)
echo "Staged dataprep: $DATAPREP"

# Get specific model versions
V1=$(yanex id -n "model-v1" -s completed -l 1)
V2=$(yanex id -n "model-v2" -s completed -l 1)
yanex compare $V1 $V2
```

### Advanced Filtering

```bash
# Combine multiple filters
yanex id -s completed -t production --started-after "1 month ago" -l 20

# Get experiment IDs for specific time window
yanex id --started-after "2025-01-01" --started-before "2025-01-31"

# Get unnamed experiments
yanex id -n ""

# Get experiments by script pattern
yanex id --script "*training*" -s completed
```

## Tips

1. **Use CSV format for dependencies**: The default CSV format is designed to work seamlessly with the `--depends-on` flag
2. **Use newline format for loops**: When processing IDs in bash loops, use `--format newline`
3. **Combine with limit**: Use `-l` to get just the most recent experiments
4. **Verify filters first**: Test your filters with `yanex list` before using in `yanex id`
5. **Check results**: The ID command never fails - it returns an empty result if no experiments match

## Empty Results

When no experiments match the filters:

```bash
yanex id --format csv       # ""
yanex id --format newline   # (empty output)
yanex id --format json      # []
```

---

**Related:**
- [`yanex list`](list.md) - List experiments with details
- [`yanex show`](show.md) - Show experiment details
- [`yanex run`](run.md) - Run experiments with dependencies
