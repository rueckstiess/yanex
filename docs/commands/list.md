# `yanex list` - List and Filter Experiments

List experiments with powerful filtering and search capabilities.

## Quick Start

```bash
# List all experiments
yanex list

# Filter by status
yanex list --status completed

# Recent experiments
yanex list --started-after "1 week ago"
```

## Basic Usage

```bash
# List all experiments (most recent first)
yanex list

# Include archived experiments
yanex list --archived

# Limit number of results
yanex list --limit 10
```

## Filtering Options

### By Status

```bash
yanex list --status created     # Not yet started
yanex list --status running     # Currently executing
yanex list --status completed   # Finished successfully
yanex list --status failed      # Failed with error
yanex list --status cancelled   # Manually stopped
```

### By Name Pattern

```bash
# Exact match
yanex list --name "baseline-model"

# Wildcard patterns
yanex list --name "*baseline*"
yanex list --name "model-v*"
yanex list --name "*-prod"
```

### By Tags

```bash
# Single tag
yanex list --tag production

# Multiple tags (experiments must have ALL tags)
yanex list --tag baseline --tag validated

# Tag patterns
yanex list --tag "*test*"
```

### By Time Range

```bash
# Started after date
yanex list --started-after "2023-12-01"
yanex list --started-after "1 week ago"
yanex list --started-after "yesterday"

# Started before date
yanex list --started-before "2023-12-31"
yanex list --started-before "1 month ago"

# Date range
yanex list --started-after "2023-12-01" --started-before "2023-12-31"

# Ended after/before
yanex list --ended-after "yesterday"
yanex list --ended-before "1 week ago"
```

## Examples

### Daily Workflow

```bash
# Check today's experiments
yanex list --started-after today

# Recent successful experiments
yanex list --status completed --started-after "3 days ago"

# Failed experiments that need attention
yanex list --status failed --started-after "1 week ago"
```

### Project Management

```bash
# Production experiments
yanex list --tag production --status completed

# Development experiments
yanex list --tag development --started-after "1 week ago"

# Specific model family
yanex list --name "*resnet*" --status completed
```

### Research Analysis

```bash
# Hyperparameter sweep results
yanex list --tag "hp-sweep" --status completed

# Ablation studies
yanex list --tag ablation --started-after "1 month ago"

# Baseline comparisons
yanex list --tag baseline
```

## Output Format

The list shows key experiment information:

```
ID       Name                Started              Duration   Status     Tags
abc1234  baseline-model      2023-12-15 10:30:00  01:23:45   completed  baseline, prod
def5678  improved-model      2023-12-15 14:20:00  02:10:30   completed  improved, prod
ghi9012  ablation-study      2023-12-15 16:45:00  00:45:20   failed     ablation
```

### Columns

- **ID**: Unique experiment identifier (8 characters)
- **Name**: Human-readable experiment name
- **Started**: Experiment start timestamp
- **Duration**: Runtime (HH:MM:SS, or "running" for active experiments)
- **Status**: Current experiment status
- **Tags**: Comma-separated list of tags

## Advanced Usage

### Combining Filters

```bash
# Complex filtering
yanex list \
  --status completed \
  --tag production \
  --started-after "1 month ago" \
  --name "*model*" \
  --limit 20
```

### Integration with Other Commands

```bash
# Get IDs for comparison
EXPERIMENTS=$(yanex list --tag baseline --status completed | tail -n +2 | cut -d' ' -f1)
yanex compare $EXPERIMENTS

# Archive old failed experiments
yanex list --status failed --started-before "3 months ago" | \
  tail -n +2 | cut -d' ' -f1 | \
  xargs yanex archive --confirm
```

### Shell Scripting

```bash
#!/bin/bash
# Find best performing experiments
yanex list --tag hyperparameter-search --status completed | \
  while read id name started duration status tags; do
    accuracy=$(yanex show $id --raw | jq -r '.results.accuracy // 0')
    echo "$id $accuracy $name"
  done | sort -k2 -nr | head -5
```

---

**Related:**
- [`yanex show`](show.md) - Detailed experiment information
- [`yanex compare`](compare.md) - Compare experiments
- [Filtering Guide](../filtering.md)