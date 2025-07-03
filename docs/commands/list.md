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
yanex list --status running     # Currently executing
yanex list --status completed   # Finished successfully
yanex list --status failed      # Failed with error
yanex list --status cancelled   # Manually stopped
yanex list --status staged      # Staged experiments
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

---

**Related:**
- [`yanex show`](show.md) - Detailed experiment information
- [`yanex compare`](compare.md) - Compare experiments