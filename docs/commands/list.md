# `yanex list` - List and Filter Experiments

List experiments with powerful filtering and search capabilities.

## Quick Start

```bash
# List all experiments
yanex list

# Filter by status
yanex list -s completed

# Recent experiments
yanex list --started-after "1 week ago"
```

## Basic Usage

```bash
# List all experiments (most recent first)
yanex list

# Include archived experiments
yanex list -a

# Limit number of results
yanex list -l 10
```

## Filtering Options

### By Status

Filter using `--status` or `-s`:

```bash
yanex list -s running     # Currently executing
yanex list -s completed   # Finished successfully
yanex list -s failed      # Failed with error
yanex list -s cancelled   # Manually stopped
yanex list -s staged      # Staged experiments
```

### By Name Pattern

Filter using `--name` or `-n`:

```bash
# Exact match
yanex list -n "baseline-model"

# Wildcard patterns
yanex list -n "*baseline*"
yanex list -n "model-v*"
yanex list -n "*-prod"
```

### By Tags

Filter using `--tag` or `-t`:

```bash
# Single tag
yanex list -t production

# Multiple tags (experiments must have ALL tags)
yanex list -t baseline -t validated
```

### By Experiment IDs

Filter using `--ids` or `-i`:

```bash
# Single ID
yanex list --ids abc12345

# Multiple IDs (comma-separated)
yanex list --ids abc123,def456,ghi789
yanex list -i a1,b2,c3

# Combined with other filters
yanex list --ids abc123,def456 -s completed
```

This is particularly useful for:
- Viewing a specific set of experiments from a sweep
- Filtering based on output from other commands (e.g., lineage queries)

```bash
# List all upstream dependencies of an experiment
yanex list --ids $(yanex get upstream d7742130 -F sweep)
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
yanex list -s completed --started-after "3 days ago"

# Failed experiments that need attention
yanex list -s failed --started-after "1 week ago"
```

### Project Management

```bash
# Production experiments
yanex list -t production -s completed

# Development experiments
yanex list -t development --started-after "1 week ago"

# Specific model family
yanex list -n "*resnet*" -s completed
```

## Output Format

Control output format with `--format` or `-F`:

```bash
# Default: human-readable table
yanex list

# JSON output for scripting
yanex list -F json

# CSV for spreadsheets
yanex list -F csv

# Markdown for documentation
yanex list -F markdown
```

**Available formats:**

| Format | Description |
|--------|-------------|
| `default` | Rich terminal table with colors and formatting |
| `json` | JSON array of experiment objects |
| `csv` | Comma-separated values with headers |
| `markdown` | GitHub-flavored markdown table |

## Short Aliases

All options have convenient short aliases:

- `-i` for `--ids`
- `-s` for `--status`
- `-n` for `--name`
- `-t` for `--tag`
- `-a` for `--archived`
- `-l` for `--limit`
- `-F` for `--format`

---

**Related:**
- [`yanex show`](show.md) - Detailed experiment information
- [`yanex compare`](compare.md) - Compare experiments
- [`yanex get`](get.md) - Get specific field values