# `yanex delete` - Permanently Delete Experiments

Permanently delete experiments and all associated data.

## Quick Start

```bash
# Delete specific experiments
yanex delete a1b2c3d4 exp-name

# Delete all failed experiments
yanex delete -s failed

# Delete old archived experiments
yanex delete -a --ended-before "6 months ago"
```

## Overview

⚠️ **WARNING: This operation cannot be undone!**

The `yanex delete` command permanently removes experiments and all their associated data including:
- Metadata (name, tags, description, status, timestamps)
- Configuration parameters 
- Results and metrics
- Artifacts and output files
- Environment and Git information

Unlike archiving, deleted experiments cannot be recovered. Use this command only when you're certain you no longer need the experiment data.

## Command Options

### Required Arguments

- `experiment_identifiers`: Experiment IDs or names to delete (optional if using filter options)

### Filter Options

- `--ids`, `-i IDS`: Delete experiments by ID(s), comma-separated (e.g., `--ids a1,b2,c3`)
- `--status`, `-s STATUS`: Delete experiments with specific status (completed, failed, cancelled, running, staged)
- `--name`, `-n PATTERN`: Delete experiments matching name pattern (supports glob syntax like `*test*`)
- `--tag`, `-t TAG`: Delete experiments with ALL specified tags (repeatable)
- `--started-after DATE`: Delete experiments started after date/time
- `--started-before DATE`: Delete experiments started before date/time
- `--ended-after DATE`: Delete experiments ended after date/time
- `--ended-before DATE`: Delete experiments ended before date/time

### Location Options

- `--archived`, `-a`: Delete from archived experiments (default: delete from regular experiments)

### Control Options

- `--force`: Skip confirmation prompt
- `--format`, `-F FORMAT`: Output format (default, json, csv, markdown)
- `--help`: Show help message and exit

## Usage

### Delete Specific Experiments

```bash
yanex delete a1b2c3d4 exp-name
```

Permanently deletes the specified experiments by ID or name.

### Delete Failed Experiments

```bash
yanex delete -s failed
```

Removes all failed experiments to clean up unsuccessful runs.

### Delete Old Archived Experiments

```bash
yanex delete -a --ended-before "6 months ago"
```

Permanently removes archived experiments older than 6 months to free up storage space.

## Safety Features

- **Double confirmation**: Bulk deletions require additional confirmation
- **Clear warnings**: Multiple warnings about permanent data loss
- **Preview**: Shows which experiments will be deleted before proceeding
- **Selective targeting**: Can target regular or archived experiments separately

## Date Formats

The date/time options accept flexible formats:
- **Relative**: "1 week ago", "yesterday", "2 months ago"
- **Absolute**: "2024-01-01", "2024-01-15 14:30"
- **ISO format**: "2024-01-01T14:30:00"

## Output Format

Control output format with `--format` or `-F`:

```bash
# Default: human-readable progress with warnings
yanex delete -s failed

# JSON for scripting
yanex delete -s failed -F json --force

# CSV for reporting
yanex delete -s failed -F csv --force
```

---

**Related:**
- [`yanex archive`](archive.md) - Archive experiments (reversible alternative to deletion)
- [`yanex list`](list.md) - List experiments before deleting