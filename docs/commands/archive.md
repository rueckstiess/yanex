# `yanex archive` - Archive Experiments

Move experiments to archived directory for long-term storage while keeping them accessible for reference.

## Quick Start

```bash
# Archive specific experiments
yanex archive a1b2c3d4 exp-name

# Archive all failed experiments
yanex archive -s failed

# Archive old completed experiments
yanex archive -s completed --ended-before "1 month ago"
```

## Overview

The `yanex archive` command moves experiments from the active directory to the archived directory. Archived experiments:

- **Remain accessible**: Can be viewed with `yanex show --archived` and `yanex list --archived`
- **Are preserved**: All data (metadata, results, artifacts) is kept intact
- **Free up space**: Remove clutter from the main experiments directory
- **Can be restored**: Use `yanex unarchive` to move them back

This is useful for managing completed experiments that you want to keep for reference but don't need in your active workspace.

## Command Options

### Required Arguments

- `experiment_identifiers`: Experiment IDs or names to archive (optional if using filter options)

### Filter Options

- `--ids`, `-i IDS`: Archive experiments by ID(s), comma-separated (e.g., `--ids a1,b2,c3`)
- `--status`, `-s STATUS`: Archive experiments with specific status (completed, failed, cancelled, running, staged)
- `--name`, `-n PATTERN`: Archive experiments matching name pattern (supports glob syntax like `*training*`)
- `--tag`, `-t TAG`: Archive experiments with ALL specified tags (repeatable)
- `--started-after DATE`: Archive experiments started after date/time
- `--started-before DATE`: Archive experiments started before date/time
- `--ended-after DATE`: Archive experiments ended after date/time
- `--ended-before DATE`: Archive experiments ended before date/time

### Control Options

- `--force`: Skip confirmation prompt
- `--format`, `-F FORMAT`: Output format (default, json, csv, markdown)
- `--help`: Show help message and exit

## Usage

### Archive Specific Experiments

```bash
yanex archive a1b2c3d4 exp-name
```

Archives the specified experiments by ID or name.

### Archive by Status

```bash
yanex archive -s completed
```

Archives all completed experiments. Useful for regular cleanup of finished work.

### Archive Old Experiments

```bash
yanex archive -s completed --ended-before "1 month ago"
```

Archives completed experiments that finished more than a month ago. Date formats support natural language like "yesterday", "1 week ago", or specific dates like "2024-01-01".

## Date Formats

The date/time options accept flexible formats:
- **Relative**: "1 week ago", "yesterday", "2 months ago"
- **Absolute**: "2024-01-01", "2024-01-15 14:30"
- **ISO format**: "2024-01-01T14:30:00"

## Output Format

Control output format with `--format` or `-F`:

```bash
# Default: human-readable progress
yanex archive -s completed

# JSON for scripting
yanex archive -s completed -F json

# CSV for reporting
yanex archive -s completed -F csv
```

---

**Related:**
- [`yanex unarchive`](unarchive.md) - Restore archived experiments
- [`yanex delete`](delete.md) - Permanently delete experiments
- [`yanex list`](list.md) - List experiments (use `--archived` to see archived ones)