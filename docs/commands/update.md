# `yanex update` - Update Experiment Metadata

Modify experiment metadata including names, descriptions, status, and tags.

## Quick Start

```bash
# Update single experiment
yanex update a1b2c3d4 --set-name "New Name" --set-description "Updated description"

# Add/remove tags
yanex update exp-name --add-tag production --remove-tag testing

# Bulk update by filter
yanex update -s failed --set-description "Failed batch run"
```

## Overview

The `yanex update` command allows you to modify experiment metadata without affecting the actual experiment data (configuration, results, artifacts). You can update:

- **Name**: Change or clear experiment names
- **Description**: Update or clear experiment descriptions  
- **Status**: Manually set experiment status
- **Tags**: Add or remove tags for better organization

Updates can be applied to single experiments or in bulk using filter criteria.

## Command Options

### Required Arguments

- `experiment_identifiers`: Experiment IDs or names to update (optional if using filter options)

### Update Options

- `--set-name NAME`: Set experiment name (use empty string to clear)
- `--set-description DESC`: Set experiment description (use empty string to clear)
- `--set-status STATUS`: Set experiment status (completed, failed, cancelled, running, staged)
- `--add-tag TAG`: Add tag to experiment(s) (repeatable)
- `--remove-tag TAG`: Remove tag from experiment(s) (repeatable)

### Filter Options (for bulk updates)

- `--status`, `-s STATUS`: Filter experiments by status
- `--name`, `-n PATTERN`: Filter experiments by name pattern (supports glob syntax)
- `--tag`, `-t TAG`: Filter experiments with ALL specified tags (repeatable)
- `--started-after DATE`: Filter experiments started after date/time
- `--started-before DATE`: Filter experiments started before date/time
- `--ended-after DATE`: Filter experiments ended after date/time
- `--ended-before DATE`: Filter experiments ended before date/time

### Control Options

- `--archived`, `-a`: Update archived experiments
- `--force`: Skip confirmation prompt for bulk operations
- `--dry-run`: Show what would be updated without making changes
- `--format`, `-F FORMAT`: Output format (default, json, csv, markdown)
- `--help`: Show help message and exit

## Usage

### Update Single Experiment

```bash
yanex update a1b2c3d4 --set-name "Baseline Model v2" --set-description "Updated baseline with new architecture"
```

Updates the name and description of a specific experiment.

### Manage Tags

```bash
yanex update exp-name --add-tag production --add-tag validated --remove-tag testing
```

Adds "production" and "validated" tags while removing the "testing" tag.

### Bulk Updates

```bash
yanex update -s failed --set-description "Failed during data preprocessing" --add-tag needs-review
```

Updates all failed experiments with a common description and adds a review tag.

### Preview Changes

```bash
yanex update a1b2c3d4 --set-name "New Name" --dry-run
```

Shows what would be changed without actually making the update.

## Date Formats

The date/time filter options accept flexible formats:
- **Relative**: "1 week ago", "yesterday", "2 months ago"
- **Absolute**: "2024-01-01", "2024-01-15 14:30"
- **ISO format**: "2024-01-01T14:30:00"

## Output Format

Control output format with `--format` or `-F`:

```bash
# Default: human-readable progress
yanex update a1b2c3d4 --set-name "New Name"

# JSON for scripting
yanex update -s completed --add-tag reviewed -F json --force

# CSV for reporting
yanex update -s completed --add-tag reviewed -F csv --force
```

---

**Related:**
- [`yanex show`](show.md) - View current experiment metadata
- [`yanex list`](list.md) - List experiments to identify candidates for updates