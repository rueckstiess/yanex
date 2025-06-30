# CLI Commands Reference

Complete reference for all Yanex command-line tools.

## Core Commands

### `yanex run` - Execute Experiments

Run Python scripts as tracked experiments.

```bash
# Basic usage
yanex run script.py

# With parameter overrides
yanex run script.py --param learning_rate=0.01 --param epochs=100

# With metadata
yanex run script.py --name "baseline-model" --description "Initial training run"

# With tags
yanex run script.py --tag baseline --tag production
```

**Options:**
- `--param KEY=VALUE` - Override configuration parameters
- `--name NAME` - Set experiment name
- `--description DESC` - Add experiment description  
- `--tag TAG` - Add tags (can be used multiple times)
- `--config PATH` - Use specific config file
- `--force-dirty` - Allow running with uncommitted Git changes

[**→ Detailed Documentation**](commands/run.md)

---

### `yanex list` - List Experiments

List and filter your experiments.

```bash
# List all experiments
yanex list

# Filter by status
yanex list --status completed
yanex list --status running

# Filter by time
yanex list --started-after "1 week ago"
yanex list --started-before "2023-12-01"

# Filter by tags
yanex list --tag production --tag baseline

# Filter by name pattern
yanex list --name "*baseline*"

# Combine filters
yanex list --status completed --tag production --started-after "1 month ago"
```

**Options:**
- `--status STATUS` - Filter by experiment status
- `--name PATTERN` - Filter by name (supports wildcards)
- `--tag TAG` - Filter by tags (multiple allowed)
- `--started-after DATE` - Show experiments started after date
- `--started-before DATE` - Show experiments started before date
- `--ended-after DATE` - Show experiments ended after date  
- `--ended-before DATE` - Show experiments ended before date
- `--archived` - Include archived experiments
- `--limit N` - Limit number of results

[**→ Detailed Documentation**](commands/list.md)

---

### `yanex show` - Show Experiment Details

Display detailed information about specific experiments.

```bash
# Show by ID
yanex show abc12345

# Show by name (if unique)
yanex show my-experiment

# Show multiple experiments
yanex show exp1 exp2 exp3

# Include additional details
yanex show abc12345 --files --git-info
```

**Options:**
- `--files` - Show tracked files and artifacts
- `--git-info` - Show Git commit details
- `--raw` - Show raw JSON metadata

[**→ Detailed Documentation**](commands/show.md)

---

### `yanex compare` - Compare Experiments

Interactive comparison of experiment parameters and results.

```bash
# Interactive comparison table (all experiments)
yanex compare

# Compare specific experiments
yanex compare exp1 exp2 exp3

# Filter experiments to compare
yanex compare --status completed --tag baseline

# Show only different columns
yanex compare --only-different

# Limit to specific parameters/metrics
yanex compare --params learning_rate,epochs --metrics accuracy,loss

# Export to CSV
yanex compare --export results.csv

# Static table output
yanex compare --no-interactive
```

**Options:**
- `--status STATUS` - Filter by experiment status
- `--name PATTERN` - Filter by name pattern
- `--tag TAG` - Filter by tags
- `--started-after DATE` - Filter by start time
- `--params LIST` - Show only specified parameters
- `--metrics LIST` - Show only specified metrics
- `--only-different` - Hide columns with identical values
- `--export PATH` - Export to CSV file
- `--no-interactive` - Show static table instead
- `--max-rows N` - Limit number of experiments

**Interactive Controls:**
- `↑/↓` or `j/k` - Navigate rows
- `←/→` or `h/l` - Navigate columns
- `s/S` - Sort ascending/descending
- `1/2` - Numeric sort ascending/descending
- `r/R` - Reset/reverse sort
- `e` - Export to CSV
- `?` - Show help
- `q` - Quit

[**→ Detailed Documentation**](commands/compare.md)

---

## Management Commands

### `yanex archive` - Archive Experiments

Move experiments to archived storage.

```bash
# Archive by ID
yanex archive abc12345

# Archive multiple experiments
yanex archive exp1 exp2 exp3

# Archive by filter criteria
yanex archive --status failed --started-before "1 month ago"

# Archive with confirmation
yanex archive --status completed --confirm
```

[**→ Detailed Documentation**](commands/archive.md)

---

### `yanex unarchive` - Restore Experiments

Restore experiments from archived storage.

```bash
# Restore by ID
yanex unarchive abc12345

# Restore multiple experiments
yanex unarchive exp1 exp2 exp3

# Restore by criteria
yanex unarchive --tag important --confirm
```

[**→ Detailed Documentation**](commands/unarchive.md)

---

### `yanex delete` - Delete Experiments

Permanently delete experiments (cannot be undone).

```bash
# Delete by ID (with confirmation)
yanex delete abc12345

# Delete multiple experiments
yanex delete exp1 exp2 exp3 --confirm

# Delete by criteria
yanex delete --status failed --started-before "6 months ago" --confirm
```

**⚠️ Warning:** Deletion is permanent and cannot be undone!

[**→ Detailed Documentation**](commands/delete.md)

---

### `yanex update` - Update Experiment Metadata

Modify experiment names, descriptions, and tags.

```bash
# Update experiment name
yanex update abc12345 --name "new-experiment-name"

# Update description
yanex update abc12345 --description "Updated description"

# Add tags
yanex update abc12345 --add-tag production --add-tag validated

# Remove tags
yanex update abc12345 --remove-tag draft

# Replace all tags
yanex update abc12345 --tag production --tag final
```

[**→ Detailed Documentation**](commands/update.md)

---

## Global Options

These options work with all commands:

- `--help` - Show command help
- `--verbose` - Enable verbose output
- `--quiet` - Suppress non-essential output
- `--config PATH` - Use specific config file

## Date/Time Formats

Yanex accepts flexible date/time formats:

```bash
# Absolute dates
--started-after "2023-12-01"
--started-after "2023-12-01 15:30"

# Relative dates
--started-after "1 week ago"
--started-after "3 days ago"
--started-after "yesterday"

# Keywords
--started-after "today"
--started-after "last week"
--started-after "last month"
```

## Examples

### Common Workflows

```bash
# Daily workflow: check today's experiments
yanex list --started-after today

# Weekly review: compare recent successful experiments
yanex compare --status completed --started-after "1 week ago"

# Cleanup: archive old failed experiments
yanex archive --status failed --started-before "1 month ago" --confirm

# Research: find experiments with specific parameters
yanex list --tag hyperparameter-search | grep "learning_rate=0.01"

# Analysis: export comparison data for external analysis
yanex compare --tag production --export production_results.csv
```

### Parameter Sweeps

```bash
# Run parameter sweep
for lr in 0.001 0.01 0.1; do
    yanex run train.py --param learning_rate=$lr --tag lr-sweep
done

# Compare results
yanex compare --tag lr-sweep --only-different

# Export for analysis
yanex compare --tag lr-sweep --export lr_sweep_results.csv
```

---

**Next:** [Python API Reference](python-api.md)