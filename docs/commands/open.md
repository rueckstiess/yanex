# yanex open

Open an experiment's directory in your system's file manager.

## Synopsis

```bash
yanex open EXPERIMENT_ID [EXPERIMENT_ID ...]
yanex open --latest [FILTER_OPTIONS]
```

## Description

The `open` command opens the experiment directory in your system's default file manager, allowing you to quickly inspect artifacts, logs, and other experiment files. This is useful for examining output files, checking generated artifacts, or debugging experiment issues.

## Arguments

### `EXPERIMENT_ID`
One or more experiment IDs to open. Supports ID prefix matching.

## Options

### `--latest`
Open the most recently created experiment instead of specifying an ID

### Filter Options (with `--latest`)
When using `--latest`, you can filter which experiment to open:

- `--status STATUS` / `-s STATUS` - Filter by status (completed, failed, etc.)
- `--tag TAG` / `-t TAG` - Filter by tag (can be repeated)
- `--name PATTERN` / `-n PATTERN` - Filter by name pattern

## Examples

### Open by ID

```bash
# Open by full ID
yanex open abc12345

# Open by ID prefix (if unique)
yanex open abc

# Open multiple experiments
yanex open abc123 def456 ghi789
```

### Open Latest Experiment

```bash
# Open most recent experiment
yanex open --latest

# Open latest completed experiment
yanex open --latest -s completed

# Open latest with specific tag
yanex open --latest -t production

# Open latest matching name pattern
yanex open --latest -n "training-*"
```

## Platform Behavior

The `open` command uses your system's default file manager:

### macOS
Opens directory in **Finder**:
```bash
yanex open abc123
# Opens /Users/username/.yanex/experiments/abc12345 in Finder
```

### Windows
Opens directory in **File Explorer**:
```bash
yanex open abc123
# Opens C:\Users\username\.yanex\experiments\abc12345 in Explorer
```

### Linux
Opens directory in your default file manager (typically **Nautilus**, **Dolphin**, or **Thunar**):
```bash
yanex open abc123
# Opens /home/username/.yanex/experiments/abc12345 in file manager
```

## Use Cases

### Inspect Artifacts

Quickly access experiment artifacts:

```bash
# Open experiment to view saved models, plots, etc.
yanex open abc123
```

Navigate to the `artifacts/` subdirectory to see logged files.

### Check Logs

Review experiment logs and output:

```bash
# Open to inspect stdout.txt and stderr.txt
yanex open abc123
```

### Debug Failed Experiments

Investigate why an experiment failed:

```bash
# Open latest failed experiment
yanex open --latest -s failed
```

Check error logs and output files to diagnose issues.

### Review Git Patches

Inspect uncommitted changes that were captured:

```bash
# Open experiment directory
yanex open abc123

# Navigate to artifacts/git_diff.patch
```

### Access Config Files

View the exact configuration used:

```bash
# Open experiment
yanex open abc123

# Check config.yaml in the directory
```

## Directory Structure

When you open an experiment directory, you'll find:

```
abc12345/
├── metadata.json          # Experiment metadata
├── config.yaml            # Parameters used
├── metrics.json           # Logged metrics
├── script_runs.json       # Bash script execution logs (if any)
├── stdout.txt             # Standard output
├── stderr.txt             # Standard error
├── environment.txt        # Python environment info
└── artifacts/             # Logged artifacts
    ├── model.pth
    ├── training_curve.png
    └── git_diff.patch     # Uncommitted changes (if any)
```

See [Experiment Structure](../experiment-structure.md) for complete details.

## Tips

### Quick Inspection Workflow

```bash
# 1. Find experiments of interest
yanex list -s completed -t experiment

# 2. Compare to find best
yanex compare -t experiment

# 3. Open best experiment directory
yanex open abc123
```

### Multiple Experiments

Open several experiments at once for side-by-side inspection:

```bash
yanex open abc123 def456 ghi789
```

This opens three file manager windows, useful for comparing artifacts visually.

### Copy Files Out

After opening the directory, you can:
- Copy artifacts to another location
- Share specific files with team members
- Move models to production deployment

## Error Handling

### Experiment Not Found

```bash
$ yanex open xyz999
Error: Experiment xyz999 not found
```

Verify the ID with `yanex list` first.

### Multiple Matches

If an ID prefix matches multiple experiments:

```bash
$ yanex open abc
Error: Multiple experiments match 'abc': abc123, abc456
Use a more specific prefix
```

Provide more characters to uniquely identify the experiment.

### No File Manager Available

On some minimal Linux systems without a graphical file manager:

```bash
$ yanex open abc123
Warning: Could not open file manager
Directory: /home/username/.yanex/experiments/abc12345
```

The directory path is printed so you can navigate manually.

## Related Commands

- [`show`](show.md) - Display experiment details in terminal
- [`list`](list.md) - Find experiments to open
- [`compare`](compare.md) - Compare before selecting which to open

## See Also

- [Experiment Structure](../experiment-structure.md) - Directory layout and files
- [CLI Commands Overview](../cli-commands.md)
- [Best Practices](../best-practices.md) - File management patterns
