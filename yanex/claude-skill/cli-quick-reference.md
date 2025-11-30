# CLI Quick Reference

## yanex run

Execute experiments with tracking.

**Note**: Run in background (use Bash tool's `run_in_background` parameter) to avoid blocking on long experiments.

```bash
yanex run script.py [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--config PATH` | `-c` | Config file (repeatable, merged left-to-right) |
| `--param KEY=VALUE` | `-p` | Override parameter (repeatable) |
| `--depends-on ID` | `-D` | Dependency with optional slot: `-D data=abc123` |
| `--name NAME` | `-n` | Experiment name |
| `--tag TAG` | `-t` | Add tag (repeatable) |
| `--description DESC` | | Experiment description |
| `--clone-from ID` | | Clone config from existing experiment |
| `--parallel N` | `-j` | Parallel workers (0=auto-detect CPUs) |
| `--stage` | | Stage for later execution |
| `--staged` | | Run all staged experiments |
| `--dry-run` | | Validate without executing |

### Sweep Syntax

```bash
# Comma-separated
-p "lr=0.001,0.01,0.1"

# Range (integers)
-p "epochs=range(10, 100, 10)"

# Linspace (evenly spaced)
-p "dropout=linspace(0.1, 0.5, 5)"

# Logspace (log-spaced, for learning rates)
-p "lr=logspace(-4, -1, 10)"
```

### Dependency Slots

```bash
# Named slot (recommended)
-D data=abc12345
-D model=def67890

# Multiple dependencies
-D data=abc123 -D model=def456

# Dependency sweep
-D "data=abc123,def456"
```

## yanex list

List and filter experiments.

```bash
yanex list [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--status STATUS` | `-s` | Filter by status: running, completed, failed, cancelled, staged |
| `--name PATTERN` | `-n` | Filter by name pattern (glob) |
| `--tag TAG` | `-t` | Filter by tag (repeatable, AND logic) |
| `--started-after DATE` | | Filter by start time |
| `--started-before DATE` | | Filter by start time |
| `--ended-after DATE` | | Filter by end time |
| `--ended-before DATE` | | Filter by end time |
| `--archived` | `-a` | Include archived experiments |
| `--limit N` | `-l` | Max results |

### Examples

```bash
yanex list -s completed -n "yelp-2-*"
yanex list -t training -t sweep --started-after "1 week ago"
yanex list -s failed -l 10
```

## yanex get

Extract specific field values from experiments. Optimized for AI agents and bash scripting.

```bash
yanex get FIELD [EXPERIMENT_ID] [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--status STATUS` | `-s` | Filter by status |
| `--name PATTERN` | `-n` | Filter by name pattern |
| `--tag TAG` | `-t` | Filter by tag (repeatable) |
| `--limit N` | `-l` | Max results |
| `--csv` | | Comma-separated output (for bash substitution) |
| `--json` | `-j` | JSON output |
| `--no-id` | | Omit experiment ID prefix in multi-experiment output |
| `--default VALUE` | | Value for missing fields (default: `[not_found]`) |
| `--head N` | | Return first N lines (stdout/stderr only) |
| `--tail N` | | Return last N lines (stdout/stderr only) |
| `--follow` | `-f` | Follow output in real-time (stdout/stderr, single experiment only) |

### Field Paths

| Field | Description |
|-------|-------------|
| `id` | Experiment ID |
| `name` | Experiment name |
| `status` | Status (running, completed, failed, etc.) |
| `tags` | List of tags |
| `stdout` | Standard output (supports --head/--tail/--follow) |
| `stderr` | Standard error (supports --head/--tail/--follow) |
| `cli-command` | Original CLI invocation (with sweep syntax if applicable) |
| `run-command` | Reproducible command (with resolved parameter values) |
| `params` | List available parameter names |
| `params.<key>` | Specific parameter value (e.g., `params.lr`) |
| `metrics` | List available metric names |
| `metrics.<key>` | Last logged metric value (e.g., `metrics.accuracy`) |
| `dependencies` | Dependency slot=id pairs |
| `git.branch` | Git branch name |
| `git.commit_hash` | Git commit hash |
| `created_at` | Creation timestamp |
| `started_at` | Start timestamp |
| `completed_at` | Completion timestamp |

### Single Experiment Mode

```bash
yanex get status abc12345              # Get status
yanex get params.lr abc12345           # Get learning rate parameter
yanex get metrics.accuracy abc12345    # Get last logged accuracy
yanex get params abc12345              # List available parameter names
yanex get metrics abc12345             # List available metric names
yanex get dependencies abc12345        # Get dependencies as slot=id pairs
yanex get tags abc12345 --json         # Get tags as JSON array
yanex get stdout abc12345              # Get full stdout
yanex get stdout abc12345 --tail 50    # Get last 50 lines of stdout
yanex get stdout abc12345 --head 10    # Get first 10 lines of stdout
yanex get stdout abc12345 --head 5 --tail 5  # First 5 and last 5 lines
yanex get stdout abc12345 -f           # Follow stdout in real-time
yanex get stdout abc12345 --tail 20 -f # Show last 20 lines then follow
yanex get stderr abc12345              # Get stderr output
yanex get cli-command abc12345         # Get original CLI invocation (with sweep syntax)
yanex get run-command abc12345         # Get reproducible command (resolved values)
```

### Multi-Experiment Mode (with filters)

```bash
yanex get id -s completed              # Get IDs of completed experiments
yanex get id -n "train-*" -l 5         # Get IDs of matching experiments
yanex get params.lr -s completed       # Get learning rates from all completed
yanex get status -t sweep              # Get status of all experiments with tag
yanex get stdout -s running --tail 5   # Check progress of running experiments
```

### Output Formats

```bash
# Default: human-readable
yanex get id -s completed
# abc12345
# def67890

# CSV: comma-separated (no newline, for bash substitution)
yanex get id -s completed --csv
# abc12345,def67890

# JSON: machine-readable
yanex get params.lr -s completed --json
# [{"id": "abc12345", "value": 0.001}, {"id": "def67890", "value": 0.01}]

# Multi-experiment stdout/stderr (header-separated)
yanex get stdout -s running --tail 5
# [experiment: abc12345]
# Epoch 10/100, loss=0.234
# ...
#
# [experiment: def67890]
# Processing batch 50/200
# ...
```

### Bash Substitution for Dynamic Sweeps

```bash
# Run training on multiple data prep experiments
yanex run train.py -D data=$(yanex get id -n "*-prep-*" --csv)

# Build parameter sweep from previous experiments
yanex run train.py -p lr=$(yanex get params.lr -s completed --csv)

# Chain experiments dynamically
DATA_ID=$(yanex get id -n "data-prep" -s completed -l 1)
yanex run train.py -D data=$DATA_ID
```

## yanex show

Show experiment details.

```bash
yanex show EXPERIMENT_ID [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--show-metric METRICS` | Show specific metrics (comma-separated) |
| `--archived` | Include archived in search |

### Examples

```bash
yanex show abc12345
yanex show "experiment-name"
yanex show abc123 --show-metric "accuracy,loss"
```

## yanex compare

Interactive experiment comparison.

```bash
yanex compare [OPTIONS]
```

| Option | Short | Description |
|--------|-------|-------------|
| `--only-different` | | Show only differing columns |
| `--tag TAG` | `-t` | Filter by tag |
| `--status STATUS` | `-s` | Filter by status |
| `--name PATTERN` | `-n` | Filter by name |

## yanex archive / unarchive

Move experiments to/from archive. Archiving is reversible.

```bash
yanex archive [OPTIONS]
yanex unarchive [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--force` | Skip confirmation prompt |

Uses same filters as `yanex list`.

### Examples

```bash
yanex archive -s failed --started-before "1 month ago"
yanex archive -t experiment -t test --force  # Skip confirmation
yanex unarchive abc12345
```

## yanex delete

Permanently delete experiments. **This action cannot be undone.**

```bash
yanex delete [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--force` | Skip confirmation prompt |

Uses same filters as `yanex list`.

### Examples

```bash
yanex delete abc12345 def67890
yanex delete -s failed --started-before "3 months ago" --force
```

## yanex update

Update experiment metadata.

```bash
yanex update EXPERIMENT_ID [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--name NAME` | Update name |
| `--description DESC` | Update description |
| `--tag TAG` | Add tag(s) |

## yanex open

Open experiment directory in file explorer.

```bash
yanex open EXPERIMENT_ID
```

## yanex ui

Launch web UI server.

```bash
yanex ui [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--port PORT` | Server port (default: 8000) |
| `--host HOST` | Server host (default: localhost) |

## Date/Time Formats

The CLI supports natural language dates:

- `today`, `yesterday`
- `1 week ago`, `2 days ago`
- `1 month ago`, `3 months ago`
- `2025-01-15` (ISO format)

## Status Values

- `running` - Currently executing
- `completed` - Finished successfully
- `failed` - Failed with error
- `cancelled` - Manually stopped
- `staged` - Staged for later execution
