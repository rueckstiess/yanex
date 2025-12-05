# `yanex get` - Get Specific Field Values

Retrieve specific field values from experiments for scripting, automation, and AI agent workflows.

## Quick Start

```bash
# Get status of a specific experiment
yanex get status abc123

# Get learning rate parameter
yanex get params.lr abc123

# Get last accuracy metric
yanex get metrics.accuracy abc123

# Get IDs of all completed experiments (for scripting)
yanex get id -s completed -F sweep
```

## Overview

The `yanex get` command extracts specific field values from experiments. It's designed for:

- **Scripting**: Extract values for use in shell scripts and pipelines
- **AI agents**: Machine-readable output for automated workflows
- **Bash substitution**: Build dynamic parameter sweeps from previous experiments
- **Debugging**: Quickly check stdout/stderr output from running or completed experiments

### Two Operation Modes

**1. Single Experiment Mode**

Specify an experiment by ID or name:

```bash
yanex get status abc123
yanex get params.lr "my-experiment"
```

**2. Multi-Experiment Mode with Filters**

Use filter options to query multiple experiments:

```bash
yanex get id -s completed              # All completed experiment IDs
yanex get params.lr -t baseline        # Learning rates from baseline experiments
yanex get metrics.accuracy -n "v2-*"   # Accuracy from experiments matching pattern
```

Filter options work the same as [`yanex list`](list.md). See that documentation for full filtering details.

## Available Fields

### Quick Reference Table

| Field | Type | Description | Example Output |
|-------|------|-------------|----------------|
| `id` | Scalar | Experiment ID | `abc12345` |
| `name` | Scalar | Experiment name | `baseline-v2` |
| `status` | Scalar | Current status | `completed` |
| `description` | Scalar | Experiment description | `Training run with...` |
| `tags` | List | Experiment tags | `baseline, validated` |
| `script_path` | Scalar | Path to experiment script | `/path/to/train.py` |
| `created_at` | Scalar | Creation timestamp | `2024-01-15T10:30:00` |
| `started_at` | Scalar | Start timestamp | `2024-01-15T10:30:01` |
| `completed_at` | Scalar | Completion timestamp | `2024-01-15T11:45:00` |
| `error_message` | Scalar | Error message (if failed) | `IndexError: list...` |
| `params` | List | Available parameter names | `lr, batch_size, epochs` |
| `params.<key>` | Scalar | Specific parameter value | `0.001` |
| `metrics` | List | Available metric names | `accuracy, loss, f1` |
| `metrics.<key>` | Scalar | Last logged metric value | `0.95` |
| `stdout` | Multiline | Standard output content | *(full output)* |
| `stderr` | Multiline | Standard error content | *(full output)* |
| `artifacts` | Multiline | List of artifact file paths | `/path/to/model.pt`... |
| `cli-command` | Scalar | Original CLI invocation | `yanex run train.py...` |
| `run-command` | Scalar | Reproducible command | `yanex run train.py...` |
| `experiment-dir` | Scalar | Experiment directory path | `/home/.yanex/exp/abc...` |
| `artifacts-dir` | Scalar | Artifacts directory path | `/home/.yanex/exp/abc.../artifacts` |
| `dependencies` | Complex | Dependency slot=id pairs | `data=def456 model=ghi789` |
| `git.branch` | Scalar | Git branch name | `main` |
| `git.commit_hash` | Scalar | Git commit SHA | `a1b2c3d4e5f6...` |
| `git.dirty` | Scalar | Uncommitted changes flag | `true` |
| `environment.python.version` | Scalar | Python version | `3.11.5` |
| `upstream` | Multiline | Dependency graph (what this depends on) | *(ASCII DAG)* |
| `downstream` | Multiline | Dependent graph (what depends on this) | *(ASCII DAG)* |
| `lineage` | Multiline | Full lineage (upstream + downstream) | *(ASCII DAG)* |

### Field Types

Fields are categorized by the type of value they return:

**Scalar** - Single values (strings, numbers, booleans)
```bash
yanex get status abc123          # completed
yanex get params.lr abc123       # 0.001
yanex get git.branch abc123      # main
```

**List** - Lists of values (tags, available names)
```bash
yanex get tags abc123            # baseline, validated, production
yanex get params abc123          # lr, batch_size, epochs, model_type
yanex get metrics abc123         # accuracy, loss, f1_score
```

**Complex** - Structured data (dictionaries)
```bash
yanex get dependencies abc123    # data=def456 model=ghi789
```

**Multiline** - Multi-line text content
```bash
yanex get stdout abc123          # (full stdout output)
yanex get stderr abc123          # (full stderr output)
yanex get artifacts abc123       # (list of file paths, one per line)
yanex get lineage abc123         # (ASCII DAG visualization)
```

## Output Formats

Control output format with `--format` or `-F`:

| Format | Description | Use Case |
|--------|-------------|----------|
| `default` | Human-readable output | Terminal viewing |
| `json` | JSON with `{"id": "...", "value": ...}` structure | API integration |
| `csv` | CSV with ID column and headers | Spreadsheets, data analysis |
| `markdown` | GitHub-flavored markdown table | Documentation |
| `sweep` | Comma-separated values only (no headers) | Bash command substitution |

### Format Examples

```bash
# Default: human-readable
yanex get status abc123
# Output: completed

# JSON: structured data
yanex get status abc123 -F json
# Output: {"id": "abc12345", "value": "completed"}

# CSV: with headers
yanex get params.lr -s completed -F csv
# Output:
# ID,params.lr
# abc12345,0.001
# def67890,0.01

# Markdown: for documentation
yanex get params.lr -s completed -F markdown
# Output:
# | ID | params.lr |
# | --- | --- |
# | abc12345 | 0.001 |
# | def67890 | 0.01 |

# Sweep: for bash substitution (no trailing newline)
yanex get id -s completed -F sweep
# Output: abc12345,def67890
```

## Stdout/Stderr Options

Special options for `stdout` and `stderr` fields:

### Head and Tail

```bash
# Last 50 lines of stdout
yanex get stdout abc123 --tail 50

# First 10 lines of stderr
yanex get stderr abc123 --head 10

# First 5 and last 5 lines (with ... separator)
yanex get stdout abc123 --head 5 --tail 5
```

### Follow Mode

Stream output in real-time from running experiments:

```bash
# Follow stdout as experiment runs
yanex get stdout abc123 --follow

# Show last 20 lines then follow
yanex get stdout abc123 --tail 20 --follow
```

Press `Ctrl+C` to stop following.

## Command Reconstruction

Two special fields help with reproducibility:

### `cli-command`

Returns the original CLI invocation, including sweep syntax:

```bash
yanex get cli-command abc123
# Output: yanex run train.py -p "lr=range(0.001, 0.1, 0.01)" -t sweep
```

### `run-command`

Returns a reproducible command with resolved parameter values:

```bash
yanex get run-command abc123
# Output: yanex run train.py -p "lr=0.01" -t sweep -n "abc12345"
```

## Bash Substitution

The `sweep` format is designed for building dynamic commands:

```bash
# Use completed experiment IDs as dependencies
yanex run train.py -D data=$(yanex get id -n "*-prep-*" -F sweep)

# Build parameter sweep from previous results
yanex run train.py -p "lr=$(yanex get params.lr -s completed -t best -F sweep)"

# Run on experiments matching a pattern
for id in $(yanex get id -n "baseline-*" -F sweep | tr ',' ' '); do
    echo "Processing $id"
done
```

## Lineage Visualization

The `upstream`, `downstream`, and `lineage` fields display experiment dependency graphs as ASCII DAG visualizations.

### Lineage Fields

| Field | Description |
|-------|-------------|
| `upstream` | Show dependencies (what this experiment depends on) |
| `downstream` | Show dependents (what depends on this experiment) |
| `lineage` | Show both upstream and downstream combined |

### Basic Usage

```bash
# Show what an experiment depends on
yanex get upstream abc123

# Show what depends on this experiment
yanex get downstream abc123

# Show full lineage (both directions)
yanex get lineage abc123
```

### Example Output

```
<*> indicates experiments matching the filter

• <data> 0c621736 flights-data-10k (01_prepare_data.py) ✓
└─• <encoder> 1fdbf585 flights-100-encoder (02_train_encoder.py) ✓
  └─• <*> 9f0429e8 flights-103-strategy-ucb (03_train_advisor.py) ✓
```

Output legend:
- `<*>` - Target experiment (the one you queried)
- `<slot>` - Dependency slot name (e.g., `<data>`, `<encoder>`)
- `✓` - Completed, `✗` - Failed, `●` - Running, `○` - Pending

### Multi-Experiment Lineage

Use filters to visualize lineage for multiple experiments:

```bash
# Lineage for all completed training experiments
yanex get lineage -n "train-*" -s completed

# Upstream dependencies of experiments with a tag
yanex get upstream -t production

# Downstream of specific experiments
yanex get downstream --ids abc123,def456
```

Connected experiments render as a single DAG; disconnected experiments render as separate DAGs.

### Lineage Options

| Option | Description |
|--------|-------------|
| `--depth N` | Limit traversal depth (default: 10) |

```bash
# Limit to 3 levels of dependencies
yanex get upstream abc123 --depth 3

# Get just the IDs for scripting (sweep format)
yanex get upstream abc123 -F sweep
# Output: def67890,ghi11111

# Use with bash substitution
yanex delete $(yanex get upstream abc123 -F sweep)
```

### JSON Output for Lineage

```bash
yanex get lineage abc123 -F json
```

Returns structured data with nodes, edges, and targets:
```json
{
  "targets": ["abc12345"],
  "nodes": [
    {"id": "abc12345", "name": "train-model", "status": "completed"},
    {"id": "def67890", "name": "prep-data", "status": "completed"}
  ],
  "edges": [
    {"from": "def67890", "to": "abc12345", "slot": "data"}
  ]
}
```

## Examples

### Check Experiment Status

```bash
# Single experiment
yanex get status abc123

# All running experiments
yanex get status -s running
```

### Extract Parameters

```bash
# List available parameters
yanex get params abc123

# Get specific parameter
yanex get params.learning_rate abc123

# Get nested parameter
yanex get params.model.hidden_size abc123
```

### Query Metrics

```bash
# List available metrics
yanex get metrics abc123

# Get final accuracy
yanex get metrics.accuracy abc123

# Compare accuracy across experiments
yanex get metrics.accuracy -t baseline -F csv
```

### Debug Output

```bash
# Check last 20 lines of output
yanex get stdout abc123 --tail 20

# See error output
yanex get stderr abc123

# Monitor running experiment
yanex get stdout abc123 --follow
```

### Build Pipelines

```bash
# Get best model's ID
BEST_ID=$(yanex get id -t best -l 1 -F sweep)

# Get its parameters for a new run
yanex run train.py -p "lr=$(yanex get params.lr $BEST_ID)"

# Chain experiments with dependencies
yanex run evaluate.py -D model=$(yanex get id -n "trained-model" -F sweep)
```

## Command Options

### Arguments

- `FIELD` (required): Field path to retrieve
- `EXPERIMENT_ID` (optional): Experiment ID, name, or prefix

### Filter Options

Same as [`yanex list`](list.md):

- `--ids IDS`: Comma-separated experiment IDs
- `--status`, `-s STATUS`: Filter by status
- `--name`, `-n PATTERN`: Filter by name pattern (glob syntax)
- `--tag`, `-t TAG`: Filter by tag (repeatable)
- `--started-after DATE`: Filter by start time
- `--started-before DATE`: Filter by start time
- `--ended-after DATE`: Filter by end time
- `--ended-before DATE`: Filter by end time
- `--archived`, `-a`: Include archived experiments
- `--limit`, `-l N`: Limit number of results

### Output Options

- `--format`, `-F FORMAT`: Output format (default, json, csv, markdown, sweep)
- `--default VALUE`: Value for missing fields (default: `[not_found]`)

### Stdout/Stderr Options

- `--head N`: Return first N lines
- `--tail N`: Return last N lines
- `--follow`: Stream output in real-time (single experiment only)

### Lineage Options

- `--depth N`: Limit dependency traversal depth (default: 10)
- `-F sweep`: Output comma-separated experiment IDs only (for scripting)

---

**Related:**
- [`yanex list`](list.md) - List experiments with the same filter options
- [`yanex show`](show.md) - Display full experiment details
- [`yanex compare`](compare.md) - Compare experiments side-by-side
