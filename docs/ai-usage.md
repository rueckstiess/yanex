# AI and Automation Usage

Yanex is designed to work seamlessly with AI assistants and automation scripts. This guide covers machine-readable output formats, the `yanex get` command for quick field extraction, and the Claude Code skill for AI-assisted experiment management.

## Machine-Readable Output Formats

Most yanex commands support the `--format` / `-F` option for structured output:

| Format | Flag | Description |
|--------|------|-------------|
| Default | (none) | Human-readable rich text |
| JSON | `-F json` | Structured JSON output |
| CSV | `-F csv` | Comma-separated values |
| Markdown | `-F markdown` | Markdown tables |

### Commands Supporting `--format`

| Command | JSON | CSV | Markdown | Notes |
|---------|:----:|:---:|:--------:|-------|
| `yanex list` | ✓ | ✓ | ✓ | Experiment listings |
| `yanex show` | ✓ | ✓ | ✓ | Experiment details |
| `yanex compare` | ✓ | ✓ | ✓ | Side-by-side comparison |
| `yanex get` | ✓ | ✓ | ✓ | Field extraction (also supports `-F sweep`) |
| `yanex archive` | ✓ | ✓ | ✓ | Operation results |
| `yanex unarchive` | ✓ | ✓ | ✓ | Operation results |
| `yanex delete` | ✓ | ✓ | ✓ | Operation results |
| `yanex update` | ✓ | ✓ | ✓ | Operation results |
| `yanex migrate` | ✓ | ✓ | ✓ | Migration results |

### Examples

```bash
# JSON output for parsing
yanex list -s completed -F json | jq '.[] | .id'

# CSV for spreadsheets
yanex compare -t sweep -F csv > comparison.csv

# Markdown for documentation
yanex show abc12345 -F markdown >> experiment-notes.md
```

## The `yanex get` Command

The `yanex get` command is specifically designed for AI agents and scripts to extract individual field values without parsing complex output.

**[→ Complete `yanex get` Reference](commands/get.md)**

### Quick Examples

```bash
# Get single values
yanex get status abc12345              # → "completed"
yanex get params.lr abc12345           # → "0.001"
yanex get metrics.accuracy abc12345    # → "0.9523"

# List available fields
yanex get params abc12345              # → learning_rate, batch_size, epochs
yanex get metrics abc12345             # → loss, accuracy, f1_score

# Get stdout/stderr
yanex get stdout abc12345              # Full stdout
yanex get stdout abc12345 --tail 20    # Last 20 lines
yanex get stderr abc12345 -f           # Follow in real-time

# Get paths
yanex get experiment-dir abc12345      # → /path/to/experiment
yanex get artifacts-dir abc12345       # → /path/to/experiment/artifacts
```

### Multi-Experiment Queries

Use filters instead of an experiment ID to query multiple experiments:

```bash
# Get IDs of all completed experiments
yanex get id -s completed

# Get learning rates from a sweep
yanex get params.lr -n "hpo-*"

# Check status of running experiments
yanex get status -s running
```

### Bash Substitution (`-F sweep`)

The `-F sweep` format outputs comma-separated values for use in bash substitution:

```bash
# Run training on multiple data prep experiments
yanex run train.py -D data=$(yanex get id -n "*-prep-*" -F sweep)
# Expands to: yanex run train.py -D data=abc123,def456,ghi789

# Build parameter sweep from previous results
yanex run train.py -p lr=$(yanex get params.lr -s completed -F sweep)
# Expands to: yanex run train.py -p lr=0.001,0.01,0.1
```

### Lineage Visualization

Visualize experiment dependency graphs:

```bash
# What does this experiment depend on?
yanex get upstream abc12345

# What depends on this experiment?
yanex get downstream abc12345

# Full lineage (both directions)
yanex get lineage abc12345

# Get just the IDs for scripting
yanex get upstream abc12345 -F sweep
# → abc12345,parent1,parent2
```

### Command Reconstruction

Two fields help reproduce experiments:

| Field | Purpose | Example |
|-------|---------|---------|
| `cli-command` | Original CLI (preserves sweep syntax) | `yanex run train.py -p "lr=0.001,0.01,0.1"` |
| `run-command` | Reproducible (resolved values) | `yanex run train.py -p lr=0.01` |

```bash
# Log the original command that created a sweep
yanex get cli-command abc12345

# Re-run a specific experiment from a sweep
$(yanex get run-command abc12345)
```

## Claude Code Skill

Yanex includes a [Claude Code](https://docs.anthropic.com/en/docs/claude-code) skill that enables AI-assisted experiment management. The skill teaches Claude how to:

- Run experiments with proper parameter syntax
- Query and filter experiments effectively
- Extract values using `yanex get`
- Compare results and find best experiments
- Manage experiment lifecycle (archive, update, delete)
- Follow safety guidelines for destructive operations

### Installation

Install the skill by creating a symlink to your Claude skills directory:

```bash
# Find your yanex installation path
YANEX_PATH=$(python -c "import yanex; print(yanex.__path__[0])")

# Option 1: Install globally (available in all projects)
ln -s "$YANEX_PATH/claude-skill" ~/.claude/skills/tracking-yanex-experiments

# Option 2: Install per-project (in your project's .claude folder)
mkdir -p .claude/skills
ln -s "$YANEX_PATH/claude-skill" .claude/skills/tracking-yanex-experiments
```

### What the Skill Enables

With the skill installed, Claude can help you:

**Run experiments:**
```
"Run a learning rate sweep from 0.0001 to 0.1 with 10 values"
→ yanex run train.py -p "lr=logspace(-4, -1, 10)" --parallel 0
```

**Query results:**
```
"Show me the best performing experiment from yesterday's sweep"
→ Uses Results API to find and display best experiment
```

**Monitor progress:**
```
"Check the status of my running experiments"
→ yanex get stdout -s running --tail 10
```

**Analyze results:**
```
"Compare the accuracy across all experiments tagged 'baseline'"
→ yanex compare -t baseline -F markdown
```

### Safety Guidelines

The skill includes safety guidelines for destructive operations:

- **`yanex delete`**: Always requires user confirmation before execution
- **`yanex archive`**: Safe to run autonomously (reversible with `unarchive`)
- **`--force` flag**: Documented for bypassing CLI confirmation prompts

## Automation Patterns

### CI/CD Integration

```bash
#!/bin/bash
# Run sweep and check for failures
yanex run train.py -p "lr=0.001,0.01,0.1" --parallel 4

# Check if any failed
FAILED=$(yanex get id -s failed -n "train-*" --started-after "1 hour ago")
if [ -n "$FAILED" ]; then
    echo "Failed experiments: $FAILED"
    exit 1
fi

# Get best result
BEST_ACC=$(yanex get metrics.accuracy $(yanex get id -s completed -l 1 --sort-by metrics.accuracy))
echo "Best accuracy: $BEST_ACC"
```

### Experiment Pipelines

```bash
#!/bin/bash
# Multi-stage pipeline with dependencies

# Stage 1: Data preparation
yanex run prepare_data.py -n "data-prep"
DATA_ID=$(yanex get id -n "data-prep" -s completed -l 1)

# Stage 2: Training (depends on data prep)
yanex run train.py -D data=$DATA_ID -p "lr=0.001,0.01" -n "train"

# Stage 3: Evaluation (depends on all training runs)
TRAIN_IDS=$(yanex get id -n "train-*" -s completed -F sweep)
yanex run evaluate.py -D models=$TRAIN_IDS -n "eval"
```

### Monitoring Dashboard

```bash
#!/bin/bash
# Simple monitoring script
watch -n 30 '
echo "=== Running Experiments ==="
yanex list -s running -F markdown

echo ""
echo "=== Recent Completions ==="
yanex list -s completed --started-after "1 hour ago" -F markdown

echo ""
echo "=== Failures ==="
yanex list -s failed --started-after "1 day ago" -F markdown
'
```

## Tips for AI Assistants

1. **Use `yanex get` for field extraction** - It's faster and cleaner than parsing `yanex show` output

2. **Prefer `-F json` for complex data** - JSON is easier to parse programmatically than default output

3. **Use `-F sweep` for bash substitution** - Generates comma-separated values ready for parameter sweeps

4. **Check status before operations** - Use `yanex get status <id>` to verify experiment state

5. **Use filters over iteration** - Commands like `yanex get id -s completed` are more efficient than listing and filtering

6. **Follow safety guidelines** - Always confirm with users before running `yanex delete`
