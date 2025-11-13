# 03: Logging Artifacts

## What This Example Demonstrates

- Logging file artifacts: `log_artifact(name, file_path)`
- Logging text content: `log_text(content, filename)`
- Logging matplotlib figures: `log_matplotlib_figure(fig, filename)`
- How artifacts are organized in the experiment directory
- Working with temporary files (create → log → clean up)

## Files

- `analyze.py` - Data analysis script that generates multiple artifact types

## How to Run

### Basic usage
```bash
yanex run analyze.py
```

### With different sample sizes
```bash
yanex run analyze.py -p num_samples=100
yanex run analyze.py -p num_samples=200 --name "large-dataset-analysis"
```

## Expected Output

```
Analyzing 50 data points...
Analysis complete: avg=54.32, max=98.45, min=12.67
✓ Logged CSV artifact
✓ Logged text summary
✓ Logged matplotlib figure

All artifacts logged successfully!
```

## Generated Artifacts

After running, check the experiment's `artifacts` directory for these artifacts:

1. **results.csv** - CSV file with all data points
2. **summary.txt** - Text summary of analysis results
3. **analysis_plot.png** - Matplotlib visualization (if matplotlib is installed)

## How to View Artifacts

```bash
# Show experiment details (includes artifacts list)
yanex show <id>

# Open the experiment directory in your file browser
yanex open <id>

# Artifacts are stored in: ~/.yanex/experiments/<id>/artifacts/
```

## Artifact Types Explained

### 1. File Artifacts (`log_artifact`)

For existing files you want to preserve:

```python
# Create a file locally
csv_file = Path("data_results.csv")
with open(csv_file, 'w') as f:
    f.write("index,value\n")
    # ... write data ...

# Log it to experiment artifacts
yanex.log_artifact("results.csv", csv_file)

# Clean up local file (optional)
csv_file.unlink()
```

**Use cases**: CSV files, model checkpoints, processed datasets, configuration dumps

### 2. Text Artifacts (`log_text`)

For generating text content directly:

```python
# Create text content
summary = f"""Analysis Summary
Results: {results}
"""

# Save directly as artifact (no intermediate file needed)
yanex.log_text(summary, "summary.txt")
```

**Use cases**: Summary reports, logs, JSON/YAML data, error messages

### 3. Matplotlib Figures (`log_matplotlib_figure`)

For saving plots and visualizations:

```python
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots()
ax.plot(x_data, y_data)
ax.set_title("Results")

# Save as artifact
yanex.log_matplotlib_figure(fig, "plot.png", dpi=150)
plt.close(fig)  # Clean up
```

**Use cases**: Charts, graphs, training curves, diagnostic plots

## Artifact Storage

All artifacts are stored in the experiment's `artifacts/` directory:

```
~/.yanex/experiments/<id>/
├── metadata.json
├── config.json
├── metrics.json
└── artifacts/
    ├── results.csv           # Your logged files
    ├── summary.txt
    ├── analysis_plot.png
    └── stdout.txt            # Automatic: captured stdout
```

## Key Concepts

- **Artifacts are copied**: Files are copied to the experiment directory, so you can delete the originals
- **No-op in standalone mode**: All logging functions (`log_artifact()`, `log_text()`, `log_matplotlib_figure()`) do nothing when run without yanex - your script still works!
- **Mode detection**: Use `yanex.is_standalone()` to check if running with or without yanex tracking
- **Automatic stdout/stderr**: Yanex automatically captures script output as artifacts
- **Any file type**: Log CSVs, images, models, configs - anything you want to preserve

## Next Steps

- View your artifacts: `yanex open <id>` opens the experiment directory
- Compare results across experiments with different parameters
- See example 04 to learn about experiment metadata and tags
