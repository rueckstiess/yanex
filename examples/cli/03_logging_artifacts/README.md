# 03: Logging Artifacts

## What This Example Demonstrates

- Copying file artifacts: `copy_artifact(src_path, filename)`
- Saving text content: `save_artifact(text, filename)`
- Saving matplotlib figures: `save_artifact(fig, filename)`
- Automatic format detection based on file extension
- How artifacts are organized in the experiment directory
- Working with temporary files (create → save → clean up)

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
✓ Saved CSV artifact
✓ Saved text summary
✓ Saved matplotlib figure

All artifacts saved successfully!
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

### 1. Copying Files (`copy_artifact`)

For existing files you want to copy to the experiment:

```python
# Create a file locally
csv_file = Path("data_results.csv")
with open(csv_file, 'w') as f:
    f.write("index,value\n")
    # ... write data ...

# Copy it to experiment artifacts
yanex.copy_artifact(csv_file, "results.csv")

# Clean up local file (optional)
csv_file.unlink()
```

**Use cases**: CSV files, model checkpoints, processed datasets, configuration dumps

### 2. Saving Python Objects (`save_artifact`)

For saving Python objects with automatic format detection:

**Text content:**
```python
# Create text content
summary = f"""Analysis Summary
Results: {results}
"""

# Save directly as artifact (format auto-detected from .txt extension)
yanex.save_artifact(summary, "summary.txt")
```

**Matplotlib figures:**
```python
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots()
ax.plot(x_data, y_data)
ax.set_title("Results")

# Save as artifact (format auto-detected from .png extension)
yanex.save_artifact(fig, "plot.png")
plt.close(fig)  # Clean up
```

**Other formats:**
```python
# JSON (auto-detected from .json extension)
yanex.save_artifact({"accuracy": 0.95}, "metrics.json")

# PyTorch model (auto-detected from .pt extension)
yanex.save_artifact(model.state_dict(), "model.pt")

# Pandas DataFrame as CSV (auto-detected from .csv extension)
import pandas as pd
yanex.save_artifact(df, "results.csv")
```

**Use cases**: Text reports, JSON/YAML data, plots, PyTorch models, pandas DataFrames, numpy arrays

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
- **Standalone mode support**: Artifact functions work in both modes:
  - **With tracking** (`yanex run`): Saves to experiment directory
  - **Standalone** (direct `python`): Saves to `./artifacts/` directory
- **Automatic format detection**: File extension determines how to save/load (`.txt`, `.json`, `.pt`, `.png`, etc.)
- **Mode detection**: Use `yanex.is_standalone()` to check if running with or without yanex tracking
- **Automatic stdout/stderr**: Yanex automatically captures script output as artifacts
- **Any file type**: Save CSVs, images, models, configs - anything you want to preserve

## Next Steps

- View your artifacts: `yanex open <id>` opens the experiment directory
- Compare results across experiments with different parameters
- See example 04 to learn about experiment metadata and tags
