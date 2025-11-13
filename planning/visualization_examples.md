# Yanex Visualization - API Examples

**Companion to**: `visualization_design.md`

This document shows concrete examples of how the API will be used in practice.

---

## Basic Examples

### Example 1: Single Experiment Training Curve
```python
import yanex.results as yr

# Plot accuracy over time for one experiment
yr.plot_metrics("accuracy", ids=["abc12345"])
```

**Expected behavior:**
- Single line plot
- X-axis: "Step"
- Y-axis: "Accuracy"
- No legend (single line)
- Title: auto-generated or None

---

### Example 2: Multiple Metrics, Single Experiment
```python
yr.plot_metrics(
    ["accuracy", "loss", "f1"],
    ids=["abc12345"]
)
```

**Expected behavior:**
- 3 subplots in single row (1×3 grid)
- Each subplot shows one metric
- Same experiment across all subplots
- Subplot titles: "Accuracy", "Loss", "F1"

---

### Example 3: Compare Multiple Experiments
```python
yr.plot_metrics(
    "accuracy",
    tags=["baseline_comparison"]
)
```

**Expected behavior:**
- Single plot with multiple lines (one per experiment)
- Default `label_by="id"` → legend shows experiment IDs
- Different color per experiment
- Shared X/Y axes

---

### Example 4: Compare with Custom Labels
```python
yr.plot_metrics(
    "loss",
    tags=["lr_sweep"],
    label_by="learning_rate"
)
```

**Expected behavior:**
- Multiple lines colored by learning rate
- Legend: "lr=0.001", "lr=0.01", "lr=0.1"
- Much clearer than experiment IDs

---

## Intermediate Examples

### Example 5: Bar Chart (Single-Step Metrics)
```python
yr.plot_metrics(
    "final_accuracy",
    tags=["model_comparison"]
)
```

**Expected behavior:**
- Auto-detected: metrics have no steps → bar chart
- X-axis: Experiment IDs (or names if available)
- Y-axis: "Final Accuracy"
- One bar per experiment

---

### Example 6: Subplots by Parameter
```python
yr.plot_metrics(
    "f1_score",
    tags=["model_comparison"],
    label_by="learning_rate",
    subplot_by="model_type"
)
```

**Scenario:**
- 3 model types: cnn, rnn, transformer
- 4 learning rates per model
- Total: 12 experiments

**Expected behavior:**
- 1×3 subplot grid (single row)
- Subplot 1: "cnn" with 4 colored lines (one per LR)
- Subplot 2: "rnn" with 4 colored lines
- Subplot 3: "transformer" with 4 colored lines
- Shared legend showing LR values
- Shared Y-axis label: "F1 Score"

---

### Example 7: 2D Subplot Grid
```python
yr.plot_metrics(
    "accuracy",
    tags=["grid_search"],
    label_by="learning_rate",
    subplot_by=("batch_size", "optimizer"),
    subplot_layout=(2, 3)
)
```

**Scenario:**
- 2 batch sizes: 16, 32
- 3 optimizers: adam, sgd, rmsprop
- 4 learning rates per combination
- Total: 2×3×4 = 24 experiments

**Expected behavior:**
- 2×3 subplot grid
- Each cell: one (batch_size, optimizer) combination
- Each cell contains 4 lines (learning rates)
- Subplot titles: "bs=16, opt=adam", "bs=16, opt=sgd", ...
- Shared legend for learning rates

---

## Advanced Examples

### Example 8: Statistical Aggregation
```python
yr.plot_metrics(
    "accuracy",
    tags=["repeated_runs"],
    group_by="random_seed",
    label_by="learning_rate",
    show_ci=True
)
```

**Scenario:**
- 3 learning rates: 0.001, 0.01, 0.1
- 5 random seeds per LR
- Total: 15 experiments

**Expected behavior:**
- 3 bold lines (one per LR) showing mean across seeds
- 3 shaded bands showing 95% confidence interval
- 15 faint individual lines behind (if `show_individuals=True`)
- Legend shows learning rates only (not seeds)

---

### Example 9: Complex Multi-Dimensional Analysis
```python
yr.plot_metrics(
    ["accuracy", "loss"],
    tags=["full_sweep"],
    group_by="random_seed",
    label_by="learning_rate",
    subplot_by="model_type",
    show_std=True,
    show_individuals=False
)
```

**Scenario:**
- 2 metrics: accuracy, loss
- 2 model types: cnn, rnn
- 3 learning rates per model
- 5 random seeds per (model, LR) combination
- Total: 2×3×5 = 30 experiments

**Expected behavior:**
- 2×2 subplot grid:
  ```
  Row 1: Accuracy-CNN    Accuracy-RNN
  Row 2: Loss-CNN        Loss-RNN
  ```
- Each subplot: 3 bold lines (LRs) with std bands
- No individual lines (show_individuals=False)
- Shared legend for LRs

**Alternative layout** (if we decide metrics go on columns):
```
Row 1: CNN-Accuracy     CNN-Loss
Row 2: RNN-Accuracy     RNN-Loss
```

---

### Example 10: Custom Lambda Functions
```python
yr.plot_metrics(
    "f1_score",
    tags=["architecture_search"],
    label_by=lambda exp: f"{exp.get_param('model')}-{exp.get_param('layers')}L",
    subplot_by=lambda exp: "Deep" if exp.get_param('layers') > 5 else "Shallow"
)
```

**Scenario:**
- Various models with different layer counts
- Want to categorize as "Shallow" (≤5 layers) vs "Deep" (>5 layers)
- Want labels like "cnn-3L", "rnn-8L"

**Expected behavior:**
- 2 subplots: "Shallow", "Deep"
- Custom labels in legend based on lambda
- Full control over categorization

---

## Customization Examples

### Example 11: Full Styling Control
```python
yr.plot_metrics(
    "accuracy",
    ids=["abc123"],
    title="Training Progress - ResNet50",
    xlabel="Epoch",
    ylabel="Validation Accuracy",
    figsize=(10, 6),
    grid=True,
    colors=["#FF5733"]  # Custom color
)
```

**Expected behavior:**
- Custom title and labels
- Larger figure size
- Custom color (overrides default palette)

---

### Example 12: Advanced Post-Processing
```python
fig, axes = yr.plot_metrics(
    "loss",
    tags=["training"],
    label_by="optimizer",
    return_axes=True,
    show=False
)

# Advanced matplotlib customization
axes.set_yscale('log')
axes.axhline(y=0.1, color='r', linestyle='--', label='Target')
axes.legend()
fig.savefig("custom_plot.pdf", dpi=300)
plt.show()
```

**Expected behavior:**
- Returns (fig, axes) for customization
- Doesn't show immediately (show=False)
- User adds log scale and reference line
- Saves as high-DPI PDF

---

## Web UI Integration Examples

### Example 13: Data Export for JavaScript
```python
from yanex.visualization.metrics_data import extract_metrics
from yanex.visualization.metrics_grouping import organize_for_plotting
import yanex.results as yr
import json

# Get experiments and organize data
experiments = yr.get_experiments(tags=["training"])
data = extract_metrics(experiments, ["accuracy", "loss"])
structure = organize_for_plotting(
    data,
    ["accuracy", "loss"],
    label_by="learning_rate",
    subplot_by="model_type"
)

# Serialize to JSON (assuming we add to_dict() method)
json_data = {
    "plotType": structure.plot_type,
    "subplotGrid": structure.subplot_grid,
    "groups": [
        {
            "subplotKey": group.subplot_key,
            "metricName": group.metric_name,
            "lines": [
                {
                    "label": line.label,
                    "steps": line.steps,
                    "values": line.values
                }
                for line in group.lines
            ]
        }
        for group in structure.groups
    ],
    "colorPalette": structure.color_palette
}

# Send to frontend
print(json.dumps(json_data, indent=2))
```

**Expected output:**
```json
{
  "plotType": "line",
  "subplotGrid": [1, 2],
  "groups": [
    {
      "subplotKey": "cnn",
      "metricName": "accuracy",
      "lines": [
        {
          "label": "lr=0.001",
          "steps": [0, 1, 2, 3],
          "values": [0.8, 0.85, 0.9, 0.92]
        },
        ...
      ]
    },
    ...
  ],
  "colorPalette": ["#1f77b4", "#ff7f0e", "#2ca02c"]
}
```

---

## Advanced Composition Examples

### Example 14: Custom Plot Using Building Blocks
```python
from yanex.visualization.metrics_data import extract_metrics
from yanex.visualization.metrics_grouping import organize_for_plotting
from yanex.visualization.metrics_plotting import plot_line, apply_styling
import matplotlib.pyplot as plt
import yanex.results as yr

# Extract and organize data
experiments = yr.get_experiments(tags=["custom_analysis"])
data = extract_metrics(experiments, ["accuracy"])
structure = organize_for_plotting(data, ["accuracy"], label_by="model")

# Custom layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left plot: All models together
group = structure.groups[0]
for line in group.lines:
    plot_line(ax1, line)
apply_styling(ax1, title="All Models", xlabel="Step", ylabel="Accuracy")

# Right plot: Best model only (custom filter)
best_line = max(group.lines, key=lambda l: max(l.values))
plot_line(ax2, best_line, color='red')
ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='Target')
apply_styling(ax2, title=f"Best: {best_line.label}", xlabel="Step", ylabel="Accuracy")

plt.tight_layout()
plt.show()
```

**Expected behavior:**
- Side-by-side comparison
- Left: All models overlaid
- Right: Best model highlighted with target line
- Full control over layout and content

---

## Error Handling Examples

### Example 15: No Experiments Found
```python
yr.plot_metrics("accuracy", tags=["nonexistent"])
```

**Expected error:**
```
ValueError: No experiments found matching filters
```

---

### Example 16: Metric Not Found
```python
yr.plot_metrics("nonexistent_metric", ids=["abc123"])
```

**Expected error:**
```
ValueError: Metric 'nonexistent_metric' not found in any experiments
```

---

### Example 17: Mixed Single/Multi-Step
```python
# Experiment 1: accuracy has 100 steps
# Experiment 2: accuracy has 1 step (final value only)
yr.plot_metrics("accuracy", ids=["exp1", "exp2"])
```

**Expected error:**
```
ValueError: Metric 'accuracy' has inconsistent step counts across experiments.
  - Experiment exp1: 100 steps
  - Experiment exp2: 1 step
Cannot mix single-step and multi-step metrics in the same plot.
```

---

### Example 18: Subplot Layout Mismatch
```python
yr.plot_metrics(
    "accuracy",
    subplot_by="model_type",  # 5 model types
    subplot_layout=(2, 2)      # Only 4 subplots
)
```

**Expected error:**
```
ValueError: subplot_layout (2, 2) = 4 subplots, but subplot_by='model_type' creates 5 subplots.
Specify subplot_layout=(rows, cols) where rows * cols = 5, or use subplot_layout=None for default (single row).
```

---

### Example 19: Aggregation Without Grouping
```python
yr.plot_metrics(
    "accuracy",
    label_by="learning_rate",
    show_ci=True  # Requires group_by
)
```

**Expected error:**
```
ValueError: show_ci=True requires group_by to be specified.
Confidence intervals can only be computed when aggregating multiple experiments.
```

---

## Edge Cases

### Example 20: Single Experiment, Multiple Metrics, Single Step
```python
# Experiment has single-step metrics only
yr.plot_metrics(
    ["final_accuracy", "final_loss", "final_f1"],
    ids=["abc123"]
)
```

**Expected behavior:**
- Bar chart with 3 bars
- X-axis: metric names
- Y-axis: values
- Title: Experiment name or ID

---

### Example 21: Many Unique Labels (>20)
```python
# 30 experiments with different learning rates
yr.plot_metrics(
    "accuracy",
    tags=["massive_sweep"],
    label_by="learning_rate"
)
```

**Expected behavior:**
- Plot created successfully
- Warning printed:
  ```
  Warning: 30 unique labels but only 20 colors available. Colors will cycle.
  Consider using group_by or subplot_by to reduce the number of lines per plot.
  ```
- Colors cycle through 20-color palette

---

### Example 22: Experiments with Different Step Counts
```python
# Experiment 1: 100 steps
# Experiment 2: 50 steps
# Experiment 3: 75 steps
yr.plot_metrics("loss", ids=["exp1", "exp2", "exp3"])
```

**Expected behavior:**
- All 3 lines plotted
- X-axis goes to 100 (max step count)
- Shorter experiments end early (lines stop)
- Warning (optional):
  ```
  Warning: Experiments have different step counts (50-100 steps).
  Lines will end at their respective final steps.
  ```

---

## Performance Considerations

### Example 23: Large Dataset
```python
# 1000 experiments, each with 10,000 steps
yr.plot_metrics(
    "loss",
    status="completed",  # Matches 1000 experiments
    label_by="learning_rate"
)
```

**Considerations:**
- Data extraction could be slow (10M data points)
- Plotting 1000 lines may be unreadable
- Should warn/error or suggest alternatives:
  ```
  Warning: Plotting 1000 experiments with 10,000 steps each (10M data points).
  Consider:
  - Using group_by to aggregate experiments
  - Using subplot_by to separate into smaller plots
  - Filtering to fewer experiments
  - Plotting fewer steps (future feature: step_range parameter)
  ```

---

## Comparison with Existing Tools

### TensorBoard-like Usage
```python
# TensorBoard: tensorboard --logdir=./runs
# Yanex equivalent:
yr.plot_metrics(
    ["train_loss", "val_loss", "train_acc", "val_acc"],
    tags=["current_run"]
)
```

### Weights & Biases-like Usage
```python
# W&B: wandb.log({"accuracy": 0.9})
import yanex
yanex.log_metrics({"accuracy": 0.9})

# W&B: Plot comparison in UI
# Yanex equivalent:
yr.plot_metrics(
    "accuracy",
    tags=["project_alpha"],
    label_by="experiment_name",
    subplot_by="model_architecture"
)
```

---

## Future Extensions (Not in Initial Version)

### Time-based X-axis
```python
# Future feature
yr.plot_metrics(
    "accuracy",
    ids=["abc123"],
    x_axis="time"  # Instead of step number
)
```

### Step range filtering
```python
# Future feature
yr.plot_metrics(
    "loss",
    tags=["training"],
    step_range=(0, 100)  # Only plot first 100 steps
)
```

### Smoothing
```python
# Future feature
yr.plot_metrics(
    "loss",
    ids=["abc123"],
    smooth=0.9  # Exponential smoothing
)
```

### Statistical tests
```python
# Future feature
yr.plot_metrics(
    "accuracy",
    group_by="random_seed",
    label_by="learning_rate",
    show_significance=True  # Show significance markers
)
```
