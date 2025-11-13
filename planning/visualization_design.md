# Yanex Visualization Utilities - Design Plan

**Version**: 0.1
**Date**: 2025-11-13
**Status**: Draft for Review

## Overview

Design for modular, composable visualization utilities for yanex experiment tracking. The architecture separates data fetching, grouping/organization, and visualization into reusable layers.

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│  High-Level API: yr.plot_metrics()                      │
│  - User-facing convenience function                     │
│  - Delegates to lower layers                            │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Data Layer: metrics_data.py                            │
│  - Filter experiments (reuses yr.get_experiments)       │
│  - Extract metrics from experiments                     │
│  - Handle single-step vs multi-step                     │
│  - Returns structured data (dicts/dataclasses)          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Organization Layer: metrics_grouping.py                │
│  - Group by fields (group_by, label_by, subplot_by)     │
│  - Aggregate statistics (mean, median, CI, std)         │
│  - Resolve field values (meta → params → error)         │
│  - Create hierarchical structure for plotting           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Visualization Layer: metrics_plotting.py               │
│  - Plot individual lines, bars, bands                   │
│  - Create subplots, legends, labels                     │
│  - Apply styling (colors, fonts, accessibility)         │
│  - Takes (fig, ax) and data structures                  │
└─────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
yanex/
├── visualization/
│   ├── __init__.py              # Public API exports
│   ├── metrics_data.py          # Data extraction layer
│   ├── metrics_grouping.py      # Grouping & organization layer
│   ├── metrics_plotting.py      # Visualization layer
│   ├── styles.py                # Plot styling & themes
│   └── utils.py                 # Shared utilities (colors, validation)
│
└── results/
    └── __init__.py              # Add: plot_metrics() method
```

---

## Component APIs

### 1. Data Layer: `metrics_data.py`

**Purpose**: Extract metric values from experiments, handle single vs multi-step

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class MetricValue:
    """Single metric value at a point."""
    experiment_id: str
    metric_name: str
    step: int | None          # None for single-step metrics
    value: float
    timestamp: str | None

@dataclass
class ExperimentMetrics:
    """All metrics for a single experiment."""
    experiment_id: str
    experiment: Experiment    # Full experiment object for field resolution
    metric_values: dict[str, list[MetricValue]]  # metric_name -> list of values
    is_multi_step: bool

def extract_metrics(
    experiments: list[Experiment],
    metric_names: list[str]
) -> list[ExperimentMetrics]:
    """
    Extract specified metrics from experiments.

    Parameters
    ----------
    experiments : list[Experiment]
        Experiments to extract from
    metric_names : list[str]
        Names of metrics to extract

    Returns
    -------
    list[ExperimentMetrics]
        Structured metric data for each experiment

    Raises
    ------
    ValueError
        If metric not found in any experiment, or if mixing single/multi-step
    """
    pass

def validate_metric_consistency(
    experiment_metrics: list[ExperimentMetrics],
    metric_names: list[str]
) -> dict[str, bool]:
    """
    Check if metrics are consistently single-step or multi-step.

    Returns
    -------
    dict[str, bool]
        {metric_name: is_multi_step} for each metric

    Raises
    ------
    ValueError
        If same metric is single-step in some experiments, multi-step in others
    """
    pass
```

**Key Decisions:**
- Returns structured data, not raw dicts
- Validates metric consistency across experiments (can't mix single/multi-step)
- Keeps reference to full Experiment object for field resolution in next layer

---

### 2. Organization Layer: `metrics_grouping.py`

**Purpose**: Group, label, and organize data for plotting

```python
from dataclasses import dataclass
from typing import Callable, Any

# Type alias for field spec
FieldSpec = str | tuple[str, ...] | Callable[[Experiment], Any]

@dataclass
class PlotGroup:
    """Data for a single plot (one subplot)."""
    subplot_key: Any           # Value(s) from subplot_by
    lines: list['LineData']    # Individual lines or aggregated lines
    metric_name: str

@dataclass
class LineData:
    """Data for a single line in a plot."""
    label: str                 # Legend label
    steps: list[int]           # X-axis values (step numbers)
    values: list[float]        # Y-axis values (metric values)
    color: str | None          # Color (assigned by plotter)

    # For aggregated lines (when group_by is used)
    is_aggregated: bool = False
    ci_lower: list[float] | None = None
    ci_upper: list[float] | None = None
    std_lower: list[float] | None = None
    std_upper: list[float] | None = None
    individual_lines: list['LineData'] | None = None  # Individual lines behind mean

@dataclass
class BarData:
    """Data for bar chart (single-step metrics)."""
    labels: list[str]          # X-axis labels (experiment labels)
    values: list[float]        # Y-axis values (metric values)
    colors: list[str] | None   # Bar colors
    subplot_key: Any = None    # If using subplot_by

@dataclass
class PlotStructure:
    """Complete plot structure ready for visualization."""
    plot_type: str             # "line" or "bar"
    subplot_grid: tuple[int, int] | None  # (rows, cols) or None for single plot
    groups: list[PlotGroup] | list[BarData]  # Organized by subplot
    metric_names: list[str]    # All metrics being plotted

    # Metadata for styling
    unique_labels: list[str]   # All unique labels across all plots
    color_palette: list[str]   # Assigned colors

def resolve_field_value(
    field_spec: FieldSpec,
    experiment: Experiment
) -> Any:
    """
    Resolve field value from experiment.

    Order: metadata fields → params → raise error

    Parameters
    ----------
    field_spec : str | tuple | callable
        Field specification
    experiment : Experiment
        Experiment to extract from

    Returns
    -------
    Any
        Resolved value (single value or tuple for multi-field specs)

    Examples
    --------
    resolve_field_value("learning_rate", exp)  # Returns 0.01
    resolve_field_value(("lr", "bs"), exp)     # Returns (0.01, 32)
    resolve_field_value(lambda e: e.id[:4], exp)  # Returns "abc1"
    """
    pass

def format_label(field_spec: FieldSpec, value: Any) -> str:
    """
    Format field value as legend label.

    Examples
    --------
    format_label("learning_rate", 0.01)              # "lr=0.01"
    format_label(("lr", "bs"), (0.01, 32))          # "lr=0.01, bs=32"
    format_label(lambda e: ..., "custom")            # "custom"
    """
    pass

def organize_for_plotting(
    experiment_metrics: list[ExperimentMetrics],
    metric_names: list[str],
    *,
    group_by: FieldSpec | None = None,
    label_by: FieldSpec | None = None,
    subplot_by: FieldSpec | None = None,
    aggregation: str = "mean",
    show_ci: bool = False,
    show_std: bool = False,
    show_individuals: bool = True,
    ci_level: float = 0.95
) -> PlotStructure:
    """
    Organize experiment metrics into plot structure.

    This is the main function that handles all grouping logic.

    Process:
    1. Determine plot_type (line vs bar) from metric step counts
    2. Group experiments by subplot_by (if specified)
    3. Within each subplot group:
       a. If group_by: aggregate experiments, compute stats
       b. If label_by: assign labels and organize as separate lines
    4. Assign colors to unique labels
    5. Return complete PlotStructure

    Parameters
    ----------
    experiment_metrics : list[ExperimentMetrics]
        Extracted metric data from data layer
    metric_names : list[str]
        Metrics to organize
    group_by : FieldSpec, optional
        Field to aggregate by (compute mean/CI/std across)
    label_by : FieldSpec, optional
        Field to assign labels/colors by
        Default: None for single experiment, "id" for multiple
    subplot_by : FieldSpec, optional
        Field to create subplots by
    aggregation : str
        "mean" or "median"
    show_ci : bool
        Show confidence interval (requires group_by)
    show_std : bool
        Show standard deviation (requires group_by)
    show_individuals : bool
        Show individual lines behind aggregation (requires group_by)

    Returns
    -------
    PlotStructure
        Complete structure ready for visualization layer

    Raises
    ------
    ValueError
        If group_by used without label_by, or invalid field specs
    """
    pass

def compute_aggregation(
    lines: list[LineData],
    aggregation: str = "mean",
    ci_level: float = 0.95
) -> LineData:
    """
    Aggregate multiple lines into one with statistics.

    Handles different step counts by aligning on step numbers.

    Returns
    -------
    LineData
        Aggregated line with ci_lower, ci_upper, std_lower, std_upper filled
    """
    pass
```

**Key Decisions:**
- `PlotStructure` is the contract between organization and visualization layers
- Field resolution follows: metadata → params → error
- Label formatting: `"lr=0.01, bs=32"` for tuples
- Returns plot-ready data structures (web UI can serialize these to JSON)
- Default `label_by`: None (single exp) or "id" (multiple exps)

---

### 3. Visualization Layer: `metrics_plotting.py`

**Purpose**: Render plots using matplotlib, apply styling

```python
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

def create_plot(
    plot_structure: PlotStructure,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    grid: bool = True,
    legend: bool = True,
    legend_position: str = "best",
    show: bool = True,
    return_axes: bool = False
) -> Figure | tuple[Figure, Axes | np.ndarray]:
    """
    Create complete plot from PlotStructure.

    Main entry point for visualization layer.

    Parameters
    ----------
    plot_structure : PlotStructure
        Organized data from organization layer
    title : str, optional
        Figure title (default: auto-generated from metrics)
    xlabel : str, optional
        X-axis label (default: "Step" for line, "Experiment" for bar)
    ylabel : str, optional
        Y-axis label (default: metric name if single, "Value" if multiple)
    figsize : tuple, optional
        Figure size (default: auto-computed based on subplots)
    grid : bool
        Show grid
    legend : bool
        Show legend
    legend_position : str
        Matplotlib legend location
    show : bool
        Display plot immediately
    return_axes : bool
        Return (fig, axes) instead of just fig

    Returns
    -------
    Figure or tuple[Figure, Axes]
    """
    pass

def plot_line(
    ax: Axes,
    line_data: LineData,
    *,
    color: str | None = None,
    alpha: float = 1.0
) -> None:
    """
    Plot a single line on given axes.

    Handles regular lines and aggregated lines with CI/std bands.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    line_data : LineData
        Line data to plot
    color : str, optional
        Line color (uses line_data.color if None)
    alpha : float
        Line transparency
    """
    pass

def plot_confidence_band(
    ax: Axes,
    steps: list[int],
    lower: list[float],
    upper: list[float],
    color: str,
    alpha: float = 0.2,
    label: str | None = None
) -> None:
    """
    Plot shaded confidence/std band.

    Uses ax.fill_between()
    """
    pass

def plot_bar_chart(
    ax: Axes,
    bar_data: BarData,
    *,
    colors: list[str] | None = None
) -> None:
    """
    Plot bar chart for single-step metrics.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes
    bar_data : BarData
        Bar chart data
    colors : list[str], optional
        Bar colors
    """
    pass

def create_subplot_grid(
    nrows: int,
    ncols: int,
    figsize: tuple[float, float] | None = None
) -> tuple[Figure, np.ndarray]:
    """
    Create figure with subplot grid.

    Parameters
    ----------
    nrows, ncols : int
        Grid dimensions
    figsize : tuple, optional
        Figure size (auto-computed if None based on grid size)

    Returns
    -------
    tuple[Figure, np.ndarray]
        Figure and array of axes
    """
    pass

def apply_styling(
    ax: Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid: bool = True,
    legend: bool = True,
    legend_position: str = "best"
) -> None:
    """
    Apply consistent styling to axes.

    Uses publication-ready defaults:
    - White background
    - Professional fonts
    - Grid with subtle color
    - Proper tick sizes
    """
    pass
```

**Key Decisions:**
- All plot functions take `(fig, ax)` as input (or create them)
- Modular: `plot_line()`, `plot_bar_chart()` can be used independently
- Styling is separate and consistent across all plot types
- Returns matplotlib objects for further customization

---

### 4. Styling & Utilities: `styles.py` and `utils.py`

```python
# styles.py

def get_color_palette(n_colors: int, colorblind_safe: bool = True) -> list[str]:
    """
    Get color palette for n_colors.

    Strategy:
    - n <= 10: Use 10-color palette (colorblind-safe or default)
    - 10 < n <= 20: Use 20-color palette
    - n > 20: Use 20-color palette, warn about cycling

    Returns
    -------
    list[str]
        Hex color codes
    """
    pass

def get_plot_style() -> dict:
    """
    Get yanex default plot style.

    Returns dict suitable for plt.rcParams.update()

    Style:
    - White background
    - Professional serif font (if available) or sans-serif
    - Larger font sizes for readability
    - Subtle grid
    - High DPI for PDF export
    """
    pass

# utils.py

def validate_subplot_layout(
    layout: tuple[int, int] | None,
    n_subplots: int
) -> tuple[int, int]:
    """
    Validate and compute subplot layout.

    Parameters
    ----------
    layout : tuple | None
        User-specified (rows, cols) or None for default (single row)
    n_subplots : int
        Number of subplots needed

    Returns
    -------
    tuple[int, int]
        Validated (rows, cols)

    Raises
    ------
    ValueError
        If layout doesn't match n_subplots (rows * cols != n_subplots)
    """
    pass

def auto_compute_figsize(
    nrows: int,
    ncols: int,
    base_width: float = 6.0,
    base_height: float = 4.0
) -> tuple[float, float]:
    """
    Compute figure size based on subplot grid.

    Returns
    -------
    tuple[float, float]
        (width, height) in inches
    """
    pass

def normalize_metric_name(metric: str | list[str]) -> list[str]:
    """
    Normalize metric input to list.

    "accuracy" -> ["accuracy"]
    ["acc", "loss"] -> ["acc", "loss"]
    """
    pass
```

---

### 5. High-Level API: `yanex/results/__init__.py`

```python
def plot_metrics(
    metrics: str | list[str],
    *,
    # Filtering (passed to get_experiments)
    ids: list[str] | None = None,
    status: str | list[str] | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
    started_after: str | None = None,
    started_before: str | None = None,
    # ... all other filters from get_experiments

    # Dimensionality control
    group_by: FieldSpec | None = None,
    label_by: FieldSpec | None = None,
    subplot_by: FieldSpec | None = None,

    # Aggregation (requires group_by)
    aggregation: str = "mean",
    show_ci: bool = False,
    show_std: bool = False,
    show_individuals: bool = True,
    ci_level: float = 0.95,

    # Plot type
    plot_type: str = "auto",

    # Subplot layout
    subplot_layout: tuple[int, int] | None = None,

    # Styling
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    colors: list[str] | None = None,
    colorblind_safe: bool = True,
    grid: bool = True,
    legend: bool = True,
    legend_position: str = "best",

    # Output control
    show: bool = True,
    return_axes: bool = False
) -> Figure | tuple[Figure, Axes | np.ndarray]:
    """
    Plot metrics with automatic layout and styling.

    Implementation:
    1. Call get_experiments(**filters) to get experiments
    2. Call extract_metrics(experiments, metrics) [Data Layer]
    3. Call organize_for_plotting(...) [Organization Layer]
    4. Call create_plot(plot_structure, ...) [Visualization Layer]
    5. Return result
    """
    from yanex.visualization.metrics_data import extract_metrics
    from yanex.visualization.metrics_grouping import organize_for_plotting
    from yanex.visualization.metrics_plotting import create_plot
    from yanex.visualization.utils import normalize_metric_name

    # Step 1: Get experiments
    experiments = get_experiments(
        ids=ids, status=status, tags=tags, name=name,
        started_after=started_after, started_before=started_before,
        # ... pass all filters
    )

    if not experiments:
        raise ValueError("No experiments found matching filters")

    # Step 2: Normalize and extract metrics
    metric_names = normalize_metric_name(metrics)
    experiment_metrics = extract_metrics(experiments, metric_names)

    # Step 3: Organize for plotting
    plot_structure = organize_for_plotting(
        experiment_metrics,
        metric_names,
        group_by=group_by,
        label_by=label_by,
        subplot_by=subplot_by,
        aggregation=aggregation,
        show_ci=show_ci,
        show_std=show_std,
        show_individuals=show_individuals,
        ci_level=ci_level
    )

    # Step 4: Create plot
    return create_plot(
        plot_structure,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        grid=grid,
        legend=legend,
        legend_position=legend_position,
        show=show,
        return_axes=return_axes
    )
```

---

## Data Flow Example

### Example 1: Simple Training Curve
```python
yr.plot_metrics("accuracy", ids=["abc123"])
```

**Flow:**
1. **Data Layer**: Extract `accuracy` metric from experiment `abc123`
   ```python
   ExperimentMetrics(
       experiment_id="abc123",
       metric_values={"accuracy": [MetricValue(step=0, value=0.8), ...]},
       is_multi_step=True
   )
   ```

2. **Organization Layer**: Single experiment, no grouping
   ```python
   PlotStructure(
       plot_type="line",
       subplot_grid=None,  # Single plot
       groups=[
           PlotGroup(
               subplot_key=None,
               lines=[
                   LineData(
                       label=None,  # No label for single line
                       steps=[0, 1, 2, ...],
                       values=[0.8, 0.85, 0.9, ...]
                   )
               ],
               metric_name="accuracy"
           )
       ]
   )
   ```

3. **Visualization Layer**: Single line plot, no legend

---

### Example 2: Compare Multiple Experiments
```python
yr.plot_metrics("loss", tags=["baseline"], label_by="learning_rate")
```

**Assumptions:** 3 experiments with different learning rates

**Flow:**
1. **Data Layer**: Extract `loss` from 3 experiments

2. **Organization Layer**: Group by label_by
   ```python
   PlotStructure(
       plot_type="line",
       subplot_grid=None,
       groups=[
           PlotGroup(
               subplot_key=None,
               lines=[
                   LineData(label="lr=0.001", steps=[...], values=[...]),
                   LineData(label="lr=0.01", steps=[...], values=[...]),
                   LineData(label="lr=0.1", steps=[...], values=[...])
               ],
               metric_name="loss"
           )
       ],
       unique_labels=["lr=0.001", "lr=0.01", "lr=0.1"],
       color_palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
   )
   ```

3. **Visualization Layer**: 3 colored lines with legend

---

### Example 3: Subplots with Grouping
```python
yr.plot_metrics(
    "f1_score",
    tags=["experiment"],
    group_by="random_seed",
    label_by="learning_rate",
    subplot_by="model_type",
    show_ci=True
)
```

**Assumptions:**
- 2 model types: "cnn", "rnn"
- 3 learning rates: 0.001, 0.01, 0.1
- 5 random seeds per combination

**Flow:**
1. **Data Layer**: Extract `f1_score` from 2×3×5 = 30 experiments

2. **Organization Layer**:
   - Split by `subplot_by="model_type"` → 2 groups
   - Within each group, group by `random_seed` → 3 LR groups (5 seeds each)
   - Aggregate each LR group → mean + CI

   ```python
   PlotStructure(
       plot_type="line",
       subplot_grid=(1, 2),  # 2 subplots (single row default)
       groups=[
           PlotGroup(
               subplot_key="cnn",
               lines=[
                   LineData(
                       label="lr=0.001",
                       steps=[...],
                       values=[...],  # Mean across 5 seeds
                       is_aggregated=True,
                       ci_lower=[...],
                       ci_upper=[...],
                       individual_lines=[...]  # 5 faint lines
                   ),
                   # ... lr=0.01, lr=0.1
               ],
               metric_name="f1_score"
           ),
           PlotGroup(
               subplot_key="rnn",
               lines=[...]  # Same structure
           )
       ],
       unique_labels=["lr=0.001", "lr=0.01", "lr=0.1"],
       color_palette=["#1f77b4", "#ff7f0e", "#2ca02c"]
   )
   ```

3. **Visualization Layer**:
   - Create 1×2 subplot grid
   - Left subplot: "cnn" with 3 lines + CI bands + faint individuals
   - Right subplot: "rnn" with 3 lines + CI bands + faint individuals
   - Shared legend

---

## Design Decisions Summary

### 1. Default Behaviors
- **label_by**: `None` (single exp) or `"id"` (multiple exps)
- **subplot_layout**: `None` → single row, `(rows, cols)` → must match count
- **plot_type**: `"auto"` → line for multi-step, bar for single-step
- **colors**: 10-color palette (≤10 labels), 20-color (>10), warn+cycle (>20)

### 2. Field Resolution
- Order: metadata → params → error
- Tuple fields: combine with `", "` separator
- Lambda: user controls exact output

### 3. Validation
- Must specify `metrics` (required)
- Single/multi-step must be consistent across experiments for same metric
- `subplot_layout` tuple must match subplot count
- `show_ci`/`show_std` require `group_by`

### 4. Plot Style
- White background
- Professional fonts
- Publication-ready
- Colorblind-safe by default
- Subtle grid

### 5. Modularity
Each layer can be used independently:
```python
# Web UI can do this:
from yanex.visualization.metrics_data import extract_metrics
from yanex.visualization.metrics_grouping import organize_for_plotting

experiments = yr.get_experiments(tags=["sweep"])
data = extract_metrics(experiments, ["accuracy"])
structure = organize_for_plotting(data, ["accuracy"], label_by="lr")
# Convert structure to JSON, send to frontend, plot with D3.js

# Advanced user can do this:
structure = organize_for_plotting(...)
fig, axes = plt.subplots(2, 2)
for i, group in enumerate(structure.groups):
    for line in group.lines:
        plot_line(axes.flat[i], line)
    apply_styling(axes.flat[i], title=f"Model: {group.subplot_key}")
plt.show()
```

---

## File Locations

```
yanex/
├── visualization/
│   ├── __init__.py                    # Export public utilities
│   ├── metrics_data.py                # Data extraction (200-300 LOC)
│   ├── metrics_grouping.py            # Organization (400-500 LOC)
│   ├── metrics_plotting.py            # Visualization (300-400 LOC)
│   ├── styles.py                      # Styling (100-150 LOC)
│   └── utils.py                       # Utilities (100-150 LOC)
│
├── results/
│   └── __init__.py                    # Add plot_metrics() (50-100 LOC)
│
└── tests/
    └── visualization/
        ├── test_metrics_data.py
        ├── test_metrics_grouping.py
        ├── test_metrics_plotting.py
        └── test_integration.py
```

---

## Open Questions

1. **Multiple metrics + subplot_by**: Should metrics be rows or columns?
   - **Proposal**: Metrics as rows, subplot_by as columns
   - Example: 2 metrics × 3 batch sizes → 2×3 grid

2. **Missing metrics**: What if metric exists in some experiments but not others?
   - **Proposal**: Warn and skip experiments missing that metric

3. **Timestamp vs Step**: Should we support plotting by timestamp instead of step?
   - **Proposal**: Add later as `x_axis="step"` or `x_axis="time"` parameter

4. **Export to JSON**: Should `PlotStructure` be serializable?
   - **Proposal**: Add `to_dict()` method for web UI JSON export

5. **Subplot titles**: When using `subplot_by`, show subplot key in title?
   - **Proposal**: Yes, auto-generate: "f1_score (model_type=cnn)"

6. **Step alignment**: Different experiments have different step counts. Align on step number or interpolate?
   - **Proposal**: Align on step number (no interpolation), warn if very mismatched

---

## Next Steps

1. **Review this design** - Get feedback on API and architecture
2. **Implement data layer** - Start with `metrics_data.py`
3. **Implement organization layer** - Then `metrics_grouping.py`
4. **Implement visualization layer** - Then `metrics_plotting.py`
5. **Add high-level API** - Finally `plot_metrics()` in results
6. **Write tests** - Comprehensive test coverage
7. **Documentation** - Docstrings and examples
8. **Web UI integration** - Work with student on JSON serialization

---

## Dependencies

**New dependencies needed:**
- `matplotlib` (plotting)
- `numpy` (array operations for aggregation)
- `scipy` (for confidence intervals)

**Already available:**
- `pandas` (already used in comparison)

Add to `pyproject.toml`:
```toml
[project]
dependencies = [
    # ... existing ...
    "matplotlib>=3.5.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
]
```
