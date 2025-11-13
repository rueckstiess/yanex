# Yanex Visualization Utilities - Pandas-Based Design

**Version**: 0.2 (Alternative Design)
**Date**: 2025-11-13
**Status**: Draft for Review

## Overview

Alternative design using **pandas DataFrames** as the primary intermediate data structure, eliminating custom implementations of grouping, aggregation, and pivoting logic.

---

## Why Pandas?

### Problems with Original Design
- Custom data structures (`ExperimentMetrics`, `LineData`, `PlotGroup`)
- Reimplementing groupby, aggregation, pivoting from scratch
- 400-500 LOC for organization layer doing what pandas does natively
- Separate serialization logic for web UI

### Pandas Advantages
- **Built-in operations**: `groupby()`, `agg()`, `pivot()`, `rolling()`, statistical functions
- **Less code**: ~50% reduction in organization layer complexity
- **Familiar**: Users know pandas, easier customization
- **Tested & optimized**: Battle-tested library with numpy backend
- **Web UI**: `df.to_dict()` / `df.to_json()` built-in
- **Consistency**: `yr.compare()` already returns DataFrames

---

## Architecture Layers (Pandas-Based)

```
┌─────────────────────────────────────────────────────────┐
│  High-Level API: yr.plot_metrics()                      │
│  - User-facing convenience function                     │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Data Layer: metrics_data.py                            │
│  - Extract metrics into long-format DataFrame           │
│  - Columns: experiment_id, step, metric_name, value,    │
│             timestamp, plus metadata/params              │
│  - Returns: pd.DataFrame                                │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Organization Layer: metrics_grouping.py                │
│  - Uses pandas groupby/agg/pivot operations             │
│  - Resolves field specs to column names                 │
│  - Returns: Organized DataFrame(s) ready for plotting   │
│  - Much simpler than custom implementation!             │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Visualization Layer: metrics_plotting.py               │
│  - Takes DataFrame(s) and plots                         │
│  - Iterates over groups/subplots from DataFrame         │
│  - Apply styling (colors, fonts, accessibility)         │
└─────────────────────────────────────────────────────────┘
```

---

## Component APIs (Pandas Version)

### 1. Data Layer: `metrics_data.py`

**Purpose**: Extract metrics into pandas long-format DataFrame

```python
import pandas as pd
from yanex.results.experiment import Experiment

def extract_metrics_df(
    experiments: list[Experiment],
    metric_names: list[str],
    include_metadata: list[str] | None = None,
    include_params: list[str] | None = None
) -> pd.DataFrame:
    """
    Extract metrics into long-format DataFrame.

    Parameters
    ----------
    experiments : list[Experiment]
        Experiments to extract from
    metric_names : list[str]
        Metrics to extract
    include_metadata : list[str], optional
        Metadata fields to include (e.g., ["status", "name"])
        If None, auto-discovers commonly used fields
    include_params : list[str], optional
        Parameter fields to include (e.g., ["learning_rate", "batch_size"])
        If None, auto-discovers all params

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns:
        - experiment_id: str
        - step: int | None (None for single-step metrics)
        - timestamp: str
        - <metric_1>: float (one column per metric)
        - <metric_2>: float
        - ...
        - <metadata_fields>: various types (e.g., status, name, tags)
        - <param_fields>: various types (e.g., learning_rate, batch_size)

    Raises
    ------
    ValueError
        If metric not found or inconsistent step counts

    Examples
    --------
    >>> df = extract_metrics_df(experiments, ["accuracy", "loss"])
    >>> df.head()
       experiment_id  step  accuracy  loss  learning_rate model_type
    0  abc12345      0     0.80      0.20  0.001         cnn
    1  abc12345      1     0.85      0.15  0.001         cnn
    2  abc12345      2     0.90      0.10  0.001         cnn
    3  def67890      0     0.75      0.25  0.01          rnn
    4  def67890      1     0.82      0.18  0.01          rnn

    >>> # For single-step metrics, step is None
    >>> df = extract_metrics_df(experiments, ["final_accuracy"])
    >>> df.head()
       experiment_id  step  final_accuracy  learning_rate
    0  abc12345      None  0.95            0.001
    1  def67890      None  0.93            0.01
    """
    rows = []

    for exp in experiments:
        # Get all metrics for this experiment
        metrics_list = exp.get_metrics()  # List of dicts with step/metrics

        # Detect single-step vs multi-step
        if isinstance(metrics_list, dict):
            metrics_list = [metrics_list]

        # Extract metadata/params once per experiment
        metadata = _extract_metadata(exp, include_metadata)
        params = exp.get_params() if include_params is None else {
            k: exp.get_param(k) for k in include_params
        }

        # Create row for each step
        for metric_entry in metrics_list:
            row = {
                'experiment_id': exp.id,
                'step': metric_entry.get('step'),
                'timestamp': metric_entry.get('timestamp'),
                **{name: metric_entry.get(name) for name in metric_names},
                **metadata,
                **params
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Validation
    _validate_metrics_consistency(df, metric_names)

    return df


def _extract_metadata(
    exp: Experiment,
    fields: list[str] | None
) -> dict:
    """Extract metadata fields from experiment."""
    if fields is None:
        # Auto-detect common fields
        fields = ['name', 'status', 'tags']

    metadata = {}
    for field in fields:
        if hasattr(exp, field):
            value = getattr(exp, field)
            # Convert lists to strings for DataFrame compatibility
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            metadata[field] = value

    return metadata


def _validate_metrics_consistency(
    df: pd.DataFrame,
    metric_names: list[str]
) -> None:
    """
    Validate that metrics are consistently single/multi-step.

    Raises ValueError if same metric is single-step in some experiments
    and multi-step in others.
    """
    for metric in metric_names:
        # Check if metric exists
        if metric not in df.columns or df[metric].isna().all():
            raise ValueError(f"Metric '{metric}' not found in any experiments")

    # Check consistency: within each experiment_id, step should be all None or all not-None
    for exp_id in df['experiment_id'].unique():
        exp_df = df[df['experiment_id'] == exp_id]
        has_steps = exp_df['step'].notna()

        if has_steps.any() and not has_steps.all():
            raise ValueError(
                f"Experiment {exp_id} has inconsistent step data "
                f"(some metrics with steps, some without)"
            )


def detect_plot_type(df: pd.DataFrame) -> str:
    """
    Auto-detect plot type from DataFrame.

    Returns "line" for multi-step, "bar" for single-step.
    """
    if df['step'].isna().all():
        return "bar"
    return "line"
```

**Key Points:**
- Returns **long-format** DataFrame (tidy data)
- One row per (experiment, step) combination
- Metric values as columns (accuracy, loss, etc.)
- Metadata/params as additional columns
- Step is `None` for single-step metrics
- Validation happens here

---

### 2. Organization Layer: `metrics_grouping.py`

**Purpose**: Use pandas operations to organize data for plotting

```python
import pandas as pd
from typing import Callable, Any
from yanex.results.experiment import Experiment

# Type alias
FieldSpec = str | tuple[str, ...] | Callable[[Experiment], Any]

def organize_for_plotting(
    df: pd.DataFrame,
    metric_names: list[str],
    *,
    group_by: str | list[str] | None = None,
    label_by: str | list[str] | None = None,
    subplot_by: str | list[str] | None = None,
    aggregation: str = "mean",
    ci_level: float = 0.95
) -> dict[str, pd.DataFrame]:
    """
    Organize DataFrame using pandas operations for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame from extract_metrics_df()
    metric_names : list[str]
        Metrics to plot
    group_by : str | list[str], optional
        Column(s) to group by for aggregation
    label_by : str | list[str], optional
        Column(s) for legend labels
    subplot_by : str | list[str], optional
        Column(s) for subplot splitting
    aggregation : str
        "mean" or "median"
    ci_level : float
        Confidence interval level (e.g., 0.95 for 95%)

    Returns
    -------
    dict[str, pd.DataFrame]
        {
            "plot_type": "line" or "bar",
            "subplots": {
                subplot_key: {
                    "data": pd.DataFrame (organized for this subplot),
                    "metric": metric_name
                },
                ...
            },
            "metadata": {
                "unique_labels": [...],
                "color_palette": [...],
                "has_aggregation": bool,
                "has_individuals": bool (if group_by)
            }
        }

    Examples
    --------
    # Simple case: plot all experiments as separate lines
    >>> result = organize_for_plotting(df, ["accuracy"], label_by="experiment_id")
    >>> result["subplots"]["accuracy"]["data"]
       step  experiment_id  accuracy
    0  0     exp1          0.80
    1  1     exp1          0.85
    2  0     exp2          0.75
    3  1     exp2          0.82

    # Aggregation case: group by random_seed, label by learning_rate
    >>> result = organize_for_plotting(
    ...     df, ["accuracy"],
    ...     group_by="random_seed",
    ...     label_by="learning_rate"
    ... )
    >>> result["subplots"]["accuracy"]["data"]
       step  learning_rate  mean  ci_lower  ci_upper  std
    0  0     0.001         0.80  0.75      0.85      0.05
    1  1     0.001         0.85  0.80      0.90      0.05
    2  0     0.01          0.78  0.73      0.83      0.05
    """
    plot_type = detect_plot_type(df)

    # Apply default label_by
    if label_by is None:
        n_experiments = df['experiment_id'].nunique()
        label_by = None if n_experiments == 1 else 'experiment_id'

    # Normalize to lists
    if isinstance(label_by, str):
        label_by = [label_by]
    if isinstance(group_by, str):
        group_by = [group_by]
    if isinstance(subplot_by, str):
        subplot_by = [subplot_by]

    # Create subplot groups
    subplot_groups = _create_subplot_groups(df, subplot_by, metric_names)

    # Process each subplot
    result = {
        "plot_type": plot_type,
        "subplots": {},
        "metadata": {
            "unique_labels": [],
            "color_palette": [],
            "has_aggregation": group_by is not None,
        }
    }

    for subplot_key, subplot_df in subplot_groups.items():
        metric = subplot_key[1] if isinstance(subplot_key, tuple) else subplot_key

        if group_by is not None:
            # Apply aggregation
            organized_df = _apply_aggregation(
                subplot_df,
                metric,
                group_by=group_by,
                label_by=label_by,
                aggregation=aggregation,
                ci_level=ci_level
            )
        else:
            # No aggregation, just organize by label_by
            if label_by:
                organized_df = subplot_df.sort_values(label_by + ['step'])
            else:
                organized_df = subplot_df

        result["subplots"][subplot_key] = {
            "data": organized_df,
            "metric": metric
        }

    # Collect unique labels for color assignment
    if label_by:
        unique_labels = df[label_by].drop_duplicates().values.tolist()
        if len(label_by) > 1:
            # Format tuples as "lr=0.01, bs=32"
            unique_labels = [_format_label(label_by, vals) for vals in unique_labels]
        result["metadata"]["unique_labels"] = unique_labels
        result["metadata"]["color_palette"] = _get_color_palette(len(unique_labels))

    return result


def _create_subplot_groups(
    df: pd.DataFrame,
    subplot_by: list[str] | None,
    metric_names: list[str]
) -> dict[Any, pd.DataFrame]:
    """
    Split DataFrame into subplot groups.

    Returns dict mapping subplot_key -> DataFrame subset
    """
    if subplot_by is None and len(metric_names) == 1:
        # Single subplot
        return {metric_names[0]: df}

    groups = {}

    # Split by metrics (always first dimension)
    for metric in metric_names:
        if subplot_by is None:
            # Just split by metric
            groups[metric] = df
        else:
            # Split by metric AND subplot_by fields
            for subplot_vals, subplot_df in df.groupby(subplot_by):
                key = (tuple(subplot_vals) if len(subplot_by) > 1 else subplot_vals, metric)
                groups[key] = subplot_df

    return groups


def _apply_aggregation(
    df: pd.DataFrame,
    metric: str,
    group_by: list[str],
    label_by: list[str] | None,
    aggregation: str,
    ci_level: float
) -> pd.DataFrame:
    """
    Apply pandas groupby aggregation.

    Groups by: label_by + ['step'], aggregates over group_by experiments.

    Returns DataFrame with columns:
    - step
    - label_by columns
    - mean/median
    - ci_lower, ci_upper
    - std
    """
    # Group by label_by + step
    groupby_cols = (label_by if label_by else []) + ['step']
    grouped = df.groupby(groupby_cols)[metric]

    # Compute statistics
    agg_funcs = {
        'mean': 'mean' if aggregation == 'mean' else lambda x: x.median(),
        'std': 'std',
        'ci_lower': lambda x: x.quantile((1 - ci_level) / 2),
        'ci_upper': lambda x: x.quantile((1 + ci_level) / 2)
    }

    result = grouped.agg(**agg_funcs).reset_index()

    # Also store individual lines for show_individuals
    # (Keep original df for plotting faint lines)
    result.attrs['individual_data'] = df

    return result


def _format_label(fields: list[str], values: tuple | Any) -> str:
    """
    Format field values as label string.

    Examples:
    - ("learning_rate",), (0.01,) -> "lr=0.01"
    - ("lr", "bs"), (0.01, 32) -> "lr=0.01, bs=32"
    """
    if not isinstance(values, tuple):
        values = (values,)

    parts = []
    for field, value in zip(fields, values):
        # Shorten common field names
        short_field = field.replace('learning_rate', 'lr').replace('batch_size', 'bs')
        parts.append(f"{short_field}={value}")

    return ", ".join(parts)


def _get_color_palette(n_colors: int) -> list[str]:
    """Get color palette (same as original design)."""
    # Implementation from styles.py
    pass


def resolve_field_columns(
    field_spec: FieldSpec,
    df: pd.DataFrame,
    experiments: list[Experiment]
) -> list[str]:
    """
    Resolve field specification to DataFrame column names.

    For strings/tuples: return column names directly (already in df)
    For callables: create new column by applying to experiments

    Parameters
    ----------
    field_spec : str | tuple | callable
        Field specification
    df : pd.DataFrame
        DataFrame to add column to (if callable)
    experiments : list[Experiment]
        Experiments (needed for callable resolution)

    Returns
    -------
    list[str]
        Column name(s) in DataFrame
    """
    if callable(field_spec):
        # Create new column from lambda
        exp_map = {exp.id: field_spec(exp) for exp in experiments}
        col_name = '_custom_field'
        df[col_name] = df['experiment_id'].map(exp_map)
        return [col_name]

    if isinstance(field_spec, str):
        if field_spec not in df.columns:
            raise ValueError(f"Field '{field_spec}' not found in DataFrame columns")
        return [field_spec]

    # Tuple
    for field in field_spec:
        if field not in df.columns:
            raise ValueError(f"Field '{field}' not found in DataFrame columns")
    return list(field_spec)
```

**Key Points:**
- Uses pandas `groupby()`, `agg()`, `pivot()` directly
- Returns organized DataFrames (not custom structures)
- Much simpler than custom implementation
- Field resolution maps specs to DataFrame columns
- Stores individual data in DataFrame `.attrs` for show_individuals

---

### 3. Visualization Layer: `metrics_plotting.py`

**Purpose**: Plot organized DataFrames

```python
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

def create_plot(
    plot_structure: dict,
    *,
    show_ci: bool = False,
    show_std: bool = False,
    show_individuals: bool = True,
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
    Create plot from organized DataFrames.

    Parameters
    ----------
    plot_structure : dict
        Result from organize_for_plotting()

    Returns
    -------
    Figure or tuple[Figure, Axes]
    """
    plot_type = plot_structure["plot_type"]
    subplots = plot_structure["subplots"]
    metadata = plot_structure["metadata"]

    # Create subplot grid
    n_subplots = len(subplots)
    if n_subplots == 1:
        fig, ax = plt.subplots(figsize=figsize or (8, 6))
        axes = [ax]
    else:
        nrows, ncols = _compute_subplot_layout(n_subplots)
        figsize = figsize or (6 * ncols, 4 * nrows)
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes_arr.flat

    # Plot each subplot
    color_map = dict(zip(metadata["unique_labels"], metadata["color_palette"]))

    for ax, (subplot_key, subplot_data) in zip(axes, subplots.items()):
        df = subplot_data["data"]
        metric = subplot_data["metric"]

        if plot_type == "line":
            _plot_line_from_df(
                ax, df, metric,
                color_map=color_map,
                has_aggregation=metadata["has_aggregation"],
                show_ci=show_ci,
                show_std=show_std,
                show_individuals=show_individuals
            )
        else:  # bar
            _plot_bar_from_df(ax, df, metric, color_map=color_map)

        # Styling
        ax.set_title(subplot_key if n_subplots > 1 else title)
        ax.set_xlabel(xlabel or ("Step" if plot_type == "line" else "Experiment"))
        ax.set_ylabel(ylabel or metric)
        if grid:
            ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(loc=legend_position)

    plt.tight_layout()

    if show:
        plt.show()

    return (fig, axes) if return_axes else fig


def _plot_line_from_df(
    ax: Axes,
    df: pd.DataFrame,
    metric: str,
    color_map: dict,
    has_aggregation: bool,
    show_ci: bool,
    show_std: bool,
    show_individuals: bool
) -> None:
    """
    Plot line(s) from DataFrame.

    Handles both simple (no aggregation) and aggregated data.
    """
    if not has_aggregation:
        # Simple case: plot each experiment/label as separate line
        label_cols = [col for col in df.columns
                      if col not in ['step', metric, 'timestamp', 'experiment_id']]

        if label_cols:
            for label_vals, group in df.groupby(label_cols):
                label = _format_label_from_values(label_cols, label_vals)
                color = color_map.get(label, None)
                ax.plot(group['step'], group[metric], label=label, color=color)
        else:
            # Single line, no label
            ax.plot(df['step'], df[metric])
    else:
        # Aggregated case: plot mean + bands + individuals

        # Plot individuals first (faint lines)
        if show_individuals and 'individual_data' in df.attrs:
            individual_df = df.attrs['individual_data']
            for exp_id, exp_group in individual_df.groupby('experiment_id'):
                ax.plot(exp_group['step'], exp_group[metric],
                       alpha=0.2, linewidth=0.5, color='gray', zorder=1)

        # Plot aggregated lines with bands
        label_cols = [col for col in df.columns
                      if col not in ['step', 'mean', 'std', 'ci_lower', 'ci_upper']]

        for label_vals, group in df.groupby(label_cols):
            label = _format_label_from_values(label_cols, label_vals)
            color = color_map.get(label, None)

            # Main line
            ax.plot(group['step'], group['mean'], label=label, color=color,
                   linewidth=2, zorder=3)

            # Confidence interval band
            if show_ci and 'ci_lower' in group.columns:
                ax.fill_between(group['step'], group['ci_lower'], group['ci_upper'],
                               alpha=0.2, color=color, zorder=2)

            # Std band
            if show_std and 'std' in group.columns:
                ax.fill_between(group['step'],
                               group['mean'] - group['std'],
                               group['mean'] + group['std'],
                               alpha=0.15, color=color, zorder=2)


def _plot_bar_from_df(
    ax: Axes,
    df: pd.DataFrame,
    metric: str,
    color_map: dict
) -> None:
    """Plot bar chart from DataFrame."""
    # For single-step metrics
    labels = df['experiment_id'].values  # Or name if available
    values = df[metric].values
    colors = [color_map.get(label, None) for label in labels]

    ax.bar(range(len(labels)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')


def _format_label_from_values(fields: list[str], values: tuple | Any) -> str:
    """Format label from field names and values."""
    # Same as _format_label in grouping
    pass


def _compute_subplot_layout(n_subplots: int) -> tuple[int, int]:
    """Compute subplot layout (single row by default)."""
    return (1, n_subplots)
```

**Key Points:**
- Takes organized DataFrames and plots them
- Iterates over DataFrame groups using pandas operations
- Cleaner code since data is already organized
- Can access individual data via `df.attrs`

---

## Simplified High-Level API

```python
# yanex/results/__init__.py

def plot_metrics(
    metrics: str | list[str],
    *,
    # Filtering
    **filters,

    # Organization
    group_by: str | list[str] | Callable | None = None,
    label_by: str | list[str] | Callable | None = None,
    subplot_by: str | list[str] | Callable | None = None,

    # ... other params ...
) -> Figure | tuple[Figure, Axes]:
    """Plot metrics using pandas-based pipeline."""

    from yanex.visualization.metrics_data import extract_metrics_df
    from yanex.visualization.metrics_grouping import (
        organize_for_plotting,
        resolve_field_columns
    )
    from yanex.visualization.metrics_plotting import create_plot

    # 1. Get experiments
    experiments = get_experiments(**filters)

    # 2. Resolve field specs to columns (for lambdas)
    # This step handles lambda functions by creating computed columns

    # 3. Extract to DataFrame
    df = extract_metrics_df(experiments, metric_names)

    # 4. Organize using pandas
    plot_structure = organize_for_plotting(
        df, metric_names,
        group_by=group_by,
        label_by=label_by,
        subplot_by=subplot_by,
        ...
    )

    # 5. Plot
    return create_plot(plot_structure, ...)
```

---

## Web UI Integration (Much Simpler!)

```python
# For web UI: just export the DataFrame to JSON
df = extract_metrics_df(experiments, ["accuracy", "loss"])
json_data = df.to_dict(orient='records')

# Or with organization:
plot_structure = organize_for_plotting(df, ["accuracy"], label_by="learning_rate")

# Export organized data
web_data = {
    "plot_type": plot_structure["plot_type"],
    "subplots": {
        key: {
            "data": subplot["data"].to_dict(orient='records'),
            "metric": subplot["metric"]
        }
        for key, subplot in plot_structure["subplots"].items()
    },
    "metadata": plot_structure["metadata"]
}

import json
json.dumps(web_data)  # Send to frontend
```

---

## Code Reduction Estimate

| Layer | Original (LOC) | Pandas-Based (LOC) | Reduction |
|-------|----------------|-------------------|-----------|
| Data | 200-300 | 150-200 | ~30% |
| Organization | 400-500 | 150-200 | ~60% |
| Visualization | 300-400 | 250-350 | ~20% |
| **Total** | **900-1200** | **550-750** | **~40%** |

**Main savings**:
- No custom data structures
- No manual aggregation logic (use pandas `groupby().agg()`)
- No manual pivoting (use pandas `pivot()`)
- Web UI export built-in (`df.to_dict()`)

---

## Advantages Summary

### Code Quality
- ✅ Less code to write, test, maintain
- ✅ Using battle-tested pandas operations
- ✅ Fewer bugs (pandas handles edge cases)
- ✅ Better performance (optimized numpy backend)

### Usability
- ✅ Familiar API for advanced users
- ✅ Easy to customize (use pandas operations on intermediate results)
- ✅ Consistent with `yr.compare()` (already returns DataFrame)
- ✅ Better debugging (can inspect DataFrames at each step)

### Extensibility
- ✅ Easy to add features (pandas has tons of operations)
- ✅ Web UI integration simpler (built-in JSON export)
- ✅ Can leverage entire pandas ecosystem (seaborn, plotly, etc.)

---

## Example: Before & After

### Original Design (Custom Structures)
```python
# Custom data structure
experiment_metrics = extract_metrics(experiments, ["accuracy"])
# Returns: list[ExperimentMetrics(experiment_id, metric_values, ...)]

# Custom aggregation logic
plot_structure = organize_for_plotting(
    experiment_metrics, ["accuracy"],
    group_by="random_seed",
    label_by="learning_rate"
)
# Manually implements: grouping, mean calculation, CI computation, pivoting
# Returns: PlotStructure(plot_type, groups, unique_labels, ...)
```

### Pandas Design
```python
# Pandas DataFrame
df = extract_metrics_df(experiments, ["accuracy"])
# Returns: pd.DataFrame with columns [experiment_id, step, accuracy, ...]

# Pandas operations
plot_structure = organize_for_plotting(
    df, ["accuracy"],
    group_by="random_seed",
    label_by="learning_rate"
)
# Uses: df.groupby().agg(), df.quantile(), df.pivot()
# Returns: dict with organized DataFrames
```

**Result**: Same functionality, ~60% less code in organization layer!

---

## Open Questions (Pandas Version)

1. **DataFrame memory**: For 1000 experiments × 10k steps, DataFrame could be large (~80MB). Acceptable?
   - **Mitigation**: Only extract needed columns, use categorical dtypes

2. **Lambda field resolution**: How to handle lambdas with pandas?
   - **Option A**: Pre-compute lambda values and add as DataFrame column
   - **Option B**: Keep experiments dict alongside DataFrame for lambda eval
   - **Proposal**: Option A (cleaner)

3. **Individual lines storage**: Store in `df.attrs` or separate dict?
   - **Proposal**: Use `df.attrs['individual_data']` (keeps things together)

4. **Multiple metrics + subplot_by**: Same question as original design
   - **Proposal**: Metrics as rows (first dimension)

---

## Recommendation

**Use pandas-based design** because:
1. ~40% less code overall
2. Leverages proven, optimized library
3. Easier for users to customize (familiar API)
4. Simpler web UI integration
5. Consistent with existing `yr.compare()` DataFrame return

The only downside is slightly higher memory usage, but for typical use cases (100s of experiments, 1000s of steps), this is negligible (~10-50MB).
