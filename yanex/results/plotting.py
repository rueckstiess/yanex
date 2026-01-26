"""Metrics visualization utilities for yanex experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_metrics(
    df: pd.DataFrame,
    metrics: str | list[str] | None = None,
    group_by: str
    | list[str]
    | Literal["params"]
    | Callable[[pd.Series], str]
    | None = None,
    sort_by: Literal["value", "group"] | None = None,
    smooth_window: int | None = None,
    show_individual: bool = True,
    alpha_individual: float = 0.3,
    colors: str | list[str] | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[Figure, np.ndarray]:
    """
    Plot metrics from a yr.get_metrics() DataFrame.

    Args:
        df: DataFrame from yr.get_metrics() with columns:
            experiment_id, step, metric_name, value, plus any param/meta columns

        metrics: Which metrics to plot. If None, plots all unique metric_name values.

        group_by: How to group experiments for coloring/aggregation:
            - None: Each experiment is its own group (default)
            - str: Single column name to group by (e.g., "lr", "name")
            - list[str]: Multiple columns - creates combined group key
            - "params": Auto-detect all param columns and group by their combinations
            - Callable[[pd.Series], str]: Custom function receiving first row per experiment

        sort_by: How to order groups in the plot:
            - None: Alphabetical sort (default)
            - "value": Sort ascending by metric value (mean for multi-experiment groups)
            - "group": Sort ascending by group parameters with smart numeric sort

        smooth_window: EMA smoothing span. Smooths individual line (1 per group)
            or mean line (multiple per group). None = no smoothing.

        show_individual: Show faint individual runs when multiple experiments per group.

        alpha_individual: Transparency for individual run lines/points (0.0-1.0).

        colors: Color scheme for groups:
            - None: matplotlib default color cycle
            - str: matplotlib/seaborn colormap name (e.g., "tab10", "Set2", "viridis")
            - list[str]: List of color strings to cycle through

        figsize: Figure size as (width, height). Auto-calculated if None.

    Returns:
        (fig, axes): matplotlib Figure and array of Axes for further customization.
            axes is always a 1D numpy array, even for single plots.

    Example:
        >>> import yanex.results as yr
        >>> df = yr.get_metrics(tags=["sweep"], params="auto")
        >>> fig, axes = yr.plot_metrics(df, metrics=["train_loss", "val_loss"])
        >>> fig.savefig("metrics.png")
    """
    import matplotlib.pyplot as plt

    # Validate required columns
    required_cols = {"experiment_id", "step", "metric_name", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Handle empty DataFrame
    if df.empty:
        fig, ax = plt.subplots(1, 1, figsize=figsize or (10, 6))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig, np.array([ax])

    # Determine metrics to plot
    if metrics is None:
        metrics_list = df["metric_name"].unique().tolist()
    elif isinstance(metrics, str):
        metrics_list = [metrics]
    else:
        metrics_list = list(metrics)

    if not metrics_list:
        raise ValueError("No metrics to plot")

    # Resolve grouping
    df = df.copy()
    df["_group"] = _resolve_group_by(df, group_by)

    # Calculate layout
    n_metrics = len(metrics_list)
    rows, cols, default_figsize = _calculate_layout(n_metrics)
    if figsize is None:
        figsize = default_figsize

    # Create figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # Get colors
    n_groups = df["_group"].nunique()
    color_list = _get_colors(colors, n_groups)

    # Plot each metric
    for idx, metric_name in enumerate(metrics_list):
        ax = axes_flat[idx]
        metric_df = df[df["metric_name"] == metric_name]

        if metric_df.empty:
            ax.text(
                0.5,
                0.5,
                f"No data for '{metric_name}'",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(_format_metric_title(metric_name))
            continue

        # Detect if single-step or multi-step
        if _is_single_step_metric(metric_df):
            _plot_bar_metric(
                ax=ax,
                metric_df=metric_df,
                show_individual=show_individual,
                alpha_individual=alpha_individual,
                colors=color_list,
                sort_by=sort_by,
            )
        else:
            _plot_line_metric(
                ax=ax,
                metric_df=metric_df,
                smooth_window=smooth_window,
                show_individual=show_individual,
                alpha_individual=alpha_individual,
                colors=color_list,
                sort_by=sort_by,
            )

        ax.set_title(_format_metric_title(metric_name))

    # Hide unused subplots
    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    return fig, axes_flat[:n_metrics]


def _resolve_column(df: pd.DataFrame, query: str) -> str:
    """Resolve a column name query to actual column name.

    Supports suffix-based resolution similar to AccessResolver:
    - Exact match: "origami.pipeline.lr" -> "origami.pipeline.lr"
    - Suffix match: "lr" -> "origami.pipeline.lr" (if unambiguous)

    Args:
        df: DataFrame with columns to search
        query: Column name or suffix to resolve

    Returns:
        Resolved column name

    Raises:
        ValueError: If column not found or ambiguous
    """
    # Standard columns that shouldn't be matched
    standard_cols = {"experiment_id", "step", "metric_name", "value", "_group"}

    # Available columns for resolution
    available = [c for c in df.columns if c not in standard_cols]

    # Exact match
    if query in available:
        return query

    # Suffix match: find columns ending with .{query}
    matches = [c for c in available if c.endswith(f".{query}")]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(
            f"Ambiguous column '{query}' matches multiple columns: {matches}"
        )
    else:
        raise ValueError(f"Column '{query}' not found. Available: {available}")


def _resolve_group_by(
    df: pd.DataFrame,
    group_by: str | list[str] | Literal["params"] | Callable[[pd.Series], str] | None,
) -> pd.Series:
    """Create a Series of group labels for each row."""
    if group_by is None:
        # Each experiment is its own group
        return df["experiment_id"].astype(str)

    if group_by == "params":
        # Auto-detect param columns (everything except standard columns)
        standard_cols = {"experiment_id", "step", "metric_name", "value"}
        param_cols = sorted([c for c in df.columns if c not in standard_cols])
        if not param_cols:
            # No param columns, fall back to experiment_id
            return df["experiment_id"].astype(str)
        group_by = param_cols

    if isinstance(group_by, str):
        group_by = [group_by]

    if isinstance(group_by, list):
        # Resolve column names (supports suffix matching like "lr" -> "train.lr")
        resolved_cols = [_resolve_column(df, col) for col in group_by]

        # Create combined label from columns
        def make_label(row: pd.Series) -> str:
            parts = []
            for col in resolved_cols:
                # Shorten column name (take last part after '.')
                short_name = col.split(".")[-1]
                parts.append(f"{short_name}={row[col]}")
            return ", ".join(parts)

        return df.apply(make_label, axis=1)

    if callable(group_by):
        # Apply callable to first row of each experiment
        exp_to_group: dict[str, str] = {}
        for exp_id in df["experiment_id"].unique():
            first_row = df[df["experiment_id"] == exp_id].iloc[0]
            exp_to_group[exp_id] = str(group_by(first_row))
        return df["experiment_id"].map(exp_to_group)

    raise ValueError(f"Invalid group_by value: {group_by}")


def _get_colors(colors_arg: str | list[str] | None, n_groups: int) -> list:
    """Resolve colors argument to list of color values."""
    import matplotlib.pyplot as plt

    if colors_arg is None:
        # Use matplotlib's default prop_cycle
        return [f"C{i}" for i in range(n_groups)]

    if isinstance(colors_arg, str):
        # Treat as colormap name
        cmap = plt.get_cmap(colors_arg)
        if n_groups == 1:
            return [cmap(0.5)]
        return [cmap(i / (n_groups - 1)) for i in range(n_groups)]

    # List of colors - cycle if needed
    return [colors_arg[i % len(colors_arg)] for i in range(n_groups)]


def _calculate_layout(n_metrics: int) -> tuple[int, int, tuple[float, float]]:
    """Calculate (rows, cols, figsize) for given number of metrics."""
    if n_metrics == 1:
        return (1, 1, (10, 6))
    elif n_metrics == 2:
        return (1, 2, (14, 5))
    elif n_metrics <= 4:
        rows = 2
        cols = 2
        return (rows, cols, (12, 10))
    else:
        cols = 3
        rows = (n_metrics + cols - 1) // cols
        return (rows, cols, (5 * cols, 4 * rows))


def _is_single_step_metric(metric_df: pd.DataFrame) -> bool:
    """Check if metric has only one step per experiment."""
    steps_per_exp = metric_df.groupby("experiment_id")["step"].nunique()
    return steps_per_exp.max() == 1


def _format_metric_title(metric_name: str) -> str:
    """Format metric name as title (replace underscores with spaces, title case)."""
    return metric_name.replace("_", " ").title()


def _parse_group_sort_key(group_label: str) -> tuple:
    """Parse group label into sortable tuple with smart numeric handling.

    Group labels like "lr=0.01, epochs=10" are split into key=value pairs.
    Values are converted to floats when possible for numeric sorting.

    Args:
        group_label: Group label string (e.g., "lr=0.01, epochs=10")

    Returns:
        Tuple of (type_priority, value) pairs for sorting.
        Numeric values get type_priority=0, strings get type_priority=1.
    """
    parts = []
    for item in group_label.split(", "):
        if "=" in item:
            _, value = item.split("=", 1)
            try:
                parts.append((0, float(value)))  # Numeric: sort as float
            except ValueError:
                parts.append((1, value))  # String: sort after numbers
        else:
            parts.append((1, item))  # No '=': treat as string
    return tuple(parts)


def _sort_groups(
    metric_df: pd.DataFrame,
    groups: list[str],
    sort_by: Literal["value", "group"] | None,
) -> list[str]:
    """Sort groups based on sort_by parameter.

    Args:
        metric_df: DataFrame with _group and value columns
        groups: List of unique group labels to sort
        sort_by: Sorting mode (None, "value", or "group")

    Returns:
        Sorted list of group labels
    """
    if sort_by is None:
        return sorted(groups)  # Default alphabetical sort

    if sort_by == "value":
        # Sort by mean metric value ascending
        group_means = {
            g: metric_df[metric_df["_group"] == g]["value"].mean() for g in groups
        }
        return sorted(groups, key=lambda g: group_means[g])

    if sort_by == "group":
        return sorted(groups, key=_parse_group_sort_key)

    return sorted(groups)


def _plot_line_metric(
    ax: Axes,
    metric_df: pd.DataFrame,
    smooth_window: int | None,
    show_individual: bool,
    alpha_individual: float,
    colors: list,
    sort_by: Literal["value", "group"] | None = None,
) -> None:
    """Plot multi-step metric as line chart."""
    all_groups = list(metric_df["_group"].unique())

    # Assign colors based on alphabetical order for consistency across sort modes
    alphabetical_groups = sorted(all_groups)
    group_to_color = {
        g: colors[i % len(colors)] for i, g in enumerate(alphabetical_groups)
    }

    # Get display order based on sort_by
    unique_groups = _sort_groups(metric_df, all_groups, sort_by)

    for group_label in unique_groups:
        group_data = metric_df[metric_df["_group"] == group_label]
        color = group_to_color[group_label]
        exp_ids = group_data["experiment_id"].unique()
        n_experiments = len(exp_ids)

        if n_experiments == 1:
            # Single experiment in group - just plot the line
            exp_data = group_data.sort_values("step")
            y = exp_data["value"].values
            if smooth_window:
                y = pd.Series(y).ewm(span=smooth_window, adjust=False).mean().values
            ax.plot(exp_data["step"], y, color=color, linewidth=1, label=group_label)
        else:
            # Multiple experiments - show individuals + mean
            if show_individual:
                for exp_id in exp_ids:
                    exp_data = group_data[
                        group_data["experiment_id"] == exp_id
                    ].sort_values("step")
                    ax.plot(
                        exp_data["step"],
                        exp_data["value"],
                        color=color,
                        alpha=alpha_individual,
                        linewidth=1,
                    )

            # Calculate and plot mean
            mean_data = group_data.groupby("step")["value"].mean()
            y = mean_data.values
            if smooth_window:
                y = pd.Series(y).ewm(span=smooth_window, adjust=False).mean().values
            ax.plot(mean_data.index, y, color=color, linewidth=2, label=group_label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_bar_metric(
    ax: Axes,
    metric_df: pd.DataFrame,
    show_individual: bool,
    alpha_individual: float,
    colors: list,
    sort_by: Literal["value", "group"] | None = None,
) -> None:
    """Plot single-step metric as bar chart."""
    all_groups = list(metric_df["_group"].unique())

    # Assign colors based on alphabetical order for consistency across sort modes
    alphabetical_groups = sorted(all_groups)
    group_to_color = {
        g: colors[i % len(colors)] for i, g in enumerate(alphabetical_groups)
    }

    # Get display order based on sort_by
    unique_groups = _sort_groups(metric_df, all_groups, sort_by)
    x_positions = np.arange(len(unique_groups))

    means = []
    stds = []

    for group_label in unique_groups:
        group_values = metric_df[metric_df["_group"] == group_label]["value"]
        means.append(group_values.mean())
        stds.append(group_values.std() if len(group_values) > 1 else 0)

    bar_colors = [group_to_color[g] for g in unique_groups]
    ax.bar(x_positions, means, yerr=stds, capsize=5, color=bar_colors, alpha=0.8)

    # Add individual points
    if show_individual:
        for i, group_label in enumerate(unique_groups):
            group_values = metric_df[metric_df["_group"] == group_label]["value"]
            if len(group_values) > 1:
                ax.scatter(
                    [i] * len(group_values),
                    group_values,
                    color=bar_colors[i],
                    s=30,
                    zorder=3,
                    alpha=alpha_individual,
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(unique_groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3, axis="y")
