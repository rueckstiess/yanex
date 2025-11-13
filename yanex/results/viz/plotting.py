"""Visualization layer for yanex metrics.

Provides matplotlib plotting functions for organized DataFrames.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from yanex.results.viz.grouping import validate_subplot_layout
from yanex.results.viz.styles import get_plot_style


def create_plot(
    plot_structure: dict,
    *,
    show_ci: bool = False,
    show_std: bool = False,
    show_individuals: bool = True,
    subplot_layout: tuple[int, int] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    grid: bool = True,
    legend: bool = True,
    legend_position: str = "best",
    show: bool = True,
    return_axes: bool = False,
) -> Figure | tuple[Figure, np.ndarray | Axes]:
    """
    Create complete plot from organized plot structure.

    Parameters
    ----------
    plot_structure : dict
        Result from organize_for_plotting()
    show_ci : bool, default=False
        Show confidence interval bands (requires aggregation)
    show_std : bool, default=False
        Show standard deviation bands (requires aggregation)
    show_individuals : bool, default=True
        Show individual lines behind aggregation (requires aggregation)
    subplot_layout : tuple[int, int], optional
        (rows, cols) for subplot grid. If None, uses single row.
    title : str, optional
        Figure title (default: auto-generated from metrics)
    xlabel : str, optional
        X-axis label (default: "Step" for line, "Experiment" for bar)
    ylabel : str, optional
        Y-axis label (default: metric name if single, "Value" if multiple)
    figsize : tuple[float, float], optional
        Figure size in inches (default: auto-computed)
    grid : bool, default=True
        Show grid
    legend : bool, default=True
        Show legend
    legend_position : str, default="best"
        Matplotlib legend location
    show : bool, default=True
        Display plot immediately
    return_axes : bool, default=False
        Return (fig, axes) instead of just fig

    Returns
    -------
    Figure or tuple[Figure, Axes]
        Matplotlib figure, or (figure, axes) if return_axes=True

    Examples
    --------
    >>> from yanex.results.viz import extract_metrics_df, organize_for_plotting
    >>> from yanex.results.viz.plotting import create_plot
    >>> import yanex.results as yr
    >>> experiments = yr.get_experiments(tags=["training"])
    >>> df = extract_metrics_df(experiments, ["accuracy"])
    >>> structure = organize_for_plotting(df, ["accuracy"])
    >>> fig = create_plot(structure)
    """
    plot_type = plot_structure["plot_type"]
    subplots = plot_structure["subplots"]
    metadata = plot_structure["metadata"]

    # Validate aggregation flags
    if not metadata["has_aggregation"] and (show_ci or show_std):
        raise ValueError(
            "show_ci and show_std require aggregation. "
            "Specify group_by parameter to enable aggregation."
        )

    # Apply yanex style
    plt.rcParams.update(get_plot_style())

    # Create subplot grid
    n_subplots = len(subplots)
    nrows, ncols = validate_subplot_layout(subplot_layout, n_subplots)

    if n_subplots == 1:
        figsize = figsize or (8, 6)
        fig, ax = plt.subplots(figsize=figsize)
        axes = np.array([ax])
    else:
        figsize = figsize or (6 * ncols, 4 * nrows)
        fig, axes_arr = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes_arr.flatten()

    # Create color mapping
    color_map = {}
    if metadata["unique_labels"] and metadata["color_palette"]:
        color_map = dict(
            zip(metadata["unique_labels"], metadata["color_palette"], strict=False)
        )

    # Plot each subplot
    for ax, (subplot_key, subplot_data) in zip(axes, subplots.items(), strict=False):
        organized_df = subplot_data["data"]
        metric = subplot_data["metric"]
        individual_df = subplot_data["individual_data"]

        if plot_type == "line":
            _plot_line_subplot(
                ax,
                organized_df,
                metric,
                color_map=color_map,
                has_aggregation=metadata["has_aggregation"],
                show_ci=show_ci,
                show_std=show_std,
                show_individuals=show_individuals,
                individual_df=individual_df,
            )
        else:  # bar
            _plot_bar_subplot(ax, organized_df, metric, color_map=color_map)

        # Apply styling
        subplot_title = _format_subplot_title(subplot_key, n_subplots, title)
        ax.set_title(subplot_title)

        ax.set_xlabel(xlabel or ("Step" if plot_type == "line" else "Experiment"))
        ax.set_ylabel(ylabel or metric.replace("_", " ").title())

        if grid:
            ax.grid(True, alpha=0.3)

        if legend and (color_map or metadata["has_aggregation"]):
            ax.legend(loc=legend_position)

    plt.tight_layout()

    if show:
        plt.show()

    if return_axes:
        return fig, axes if n_subplots > 1 else axes[0]
    return fig


def _plot_line_subplot(
    ax: Axes,
    df: pd.DataFrame,
    metric: str,
    color_map: dict,
    has_aggregation: bool,
    show_ci: bool,
    show_std: bool,
    show_individuals: bool,
    individual_df: pd.DataFrame | None,
) -> None:
    """Plot line chart on given axes."""
    if not has_aggregation:
        # Simple case: plot each group as separate line
        _plot_simple_lines(ax, df, metric, color_map)
    else:
        # Aggregated case: plot mean + bands + individuals
        _plot_aggregated_lines(
            ax,
            df,
            metric,
            color_map,
            show_ci=show_ci,
            show_std=show_std,
            show_individuals=show_individuals,
            individual_df=individual_df,
        )


def _plot_simple_lines(
    ax: Axes, df: pd.DataFrame, metric: str, color_map: dict
) -> None:
    """Plot simple lines (no aggregation)."""
    # Find label columns (exclude known columns)
    exclude_cols = {"experiment_id", "step", "timestamp", metric}
    label_cols = [col for col in df.columns if col not in exclude_cols]

    if label_cols:
        # Group by label columns
        for label_vals, group in df.groupby(label_cols):
            # Format label
            if len(label_cols) == 1:
                label = str(label_vals)
            else:
                label = ", ".join(
                    f"{col}={val}"
                    for col, val in zip(label_cols, label_vals, strict=False)
                )

            color = color_map.get(label)
            ax.plot(
                group["step"], group[metric], label=label, color=color, linewidth=1.5
            )
    else:
        # Single line, no label
        ax.plot(df["step"], df[metric], linewidth=1.5)


def _plot_aggregated_lines(
    ax: Axes,
    df: pd.DataFrame,
    metric: str,
    color_map: dict,
    show_ci: bool,
    show_std: bool,
    show_individuals: bool,
    individual_df: pd.DataFrame | None,
) -> None:
    """Plot aggregated lines with confidence bands and individuals."""
    # Plot individuals first (faint lines in background)
    if show_individuals and individual_df is not None:
        for _exp_id, exp_group in individual_df.groupby("experiment_id"):
            ax.plot(
                exp_group["step"],
                exp_group[metric],
                alpha=0.15,
                linewidth=0.8,
                color="gray",
                zorder=1,
            )

    # Find label columns (exclude aggregation result columns)
    exclude_cols = {"step", "mean", "std", "ci_lower", "ci_upper"}
    label_cols = [col for col in df.columns if col not in exclude_cols]

    if label_cols:
        # Group by label columns
        for label_vals, group in df.groupby(label_cols):
            # Format label
            if len(label_cols) == 1:
                label = str(label_vals)
            else:
                label = ", ".join(
                    f"{col}={val}"
                    for col, val in zip(label_cols, label_vals, strict=False)
                )

            color = color_map.get(label)

            # Main line (mean)
            ax.plot(
                group["step"],
                group["mean"],
                label=label,
                color=color,
                linewidth=2,
                zorder=3,
            )

            # Confidence interval band
            if show_ci and "ci_lower" in group.columns:
                ax.fill_between(
                    group["step"],
                    group["ci_lower"],
                    group["ci_upper"],
                    alpha=0.25,
                    color=color,
                    zorder=2,
                )

            # Std band
            if show_std and "std" in group.columns:
                ax.fill_between(
                    group["step"],
                    group["mean"] - group["std"],
                    group["mean"] + group["std"],
                    alpha=0.15,
                    color=color,
                    zorder=2,
                )
    else:
        # Single aggregated line
        ax.plot(df["step"], df["mean"], linewidth=2, zorder=3)

        if show_ci and "ci_lower" in df.columns:
            ax.fill_between(
                df["step"], df["ci_lower"], df["ci_upper"], alpha=0.25, zorder=2
            )

        if show_std and "std" in df.columns:
            ax.fill_between(
                df["step"],
                df["mean"] - df["std"],
                df["mean"] + df["std"],
                alpha=0.15,
                zorder=2,
            )


def _plot_bar_subplot(ax: Axes, df: pd.DataFrame, metric: str, color_map: dict) -> None:
    """Plot bar chart on given axes."""
    # For single-step metrics, use experiment_id or name as x-axis
    if "name" in df.columns:
        labels = df["name"].fillna(df["experiment_id"]).values
    else:
        labels = df["experiment_id"].values

    values = df[metric].values

    # Assign colors if available
    colors = [color_map.get(str(label), None) for label in labels]

    x_pos = np.arange(len(labels))
    ax.bar(x_pos, values, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")


def _format_subplot_title(subplot_key: Any, n_subplots: int, title: str | None) -> str:
    """Format subplot title based on key."""
    if n_subplots == 1:
        # Single plot: use provided title or no title
        return title or ""

    # Multiple subplots: format key
    if isinstance(subplot_key, tuple):
        # (subplot_vals, metric)
        subplot_vals = subplot_key[:-1]  # All except last (metric)
        metric = subplot_key[-1]

        if len(subplot_vals) == 1:
            return f"{metric} ({subplot_vals[0]})"
        else:
            # Multiple subplot dimensions
            vals_str = ", ".join(str(v) for v in subplot_vals)
            return f"{metric} ({vals_str})"
    else:
        # Just metric name
        return str(subplot_key)


def auto_compute_figsize(
    nrows: int, ncols: int, base_width: float = 6.0, base_height: float = 4.0
) -> tuple[float, float]:
    """
    Compute figure size based on subplot grid.

    Parameters
    ----------
    nrows, ncols : int
        Grid dimensions
    base_width, base_height : float
        Base size for single subplot

    Returns
    -------
    tuple[float, float]
        (width, height) in inches
    """
    return (base_width * ncols, base_height * nrows)
