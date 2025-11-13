"""Data organization utilities for yanex visualizations.

Uses pandas operations to group, aggregate, and organize data for plotting.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from yanex.results.experiment import Experiment
from yanex.results.viz.data import detect_plot_type
from yanex.results.viz.styles import get_color_palette

# Type alias for field specifications
FieldSpec = str | tuple[str, ...] | Callable[[Experiment], Any]


def organize_for_plotting(
    df: pd.DataFrame,
    metric_names: list[str],
    *,
    group_by: str | list[str] | None = None,
    label_by: str | list[str] | None = None,
    subplot_by: str | list[str] | None = None,
    aggregation: str = "mean",
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """
    Organize DataFrame using pandas operations for plotting.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame from extract_metrics_df()
    metric_names : list[str]
        Metrics to plot
    group_by : str | list[str], optional
        Column(s) to group by for aggregation (e.g., "random_seed").
        When set, computes aggregated statistics across groups.
    label_by : str | list[str], optional
        Column(s) for legend labels (e.g., "learning_rate").
        Default: None for single experiment, "experiment_id" for multiple
    subplot_by : str | list[str], optional
        Column(s) for subplot splitting (e.g., "model_type")
    aggregation : str, default="mean"
        Aggregation method: "mean" or "median"
    ci_level : float, default=0.95
        Confidence interval level (e.g., 0.95 for 95%)

    Returns
    -------
    dict
        {
            "plot_type": "line" or "bar",
            "subplots": {
                subplot_key: {
                    "data": pd.DataFrame (organized for this subplot),
                    "metric": metric_name,
                    "individual_data": pd.DataFrame (if group_by, else None)
                },
                ...
            },
            "metadata": {
                "unique_labels": list,
                "color_palette": list,
                "has_aggregation": bool,
            }
        }

    Raises
    ------
    ValueError
        If group_by specified without label_by, or invalid aggregation method

    Examples
    --------
    >>> from yanex.results.viz import extract_metrics_df, organize_for_plotting
    >>> import yanex.results as yr
    >>> experiments = yr.get_experiments(tags=["training"])
    >>> df = extract_metrics_df(experiments, ["accuracy"])
    >>> result = organize_for_plotting(df, ["accuracy"], label_by="learning_rate")
    """
    if aggregation not in ["mean", "median"]:
        raise ValueError(f"Invalid aggregation: {aggregation}. Use 'mean' or 'median'")

    if group_by is not None and label_by is None:
        raise ValueError(
            "group_by requires label_by to be specified. "
            "group_by aggregates over experiments, label_by determines "
            "how to group the aggregated results for visualization."
        )

    plot_type = detect_plot_type(df)

    # Apply default label_by
    if label_by is None:
        n_experiments = df["experiment_id"].nunique()
        label_by = "experiment_id" if n_experiments > 1 else None

    # Normalize to lists
    group_by_list = _normalize_to_list(group_by)
    label_by_list = _normalize_to_list(label_by)
    subplot_by_list = _normalize_to_list(subplot_by)

    # Create subplot groups
    subplot_groups = _create_subplot_groups(df, subplot_by_list, metric_names)

    # Process each subplot
    result = {
        "plot_type": plot_type,
        "subplots": {},
        "metadata": {
            "unique_labels": [],
            "color_palette": [],
            "has_aggregation": group_by is not None,
        },
    }

    for subplot_key, subplot_df in subplot_groups.items():
        # Extract metric name from key
        if isinstance(subplot_key, tuple):
            metric = subplot_key[-1]  # Last element is always metric
        else:
            metric = subplot_key

        individual_data = None

        if group_by_list is not None:
            # Apply aggregation
            organized_df, individual_data = _apply_aggregation(
                subplot_df,
                metric,
                group_by=group_by_list,
                label_by=label_by_list,
                aggregation=aggregation,
                ci_level=ci_level,
            )
        else:
            # No aggregation, just organize by label_by
            if label_by_list:
                organized_df = subplot_df.sort_values(label_by_list + ["step"])
            else:
                organized_df = subplot_df.sort_values(["step"])

        result["subplots"][subplot_key] = {
            "data": organized_df,
            "metric": metric,
            "individual_data": individual_data,
        }

    # Collect unique labels for color assignment
    if label_by_list:
        unique_labels = _collect_unique_labels(df, label_by_list)
        result["metadata"]["unique_labels"] = unique_labels
        result["metadata"]["color_palette"] = get_color_palette(len(unique_labels))

    return result


def _normalize_to_list(field: str | list[str] | None) -> list[str] | None:
    """Normalize field specification to list or None."""
    if field is None:
        return None
    if isinstance(field, str):
        return [field]
    return field


def _create_subplot_groups(
    df: pd.DataFrame, subplot_by: list[str] | None, metric_names: list[str]
) -> dict[Any, pd.DataFrame]:
    """
    Split DataFrame into subplot groups.

    Returns dict mapping subplot_key -> DataFrame subset.
    """
    if subplot_by is None and len(metric_names) == 1:
        # Single subplot
        return {metric_names[0]: df}

    groups = {}

    # If we have multiple metrics, always split by metric first
    for metric in metric_names:
        metric_df = df.dropna(subset=[metric])

        if subplot_by is None:
            # Just split by metric
            groups[metric] = metric_df
        else:
            # Split by metric AND subplot_by fields
            for subplot_vals, subplot_df in metric_df.groupby(subplot_by):
                # Create key: (subplot_vals, metric)
                if len(subplot_by) == 1:
                    key = (subplot_vals, metric)
                else:
                    key = (tuple(subplot_vals), metric)
                groups[key] = subplot_df

    return groups


def _apply_aggregation(
    df: pd.DataFrame,
    metric: str,
    group_by: list[str],
    label_by: list[str] | None,
    aggregation: str,
    ci_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply pandas groupby aggregation.

    Groups by: label_by + ['step'], aggregates over group_by dimensions.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (aggregated_df, individual_df)
        - aggregated_df: Contains mean/median, ci_lower, ci_upper, std
        - individual_df: Original data for show_individuals
    """
    # Store original data for individual lines
    individual_df = df.copy()

    # Group by label_by + step (these are the dimensions we keep)
    groupby_cols = (label_by if label_by else []) + ["step"]
    groupby_cols = [col for col in groupby_cols if col in df.columns]

    if not groupby_cols:
        # Edge case: no grouping columns (shouldn't happen with validation)
        groupby_cols = ["step"]

    grouped = df.groupby(groupby_cols, dropna=False)[metric]

    # Compute statistics
    if aggregation == "mean":
        agg_result = grouped.mean()
    else:  # median
        agg_result = grouped.median()

    std_result = grouped.std()
    ci_lower_result = grouped.quantile((1 - ci_level) / 2)
    ci_upper_result = grouped.quantile((1 + ci_level) / 2)

    # Combine into DataFrame
    result = pd.DataFrame(
        {
            "mean": agg_result,
            "std": std_result,
            "ci_lower": ci_lower_result,
            "ci_upper": ci_upper_result,
        }
    ).reset_index()

    return result, individual_df


def _collect_unique_labels(df: pd.DataFrame, label_by: list[str]) -> list[str]:
    """
    Collect unique labels from DataFrame.

    For single field: returns list of values.
    For multiple fields: returns list of formatted strings "field1=val1, field2=val2".
    """
    if len(label_by) == 1:
        # Single field
        unique_vals = df[label_by[0]].dropna().unique()
        return [str(v) for v in sorted(unique_vals)]

    # Multiple fields: create tuples and format
    unique_tuples = df[label_by].drop_duplicates().values

    labels = []
    for vals in unique_tuples:
        label = _format_label(label_by, vals)
        labels.append(label)

    return sorted(labels)


def _format_label(fields: list[str], values: tuple | Any) -> str:
    """
    Format field values as label string.

    Examples
    --------
    >>> _format_label(["learning_rate"], (0.01,))
    'learning_rate=0.01'
    >>> _format_label(["lr", "bs"], (0.01, 32))
    'lr=0.01, bs=32'
    """
    if not isinstance(values, (tuple, list, np.ndarray)):
        values = (values,)

    parts = []
    for field, value in zip(fields, values, strict=False):
        parts.append(f"{field}={value}")

    return ", ".join(parts)


def validate_subplot_layout(
    layout: tuple[int, int] | None, n_subplots: int
) -> tuple[int, int]:
    """
    Validate and compute subplot layout.

    Parameters
    ----------
    layout : tuple[int, int] | None
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

    Examples
    --------
    >>> validate_subplot_layout(None, 3)
    (1, 3)
    >>> validate_subplot_layout((2, 2), 4)
    (2, 2)
    >>> validate_subplot_layout((2, 2), 5)
    Traceback (most recent call last):
        ...
    ValueError: subplot_layout (2, 2) = 4 subplots...
    """
    if layout is None:
        # Default: single row
        return (1, n_subplots)

    rows, cols = layout
    if rows * cols != n_subplots:
        raise ValueError(
            f"subplot_layout ({rows}, {cols}) = {rows * cols} subplots, "
            f"but {n_subplots} subplots are needed. "
            f"Specify subplot_layout=(rows, cols) where rows * cols = {n_subplots}, "
            f"or use subplot_layout=None for default (single row)."
        )

    return layout
