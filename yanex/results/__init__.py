"""
Yanex Results API - Programmatic access to experiment data.

This module provides a unified interface for accessing and analyzing experiment results,
with support for filtering, comparison, and pandas DataFrame integration.

Example usage:
    import yanex.results as yr

    # Individual experiment access
    exp = yr.get_experiment("abc12345")
    print(exp.name, exp.get_metric("accuracy"))

    # Finding experiments
    experiments = yr.find(status="completed", tags=["training"])

    # Comparison and analysis
    df = yr.compare(
        status="completed",
        params=["learning_rate", "epochs"],
        metrics=["accuracy", "loss"]
    )

    # Best experiment
    best = yr.get_best("accuracy", maximize=True, status="completed")
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from .experiment import Experiment
from .manager import ResultsManager

# Global default manager instance (lazy-loaded)
_default_manager: ResultsManager | None = None


def _get_manager() -> ResultsManager:
    """Get or create the default ResultsManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ResultsManager()
    return _default_manager


# Individual experiment access
def get_experiment(experiment_id: str) -> Experiment:
    """
    Get a single experiment by ID.

    Args:
        experiment_id: The experiment ID to retrieve

    Returns:
        Experiment instance

    Raises:
        ExperimentNotFoundError: If experiment doesn't exist

    Examples:
        >>> exp = get_experiment("abc12345")
        >>> print(f"{exp.name}: {exp.status}")
        >>> params = exp.get_params()
        >>> accuracy = exp.get_metric("accuracy")
    """
    return _get_manager().get_experiment(experiment_id)


def get_latest(**filters) -> Experiment | None:
    """
    Get the most recently created experiment matching filters.

    Args:
        **filters: Filter arguments (status, tags, etc.)

    Returns:
        Most recent Experiment or None if no matches

    Examples:
        >>> latest = get_latest(tags=["training"])
        >>> latest = get_latest(status="completed", name="model_*")
    """
    return _get_manager().get_latest(**filters)


def get_best(metric: str, maximize: bool = True, **filters) -> Experiment | None:
    """
    Get the experiment with the best value for a specific metric.

    Args:
        metric: Metric name to optimize
        maximize: True to find maximum value, False for minimum
        **filters: Filter arguments

    Returns:
        Best Experiment or None if no matches

    Examples:
        >>> best = get_best("accuracy", maximize=True, status="completed")
        >>> best = get_best("loss", maximize=False, tags=["training"])
    """
    return _get_manager().get_best(metric, maximize, **filters)


# Multiple experiment access with unified filtering
def find(**filters) -> list[dict[str, Any]]:
    """
    Find experiments matching filter criteria.

    Args:
        **filters: Unified filter arguments supporting:
            - ids: list[str] - Match any of these IDs (OR logic)
            - status: str | list[str] - Match any of these statuses (OR logic)
            - name: str - Glob pattern matching
            - tags: list[str] - Must have ALL these tags (AND logic)
            - started_after: str | datetime - Started >= this time
            - started_before: str | datetime - Started <= this time
            - ended_after: str | datetime - Ended >= this time
            - ended_before: str | datetime - Ended <= this time
            - archived: bool - True/False/None (both)
            - limit: int - Maximum number of results

    Returns:
        List of experiment metadata dictionaries

    Examples:
        >>> # By IDs
        >>> experiments = find(ids=["abc123", "def456"])

        >>> # By status and tags
        >>> experiments = find(status="completed", tags=["training"])

        >>> # Complex filtering
        >>> experiments = find(
        ...     status=["completed", "failed"],
        ...     tags=["training", "cnn"],
        ...     started_after="2024-01-01",
        ...     limit=10
        ... )

        >>> # Mixed ID and filter criteria
        >>> experiments = find(
        ...     ids=["abc123", "def456"],
        ...     status="completed"
        ... )
    """
    return _get_manager().find(**filters)


def get_experiments(**filters) -> list[Experiment]:
    """
    Get multiple experiments as Experiment objects.

    Args:
        **filters: Same filter arguments as find()

    Returns:
        List of Experiment instances

    Examples:
        >>> experiments = get_experiments(status="completed")
        >>> for exp in experiments:
        ...     print(f"{exp.name}: {exp.get_metric('accuracy'):.3f}")

        >>> experiments = get_experiments(
        ...     tags=["training"],
        ...     started_after="1 week ago"
        ... )
    """
    return _get_manager().get_experiments(**filters)


def list_experiments(limit: int = 10, **filters) -> list[dict[str, Any]]:
    """
    List experiments with a default limit (convenience method).

    Args:
        limit: Maximum number of experiments to return
        **filters: Filter arguments

    Returns:
        List of experiment metadata dictionaries

    Examples:
        >>> recent = list_experiments(limit=5)
        >>> training = list_experiments(limit=20, tags=["training"])
    """
    return _get_manager().list_experiments(limit, **filters)


# Comparison and DataFrames
def compare(
    params: list[str] | None = None,
    metrics: list[str] | None = None,
    only_different: bool = False,
    **filters,
) -> "pd.DataFrame":
    """
    Compare experiments and return pandas DataFrame.

    Args:
        params: List of parameter names to include (None for auto-discovery)
        metrics: List of metric names to include (None for auto-discovery)
        only_different: If True, only show columns where values differ
        **filters: Filter arguments to select experiments

    Returns:
        pandas DataFrame with hierarchical columns for comparison

    Raises:
        ImportError: If pandas is not available

    Examples:
        >>> # Compare specific experiments
        >>> df = compare(
        ...     ids=["abc123", "def456", "ghi789"],
        ...     params=["learning_rate", "epochs"],
        ...     metrics=["accuracy", "loss"]
        ... )

        >>> # Compare by filter criteria
        >>> df = compare(
        ...     status="completed",
        ...     tags=["training"],
        ...     params=["learning_rate", "batch_size"],
        ...     metrics=["accuracy", "f1_score"]
        ... )

        >>> # Access data
        >>> print(df[("param", "learning_rate")])  # Parameter column
        >>> print(df[("metric", "accuracy")].max())  # Best accuracy
        >>> params_df = df.xs("param", axis=1, level=0)  # All parameters
    """
    return _get_manager().compare_experiments(
        params, metrics, only_different, **filters
    )


# Visualization
def plot_metrics(
    metrics: str | list[str],
    *,
    # Filtering (same as other methods)
    ids: list[str] | None = None,
    status: str | list[str] | None = None,
    tags: list[str] | None = None,
    name: str | None = None,
    script_pattern: str | None = None,
    started_after: str | None = None,
    started_before: str | None = None,
    ended_after: str | None = None,
    ended_before: str | None = None,
    archived: bool | None = False,
    include_all: bool = False,
    limit: int | None = None,
    # Organization
    group_by: str | list[str] | None = None,
    label_by: str | list[str] | None = None,
    subplot_by: str | list[str] | None = None,
    # Aggregation
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
    return_axes: bool = False,
):
    """
    Plot metrics from experiments with automatic layout and styling.

    This function provides flexible visualization of experiment metrics with support
    for grouping, aggregation, and multi-dimensional comparison. It automatically
    adapts to the data shape and provides publication-ready plots by default.

    Parameters
    ----------
    metrics : str | list[str]
        Metric name(s) to plot. Required.

    **Filtering (same as find() and get_experiments())**
    ids : list[str], optional
        Experiment IDs to include
    status : str | list[str], optional
        Filter by status (e.g., "completed", ["completed", "failed"])
    tags : list[str], optional
        Filter by tags (must have ALL tags)
    name : str, optional
        Filter by name (glob pattern)
    script_pattern : str, optional
        Filter by script name (glob pattern)
    started_after, started_before : str, optional
        Filter by start time (ISO format or natural language)
    ended_after, ended_before : str, optional
        Filter by end time
    archived : bool | None, optional
        Include archived experiments (default: False)
    include_all : bool, optional
        Include all experiments without limit (default: False)
    limit : int, optional
        Maximum number of experiments to include

    **Organization**
    group_by : str | list[str], optional
        Column(s) to aggregate by (e.g., "random_seed").
        Computes mean/median across these dimensions.
    label_by : str | list[str], optional
        Column(s) for legend labels (e.g., "learning_rate").
        Default: None for single experiment, "experiment_id" for multiple.
    subplot_by : str | list[str], optional
        Column(s) to create subplot grid by (e.g., "model_type")

    **Aggregation (requires group_by)**
    aggregation : str, default="mean"
        Aggregation method: "mean" or "median"
    show_ci : bool, default=False
        Show confidence interval bands
    show_std : bool, default=False
        Show standard deviation bands
    show_individuals : bool, default=True
        Show individual lines behind aggregation
    ci_level : float, default=0.95
        Confidence interval level (e.g., 0.95 for 95%)

    **Plot Type**
    plot_type : str, default="auto"
        Plot type: "auto", "line", "bar"
        Auto-detects: line for multi-step metrics, bar for single-step

    **Subplot Layout**
    subplot_layout : tuple[int, int], optional
        Subplot arrangement as (rows, cols).
        If None, uses single row. Must match number of subplots if specified.

    **Styling**
    title : str, optional
        Figure title
    xlabel : str, optional
        X-axis label (default: "Step" for line, "Experiment" for bar)
    ylabel : str, optional
        Y-axis label (default: metric name if single, "Value" if multiple)
    figsize : tuple[float, float], optional
        Figure size in inches (default: auto-computed)
    colors : list[str], optional
        Custom color palette (hex codes)
    colorblind_safe : bool, default=True
        Use colorblind-friendly color palette
    grid : bool, default=True
        Show grid
    legend : bool, default=True
        Show legend
    legend_position : str, default="best"
        Matplotlib legend location

    **Output Control**
    show : bool, default=True
        Display plot immediately
    return_axes : bool, default=False
        Return (fig, axes) instead of just fig for advanced customization

    Returns
    -------
    Figure or tuple[Figure, Axes]
        Matplotlib figure, or (figure, axes) if return_axes=True

    Raises
    ------
    ValueError
        If no experiments found, metrics not found, or invalid parameters
    ImportError
        If matplotlib is not available

    Examples
    --------
    **Basic Usage**

    Plot single training curve:
        >>> import yanex.results as yr
        >>> yr.plot_metrics("accuracy", ids=["abc123"])

    Compare multiple experiments:
        >>> yr.plot_metrics("loss", tags=["baseline_comparison"])

    Multi-metric dashboard:
        >>> yr.plot_metrics(["accuracy", "loss", "f1"], status="completed")

    **Advanced Usage**

    Label by parameter instead of ID:
        >>> yr.plot_metrics(
        ...     "accuracy",
        ...     tags=["lr_sweep"],
        ...     label_by="learning_rate"
        ... )

    Create subplot grid:
        >>> yr.plot_metrics(
        ...     "f1_score",
        ...     tags=["model_comparison"],
        ...     label_by="learning_rate",
        ...     subplot_by="model_type"
        ... )

    Statistical aggregation:
        >>> yr.plot_metrics(
        ...     "accuracy",
        ...     tags=["repeated_runs"],
        ...     group_by="random_seed",
        ...     label_by="learning_rate",
        ...     show_ci=True
        ... )

    Complex multi-dimensional:
        >>> yr.plot_metrics(
        ...     ["accuracy", "loss"],
        ...     tags=["grid_search"],
        ...     group_by="random_seed",
        ...     label_by="learning_rate",
        ...     subplot_by="model_type",
        ...     subplot_layout=(2, 2),
        ...     show_std=True
        ... )

    **Customization**

    Custom styling:
        >>> yr.plot_metrics(
        ...     "accuracy",
        ...     ids=["abc123"],
        ...     title="Training Progress - ResNet50",
        ...     xlabel="Epoch",
        ...     ylabel="Validation Accuracy",
        ...     figsize=(10, 6)
        ... )

    Advanced post-processing:
        >>> fig, axes = yr.plot_metrics(
        ...     "loss",
        ...     tags=["training"],
        ...     return_axes=True,
        ...     show=False
        ... )
        >>> axes.set_yscale('log')
        >>> axes.axhline(y=0.1, color='r', linestyle='--', label='Target')
        >>> fig.savefig("custom_plot.pdf", dpi=300)

    **Building Blocks**

    For more control, use the underlying components:
        >>> from yanex.results.viz import extract_metrics_df, organize_for_plotting
        >>> experiments = yr.get_experiments(tags=["training"])
        >>> df = extract_metrics_df(experiments, ["accuracy"])
        >>> # Use pandas operations on df
        >>> result = organize_for_plotting(df, ["accuracy"], label_by="learning_rate")
        >>> # Custom plotting with the organized data
    """
    try:
        from yanex.results.viz.data import extract_metrics_df
        from yanex.results.viz.grouping import organize_for_plotting
        from yanex.results.viz.plotting import create_plot
    except ImportError as e:
        raise ImportError(
            "Visualization requires matplotlib. Install with: pip install matplotlib"
        ) from e

    # Normalize metrics to list
    if isinstance(metrics, str):
        metric_names = [metrics]
    else:
        metric_names = metrics

    # Gather filter arguments
    filter_args = {
        "ids": ids,
        "status": status,
        "tags": tags,
        "name": name,
        "script_pattern": script_pattern,
        "started_after": started_after,
        "started_before": started_before,
        "ended_after": ended_after,
        "ended_before": ended_before,
        "archived": archived,
        "include_all": include_all,
        "limit": limit,
    }
    # Remove None values
    filter_args = {k: v for k, v in filter_args.items() if v is not None}

    # Get experiments
    experiments = get_experiments(**filter_args)

    if not experiments:
        raise ValueError("No experiments found matching filters")

    # Extract metrics to DataFrame
    df = extract_metrics_df(experiments, metric_names)

    # Organize for plotting
    plot_structure = organize_for_plotting(
        df,
        metric_names,
        group_by=group_by,
        label_by=label_by,
        subplot_by=subplot_by,
        aggregation=aggregation,
        ci_level=ci_level,
    )

    # Create plot
    return create_plot(
        plot_structure,
        show_ci=show_ci,
        show_std=show_std,
        show_individuals=show_individuals,
        subplot_layout=subplot_layout,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        figsize=figsize,
        grid=grid,
        legend=legend,
        legend_position=legend_position,
        show=show,
        return_axes=return_axes,
    )


# Bulk operations
def archive_experiments(**filters) -> int:
    """
    Archive experiments matching filters.

    Args:
        **filters: Filter arguments to select experiments

    Returns:
        Number of experiments successfully archived

    Examples:
        >>> # Archive failed experiments older than 1 month
        >>> count = archive_experiments(
        ...     status="failed",
        ...     ended_before="1 month ago"
        ... )
        >>> print(f"Archived {count} experiments")

        >>> # Archive specific experiments
        >>> count = archive_experiments(ids=["abc123", "def456"])
    """
    return _get_manager().archive_experiments(**filters)


def delete_experiments(**filters) -> int:
    """
    Permanently delete experiments matching filters.

    Args:
        **filters: Filter arguments to select experiments

    Returns:
        Number of experiments successfully deleted

    Examples:
        >>> # Delete failed experiments older than 1 month
        >>> count = delete_experiments(
        ...     status="failed",
        ...     ended_before="1 month ago"
        ... )
        >>> print(f"Deleted {count} experiments")

        >>> # Delete specific experiments
        >>> count = delete_experiments(ids=["abc123", "def456"])
    """
    return _get_manager().delete_experiments(**filters)


def export_experiments(path: str, format: str = "json", **filters) -> None:
    """
    Export experiments matching filters to a file.

    Args:
        path: Output file path
        format: Export format ("json", "csv", "yaml")
        **filters: Filter arguments to select experiments

    Raises:
        ValueError: If format is not supported
        IOError: If file cannot be written

    Examples:
        >>> # Export training results to JSON
        >>> export_experiments(
        ...     "training_results.json",
        ...     format="json",
        ...     tags=["training"],
        ...     status="completed"
        ... )

        >>> # Export comparison data to CSV
        >>> export_experiments(
        ...     "comparison.csv",
        ...     format="csv",
        ...     ids=["abc123", "def456", "ghi789"]
        ... )
    """
    return _get_manager().export_experiments(path, format, **filters)


# Utility functions
def get_experiment_count(**filters) -> int:
    """
    Get count of experiments matching filters.

    Args:
        **filters: Filter arguments

    Returns:
        Number of matching experiments

    Examples:
        >>> total = get_experiment_count()
        >>> completed = get_experiment_count(status="completed")
        >>> recent_training = get_experiment_count(
        ...     tags=["training"],
        ...     started_after="1 week ago"
        ... )
    """
    return _get_manager().get_experiment_count(**filters)


def experiment_exists(experiment_id: str, include_archived: bool = True) -> bool:
    """
    Check if an experiment exists.

    Args:
        experiment_id: Experiment ID to check
        include_archived: Whether to include archived experiments

    Returns:
        True if experiment exists, False otherwise

    Examples:
        >>> if experiment_exists("abc12345"):
        ...     exp = get_experiment("abc12345")
        ...     print(f"Found: {exp.name}")
    """
    return _get_manager().experiment_exists(experiment_id, include_archived)


# Manager access for advanced usage
def get_manager(storage_path: Path | None = None) -> ResultsManager:
    """
    Get a ResultsManager instance.

    Args:
        storage_path: Optional custom storage path

    Returns:
        ResultsManager instance

    Examples:
        >>> # Use default storage
        >>> manager = get_manager()

        >>> # Use custom storage path
        >>> custom_manager = get_manager(Path("/path/to/experiments"))
    """
    if storage_path:
        return ResultsManager(storage_path)
    return _get_manager()


# Export commonly used classes and functions
__all__ = [
    # Core classes
    "Experiment",
    "ResultsManager",
    # Individual experiment access
    "get_experiment",
    "get_latest",
    "get_best",
    # Multiple experiment access
    "find",
    "get_experiments",
    "list_experiments",
    # Comparison and DataFrames
    "compare",
    # Visualization
    "plot_metrics",
    # Bulk operations
    "archive_experiments",
    "delete_experiments",
    "export_experiments",
    # Utilities
    "get_experiment_count",
    "experiment_exists",
    "get_manager",
]
