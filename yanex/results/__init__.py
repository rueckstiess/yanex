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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

from .experiment import Experiment
from .manager import ResultsManager
from .plotting import plot_metrics

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
    params: str | list[str] = "auto",
    metrics: str | list[str] = "auto",
    meta: str | list[str] = "auto",
    include_dep_params: bool = False,
    **filters,
) -> "pd.DataFrame":
    """
    Compare experiments and return pandas DataFrame.

    Args:
        params: Parameter columns to include:
            - "auto" (default): Only parameters that differ across experiments
            - "all": All parameters
            - "none": No parameter columns
            - list[str]: Specific parameter names (supports patterns like "*.lr")
        metrics: Metric columns to include:
            - "auto" (default): Only metrics that differ across experiments
            - "all": All metrics
            - "none": No metric columns
            - list[str]: Specific metric names (supports patterns like "train.*")
        meta: Metadata columns to include:
            - "auto" (default): id, name, status
            - "all": All metadata fields
            - "none": No metadata columns
            - list[str]: Specific metadata fields
        include_dep_params: If True, include parameters from dependencies
            (merged with local params, local values take precedence)
        **filters: Filter arguments to select experiments

    Returns:
        pandas DataFrame with flat group:path columns (e.g., "param:lr", "metric:accuracy")

    Raises:
        ImportError: If pandas is not available

    Examples:
        >>> # Default: show differing params and metrics
        >>> df = compare(status="completed")

        >>> # Explicit field selection
        >>> df = compare(params=["lr", "epochs"], metrics=["accuracy", "loss"])

        >>> # With patterns
        >>> df = compare(params=["*.lr"], metrics=["train.*"])

        >>> # Special values
        >>> df = compare(params="all", metrics="none")
        >>> df = compare(params="auto", metrics="all", meta=["id", "status"])

        >>> # Include parameters from dependencies
        >>> df = compare(tags=["sweep"], params=["model.n_embd"], include_dep_params=True)

        >>> # Access data with flat column names
        >>> print(df["param:lr"])  # Parameter column
        >>> print(df["metric:accuracy"].max())  # Best accuracy
        >>> param_cols = [c for c in df.columns if c.startswith("param:")]
    """
    return _get_manager().compare_experiments(
        params=params,
        metrics=metrics,
        meta=meta,
        include_dep_params=include_dep_params,
        **filters,
    )


def get_metrics(
    *,
    metrics: str | list[str] | None = None,
    params: list[str] | Literal["auto", "all", "none"] = "auto",
    meta: list[str] | None = None,
    include_dep_params: bool = False,
    as_dataframe: bool = True,
    **filters,
) -> "pd.DataFrame | dict[str, list[dict]]":
    """
    Get time-series metrics from multiple experiments in long (tidy) format.

    Returns a DataFrame optimized for visualization and analysis with matplotlib,
    seaborn, or plotly.

    Args:
        metrics: Which metrics to include:
            - None: All metrics (default)
            - str: Single metric name
            - list[str]: List of specific metric names
        params: Which parameter columns to include:
            - "auto" (default): Include only parameters that vary across experiments
            - "all": Include all parameters
            - "none": No parameter columns
            - list[str]: Include only specified parameters
        meta: List of metadata fields to include as columns for faceting:
            - None: No metadata columns (default)
            - list[str]: Include specified fields (e.g., ['name', 'status'])
        include_dep_params: If True, include parameters from dependencies
            (merged with local params, local values take precedence)
        as_dataframe: If True (default), return DataFrame. If False, return dict.
        **filters: Filter arguments to select experiments (same as get_experiments)

    Returns:
        DataFrame with columns: [experiment_id, step, metric_name, value, <meta...>, <params...>]
        or dict[str, list[dict]] mapping experiment_id to metrics list

    Raises:
        ImportError: If pandas is not available and as_dataframe=True

    Examples:
        Basic usage with matplotlib:

        >>> import yanex.results as yr
        >>> import matplotlib.pyplot as plt
        >>>
        >>> # Get metrics for all experiments with a tag
        >>> df = yr.get_metrics(tags=['sweep'])
        >>> df_loss = df[df.metric_name == 'train_loss']
        >>>
        >>> # Plot grouped by learning rate
        >>> for lr, group in df_loss.groupby('lr'):
        ...     plt.plot(group.step, group.value, label=f'lr={lr}')
        >>> plt.legend()

        Multiple metrics with subplots:

        >>> df = yr.get_metrics(tags=['training'])
        >>> fig, (ax1, ax2) = plt.subplots(2, 1)
        >>>
        >>> # Plot loss
        >>> df_loss = df[df.metric_name == 'train_loss']
        >>> for lr, group in df_loss.groupby('lr'):
        ...     ax1.plot(group.step, group.value, label=f'lr={lr}')
        >>> ax1.set_ylabel('Loss')
        >>> ax1.legend()
        >>>
        >>> # Plot accuracy
        >>> df_acc = df[df.metric_name == 'train_accuracy']
        >>> for lr, group in df_acc.groupby('lr'):
        ...     ax2.plot(group.step, group.value, label=f'lr={lr}')
        >>> ax2.set_ylabel('Accuracy')
        >>> ax2.legend()

        Get specific metric only:

        >>> df = yr.get_metrics(tags=['training'], metrics='train_loss')
        >>> # Only train_loss rows

        Control parameter inclusion:

        >>> # Only varying params (default)
        >>> df = yr.get_metrics(tags=['sweep'])
        >>>
        >>> # All params
        >>> df = yr.get_metrics(tags=['sweep'], params='all')
        >>>
        >>> # No params
        >>> df = yr.get_metrics(tags=['sweep'], params='none')
        >>>
        >>> # Specific params
        >>> df = yr.get_metrics(tags=['sweep'], params=['lr', 'epochs'])

        Include parameters from dependencies:

        >>> # Get dependency params (e.g., encoder settings from parent experiment)
        >>> df = yr.get_metrics(tags=['sweep'], params=['model.n_embd'], include_dep_params=True)
        >>> # Now model.n_embd column includes values from dependency experiments

        Include metadata for faceting (e.g., color by experiment name):

        >>> df = yr.get_metrics(tags=['sweep'], meta=['name'])
        >>> # Columns: experiment_id, step, metric_name, value, name, lr, ...
        >>> for name, group in df[df.metric_name == 'loss'].groupby('name'):
        ...     plt.plot(group.step, group.value, label=name)

        Get raw dict format:

        >>> data = yr.get_metrics(tags=['training'], as_dataframe=False)
        >>> for exp_id, metrics in data.items():
        ...     print(f"{exp_id}: {len(metrics)} steps")

    See Also:
        compare: Compare final metric values across experiments (wide format)
        get_experiments: Get experiment objects for custom processing
    """
    return _get_manager().get_metrics(
        metrics=metrics,
        params=params,
        meta=meta,
        include_dep_params=include_dep_params,
        as_dataframe=as_dataframe,
        **filters,
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
    "get_metrics",
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
