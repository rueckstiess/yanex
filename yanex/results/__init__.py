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
        >>> latest = get_latest(status="completed", name_pattern="model_*")
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
            - name_pattern: str - Glob pattern matching
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
    # Bulk operations
    "archive_experiments",
    "export_experiments",
    # Utilities
    "get_experiment_count",
    "experiment_exists",
    "get_manager",
]
