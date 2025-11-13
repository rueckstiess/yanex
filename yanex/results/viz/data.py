"""Data extraction utilities for yanex visualizations.

Converts experiment metrics into pandas DataFrames for analysis and plotting.
"""

import pandas as pd

from yanex.results.experiment import Experiment


def extract_metrics_df(
    experiments: list[Experiment],
    metric_names: list[str],
    include_metadata: list[str] | None = None,
    include_params: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract metrics into long-format pandas DataFrame.

    Parameters
    ----------
    experiments : list[Experiment]
        Experiments to extract metrics from
    metric_names : list[str]
        Names of metrics to extract
    include_metadata : list[str], optional
        Metadata fields to include as columns (e.g., ["name", "status", "tags"]).
        If None, includes commonly used fields: name, status
    include_params : list[str], optional
        Parameter fields to include as columns (e.g., ["learning_rate"]).
        If None, auto-discovers and includes all parameters

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
        - <metadata_fields>: various types
        - <param_fields>: various types

    Raises
    ------
    ValueError
        If metric not found in any experiments, or if metrics have
        inconsistent step counts across experiments

    Examples
    --------
    >>> import yanex.results as yr
    >>> from yanex.results.viz import extract_metrics_df
    >>> experiments = yr.get_experiments(tags=["training"])
    >>> df = extract_metrics_df(experiments, ["accuracy", "loss"])
    >>> df.head()
       experiment_id  step  timestamp            accuracy  loss  learning_rate
    0  abc12345      0     2024-01-15T10:00:00  0.80      0.20  0.001
    1  abc12345      1     2024-01-15T10:05:00  0.85      0.15  0.001
    2  abc12345      2     2024-01-15T10:10:00  0.90      0.10  0.001

    >>> # For single-step metrics
    >>> df = extract_metrics_df(experiments, ["final_accuracy"])
    >>> df.head()
       experiment_id  step  final_accuracy  learning_rate
    0  abc12345      None  0.95            0.001
    1  def67890      None  0.93            0.01
    """
    if not experiments:
        raise ValueError("No experiments provided")

    if not metric_names:
        raise ValueError("No metrics specified")

    rows = []

    for exp in experiments:
        # Get all metrics for this experiment
        metrics_data = exp.get_metrics()

        # Handle both single dict and list of dicts
        if isinstance(metrics_data, dict):
            # Single step
            metrics_list = [metrics_data]
        elif isinstance(metrics_data, list):
            # Multiple steps
            metrics_list = metrics_data
        else:
            # No metrics
            continue

        # Extract metadata/params once per experiment
        metadata = _extract_metadata(exp, include_metadata)
        params = _extract_params(exp, include_params)

        # Create row for each step
        for metric_entry in metrics_list:
            row = {
                "experiment_id": exp.id,
                "step": metric_entry.get("step"),
                "timestamp": metric_entry.get("timestamp"),
            }

            # Add metric values
            for metric in metric_names:
                row[metric] = metric_entry.get(metric)

            # Add metadata and params
            row.update(metadata)
            row.update(params)

            rows.append(row)

    if not rows:
        raise ValueError(
            f"No data found for metrics {metric_names} in provided experiments"
        )

    df = pd.DataFrame(rows)

    # Validation
    _validate_metrics_consistency(df, metric_names)

    return df


def _extract_metadata(exp: Experiment, fields: list[str] | None) -> dict:
    """Extract metadata fields from experiment.

    Parameters
    ----------
    exp : Experiment
        Experiment to extract from
    fields : list[str] | None
        Field names to extract, or None for defaults

    Returns
    -------
    dict
        Metadata field values
    """
    if fields is None:
        # Default: commonly used metadata fields
        fields = ["name", "status"]

    metadata = {}
    for field in fields:
        if hasattr(exp, field):
            value = getattr(exp, field)
            # Convert lists to comma-separated strings for DataFrame compatibility
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            metadata[field] = value

    return metadata


def _extract_params(exp: Experiment, fields: list[str] | None) -> dict:
    """Extract parameter fields from experiment.

    Parameters
    ----------
    exp : Experiment
        Experiment to extract from
    fields : list[str] | None
        Parameter names to extract, or None for all

    Returns
    -------
    dict
        Parameter field values
    """
    all_params = exp.get_params()

    if fields is None:
        # Include all params
        return all_params

    # Include only specified params
    params = {}
    for field in fields:
        value = exp.get_param(field, default=None)
        if value is not None:
            params[field] = value

    return params


def _validate_metrics_consistency(df: pd.DataFrame, metric_names: list[str]) -> None:
    """Validate that metrics are consistently single/multi-step.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with metrics
    metric_names : list[str]
        Metric names to validate

    Raises
    ------
    ValueError
        If metric not found or has inconsistent step counts
    """
    # Check if each metric exists in at least one experiment
    for metric in metric_names:
        if metric not in df.columns or df[metric].isna().all():
            raise ValueError(
                f"Metric '{metric}' not found in any experiments. "
                f"Available metrics vary by experiment - check individual "
                f"experiments to see what metrics were logged."
            )

    # Check consistency: within each (experiment_id, metric) pair,
    # step should be all None or all not-None
    for metric in metric_names:
        metric_df = df[df[metric].notna()]

        for exp_id in metric_df["experiment_id"].unique():
            exp_metric_df = metric_df[metric_df["experiment_id"] == exp_id]
            has_steps = exp_metric_df["step"].notna()

            if has_steps.any() and not has_steps.all():
                raise ValueError(
                    f"Experiment {exp_id} has inconsistent step data for "
                    f"metric '{metric}' (some entries with steps, some without)"
                )

    # Check cross-experiment consistency for each metric
    for metric in metric_names:
        metric_df = df[df[metric].notna()]

        # Count experiments with steps vs without
        exp_with_steps = set()
        exp_without_steps = set()

        for exp_id in metric_df["experiment_id"].unique():
            exp_metric_df = metric_df[metric_df["experiment_id"] == exp_id]
            if exp_metric_df["step"].notna().any():
                exp_with_steps.add(exp_id)
            else:
                exp_without_steps.add(exp_id)

        # If both sets are non-empty, we have inconsistency
        if exp_with_steps and exp_without_steps:
            n_with = len(exp_with_steps)
            n_without = len(exp_without_steps)
            sample_with = list(exp_with_steps)[:2]
            sample_without = list(exp_without_steps)[:2]

            raise ValueError(
                f"Metric '{metric}' has inconsistent step counts across experiments:\n"
                f"  - {n_with} experiments with multiple steps "
                f"(e.g., {', '.join(sample_with)})\n"
                f"  - {n_without} experiments with single step "
                f"(e.g., {', '.join(sample_without)})\n"
                f"Cannot mix single-step and multi-step metrics in the same plot."
            )


def detect_plot_type(df: pd.DataFrame) -> str:
    """
    Auto-detect plot type from DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from extract_metrics_df()

    Returns
    -------
    str
        "line" for multi-step metrics, "bar" for single-step metrics
    """
    # If all steps are None/NaN, it's single-step → bar chart
    if df["step"].isna().all():
        return "bar"
    return "line"
