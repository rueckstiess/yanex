"""
pandas DataFrame integration for experiment comparison.

This module provides functions to convert experiment comparison data into
pandas DataFrames with proper hierarchical columns and data types.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
else:
    # Create a dummy for runtime
    class pd:
        DataFrame = "pd.DataFrame"


def experiments_to_dataframe(comparison_data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert experiment comparison data to pandas DataFrame with hierarchical columns.

    Args:
        comparison_data: Comparison data from ExperimentComparisonData.get_comparison_data()

    Returns:
        pandas DataFrame with hierarchical columns (category, name)

    Raises:
        ImportError: If pandas is not available

    Examples:
        >>> comparison_data = extractor.get_comparison_data(...)
        >>> df = experiments_to_dataframe(comparison_data)
        >>> print(df[("param", "learning_rate")])  # Access parameter column
        >>> print(df.xs("metric", axis=1, level=0))  # All metrics
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame functionality. Install it with: pip install pandas"
        )

    rows = comparison_data.get("rows", [])

    if not rows:
        # Return empty DataFrame with proper MultiIndex structure
        return pd.DataFrame(
            columns=pd.MultiIndex.from_tuples([], names=["category", "name"])
        )

    # Extract data and build hierarchical column structure
    data_dict = {}
    column_tuples = []

    # Process first row to determine column structure
    first_row = rows[0]

    for key in first_row.keys():
        if key.startswith("param:"):
            param_name = key[6:]  # Remove "param:" prefix
            column_tuples.append(("param", param_name))
            data_dict[("param", param_name)] = []
        elif key.startswith("metric:"):
            metric_name = key[7:]  # Remove "metric:" prefix
            column_tuples.append(("metric", metric_name))
            data_dict[("metric", metric_name)] = []
        else:
            # Metadata columns
            column_tuples.append(("meta", key))
            data_dict[("meta", key)] = []

    # Fill data for all rows
    for row in rows:
        for key in first_row.keys():
            if key.startswith("param:"):
                param_name = key[6:]
                data_dict[("param", param_name)].append(row.get(key, None))
            elif key.startswith("metric:"):
                metric_name = key[7:]
                data_dict[("metric", metric_name)].append(row.get(key, None))
            else:
                data_dict[("meta", key)].append(row.get(key, None))

    # Create MultiIndex columns
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["category", "name"])

    # Create DataFrame
    df = pd.DataFrame(data_dict, columns=columns)

    # Set experiment ID as index if available
    if ("meta", "id") in df.columns:
        df = df.set_index(("meta", "id"))
        df.index.name = "experiment_id"

    # Optimize data types
    df = format_dataframe_for_analysis(df)

    return df


def format_dataframe_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame for analysis by setting proper data types.

    Args:
        df: Input DataFrame with experiment data

    Returns:
        DataFrame with optimized dtypes and formatting

    Examples:
        >>> df = format_dataframe_for_analysis(raw_df)
        >>> print(df.dtypes)  # Shows optimized data types
    """
    try:
        import pandas as pd
    except ImportError:
        return df  # Return unchanged if pandas not available

    df = df.copy()

    # Convert numeric columns
    for col in df.columns:
        if col[0] in ["param", "metric"]:  # Only process parameter and metric columns
            # Try to convert to numeric, keeping non-numeric as object
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # Keep original dtype if conversion fails

    # Convert datetime columns
    datetime_columns = [
        ("meta", "started"),
        ("meta", "started_at"),
        ("meta", "ended_at"),
        ("meta", "completed_at"),
    ]

    for col in datetime_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass  # Keep original dtype if conversion fails

    # Convert duration to timedelta if it's in HH:MM:SS format
    if ("meta", "duration") in df.columns:
        duration_col = df[("meta", "duration")]
        try:
            # Try to convert HH:MM:SS format to timedelta
            df[("meta", "duration")] = pd.to_timedelta(duration_col)
        except (ValueError, TypeError):
            pass  # Keep original format if conversion fails

    # Convert categorical columns
    categorical_columns = [
        ("meta", "status"),
        ("meta", "name"),
    ]

    for col in categorical_columns:
        if col in df.columns and df[col].dtype == "object":
            # Only convert to category if there are repeated values
            unique_ratio = df[col].nunique() / len(df[col])
            if unique_ratio < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype("category")

    return df


def flatten_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten hierarchical columns to single-level with descriptive names.

    Args:
        df: DataFrame with hierarchical columns

    Returns:
        DataFrame with flattened column names

    Examples:
        >>> flat_df = flatten_dataframe_columns(hierarchical_df)
        >>> print(flat_df.columns)  # ['param_learning_rate', 'metric_accuracy', ...]
    """
    try:
        import pandas as pd
    except ImportError:
        return df

    if not isinstance(df.columns, pd.MultiIndex):
        return df  # Already flat

    df = df.copy()

    # Create new column names
    new_columns = []
    for category, name in df.columns:
        if category == "meta":
            new_columns.append(name)
        else:
            new_columns.append(f"{category}_{name}")

    # Flatten columns
    df.columns = new_columns

    return df


def get_parameter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all parameters.

    Args:
        df: DataFrame with experiment data

    Returns:
        DataFrame with parameter summary statistics

    Examples:
        >>> summary = get_parameter_summary(df)
        >>> print(summary)  # Shows min, max, mean, etc. for each parameter
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for summary statistics")

    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()  # Can't process without hierarchical columns

    # Get all parameter columns
    param_cols = [col for col in df.columns if col[0] == "param"]

    if not param_cols:
        return pd.DataFrame()

    param_df = df[param_cols]

    # Generate summary statistics
    summary = param_df.describe(include="all")

    # Clean up column names for summary
    summary.columns = [col[1] for col in summary.columns]  # Remove "param" prefix

    return summary


def get_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all metrics.

    Args:
        df: DataFrame with experiment data

    Returns:
        DataFrame with metric summary statistics

    Examples:
        >>> summary = get_metric_summary(df)
        >>> print(summary)  # Shows min, max, mean, etc. for each metric
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for summary statistics")

    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()  # Can't process without hierarchical columns

    # Get all metric columns
    metric_cols = [col for col in df.columns if col[0] == "metric"]

    if not metric_cols:
        return pd.DataFrame()

    metric_df = df[metric_cols]

    # Generate summary statistics
    summary = metric_df.describe(include="all")

    # Clean up column names for summary
    summary.columns = [col[1] for col in summary.columns]  # Remove "metric" prefix

    return summary


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix between parameters and metrics.

    Args:
        df: DataFrame with experiment data

    Returns:
        Correlation matrix DataFrame

    Examples:
        >>> corr = correlation_analysis(df)
        >>> print(corr.loc["learning_rate", "accuracy"])  # Correlation between LR and accuracy
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for correlation analysis")

    if not isinstance(df.columns, pd.MultiIndex):
        return pd.DataFrame()

    # Get numeric parameter and metric columns
    param_cols = [col for col in df.columns if col[0] == "param"]
    metric_cols = [col for col in df.columns if col[0] == "metric"]

    all_cols = param_cols + metric_cols

    if not all_cols:
        return pd.DataFrame()

    # Select only numeric columns
    numeric_df = df[all_cols].select_dtypes(include=["number"])

    if numeric_df.empty:
        return pd.DataFrame()

    # Flatten column names for correlation matrix
    numeric_df.columns = [col[1] for col in numeric_df.columns]

    # Compute correlation matrix
    correlation_matrix = numeric_df.corr()

    return correlation_matrix


def find_best_experiments(
    df: pd.DataFrame, metric: str, maximize: bool = True, top_n: int = 5
) -> pd.DataFrame:
    """
    Find the best experiments based on a specific metric.

    Args:
        df: DataFrame with experiment data
        metric: Metric name to optimize
        maximize: True to find maximum values, False for minimum
        top_n: Number of top experiments to return

    Returns:
        DataFrame with top experiments sorted by metric

    Examples:
        >>> best = find_best_experiments(df, "accuracy", maximize=True, top_n=3)
        >>> print(best)  # Top 3 experiments with highest accuracy
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for finding best experiments")

    if not isinstance(df.columns, pd.MultiIndex):
        return df.head(0)  # Return empty DataFrame

    metric_col = ("metric", metric)

    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame")

    # Filter out rows with missing metric values
    filtered_df = df.dropna(subset=[metric_col])

    if filtered_df.empty:
        return filtered_df

    # Sort by metric
    sorted_df = filtered_df.sort_values(metric_col, ascending=not maximize)

    return sorted_df.head(top_n)


def determine_varying_params(experiments: list) -> list[str]:
    """
    Determine which parameters vary across experiments.

    Flattens nested parameter dictionaries before comparison, so parameters like
    {"train": {"lr": 0.001}} become "train.lr" in the output.

    Args:
        experiments: List of Experiment objects

    Returns:
        List of parameter names (with dot notation for nested params) that have
        different values across experiments

    Examples:
        >>> varying = determine_varying_params(experiments)
        >>> print(varying)  # ['lr', 'batch_size'] if only these vary
        >>> # Nested params are flattened:
        >>> print(varying)  # ['train.lr', 'model.n_layer']
    """
    if not experiments:
        return []

    from ..utils.dict_utils import flatten_dict

    # Collect all params from all experiments (flattened)
    all_params: dict[str, set[str]] = {}
    for exp in experiments:
        params = exp.get_params()
        # Flatten nested params to dot notation
        flat_params = flatten_dict(params)
        for key, value in flat_params.items():
            if key not in all_params:
                all_params[key] = set()
            # Convert to string for comparison (handles different types)
            all_params[key].add(str(value))

    # Return only params with >1 unique value
    return sorted([key for key, values in all_params.items() if len(values) > 1])


def metrics_to_long_dataframe(
    experiments: list,
    metrics: list[str] | None = None,
    param_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convert experiment metrics to long (tidy) format DataFrame.

    Flattens nested parameter dictionaries, so parameters like {"train": {"lr": 0.001}}
    become "train.lr" columns in the output.

    Args:
        experiments: List of Experiment objects
        metrics: List of metric names to include (None for all)
        param_cols: List of parameter names (with dot notation for nested) to include as columns

    Returns:
        DataFrame with columns: [experiment_id, step, metric_name, value, <params...>]

    Examples:
        >>> df = metrics_to_long_dataframe(experiments, metrics=['train_loss'], param_cols=['lr'])
        >>> print(df.columns)  # ['experiment_id', 'step', 'metric_name', 'value', 'lr']
        >>> # Nested params use dot notation:
        >>> df = metrics_to_long_dataframe(experiments, param_cols=['train.lr', 'model.n_layer'])
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame functionality. Install it with: pip install pandas"
        )

    if not experiments:
        # Return empty DataFrame with expected structure
        columns = ["experiment_id", "step", "metric_name", "value"]
        if param_cols:
            columns.extend(param_cols)
        return pd.DataFrame(columns=columns)

    from ..utils.dict_utils import flatten_dict

    rows = []

    for exp in experiments:
        exp_id = exp.id
        exp_metrics = exp.get_metrics(as_dataframe=False)  # Returns list[dict]

        # Get params for this experiment (flattened to support dot notation)
        exp_params = flatten_dict(exp.get_params()) if param_cols else {}

        # Process each step
        for step_data in exp_metrics:
            step = step_data.get("step")

            # Extract each metric as a separate row
            for metric_name, metric_value in step_data.items():
                # Skip non-metric fields
                if metric_name in ["step", "timestamp", "last_updated"]:
                    continue

                # Filter to requested metrics if specified
                if metrics is not None and metric_name not in metrics:
                    continue

                # Build row
                row = {
                    "experiment_id": exp_id,
                    "step": step,
                    "metric_name": metric_name,
                    "value": metric_value,
                }

                # Add param columns (from flattened params)
                if param_cols:
                    for param_name in param_cols:
                        row[param_name] = exp_params.get(param_name)

                rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Optimize data types
    if not df.empty:
        # Convert step to int if possible
        if "step" in df.columns:
            df["step"] = pd.to_numeric(df["step"], errors="coerce")

        # Convert value to numeric if possible
        if "value" in df.columns:
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

        # Convert param columns to numeric where possible
        if param_cols:
            for param_col in param_cols:
                if param_col in df.columns:
                    try:
                        df[param_col] = pd.to_numeric(df[param_col])
                    except (ValueError, TypeError):
                        pass  # Keep as object if not numeric

    return df
