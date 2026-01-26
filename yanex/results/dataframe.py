"""
pandas DataFrame integration for experiment comparison.

This module provides functions to convert experiment comparison data into
pandas DataFrames with flat group:path column names (e.g., "param:lr", "metric:accuracy").
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
    Convert experiment comparison data to pandas DataFrame with flat group:path columns.

    Args:
        comparison_data: Comparison data from ExperimentComparisonData.get_comparison_data()

    Returns:
        pandas DataFrame with flat column names using group:path format
        (e.g., "param:lr", "metric:accuracy", "meta:id")

    Raises:
        ImportError: If pandas is not available

    Examples:
        >>> comparison_data = extractor.get_comparison_data(...)
        >>> df = experiments_to_dataframe(comparison_data)
        >>> print(df["param:learning_rate"])  # Access parameter column
        >>> param_cols = [c for c in df.columns if c.startswith("param:")]
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas is required for DataFrame functionality. Install it with: pip install pandas"
        )

    rows = comparison_data.get("rows", [])

    if not rows:
        return pd.DataFrame()

    # Extract data - columns already use group:path format from comparison data
    data_dict: dict[str, list] = {}

    # Process first row to determine column structure
    first_row = rows[0]

    for key in first_row.keys():
        # Ensure all columns use canonical group:path format
        canonical_key = _ensure_canonical_column(key)
        data_dict[canonical_key] = []

    # Fill data for all rows
    for row in rows:
        for key in first_row.keys():
            canonical_key = _ensure_canonical_column(key)
            data_dict[canonical_key].append(row.get(key, None))

    # Create DataFrame
    df = pd.DataFrame(data_dict)

    # Set experiment ID as index if available
    if "meta:id" in df.columns:
        df = df.set_index("meta:id")
        df.index.name = "experiment_id"

    # Optimize data types
    df = format_dataframe_for_analysis(df)

    return df


def _ensure_canonical_column(key: str) -> str:
    """
    Ensure a column name uses canonical group:path format.

    Args:
        key: Column name (may or may not have group prefix)

    Returns:
        Canonical column name with group prefix
    """
    # Already has a valid group prefix
    if key.startswith(("param:", "metric:", "meta:")):
        return key

    # Legacy metadata fields without prefix -> add meta:
    return f"meta:{key}"


def format_dataframe_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame for analysis by setting proper data types.

    Args:
        df: Input DataFrame with experiment data (flat group:path columns)

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

    # Convert numeric columns (param: and metric: columns)
    for col in df.columns:
        if col.startswith(("param:", "metric:")):
            # Try to convert to numeric, keeping non-numeric as object
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass  # Keep original dtype if conversion fails

    # Convert datetime columns
    datetime_columns = [
        "meta:started",
        "meta:started_at",
        "meta:ended_at",
        "meta:completed_at",
    ]

    for col in datetime_columns:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                pass  # Keep original dtype if conversion fails

    # Convert duration to timedelta if it's in HH:MM:SS format
    if "meta:duration" in df.columns:
        duration_col = df["meta:duration"]
        try:
            # Try to convert HH:MM:SS format to timedelta
            df["meta:duration"] = pd.to_timedelta(duration_col)
        except (ValueError, TypeError):
            pass  # Keep original format if conversion fails

    # Convert categorical columns
    categorical_columns = [
        "meta:status",
        "meta:name",
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
    Convert group:path column names to underscore format.

    This is a legacy compatibility function. With the new flat column format,
    this converts "param:learning_rate" to "param_learning_rate".

    Args:
        df: DataFrame with group:path columns

    Returns:
        DataFrame with underscore-separated column names

    Examples:
        >>> flat_df = flatten_dataframe_columns(df)
        >>> print(flat_df.columns)  # ['param_learning_rate', 'metric_accuracy', 'id', ...]
    """
    try:
        import pandas as pd
    except ImportError:
        return df

    # Handle legacy MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        new_columns = []
        for category, name in df.columns:
            if category == "meta":
                new_columns.append(name)
            else:
                new_columns.append(f"{category}_{name}")
        df.columns = new_columns
        return df

    df = df.copy()

    # Convert group:path to group_path format
    new_columns = []
    for col in df.columns:
        if col.startswith("meta:"):
            # Strip meta: prefix for metadata columns
            new_columns.append(col[5:])
        elif col.startswith(("param:", "metric:")):
            # Replace : with _ for param/metric columns
            new_columns.append(col.replace(":", "_", 1))
        else:
            new_columns.append(col)

    df.columns = new_columns
    return df


def get_parameter_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all parameters.

    Args:
        df: DataFrame with experiment data (flat group:path columns)

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

    # Get all parameter columns (param:* format)
    param_cols = [col for col in df.columns if col.startswith("param:")]

    if not param_cols:
        return pd.DataFrame()

    param_df = df[param_cols]

    # Generate summary statistics
    summary = param_df.describe(include="all")

    # Clean up column names for summary (remove "param:" prefix)
    summary.columns = [col[6:] for col in summary.columns]

    return summary


def get_metric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all metrics.

    Args:
        df: DataFrame with experiment data (flat group:path columns)

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

    # Get all metric columns (metric:* format)
    metric_cols = [col for col in df.columns if col.startswith("metric:")]

    if not metric_cols:
        return pd.DataFrame()

    metric_df = df[metric_cols]

    # Generate summary statistics
    summary = metric_df.describe(include="all")

    # Clean up column names for summary (remove "metric:" prefix)
    summary.columns = [col[7:] for col in summary.columns]

    return summary


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix between parameters and metrics.

    Args:
        df: DataFrame with experiment data (flat group:path columns)

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

    # Get numeric parameter and metric columns
    param_cols = [col for col in df.columns if col.startswith("param:")]
    metric_cols = [col for col in df.columns if col.startswith("metric:")]

    all_cols = param_cols + metric_cols

    if not all_cols:
        return pd.DataFrame()

    # Select only numeric columns
    numeric_df = df[all_cols].select_dtypes(include=["number"])

    if numeric_df.empty:
        return pd.DataFrame()

    # Simplify column names for correlation matrix (remove group: prefix)
    new_columns = []
    for col in numeric_df.columns:
        if col.startswith("param:"):
            new_columns.append(col[6:])
        elif col.startswith("metric:"):
            new_columns.append(col[7:])
        else:
            new_columns.append(col)
    numeric_df.columns = new_columns

    # Compute correlation matrix
    correlation_matrix = numeric_df.corr()

    return correlation_matrix


def find_best_experiments(
    df: pd.DataFrame, metric: str, maximize: bool = True, top_n: int = 5
) -> pd.DataFrame:
    """
    Find the best experiments based on a specific metric.

    Args:
        df: DataFrame with experiment data (flat group:path columns)
        metric: Metric name to optimize (with or without "metric:" prefix)
        maximize: True to find maximum values, False for minimum
        top_n: Number of top experiments to return

    Returns:
        DataFrame with top experiments sorted by metric

    Examples:
        >>> best = find_best_experiments(df, "accuracy", maximize=True, top_n=3)
        >>> print(best)  # Top 3 experiments with highest accuracy
    """
    # Support both "metric:accuracy" and "accuracy" formats
    if metric.startswith("metric:"):
        metric_col = metric
    else:
        metric_col = f"metric:{metric}"

    if metric_col not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in DataFrame")

    # Filter out rows with missing metric values
    filtered_df = df.dropna(subset=[metric_col])

    if filtered_df.empty:
        return filtered_df

    # Sort by metric
    sorted_df = filtered_df.sort_values(metric_col, ascending=not maximize)

    return sorted_df.head(top_n)


def determine_varying_params(
    experiments: list, include_dep_params: bool = False
) -> list[str]:
    """
    Determine which parameters vary across experiments.

    Flattens nested parameter dictionaries before comparison, so parameters like
    {"train": {"lr": 0.001}} become "train.lr" in the output.

    Args:
        experiments: List of Experiment objects
        include_dep_params: If True, include parameters from dependencies when
            determining variation

    Returns:
        List of parameter names (with dot notation for nested params) that have
        different values across experiments

    Examples:
        >>> varying = determine_varying_params(experiments)
        >>> print(varying)  # ['lr', 'batch_size'] if only these vary
        >>> # Nested params are flattened:
        >>> print(varying)  # ['train.lr', 'model.n_layer']
        >>> # Include dependency params:
        >>> varying = determine_varying_params(experiments, include_dep_params=True)
    """
    if not experiments:
        return []

    from ..utils.dict_utils import flatten_dict

    # Collect all params from all experiments (flattened)
    all_params: dict[str, set[str]] = {}
    for exp in experiments:
        params = exp.get_params(include_deps=include_dep_params)
        # Flatten nested params to dot notation
        flat_params = flatten_dict(params)
        for key, value in flat_params.items():
            if key not in all_params:
                all_params[key] = set()
            # Convert to string for comparison (handles different types)
            all_params[key].add(str(value))

    # Return only params with >1 unique value
    return sorted([key for key, values in all_params.items() if len(values) > 1])


def _get_experiment_meta(exp: Any, meta_name: str) -> Any:
    """
    Extract a metadata value from an Experiment object.

    Args:
        exp: Experiment object
        meta_name: Metadata field name (e.g., 'name', 'status', 'tags')

    Returns:
        The metadata value, or None if not found
    """
    # Direct attribute access for common fields
    if meta_name == "id":
        return exp.id
    elif meta_name == "name":
        return exp.name
    elif meta_name == "status":
        return exp.status
    elif meta_name == "description":
        return exp.description
    elif meta_name == "tags":
        return exp.tags
    elif meta_name == "started_at":
        return exp.started_at
    elif meta_name == "completed_at":
        return exp.completed_at
    elif meta_name == "script_path":
        return str(exp.script_path) if exp.script_path else None

    # Try to get from metadata dict for other fields
    try:
        metadata = exp._load_metadata()
        if meta_name in metadata:
            return metadata[meta_name]
        # Handle nested paths like git.branch
        if "." in meta_name:
            from ..utils.dict_utils import get_nested_value

            return get_nested_value(metadata, meta_name)
    except Exception:
        pass

    return None


def metrics_to_long_dataframe(
    experiments: list,
    metrics: list[str] | None = None,
    param_cols: list[str] | None = None,
    meta_cols: list[str] | None = None,
    include_dep_params: bool = False,
) -> pd.DataFrame:
    """
    Convert experiment metrics to long (tidy) format DataFrame.

    Flattens nested parameter dictionaries, so parameters like {"train": {"lr": 0.001}}
    become "train.lr" columns in the output.

    Args:
        experiments: List of Experiment objects
        metrics: List of metric names to include (None for all)
        param_cols: List of parameter names (with dot notation for nested) to include as columns
        meta_cols: List of metadata field names to include as columns (e.g., 'name', 'status')
        include_dep_params: If True, include parameters from dependencies when extracting params

    Returns:
        DataFrame with columns: [experiment_id, step, metric_name, value, <meta...>, <params...>]

    Examples:
        >>> df = metrics_to_long_dataframe(experiments, metrics=['train_loss'], param_cols=['lr'])
        >>> print(df.columns)  # ['experiment_id', 'step', 'metric_name', 'value', 'lr']
        >>> # Nested params use dot notation:
        >>> df = metrics_to_long_dataframe(experiments, param_cols=['train.lr', 'model.n_layer'])
        >>> # Include meta columns for faceting:
        >>> df = metrics_to_long_dataframe(experiments, meta_cols=['name', 'status'])
        >>> # Include dependency params:
        >>> df = metrics_to_long_dataframe(experiments, param_cols=['model.n_embd'], include_dep_params=True)
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
        if meta_cols:
            columns.extend(meta_cols)
        if param_cols:
            columns.extend(param_cols)
        return pd.DataFrame(columns=columns)

    from ..utils.dict_utils import flatten_dict

    rows = []

    for exp in experiments:
        exp_id = exp.id
        exp_metrics = exp.get_metrics(as_dataframe=False)  # Returns list[dict]

        # Get params for this experiment (flattened to support dot notation)
        exp_params = (
            flatten_dict(exp.get_params(include_deps=include_dep_params))
            if param_cols
            else {}
        )

        # Get meta values for this experiment
        exp_meta = {}
        if meta_cols:
            for meta_name in meta_cols:
                exp_meta[meta_name] = _get_experiment_meta(exp, meta_name)

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

                # Add meta columns
                if meta_cols:
                    for meta_name in meta_cols:
                        row[meta_name] = exp_meta.get(meta_name)

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
