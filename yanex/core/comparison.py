"""
Experiment comparison data extraction and processing.

This module provides functionality to extract, process, and organize experiment data
for comparison views, including parameter and metric analysis.
"""

import json
from typing import Any

from ..utils.datetime_utils import calculate_duration_seconds, parse_iso_timestamp
from ..utils.dict_utils import flatten_dict
from ..utils.exceptions import StorageError
from .access_resolver import AUTO_META_FIELDS
from .manager import ExperimentManager


class ExperimentComparisonData:
    """Handles data extraction and processing for experiment comparison."""

    def __init__(self, manager: ExperimentManager | None = None):
        """
        Initialize comparison data processor.

        Args:
            manager: ExperimentManager instance, creates default if None
        """
        self.manager = manager or ExperimentManager()
        self.storage = self.manager.storage

    def extract_experiment_data(
        self,
        experiment_ids: list[str],
        include_archived: bool = False,
        include_dep_params: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Extract complete data for a list of experiments.

        Args:
            experiment_ids: List of experiment IDs to extract data for
            include_archived: Whether to include archived experiments
            include_dep_params: Whether to include params from dependencies

        Returns:
            List of experiment data dictionaries

        Raises:
            StorageError: If experiment data cannot be loaded
        """
        experiments_data = []

        for exp_id in experiment_ids:
            try:
                exp_data = self._extract_single_experiment(
                    exp_id, include_archived, include_dep_params
                )
                if exp_data:
                    experiments_data.append(exp_data)
            except Exception as e:
                # Log warning but continue with other experiments
                print(f"Warning: Failed to load experiment {exp_id}: {e}")
                continue

        return experiments_data

    def _extract_single_experiment(
        self,
        experiment_id: str,
        include_archived: bool = False,
        include_dep_params: bool = False,
    ) -> dict[str, Any] | None:
        """
        Extract data from a single experiment.

        Args:
            experiment_id: Experiment ID to extract
            include_archived: Whether to search archived experiments
            include_dep_params: Whether to include params from dependencies

        Returns:
            Experiment data dictionary or None if failed
        """
        try:
            # Load metadata (required)
            metadata = self.storage.load_metadata(experiment_id, include_archived)

            # Load config (optional, with optional dependency merging)
            try:
                if include_dep_params:
                    # Use Experiment object to get merged params
                    from ..results.experiment import Experiment

                    exp = Experiment(experiment_id, self.manager)
                    config = exp.get_params(include_deps=True)
                else:
                    config = self.storage.load_config(experiment_id, include_archived)
            except StorageError:
                config = {}

            # Load results (optional)
            try:
                results = self._load_results(experiment_id, include_archived)
            except StorageError:
                results = {}

            # Combine all data
            exp_data = {
                "id": experiment_id,
                "metadata": metadata,
                "config": config,
                "results": results,
                # Extract commonly used fields for easy access
                "name": metadata.get("name"),
                "description": metadata.get("description"),
                "status": metadata.get("status", "unknown"),
                "tags": metadata.get("tags", []),
                "started_at": metadata.get("started_at"),
                "ended_at": metadata.get("completed_at"),
                "script_path": metadata.get("script_path", ""),
                "archived": metadata.get("archived", False),
            }

            return exp_data

        except Exception as e:
            raise StorageError(
                f"Failed to extract experiment {experiment_id}: {e}"
            ) from e

    def _load_results(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any]:
        """
        Load results from experiment directory.

        Args:
            experiment_id: Experiment ID
            include_archived: Whether to search archived experiments

        Returns:
            Results dictionary
        """
        exp_dir = self.storage.get_experiment_directory(experiment_id, include_archived)
        metrics_path = exp_dir / "metrics.json"
        legacy_path = exp_dir / "results.json"

        # Try metrics.json first, then fall back to results.json for backward compatibility
        if metrics_path.exists():
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                raise StorageError(f"Failed to load metrics: {e}") from e
        elif legacy_path.exists():
            try:
                with legacy_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                raise StorageError(f"Failed to load legacy results: {e}") from e

        return {}

    def discover_columns(
        self,
        experiments_data: list[dict[str, Any]],
        params: str | list[str] = "auto",
        metrics: str | list[str] = "auto",
    ) -> tuple[list[str], list[str]]:
        """
        Discover available parameter and metric columns.

        Args:
            experiments_data: List of experiment data dictionaries
            params: "auto" (only differing), "all", "none", or specific list
            metrics: "auto" (only differing), "all", "none", or specific list

        Returns:
            Tuple of (parameter_columns, metric_columns)
        """
        # Auto-discover all available columns first
        all_params: set[str] = set()
        all_metrics: set[str] = set()

        for exp_data in experiments_data:
            # Collect parameter keys from flattened config
            config = exp_data.get("config", {})
            flat_config = flatten_dict(config)
            all_params.update(flat_config.keys())

            # Collect metric keys
            results = exp_data.get("results", {})
            if isinstance(results, dict):
                all_metrics.update(results.keys())
            elif isinstance(results, list):
                # Handle list of result dictionaries
                for result_entry in results:
                    if isinstance(result_entry, dict):
                        all_metrics.update(result_entry.keys())

        # Determine final params based on mode
        if params == "none":
            final_params: list[str] = []
        elif params == "all":
            final_params = sorted(all_params)
        elif params == "auto":
            # Will be filtered later by filter_different_columns
            final_params = sorted(all_params)
        elif isinstance(params, list):
            final_params = params
        else:
            final_params = sorted(all_params)

        # Determine final metrics based on mode
        if metrics == "none":
            final_metrics: list[str] = []
        elif metrics == "all":
            final_metrics = sorted(all_metrics)
        elif metrics == "auto":
            # Will be filtered later by filter_different_columns
            final_metrics = sorted(all_metrics)
        elif isinstance(metrics, list):
            final_metrics = metrics
        else:
            final_metrics = sorted(all_metrics)

        return final_params, final_metrics

    def discover_meta_columns(
        self,
        experiments_data: list[dict[str, Any]],
        meta: str | list[str] = "auto",
    ) -> list[str]:
        """
        Discover available metadata columns.

        Args:
            experiments_data: List of experiment data dictionaries
            meta: "auto" (id, name, status), "all", "none", or specific list

        Returns:
            List of metadata column names
        """
        if meta == "none":
            return []
        elif meta == "auto":
            return list(AUTO_META_FIELDS)
        elif meta == "all":
            # Return all available metadata fields
            all_meta: set[str] = set()
            for exp_data in experiments_data:
                metadata = exp_data.get("metadata", {})
                flat_meta = flatten_dict(metadata)
                all_meta.update(flat_meta.keys())
                # Add top-level convenience fields
                for key in [
                    "id",
                    "name",
                    "status",
                    "description",
                    "tags",
                    "started_at",
                    "ended_at",
                    "script_path",
                ]:
                    if key in exp_data or key in metadata:
                        all_meta.add(key)
            return sorted(all_meta)
        elif isinstance(meta, list):
            return meta
        else:
            return list(AUTO_META_FIELDS)

    def build_comparison_matrix(
        self,
        experiments_data: list[dict[str, Any]],
        param_columns: list[str],
        metric_columns: list[str],
        meta_columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build comparison data matrix with unified columns.

        Args:
            experiments_data: List of experiment data dictionaries
            param_columns: Parameter column names
            metric_columns: Metric column names
            meta_columns: Metadata column names (optional)

        Returns:
            List of row dictionaries for table display
        """
        comparison_rows = []

        for exp_data in experiments_data:
            row = self._build_experiment_row(
                exp_data, param_columns, metric_columns, meta_columns
            )
            comparison_rows.append(row)

        return comparison_rows

    def _build_experiment_row(
        self,
        exp_data: dict[str, Any],
        param_columns: list[str],
        metric_columns: list[str],
        meta_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Build a single experiment row for the comparison table.

        Args:
            exp_data: Experiment data dictionary
            param_columns: Parameter column names (using dot notation for nested params)
            metric_columns: Metric column names
            meta_columns: Metadata column names (optional)

        Returns:
            Row dictionary with all columns using group:path format
        """
        config = exp_data.get("config", {})
        results = exp_data.get("results", {})
        metadata = exp_data.get("metadata", {})

        # Flatten config for parameter extraction
        flat_config = flatten_dict(config)
        flat_metadata = flatten_dict(metadata)

        row: dict[str, Any] = {}

        # Metadata columns with meta: prefix
        if meta_columns:
            for meta_col in meta_columns:
                # Try different sources for metadata
                if meta_col in exp_data:
                    value = exp_data[meta_col]
                elif meta_col in flat_metadata:
                    value = flat_metadata[meta_col]
                elif meta_col in metadata:
                    value = metadata[meta_col]
                else:
                    value = None

                # Format special values
                if meta_col == "tags" and isinstance(value, list):
                    row[f"meta:{meta_col}"] = value
                else:
                    row[f"meta:{meta_col}"] = (
                        self._format_value(value) if value is not None else "-"
                    )

        # Parameter columns with param: prefix
        for param in param_columns:
            value = flat_config.get(param)
            row[f"param:{param}"] = self._format_value(value)

        # Metric columns with metric: prefix
        for metric in metric_columns:
            value = self._extract_metric_value(results, metric)
            row[f"metric:{metric}"] = self._format_value(value)

        return row

    def _extract_metric_value(self, results: Any, metric_name: str) -> Any:
        """Extract a metric value from results (dict or list)."""
        if isinstance(results, dict):
            return results.get(metric_name)
        elif isinstance(results, list):
            # For list of results, try to find the latest/last value
            # or aggregate if appropriate
            for result_entry in reversed(results):  # Start from most recent
                if isinstance(result_entry, dict) and metric_name in result_entry:
                    return result_entry[metric_name]
        return None

    def _format_datetime(self, dt_str: str | None) -> str:
        """Format datetime string for display."""
        if not dt_str:
            return "-"

        dt = parse_iso_timestamp(dt_str)
        if dt is None:
            return str(dt_str) if dt_str else "-"
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _format_tags(self, tags: list) -> str:
        """Format tags list for display."""
        if not tags:
            return "-"
        return ", ".join(tags)

    def _calculate_duration(
        self, start_str: str | None, end_str: str | None, metadata: dict = None
    ) -> str:
        """Calculate and format duration."""
        # Try to use duration from metadata first
        if metadata and "duration" in metadata:
            try:
                duration_seconds = float(metadata["duration"])
                hours = int(duration_seconds // 3600)
                minutes = int((duration_seconds % 3600) // 60)
                seconds = int(duration_seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            except (ValueError, TypeError):
                pass

        # Fall back to calculating from start/end times
        if not start_str:
            return "-"

        if not end_str:
            return "[running]"

        duration_seconds = calculate_duration_seconds(start_str, end_str)
        if duration_seconds is None:
            return "-"

        # Format as HH:MM:SS
        total_seconds = int(duration_seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _format_value(self, value: Any) -> str:
        """Format a value for table display."""
        if value is None:
            return "-"

        # Handle different types appropriately
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, int | float):
            if isinstance(value, float):
                # Format floats with reasonable precision
                if abs(value) >= 10000:
                    return f"{value:.2e}"
                elif abs(value) >= 1:
                    return f"{value:.4f}".rstrip("0").rstrip(".")
                else:
                    return f"{value:.6f}".rstrip("0").rstrip(".")
            return str(value)
        elif isinstance(value, list | tuple):
            # Format lists/tuples as comma-separated strings
            return ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            # Format dicts as key=value pairs
            return ", ".join(f"{k}={v}" for k, v in value.items())
        else:
            return str(value)

    def filter_different_columns(
        self,
        comparison_rows: list[dict[str, Any]],
        param_columns: list[str],
        metric_columns: list[str],
    ) -> tuple[list[str], list[str]]:
        """
        Filter out columns where all values are identical.

        Args:
            comparison_rows: Comparison matrix rows
            param_columns: Parameter column names
            metric_columns: Metric column names

        Returns:
            Tuple of (filtered_param_columns, filtered_metric_columns)
        """
        if not comparison_rows:
            return param_columns, metric_columns

        def has_different_values(column_key: str) -> bool:
            """Check if a column has different values across experiments."""
            values = set()

            for row in comparison_rows:
                value = row.get(column_key, "-")
                # Skip missing values for difference analysis
                if value != "-":
                    values.add(value)

            # Column is "different" if it has more than one unique non-missing value
            return len(values) > 1

        # Filter parameter columns
        filtered_params = []
        for param in param_columns:
            column_key = f"param:{param}"
            if has_different_values(column_key):
                filtered_params.append(param)

        # Filter metric columns
        filtered_metrics = []
        for metric in metric_columns:
            column_key = f"metric:{metric}"
            if has_different_values(column_key):
                filtered_metrics.append(metric)

        return filtered_params, filtered_metrics

    def infer_column_types(
        self,
        comparison_rows: list[dict[str, Any]],
        param_columns: list[str],
        metric_columns: list[str],
    ) -> dict[str, str]:
        """
        Infer data types for columns to enable proper sorting.

        Args:
            comparison_rows: Comparison matrix rows
            param_columns: Parameter column names
            metric_columns: Metric column names

        Returns:
            Dictionary mapping column keys to data types ('numeric', 'datetime', 'string')
        """
        column_types = {}

        # Fixed columns - we know their types
        # Using meta: prefix for metadata columns
        column_types.update(
            {
                "meta:id": "string",
                "meta:name": "string",
                "meta:script_path": "string",
                "meta:started_at": "datetime",
                "meta:ended_at": "datetime",
                "meta:status": "string",
                "meta:tags": "string",
            }
        )

        # Infer types for parameter and metric columns
        all_columns = [f"param:{param}" for param in param_columns] + [
            f"metric:{metric}" for metric in metric_columns
        ]

        for column_key in all_columns:
            column_type = self._infer_single_column_type(comparison_rows, column_key)
            column_types[column_key] = column_type

        return column_types

    def _infer_single_column_type(
        self, comparison_rows: list[dict[str, Any]], column_key: str
    ) -> str:
        """Infer the data type of a single column."""
        # Collect non-missing values
        values = []
        for row in comparison_rows:
            value = row.get(column_key, "-")
            if value != "-":
                values.append(value)

        if not values:
            return "string"

        # Try to infer type from values
        numeric_count = 0
        datetime_count = 0

        for value in values:
            # Check if numeric
            try:
                float(value)
                numeric_count += 1
                continue
            except (ValueError, TypeError):
                pass

            # Check if datetime
            if parse_iso_timestamp(str(value)) is not None:
                datetime_count += 1
                continue

        # Determine type based on majority
        total_values = len(values)
        if numeric_count > total_values * 0.8:
            return "numeric"
        elif datetime_count > total_values * 0.8:
            return "datetime"
        else:
            return "string"

    def get_comparison_data(
        self,
        experiment_ids: list[str],
        params: str | list[str] = "auto",
        metrics: str | list[str] = "auto",
        meta: str | list[str] = "auto",
        include_archived: bool = False,
        include_dep_params: bool = False,
    ) -> dict[str, Any]:
        """
        Get complete comparison data for experiments.

        Args:
            experiment_ids: List of experiment IDs
            params: "auto" (only differing), "all", "none", or specific list
            metrics: "auto" (only differing), "all", "none", or specific list
            meta: "auto" (id, name, status), "all", "none", or specific list
            include_archived: Whether to include archived experiments
            include_dep_params: Whether to include params from dependencies

        Returns:
            Dictionary containing comparison data and metadata
        """
        # Extract experiment data
        experiments_data = self.extract_experiment_data(
            experiment_ids, include_archived, include_dep_params
        )

        if not experiments_data:
            return {
                "rows": [],
                "param_columns": [],
                "metric_columns": [],
                "meta_columns": [],
                "column_types": {},
                "total_experiments": 0,
            }

        # Discover columns
        param_columns, metric_columns = self.discover_columns(
            experiments_data, params, metrics
        )
        meta_columns = self.discover_meta_columns(experiments_data, meta)

        # Build comparison matrix
        comparison_rows = self.build_comparison_matrix(
            experiments_data, param_columns, metric_columns, meta_columns
        )

        # Filter for different columns if "auto" mode
        if params == "auto" or metrics == "auto":
            filtered_params, filtered_metrics = self.filter_different_columns(
                comparison_rows, param_columns, metric_columns
            )
            # Only update columns that were in auto mode
            if params == "auto":
                param_columns = filtered_params
            if metrics == "auto":
                metric_columns = filtered_metrics
            # Rebuild matrix with filtered columns
            comparison_rows = self.build_comparison_matrix(
                experiments_data, param_columns, metric_columns, meta_columns
            )

        # Infer column types
        column_types = self.infer_column_types(
            comparison_rows, param_columns, metric_columns
        )

        return {
            "rows": comparison_rows,
            "param_columns": param_columns,
            "metric_columns": metric_columns,
            "meta_columns": meta_columns,
            "column_types": column_types,
            "total_experiments": len(experiments_data),
        }
