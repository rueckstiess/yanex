"""
Experiment comparison data extraction and processing.

This module provides functionality to extract, process, and organize experiment data
for comparison views, including parameter and metric analysis.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..utils.exceptions import StorageError
from .manager import ExperimentManager


class ExperimentComparisonData:
    """Handles data extraction and processing for experiment comparison."""

    def __init__(self, manager: Optional[ExperimentManager] = None):
        """
        Initialize comparison data processor.

        Args:
            manager: ExperimentManager instance, creates default if None
        """
        self.manager = manager or ExperimentManager()
        self.storage = self.manager.storage

    def extract_experiment_data(
        self, experiment_ids: List[str], include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Extract complete data for a list of experiments.

        Args:
            experiment_ids: List of experiment IDs to extract data for
            include_archived: Whether to include archived experiments

        Returns:
            List of experiment data dictionaries

        Raises:
            StorageError: If experiment data cannot be loaded
        """
        experiments_data = []

        for exp_id in experiment_ids:
            try:
                exp_data = self._extract_single_experiment(exp_id, include_archived)
                if exp_data:
                    experiments_data.append(exp_data)
            except Exception as e:
                # Log warning but continue with other experiments
                print(f"Warning: Failed to load experiment {exp_id}: {e}")
                continue

        return experiments_data

    def _extract_single_experiment(
        self, experiment_id: str, include_archived: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Extract data from a single experiment.

        Args:
            experiment_id: Experiment ID to extract
            include_archived: Whether to search archived experiments

        Returns:
            Experiment data dictionary or None if failed
        """
        try:
            # Load metadata (required)
            metadata = self.storage.load_metadata(experiment_id, include_archived)

            # Load config (optional)
            try:
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
    ) -> Dict[str, Any]:
        """
        Load results from experiment directory.

        Args:
            experiment_id: Experiment ID
            include_archived: Whether to search archived experiments

        Returns:
            Results dictionary
        """
        exp_dir = self.storage.get_experiment_directory(experiment_id, include_archived)
        results_path = exp_dir / "results.json"

        if not results_path.exists():
            return {}

        try:
            with results_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(f"Failed to load results: {e}") from e

    def discover_columns(
        self,
        experiments_data: List[Dict[str, Any]],
        params: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Discover available parameter and metric columns.

        Args:
            experiments_data: List of experiment data dictionaries
            params: Specific parameters to include (None for auto-discovery)
            metrics: Specific metrics to include (None for auto-discovery)

        Returns:
            Tuple of (parameter_columns, metric_columns)
        """
        if params is not None and metrics is not None:
            # Both specified - use as-is
            return params, metrics

        # Auto-discover columns
        all_params = set()
        all_metrics = set()

        for exp_data in experiments_data:
            # Collect parameter keys
            config = exp_data.get("config", {})
            all_params.update(config.keys())

            # Collect metric keys
            results = exp_data.get("results", {})
            if isinstance(results, dict):
                all_metrics.update(results.keys())
            elif isinstance(results, list):
                # Handle list of result dictionaries
                for result_entry in results:
                    if isinstance(result_entry, dict):
                        all_metrics.update(result_entry.keys())

        # Use specified or discovered columns
        final_params = params if params is not None else sorted(all_params)
        final_metrics = metrics if metrics is not None else sorted(all_metrics)

        return final_params, final_metrics

    def build_comparison_matrix(
        self,
        experiments_data: List[Dict[str, Any]],
        param_columns: List[str],
        metric_columns: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Build comparison data matrix with unified columns.

        Args:
            experiments_data: List of experiment data dictionaries
            param_columns: Parameter column names
            metric_columns: Metric column names

        Returns:
            List of row dictionaries for table display
        """
        comparison_rows = []

        for exp_data in experiments_data:
            row = self._build_experiment_row(exp_data, param_columns, metric_columns)
            comparison_rows.append(row)

        return comparison_rows

    def _build_experiment_row(
        self,
        exp_data: Dict[str, Any],
        param_columns: List[str],
        metric_columns: List[str],
    ) -> Dict[str, Any]:
        """
        Build a single experiment row for the comparison table.

        Args:
            exp_data: Experiment data dictionary
            param_columns: Parameter column names
            metric_columns: Metric column names

        Returns:
            Row dictionary with all columns
        """
        config = exp_data.get("config", {})
        results = exp_data.get("results", {})

        # Fixed columns
        row = {
            "id": exp_data["id"],
            "name": exp_data.get("name") or "[unnamed]",
            "started": self._format_datetime(exp_data.get("started_at")),
            "duration": self._calculate_duration(
                exp_data.get("started_at"),
                exp_data.get("ended_at"),
                exp_data.get("metadata", {}),
            ),
            "status": exp_data["status"],
            "tags": self._format_tags(exp_data.get("tags", [])),
        }

        # Parameter columns
        for param in param_columns:
            value = config.get(param)
            row[f"param:{param}"] = self._format_value(value)

        # Metric columns
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

    def _format_datetime(self, dt_str: Optional[str]) -> str:
        """Format datetime string for display."""
        if not dt_str:
            return "-"

        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            return str(dt_str) if dt_str else "-"

    def _format_tags(self, tags: list) -> str:
        """Format tags list for display."""
        if not tags:
            return "-"
        return ", ".join(tags)

    def _calculate_duration(
        self, start_str: Optional[str], end_str: Optional[str], metadata: dict = None
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

        try:
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            duration = end_dt - start_dt

            # Format as HH:MM:SS
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60

            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        except (ValueError, AttributeError):
            return "-"

    def _format_value(self, value: Any) -> str:
        """Format a value for table display."""
        if value is None:
            return "-"

        # Handle different types appropriately
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            if isinstance(value, float):
                # Format floats with reasonable precision
                if abs(value) >= 10000:
                    return f"{value:.2e}"
                elif abs(value) >= 1:
                    return f"{value:.4f}".rstrip("0").rstrip(".")
                else:
                    return f"{value:.6f}".rstrip("0").rstrip(".")
            return str(value)
        elif isinstance(value, (list, tuple)):
            # Format lists/tuples as comma-separated strings
            return ", ".join(str(item) for item in value)
        elif isinstance(value, dict):
            # Format dicts as key=value pairs
            return ", ".join(f"{k}={v}" for k, v in value.items())
        else:
            return str(value)

    def filter_different_columns(
        self,
        comparison_rows: List[Dict[str, Any]],
        param_columns: List[str],
        metric_columns: List[str],
    ) -> Tuple[List[str], List[str]]:
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
        comparison_rows: List[Dict[str, Any]],
        param_columns: List[str],
        metric_columns: List[str],
    ) -> Dict[str, str]:
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
        column_types.update(
            {
                "id": "string",
                "name": "string",
                "started": "datetime",
                "duration": "string",  # Duration format
                "status": "string",
                "tags": "string",
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
        self, comparison_rows: List[Dict[str, Any]], column_key: str
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
            try:
                datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                datetime_count += 1
                continue
            except (ValueError, TypeError):
                pass

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
        experiment_ids: List[str],
        params: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        only_different: bool = False,
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        """
        Get complete comparison data for experiments.

        Args:
            experiment_ids: List of experiment IDs
            params: Specific parameters to include (None for auto-discovery)
            metrics: Specific metrics to include (None for auto-discovery)
            only_different: Whether to show only columns with different values
            include_archived: Whether to include archived experiments

        Returns:
            Dictionary containing comparison data and metadata
        """
        # Extract experiment data
        experiments_data = self.extract_experiment_data(
            experiment_ids, include_archived
        )

        if not experiments_data:
            return {
                "rows": [],
                "param_columns": [],
                "metric_columns": [],
                "column_types": {},
                "total_experiments": 0,
            }

        # Discover columns
        param_columns, metric_columns = self.discover_columns(
            experiments_data, params, metrics
        )

        # Build comparison matrix
        comparison_rows = self.build_comparison_matrix(
            experiments_data, param_columns, metric_columns
        )

        # Filter for different columns if requested
        if only_different:
            param_columns, metric_columns = self.filter_different_columns(
                comparison_rows, param_columns, metric_columns
            )
            # Rebuild matrix with filtered columns
            comparison_rows = self.build_comparison_matrix(
                experiments_data, param_columns, metric_columns
            )

        # Infer column types
        column_types = self.infer_column_types(
            comparison_rows, param_columns, metric_columns
        )

        return {
            "rows": comparison_rows,
            "param_columns": param_columns,
            "metric_columns": metric_columns,
            "column_types": column_types,
            "total_experiments": len(experiments_data),
        }
