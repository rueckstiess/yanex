"""
Results manager for filtering and comparison operations.

This module provides the ResultsManager class that handles the heavy lifting
for experiment filtering, comparison, and bulk operations.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd
else:
    # Create a dummy for runtime
    class pd:
        DataFrame = "pd.DataFrame"


from ..core.access_resolver import AccessResolver, parse_canonical_key
from ..core.filtering import ExperimentFilter
from ..core.manager import ExperimentManager
from ..utils.dict_utils import flatten_dict
from ..utils.exceptions import ExperimentNotFoundError
from .experiment import Experiment

logger = logging.getLogger(__name__)


def _resolve_column_spec(
    resolver: AccessResolver,
    spec: str | list[str],
    scope: str,
) -> str | list[str]:
    """Resolve a column specification using AccessResolver.

    Args:
        resolver: AccessResolver instance
        spec: Column specification - "auto", "all", "none", or list of keys/patterns
        scope: Scope for resolution - "param", "metric", or "meta"

    Returns:
        Special value unchanged, or list of resolved paths (without group prefix)
    """
    # Special values pass through unchanged
    if spec in ("auto", "all", "none"):
        return spec

    # Resolve list of keys/patterns
    if not isinstance(spec, list):
        return spec

    # Resolve each value (handles both single keys and patterns)
    # If resolution fails, return original value to let comparison handle it
    try:
        canonical_keys = resolver.resolve_list(spec, scope=scope)
    except Exception:
        # Resolution failed - return original list
        return spec

    # Strip group prefixes to get paths for comparison extractor
    paths = []
    for key in canonical_keys:
        _, path = parse_canonical_key(key)
        paths.append(path)

    return paths


def _build_resolver_from_experiments(
    experiments: list[Experiment],
) -> AccessResolver | None:
    """Build an AccessResolver from a list of Experiment objects.

    Aggregates params, metrics, and metadata across all experiments.

    Args:
        experiments: List of Experiment objects

    Returns:
        AccessResolver instance or None if no data available
    """
    if not experiments:
        return None

    # Aggregate data from all experiments
    all_params: dict = {}
    all_metrics: dict = {}
    all_meta: dict = {}

    for exp in experiments:
        try:
            # Get params (flattened)
            params = flatten_dict(exp.get_params())
            for key, value in params.items():
                if key not in all_params:
                    all_params[key] = value

            # Get metrics from last entry
            metrics_list = exp.get_metrics(as_dataframe=False)
            if metrics_list and isinstance(metrics_list, list):
                last_entry = metrics_list[-1] if metrics_list else {}
                if isinstance(last_entry, dict):
                    for key, value in last_entry.items():
                        if key not in ["step", "timestamp"] and key not in all_metrics:
                            all_metrics[key] = value

            # Add meta fields
            for field in ["id", "name", "status", "description", "tags"]:
                if field not in all_meta:
                    value = getattr(exp, field, None)
                    if value is not None:
                        all_meta[field] = value

        except Exception:
            continue

    if not all_params and not all_metrics and not all_meta:
        return None

    return AccessResolver(params=all_params, metrics=all_metrics, meta=all_meta)


class ResultsManager:
    """
    Backend manager for experiment results access and manipulation.

    This class provides the core functionality for finding, filtering, and comparing
    experiments. It uses the unified filtering system and provides a consistent
    interface for both CLI and Python API access.
    """

    def __init__(self, storage_path: Path | None = None):
        """
        Initialize results manager.

        Args:
            storage_path: Optional path to experiments directory (uses default if None)
        """
        if storage_path:
            self._manager = ExperimentManager(experiments_dir=storage_path)
        else:
            self._manager = ExperimentManager()

        self._filter = ExperimentFilter(manager=self._manager)

    def find(self, **filters) -> list[dict[str, Any]]:
        """
        Find experiments matching filter criteria.

        Args:
            **filters: Filter arguments (ids, status, tags, etc.)

        Returns:
            List of experiment metadata dictionaries

        Examples:
            >>> manager = ResultsManager()
            >>> experiments = manager.find(status="completed", tags=["training"])
            >>> experiments = manager.find(ids=["abc123", "def456"])
        """

        if "include_all" not in filters and "limit" not in filters:
            filters["include_all"] = True

        return self._filter.filter_experiments(**filters)

    def get_experiment(self, experiment_id: str) -> Experiment:
        """
        Get a single experiment by ID.

        Args:
            experiment_id: The experiment ID to retrieve

        Returns:
            Experiment instance

        Raises:
            ExperimentNotFoundError: If experiment doesn't exist

        Examples:
            >>> manager = ResultsManager()
            >>> exp = manager.get_experiment("abc12345")
            >>> print(exp.name, exp.status)
        """
        return Experiment(experiment_id, manager=self._manager)

    def get_experiments(self, **filters) -> list[Experiment]:
        """
        Get multiple experiments as Experiment objects.

        Args:
            **filters: Filter arguments (ids, status, tags, etc.)

        Returns:
            List of Experiment instances

        Examples:
            >>> manager = ResultsManager()
            >>> experiments = manager.get_experiments(status="completed")
            >>> for exp in experiments:
            ...     metrics = exp.get_metrics()
            ...     accuracy = metrics[-1].get('accuracy') if metrics else None
            ...     print(f"{exp.name}: {accuracy}")
        """
        metadata_list = self.find(**filters)
        experiments = []

        for metadata in metadata_list:
            try:
                exp = Experiment(metadata["id"], manager=self._manager)
                experiments.append(exp)
            except ExperimentNotFoundError as e:
                # Skip experiments that can't be loaded but log the issue
                logger.warning(
                    "Skipping experiment %s: %s", metadata.get("id", "unknown"), str(e)
                )
                continue

        return experiments

    def compare_experiments(
        self,
        params: str | list[str] = "auto",
        metrics: str | list[str] = "auto",
        meta: str | list[str] = "auto",
        include_dep_params: bool = False,
        **filters,
    ) -> pd.DataFrame:
        """
        Compare experiments and return pandas DataFrame.

        Args:
            params: Parameter columns to include:
                - "auto" (default): Only parameters that differ across experiments
                - "all": All parameters
                - "none": No parameter columns
                - list[str]: Specific parameter names
            metrics: Metric columns to include:
                - "auto" (default): Only metrics that differ across experiments
                - "all": All metrics
                - "none": No metric columns
                - list[str]: Specific metric names
            meta: Metadata columns to include:
                - "auto" (default): id, name, status
                - "all": All metadata fields
                - "none": No metadata columns
                - list[str]: Specific metadata fields
            include_dep_params: If True, include parameters from dependencies
                (merged with local params, local values take precedence)
            **filters: Filter arguments to select experiments

        Returns:
            pandas DataFrame with flat group:path columns

        Raises:
            ImportError: If pandas is not available

        Examples:
            >>> manager = ResultsManager()
            >>> df = manager.compare_experiments(
            ...     status="completed",
            ...     params=["learning_rate", "epochs"],
            ...     metrics=["accuracy", "loss"]
            ... )
            >>> print(df["metric:accuracy"].max())
            >>> # Include dependency params
            >>> df = manager.compare_experiments(
            ...     tags=["sweep"],
            ...     params=["model.n_embd"],
            ...     include_dep_params=True
            ... )
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for compare_experiments. Install it with: pip install pandas"
            )

        # Find experiments matching filters
        experiments_metadata = self.find(**filters)

        if not experiments_metadata:
            # Return empty DataFrame with proper structure
            return pd.DataFrame()

        # Resolve column specs if any are lists (for sub-path resolution and patterns)
        needs_resolution = any(isinstance(s, list) for s in [params, metrics, meta])

        if needs_resolution:
            # Get Experiment objects to build resolver
            experiment_objs = self.get_experiments(**filters)
            resolver = _build_resolver_from_experiments(experiment_objs)
            if resolver:
                params = _resolve_column_spec(resolver, params, "param")
                metrics = _resolve_column_spec(resolver, metrics, "metric")
                meta = _resolve_column_spec(resolver, meta, "meta")

        # Use existing comparison system
        from ..core.comparison import ExperimentComparisonData

        comparison_extractor = ExperimentComparisonData(manager=self._manager)

        experiment_ids = [exp["id"] for exp in experiments_metadata]
        comparison_data = comparison_extractor.get_comparison_data(
            experiment_ids=experiment_ids,
            params=params,
            metrics=metrics,
            meta=meta,
            include_archived=filters.get("archived") is not False,
            include_dep_params=include_dep_params,
        )

        # Convert to pandas DataFrame
        from .dataframe import experiments_to_dataframe

        return experiments_to_dataframe(comparison_data)

    def get_metrics(
        self,
        *,
        metrics: str | list[str] | None = None,
        params: list[str] | Literal["auto", "all", "none"] = "auto",
        meta: list[str] | None = None,
        include_dep_params: bool = False,
        as_dataframe: bool = True,
        **filters,
    ) -> "pd.DataFrame | dict[str, list[dict]]":
        """
        Get time-series metrics from multiple experiments.

        By default, returns long format (tidy data) with parameter columns included,
        optimized for visualization and grouping by hyperparameters.

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
            as_dataframe: If True, return DataFrame. If False, return dict.
            **filters: Filter arguments to select experiments (same as get_experiments)

        Returns:
            DataFrame with columns: [experiment_id, step, metric_name, value, <meta...>, <params...>]
            or dict[str, list[dict]] mapping experiment_id to metrics list

        Raises:
            ImportError: If pandas is not available and as_dataframe=True

        Examples:
            >>> manager = ResultsManager()
            >>> # Get all metrics with auto param detection
            >>> df = manager.get_metrics(tags=['sweep'])
            >>> df_loss = df[df.metric_name == 'train_loss']
            >>> # Plot by learning rate
            >>> for lr, group in df_loss.groupby('lr'):
            ...     plt.plot(group.step, group.value, label=f'lr={lr}')
            >>> # Get specific metric only
            >>> df = manager.get_metrics(tags=['training'], metrics='train_loss')
            >>> # Include all params
            >>> df = manager.get_metrics(tags=['sweep'], params='all')
            >>> # No params
            >>> df = manager.get_metrics(tags=['sweep'], params='none')
            >>> # Include meta for faceting (e.g., color by experiment name)
            >>> df = manager.get_metrics(tags=['sweep'], meta=['name'])
            >>> # Include dependency params
            >>> df = manager.get_metrics(tags=['sweep'], params=['model.n_embd'], include_dep_params=True)
            >>> # Dict format
            >>> data = manager.get_metrics(tags=['training'], as_dataframe=False)
        """
        # Normalize metrics to list
        if isinstance(metrics, str):
            metrics_list = [metrics]
        else:
            metrics_list = metrics

        # Get experiment objects matching filters
        experiments = self.get_experiments(**filters)

        if not experiments:
            if as_dataframe:
                try:
                    import pandas as pd
                except ImportError:
                    raise ImportError(
                        "pandas is required for DataFrame functionality. "
                        "Install it with: pip install pandas"
                    )
                # Return empty DataFrame with expected structure
                columns = ["experiment_id", "step", "metric_name", "value"]
                if meta:
                    columns.extend(meta)
                return pd.DataFrame(columns=columns)
            else:
                return {}

        # Resolve column specs if any are lists (for sub-path resolution and patterns)
        needs_resolution = any(
            [
                metrics_list is not None and isinstance(metrics_list, list),
                isinstance(params, list),
                meta is not None and isinstance(meta, list),
            ]
        )

        if needs_resolution:
            resolver = _build_resolver_from_experiments(experiments)
            if resolver:
                if metrics_list is not None:
                    metrics_list = _resolve_column_spec(
                        resolver, metrics_list, "metric"
                    )
                if isinstance(params, list):
                    params = _resolve_column_spec(resolver, params, "param")
                if meta is not None:
                    meta = _resolve_column_spec(resolver, meta, "meta")

        # Return dict format if requested
        if not as_dataframe:
            result = {}
            for exp in experiments:
                result[exp.id] = exp.get_metrics(as_dataframe=False)
            return result

        # Determine which params to include
        if params == "auto":
            from .dataframe import determine_varying_params

            param_cols = determine_varying_params(
                experiments, include_dep_params=include_dep_params
            )
        elif params == "all":
            # Get all params from first experiment (flattened)
            if experiments:
                from ..utils.dict_utils import flatten_dict

                flat_params = flatten_dict(
                    experiments[0].get_params(include_deps=include_dep_params)
                )
                param_cols = sorted(flat_params.keys())
            else:
                param_cols = []
        elif params == "none":
            param_cols = []
        else:
            # Specific list provided
            param_cols = params

        # Convert to long format DataFrame
        from .dataframe import metrics_to_long_dataframe

        return metrics_to_long_dataframe(
            experiments=experiments,
            metrics=metrics_list,
            param_cols=param_cols,
            meta_cols=meta,
            include_dep_params=include_dep_params,
        )

    def get_latest(self, **filters) -> Experiment | None:
        """
        Get the most recently created experiment matching filters.

        Args:
            **filters: Filter arguments

        Returns:
            Most recent Experiment or None if no matches

        Examples:
            >>> manager = ResultsManager()
            >>> latest = manager.get_latest(tags=["training"])
            >>> if latest:
            ...     print(f"Latest training run: {latest.name}")
        """
        experiments = self.find(
            limit=1, sort_by="created_at", sort_desc=True, **filters
        )
        if experiments:
            return self.get_experiment(experiments[0]["id"])
        return None

    def get_best(
        self, metric: str, maximize: bool = True, **filters
    ) -> Experiment | None:
        """
        Get the experiment with the best value for a specific metric.

        Args:
            metric: Metric name to optimize
            maximize: True to find maximum value, False for minimum
            **filters: Filter arguments

        Returns:
            Best Experiment or None if no matches

        Examples:
            >>> manager = ResultsManager()
            >>> best = manager.get_best("accuracy", maximize=True, status="completed")
            >>> if best:
            ...     metrics = best.get_metrics()
            ...     if metrics:
            ...         print(f"Best accuracy: {metrics[-1].get('accuracy')}")
        """
        experiments = self.get_experiments(**filters)

        if not experiments:
            return None

        best_exp = None
        best_value = None

        for exp in experiments:
            # Get the latest metric value from the list of metrics
            metrics = exp.get_metrics(as_dataframe=False)
            value = None

            if isinstance(metrics, list):
                # Find the metric in the most recent entries (search backwards)
                for entry in reversed(metrics):
                    if isinstance(entry, dict) and metric in entry:
                        value = entry[metric]
                        break
            elif isinstance(metrics, dict):
                # Handle the case where metrics is a single dict (shouldn't happen with new API)
                value = metrics.get(metric)

            if value is not None:
                try:
                    numeric_value = float(value)
                    if best_value is None:
                        best_exp = exp
                        best_value = numeric_value
                    elif (maximize and numeric_value > best_value) or (
                        not maximize and numeric_value < best_value
                    ):
                        best_exp = exp
                        best_value = numeric_value
                except (ValueError, TypeError):
                    # Skip non-numeric values
                    continue

        return best_exp

    def archive_experiments(self, **filters) -> int:
        """
        Archive experiments matching filters.

        Args:
            **filters: Filter arguments to select experiments

        Returns:
            Number of experiments successfully archived

        Examples:
            >>> manager = ResultsManager()
            >>> count = manager.archive_experiments(status="failed", ended_before="1 month ago")
            >>> print(f"Archived {count} failed experiments")
        """
        # Find experiments to archive (exclude already archived)
        experiments = self.find(archived=False, **filters)

        archived_count = 0
        for exp_metadata in experiments:
            try:
                self._manager.storage.archive_experiment(exp_metadata["id"])
                archived_count += 1
            except Exception:
                # Continue with other experiments if one fails
                continue

        return archived_count

    def delete_experiments(self, **filters) -> int:
        """
        Permanently delete experiments matching filters.

        Args:
            **filters: Filter arguments to select experiments

        Returns:
            Number of experiments successfully deleted

        Examples:
            >>> manager = ResultsManager()
            >>> count = manager.delete_experiments(status="failed", ended_before="1 month ago")
            >>> print(f"Deleted {count} failed experiments")
        """
        # Find experiments to delete (include both archived and unarchived)
        experiments = self.find(include_all=True, **filters)

        deleted_count = 0
        for exp_metadata in experiments:
            try:
                self._manager.storage.delete_experiment(exp_metadata["id"])
                deleted_count += 1
            except Exception:
                # Continue with other experiments if one fails
                continue

        return deleted_count

    def export_experiments(self, path: str, format: str = "json", **filters) -> None:
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
            >>> manager = ResultsManager()
            >>> manager.export_experiments(
            ...     "training_results.json",
            ...     status="completed",
            ...     tags=["training"]
            ... )
        """
        experiments = self.get_experiments(**filters)
        export_data = [exp.to_dict() for exp in experiments]

        output_path = Path(path)

        if format.lower() == "json":
            import json

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

        elif format.lower() == "csv":
            # Flatten data for CSV export
            df = self.compare_experiments(**filters)
            df.to_csv(output_path)

        elif format.lower() == "yaml":
            try:
                import yaml

                with output_path.open("w", encoding="utf-8") as f:
                    yaml.dump(export_data, f, default_flow_style=False)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML export. Install it with: pip install pyyaml"
                )

        else:
            raise ValueError(
                f"Unsupported export format: {format}. Supported formats: json, csv, yaml"
            )

    def get_experiment_count(self, **filters) -> int:
        """
        Get count of experiments matching filters.

        Args:
            **filters: Filter arguments

        Returns:
            Number of matching experiments

        Examples:
            >>> manager = ResultsManager()
            >>> count = manager.get_experiment_count(status="completed")
            >>> print(f"Found {count} completed experiments")
        """
        return self._filter.get_experiment_count(**filters)

    def experiment_exists(
        self, experiment_id: str, include_archived: bool = True
    ) -> bool:
        """
        Check if an experiment exists.

        Args:
            experiment_id: Experiment ID to check
            include_archived: Whether to include archived experiments

        Returns:
            True if experiment exists, False otherwise

        Examples:
            >>> manager = ResultsManager()
            >>> if manager.experiment_exists("abc12345"):
            ...     print("Experiment exists!")
        """
        return self._filter.experiment_exists(experiment_id, include_archived)

    def list_experiments(self, limit: int = 10, **filters) -> list[dict[str, Any]]:
        """
        List experiments with a default limit (convenience method).

        Args:
            limit: Maximum number of experiments to return
            **filters: Filter arguments

        Returns:
            List of experiment metadata dictionaries

        Examples:
            >>> manager = ResultsManager()
            >>> recent = manager.list_experiments(limit=5)
            >>> for exp in recent:
            ...     print(f"{exp['id']}: {exp.get('name', '[unnamed]')}")
        """
        return self.find(limit=limit, **filters)

    @property
    def storage_path(self) -> Path:
        """Get the experiments storage directory path."""
        return self._manager.storage.experiments_dir

    def __repr__(self) -> str:
        """String representation of manager."""
        return f"ResultsManager(storage_path='{self.storage_path}')"
