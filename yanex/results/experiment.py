"""
Individual experiment access and manipulation.

This module provides the Experiment class for working with individual experiments,
including metadata access, data retrieval, and metadata updates.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
else:
    # Create a dummy for runtime
    class pd:
        DataFrame = "pd.DataFrame"


from ..core.manager import ExperimentManager
from ..utils.datetime_utils import parse_iso_timestamp
from ..utils.dict_utils import get_nested_value
from ..utils.exceptions import ExperimentNotFoundError, StorageError


class Experiment:
    """
    Represents a single experiment with convenient access to all its data.

    This class provides a high-level interface for working with individual experiments,
    including reading parameters, metrics, and metadata, as well as updating metadata.
    """

    def __init__(self, experiment_id: str, manager: ExperimentManager | None = None):
        """
        Initialize experiment instance.

        Args:
            experiment_id: The experiment ID to load
            manager: Optional ExperimentManager instance (creates default if None)

        Raises:
            ExperimentNotFoundError: If the experiment doesn't exist
        """
        self._experiment_id = experiment_id
        self._manager = manager or ExperimentManager()
        self._cached_metadata = None
        self._cached_config = None
        self._cached_metrics = None

        # Verify experiment exists
        try:
            self._load_metadata()
        except Exception as e:
            raise ExperimentNotFoundError(
                f"Experiment '{experiment_id}' not found"
            ) from e

    @property
    def id(self) -> str:
        """Get experiment ID."""
        return self._experiment_id

    @property
    def name(self) -> str | None:
        """Get experiment name."""
        metadata = self._load_metadata()
        return metadata.get("name")

    @property
    def description(self) -> str | None:
        """Get experiment description."""
        metadata = self._load_metadata()
        return metadata.get("description")

    @property
    def status(self) -> str:
        """Get experiment status."""
        metadata = self._load_metadata()
        return metadata.get("status", "unknown")

    @property
    def tags(self) -> list[str]:
        """Get experiment tags."""
        metadata = self._load_metadata()
        tags = metadata.get("tags", [])
        return sorted(tags)

    @property
    def started_at(self) -> datetime | None:
        """Get experiment start time."""
        metadata = self._load_metadata()
        started_str = metadata.get("started_at")
        if started_str:
            return parse_iso_timestamp(started_str)
        return None

    @property
    def completed_at(self) -> datetime | None:
        """Get experiment completion time."""
        metadata = self._load_metadata()
        completed_str = metadata.get("completed_at")
        if completed_str:
            return parse_iso_timestamp(completed_str)
        return None

    @property
    def duration(self) -> timedelta | None:
        """Get experiment duration."""
        started = self.started_at
        completed = self.completed_at

        if started and completed:
            return completed - started
        elif started and self.status == "running":
            return datetime.now(UTC) - started

        return None

    @property
    def script_path(self) -> Path | None:
        """Get experiment script path."""
        metadata = self._load_metadata()
        script_str = metadata.get("script_path")
        if script_str:
            return Path(script_str)
        return None

    @property
    def archived(self) -> bool:
        """Check if experiment is archived."""
        try:
            # Try loading from regular location first
            self._manager.storage.load_metadata(
                self._experiment_id, include_archived=False
            )
            return False
        except Exception:
            # If not found in regular location, check archived
            try:
                self._manager.storage.load_metadata(
                    self._experiment_id, include_archived=True
                )
                return True
            except Exception:
                raise ExperimentNotFoundError(
                    f"Experiment '{self._experiment_id}' not found"
                )

    @property
    def experiment_dir(self) -> Path:
        """Get experiment directory path."""
        return self._manager.storage.get_experiment_directory(
            self._experiment_id, include_archived=self.archived
        )

    @property
    def artifacts_dir(self) -> Path:
        """Get artifacts directory path."""
        return self.experiment_dir / "artifacts"

    def get_params(
        self, include_deps: bool = False, transitive: bool = True
    ) -> dict[str, Any]:
        """
        Get all experiment parameters, optionally including dependency parameters.

        Args:
            include_deps: If True, merge parameters from dependencies (local wins on conflict)
            transitive: If True (default), include all transitive dependencies;
                       if False, only include direct dependencies

        Returns:
            Dictionary of experiment parameters. When include_deps=True, dependency
            parameters are merged with local parameters, with local values taking
            precedence on conflicts.

        Examples:
            >>> exp = get_experiment("abc12345")
            >>> # Get local params only (default)
            >>> params = exp.get_params()
            >>>
            >>> # Get params merged with all transitive dependencies
            >>> all_params = exp.get_params(include_deps=True)
            >>>
            >>> # Get params merged with direct dependencies only
            >>> direct_params = exp.get_params(include_deps=True, transitive=False)
        """
        if self._cached_config is None:
            try:
                self._cached_config = self._manager.storage.load_config(
                    self._experiment_id, include_archived=self.archived
                )
            except StorageError:
                self._cached_config = {}

        local_params = self._cached_config.copy()

        if not include_deps:
            return local_params

        # Merge dependency params (local wins on conflict)
        from ..utils.dict_utils import deep_merge

        merged = {}
        dependencies = self.get_dependencies(transitive=transitive)

        if isinstance(dependencies, dict):
            # Direct dependencies as dict[slot, Experiment]
            deps_list = list(dependencies.values())
        else:
            # Transitive dependencies as list[Experiment]
            deps_list = dependencies

        # Merge dependency params in order (later deps override earlier)
        for dep in deps_list:
            dep_params = dep.get_params()  # Don't recurse include_deps
            merged = deep_merge(merged, dep_params)

        # Local params override dependency params
        merged = deep_merge(merged, local_params)

        return merged

    def get_param(
        self, key: str, default: Any = None, include_deps: bool = False
    ) -> Any:
        """
        Get a specific parameter with support for dot notation.

        Args:
            key: Parameter key (supports dot notation like "model.learning_rate")
            default: Default value if key not found
            include_deps: If True, search in merged params including dependencies

        Returns:
            Parameter value or default

        Examples:
            >>> exp = get_experiment("abc12345")
            >>> # Get local param only (default)
            >>> lr = exp.get_param("learning_rate")
            >>>
            >>> # Get param from merged params (includes dependencies)
            >>> n_embd = exp.get_param("model.n_embd", include_deps=True)
        """
        params = self.get_params(include_deps=include_deps)
        return get_nested_value(params, key, default=default)

    def get_metrics(
        self, step: int | None = None, as_dataframe: bool = True
    ) -> "list[dict[str, Any]] | dict[str, Any] | pd.DataFrame":
        """
        Get experiment metrics.

        Args:
            step: Optional step number. If provided, returns the metrics for that step only.
            as_dataframe: If True (default), return DataFrame. If False, return list/dict.

        Returns:
            - If as_dataframe=True and step=None: DataFrame with step as index, metrics as columns
            - If as_dataframe=True and step=N: DataFrame with single row for that step
            - If as_dataframe=False and step=None: List of metric dictionaries
            - If as_dataframe=False and step=N: Single dictionary for that step

        Examples:
            >>> # Get as DataFrame (default)
            >>> df = exp.get_metrics()
            >>> df.plot(y='train_accuracy')
            >>>
            >>> # Get as list of dicts
            >>> metrics = exp.get_metrics(as_dataframe=False)
            >>> for m in metrics:
            ...     print(m['train_loss'])
            >>>
            >>> # Get specific step as DataFrame
            >>> df_step_5 = exp.get_metrics(step=5)
            >>>
            >>> # Get specific step as dict
            >>> step_5 = exp.get_metrics(step=5, as_dataframe=False)
        """
        if self._cached_metrics is None:
            try:
                exp_dir = self._manager.storage.get_experiment_directory(
                    self._experiment_id, include_archived=self.archived
                )
                metrics_path = exp_dir / "metrics.json"

                if metrics_path.exists():
                    import json

                    with metrics_path.open("r", encoding="utf-8") as f:
                        self._cached_metrics = json.load(f)
                else:
                    self._cached_metrics = []

            except Exception:
                self._cached_metrics = []

        # Ensure we have a list
        cached = self._cached_metrics if isinstance(self._cached_metrics, list) else []

        # Get data as list/dict first
        if step is not None:
            # Return metrics for specific step
            data = {}
            for entry in cached:
                if isinstance(entry, dict) and entry.get("step") == step:
                    data = entry.copy()
                    break
        else:
            # Return entire list
            data = [
                entry.copy() if isinstance(entry, dict) else entry for entry in cached
            ]

        # Convert to DataFrame if requested
        if as_dataframe:
            try:
                import pandas as pd
            except ImportError:
                raise ImportError(
                    "pandas is required for DataFrame functionality. "
                    "Install it with: pip install pandas"
                )

            if step is not None:
                # Single step: return DataFrame with one row
                if data:
                    df = pd.DataFrame([data])
                    if "step" in df.columns:
                        df = df.set_index("step")
                    return df
                else:
                    # Empty step: return empty DataFrame
                    return pd.DataFrame()
            else:
                # All steps: return DataFrame with step as index
                if data:
                    df = pd.DataFrame(data)
                    if "step" in df.columns:
                        df = df.set_index("step")
                    # Convert numeric columns
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col])
                        except (ValueError, TypeError):
                            pass  # Keep as-is if not numeric
                    return df
                else:
                    # Empty metrics: return empty DataFrame
                    return pd.DataFrame()
        else:
            # Return raw data (list or dict)
            return data

    def get_metric(self, name: str) -> Any | None:
        """
        Get a specific metric by name.

        Args:
            name: Name of the metric to retrieve

        Returns:
            Single value if there's only one step, list of values if multiple steps,
            or None if metric not found
        """
        # Get metrics as list for backward compatibility
        metrics = self.get_metrics(as_dataframe=False)
        if not metrics:
            return None

        # Collect all values for the specified metric across all steps
        values = []
        for entry in metrics:
            if isinstance(entry, dict) and name in entry:
                # Skip the 'step' key when extracting metric values
                if name != "step":
                    values.append(entry[name])

        if not values:
            return None
        elif len(values) == 1:
            return values[0]
        else:
            return values

    def load_artifact(self, filename: str, loader: Any | None = None) -> Any | None:
        """
        Load an artifact with automatic format detection.

        Returns None if artifact doesn't exist (allows optional artifacts).

        Args:
            filename: Name of artifact to load
            loader: Optional custom loader function (path) -> object

        Supported formats (auto-detected by extension):
            .txt        - Plain text (returns str)
            .csv        - CSV (returns pandas.DataFrame)
            .json       - JSON (returns parsed dict/list)
            .jsonl      - JSON Lines (returns list[dict])
            .npy        - NumPy array (returns np.ndarray)
            .npz        - NumPy arrays (returns dict of arrays)
            .pt, .pth   - PyTorch (returns loaded object)
            .pkl        - Pickle (returns unpickled object)
            .png        - Image (returns PIL.Image)

        Returns:
            Loaded object, or None if artifact doesn't exist

        Raises:
            ValueError: If format can't be auto-detected and no custom loader provided
            ImportError: If required library not installed
            StorageError: If artifact cannot be loaded

        Examples:
            # Auto-loading with format detection
            model = exp.load_artifact("model.pt")  # Returns loaded object
            data = exp.load_artifact("data.json")  # Returns parsed dict

            # Optional artifact
            checkpoint = exp.load_artifact("checkpoint.pt")
            if checkpoint is not None:
                model.load_state_dict(checkpoint)

            # Custom loader
            def load_custom(path):
                with open(path, 'rb') as f:
                    return custom_deserialize(f)

            obj = exp.load_artifact("data.custom", loader=load_custom)
        """
        return self._manager.storage.load_artifact(
            self._experiment_id, filename, loader, include_archived=self.archived
        )

    def artifact_exists(self, filename: str) -> bool:
        """
        Check if an artifact exists without loading it.

        Args:
            filename: Name of artifact to check

        Returns:
            True if artifact exists, False otherwise

        Examples:
            if exp.artifact_exists("checkpoint.pt"):
                checkpoint = exp.load_artifact("checkpoint.pt")
        """
        return self._manager.storage.artifact_exists(
            self._experiment_id, filename, include_archived=self.archived
        )

    def list_artifacts(self) -> list[str]:
        """
        List all artifacts in the experiment.

        Returns:
            List of artifact filenames (sorted)

        Examples:
            artifacts = exp.list_artifacts()
            # Returns: ["model.pt", "metrics.json", "plot.png"]

            for artifact_name in exp.list_artifacts():
                print(f"Found artifact: {artifact_name}")
        """
        return self._manager.storage.list_artifacts(
            self._experiment_id, include_archived=self.archived
        )

    @property
    def dependencies(self) -> dict[str, str]:
        """
        Get the dependencies dict for this experiment.

        Returns:
            Dict mapping slot names to experiment IDs

        Examples:
            >>> exp = get_experiment("abc12345")
            >>> deps = exp.dependencies
            >>> print(f"Data from: {deps.get('data')}")
        """
        dep_data = self._manager.storage.dependency_storage.load_dependencies(
            self._experiment_id, include_archived=self.archived
        )
        return dep_data.get("dependencies", {})

    @property
    def dependency_ids(self) -> list[str]:
        """
        Get the dependency IDs for this experiment (deprecated, use .dependencies).

        Returns:
            List of experiment IDs that this experiment depends on

        Examples:
            >>> exp = get_experiment("abc12345")
            >>> dep_ids = exp.dependency_ids
            >>> print(f"Depends on: {dep_ids}")
        """
        return list(self.dependencies.values())

    @property
    def has_dependencies(self) -> bool:
        """
        Check if this experiment has any dependencies.

        Returns:
            True if experiment has dependencies, False otherwise

        Examples:
            >>> exp = get_experiment("abc12345")
            >>> if exp.has_dependencies:
            ...     print("This experiment depends on other experiments")
        """
        return len(self.dependencies) > 0

    def get_dependency(self, slot: str) -> "Experiment | None":
        """
        Get dependency experiment for a specific slot.

        Args:
            slot: The slot name (e.g., "data", "model", "dep1")

        Returns:
            Experiment object for the slot, or None if slot not found

        Examples:
            >>> exp = get_experiment("abc12345")
            >>> data_exp = exp.get_dependency("data")
            >>> if data_exp:
            ...     dataset = data_exp.load_artifact("dataset.pkl")
        """
        dep_id = self.dependencies.get(slot)
        if dep_id is None:
            return None

        try:
            return Experiment(dep_id, self._manager)
        except Exception:
            return None

    def get_dependencies(
        self, transitive: bool = False, include_self: bool = False
    ) -> "dict[str, Experiment] | list[Experiment]":
        """
        Get experiment dependencies as Experiment objects.

        Args:
            transitive: If True, return flat list of all transitive dependencies
            include_self: If True, include current experiment in result (only with transitive=True)

        Returns:
            If transitive=False: dict[str, Experiment] - slot name to Experiment
            If transitive=True: list[Experiment] - flat list of all dependencies

        Examples:
            # Get direct dependencies as dict
            >>> exp = get_experiment("abc12345")
            >>> deps = exp.get_dependencies()
            >>> data_exp = deps.get("data")

            # Get all dependencies (transitive) as flat list
            >>> all_deps = exp.get_dependencies(transitive=True)
            >>> for dep in all_deps:
            ...     print(f"{dep.id} artifacts: {dep.list_artifacts()}")
        """
        # Get dependency IDs
        if transitive:
            # Get all dependencies using DependencyResolver (flat list)
            from ..core.dependencies import DependencyResolver

            resolver = DependencyResolver(self._manager)
            dependency_ids = resolver.get_transitive_dependencies(
                self._experiment_id, include_self=include_self, include_archived=True
            )

            # Create Experiment objects for each dependency
            dependencies = []
            for dep_id in dependency_ids:
                try:
                    experiment = Experiment(dep_id, self._manager)
                    dependencies.append(experiment)
                except Exception:
                    continue

            return dependencies
        else:
            # Get only direct dependencies as dict
            deps_dict = self.dependencies

            # Create Experiment objects for each dependency
            dependencies = {}
            for slot, dep_id in deps_dict.items():
                try:
                    experiment = Experiment(dep_id, self._manager)
                    dependencies[slot] = experiment
                except Exception:
                    continue

            return dependencies

    def get_script_runs(self) -> list[dict[str, Any]]:
        """
        Get experiment script run history.

        Returns:
            List of script run records
        """
        try:
            return self._manager.storage.load_script_runs(
                self._experiment_id, include_archived=self.archived
            )
        except Exception:
            return []

    def set_name(self, name: str) -> None:
        """
        Set experiment name.

        Args:
            name: New experiment name

        Raises:
            ValueError: If name is invalid
            StorageError: If update fails
        """
        if not isinstance(name, str):
            raise ValueError("Name must be a string")

        metadata = self._load_metadata()
        metadata["name"] = name
        self._save_metadata(metadata)
        self._cached_metadata = None  # Clear cache

    def set_description(self, description: str) -> None:
        """
        Set experiment description.

        Args:
            description: New experiment description

        Raises:
            ValueError: If description is invalid
            StorageError: If update fails
        """
        if not isinstance(description, str):
            raise ValueError("Description must be a string")

        metadata = self._load_metadata()
        metadata["description"] = description
        self._save_metadata(metadata)
        self._cached_metadata = None  # Clear cache

    def add_tags(self, tags: list[str]) -> None:
        """
        Add tags to experiment.

        Args:
            tags: List of tags to add

        Raises:
            ValueError: If tags are invalid
            StorageError: If update fails
        """
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list of strings")
        if not all(isinstance(tag, str) for tag in tags):
            raise ValueError("All tags must be strings")

        metadata = self._load_metadata()
        current_tags = set(metadata.get("tags", []))
        current_tags.update(tags)
        metadata["tags"] = sorted(current_tags)
        self._save_metadata(metadata)
        self._cached_metadata = None  # Clear cache

    def remove_tags(self, tags: list[str]) -> None:
        """
        Remove tags from experiment.

        Args:
            tags: List of tags to remove

        Raises:
            ValueError: If tags are invalid
            StorageError: If update fails
        """
        if not isinstance(tags, list):
            raise ValueError("Tags must be a list of strings")
        if not all(isinstance(tag, str) for tag in tags):
            raise ValueError("All tags must be strings")

        metadata = self._load_metadata()
        current_tags = set(metadata.get("tags", []))
        current_tags.difference_update(tags)
        metadata["tags"] = sorted(current_tags)
        self._save_metadata(metadata)
        self._cached_metadata = None  # Clear cache

    def set_status(self, status: str) -> None:
        """
        Set experiment status.

        Args:
            status: New experiment status

        Raises:
            ValueError: If status is invalid
            StorageError: If update fails
        """
        from ..core.constants import EXPERIMENT_STATUSES_SET

        if not isinstance(status, str):
            raise ValueError("Status must be a string")
        if status not in EXPERIMENT_STATUSES_SET:
            raise ValueError(
                f"Invalid status '{status}'. Valid options: {', '.join(sorted(EXPERIMENT_STATUSES_SET))}"
            )

        metadata = self._load_metadata()
        metadata["status"] = status
        self._save_metadata(metadata)
        self._cached_metadata = None  # Clear cache

    def to_dict(self) -> dict[str, Any]:
        """
        Get complete experiment data as dictionary.

        Returns:
            Dictionary containing all experiment data
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "tags": self.tags,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "duration_seconds": self.duration.total_seconds()
            if self.duration
            else None,
            "script_path": str(self.script_path) if self.script_path else None,
            "archived": self.archived,
            "params": self.get_params(),
            "metrics": self.get_metrics(as_dataframe=False),
            "artifacts": self.list_artifacts(),
            "script_runs": self.get_script_runs(),
            "dependency_ids": self.dependency_ids,
            "has_dependencies": self.has_dependencies,
        }

    def refresh(self) -> None:
        """
        Refresh cached data by reloading from storage.
        """
        self._cached_metadata = None
        self._cached_config = None
        self._cached_metrics = None

    def _load_metadata(self) -> dict[str, Any]:
        """Load experiment metadata with caching."""
        if self._cached_metadata is None:
            self._cached_metadata = self._manager.storage.load_metadata(
                self._experiment_id, include_archived=True
            )
        return self._cached_metadata

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save experiment metadata."""
        self._manager.storage.save_metadata(
            self._experiment_id, metadata, include_archived=self.archived
        )

    def __repr__(self) -> str:
        """String representation of experiment."""
        name = self.name or "[unnamed]"
        return f"Experiment(id='{self.id}', name='{name}', status='{self.status}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        name = self.name or "[unnamed]"
        return f"{self.id} - {name} ({self.status})"
