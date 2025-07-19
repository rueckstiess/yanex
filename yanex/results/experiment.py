"""
Individual experiment access and manipulation.

This module provides the Experiment class for working with individual experiments,
including metadata access, data retrieval, and metadata updates.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..core.manager import ExperimentManager
from ..utils.datetime_utils import parse_iso_timestamp
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
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found") from e

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
            return datetime.utcnow() - started

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
            self._manager.storage.load_metadata(self._experiment_id, include_archived=False)
            return False
        except Exception:
            # If not found in regular location, check archived
            try:
                self._manager.storage.load_metadata(self._experiment_id, include_archived=True)
                return True
            except Exception:
                raise ExperimentNotFoundError(f"Experiment '{self._experiment_id}' not found")

    def get_params(self) -> dict[str, Any]:
        """
        Get all experiment parameters.

        Returns:
            Dictionary of experiment parameters
        """
        if self._cached_config is None:
            try:
                self._cached_config = self._manager.storage.load_config(
                    self._experiment_id, include_archived=self.archived
                )
            except StorageError:
                self._cached_config = {}
        return self._cached_config.copy()

    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a specific parameter with support for dot notation.

        Args:
            key: Parameter key (supports dot notation like "model.learning_rate")
            default: Default value if key not found

        Returns:
            Parameter value or default
        """
        params = self.get_params()

        # Handle dot notation for nested parameters
        if "." in key:
            keys = key.split(".")
            current = params

            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default

            return current
        else:
            return params.get(key, default)

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all experiment metrics.

        Returns:
            Dictionary of experiment metrics
        """
        if self._cached_metrics is None:
            try:
                # Try loading from metrics.json first, then results.json for backward compatibility
                exp_dir = self._manager.storage.get_experiment_directory(
                    self._experiment_id, include_archived=self.archived
                )

                metrics_path = exp_dir / "metrics.json"
                results_path = exp_dir / "results.json"

                if metrics_path.exists():
                    import json

                    with metrics_path.open("r", encoding="utf-8") as f:
                        self._cached_metrics = json.load(f)
                elif results_path.exists():
                    import json

                    with results_path.open("r", encoding="utf-8") as f:
                        self._cached_metrics = json.load(f)
                else:
                    self._cached_metrics = {}

            except Exception:
                self._cached_metrics = {}

        # Ensure we return a dict even if the cached data is a list
        cached = self._cached_metrics
        if isinstance(cached, dict):
            return cached.copy()
        elif isinstance(cached, list):
            # For list of metrics, return the latest entry or empty dict
            if cached and isinstance(cached[-1], dict):
                return cached[-1].copy()
            else:
                return {}
        else:
            return {}

    def get_metric(self, key: str, default: Any = None) -> Any:
        """
        Get a specific metric value.

        Args:
            key: Metric key
            default: Default value if key not found

        Returns:
            Metric value or default
        """
        metrics = self.get_metrics()

        # Handle list of metrics (get latest value)
        if isinstance(metrics, list):
            for entry in reversed(metrics):  # Start from most recent
                if isinstance(entry, dict) and key in entry:
                    return entry[key]
            return default
        elif isinstance(metrics, dict):
            return metrics.get(key, default)
        else:
            return default

    def get_artifacts(self) -> list[Path]:
        """
        Get list of artifact paths.

        Returns:
            List of paths to experiment artifacts
        """
        try:
            exp_dir = self._manager.storage.get_experiment_directory(
                self._experiment_id, include_archived=self.archived
            )
            artifacts_dir = exp_dir / "artifacts"

            if not artifacts_dir.exists():
                return []

            artifacts = []
            for artifact_path in artifacts_dir.iterdir():
                if artifact_path.is_file():
                    artifacts.append(artifact_path)

            return sorted(artifacts)

        except Exception:
            return []

    def get_executions(self) -> list[dict[str, Any]]:
        """
        Get experiment execution history.

        Returns:
            List of execution records
        """
        try:
            return self._manager.storage.load_executions(self._experiment_id, include_archived=self.archived)
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
            raise ValueError(f"Invalid status '{status}'. Valid options: {', '.join(sorted(EXPERIMENT_STATUSES_SET))}")

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
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration.total_seconds() if self.duration else None,
            "script_path": str(self.script_path) if self.script_path else None,
            "archived": self.archived,
            "params": self.get_params(),
            "metrics": self.get_metrics(),
            "artifacts": [str(p) for p in self.get_artifacts()],
            "executions": self.get_executions(),
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
            self._cached_metadata = self._manager.storage.load_metadata(self._experiment_id, include_archived=True)
        return self._cached_metadata

    def _save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save experiment metadata."""
        self._manager.storage.save_metadata(self._experiment_id, metadata, include_archived=self.archived)

    def __repr__(self) -> str:
        """String representation of experiment."""
        name = self.name or "[unnamed]"
        return f"Experiment(id='{self.id}', name='{name}', status='{self.status}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        name = self.name or "[unnamed]"
        return f"{self.id} - {name} ({self.status})"
