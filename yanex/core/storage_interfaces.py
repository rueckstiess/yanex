"""Storage interfaces for Yanex experiment management."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any


class ExperimentDirectoryManager(ABC):
    """Interface for managing experiment directories."""

    @abstractmethod
    def create_experiment_directory(self, experiment_id: str) -> Path:
        """Create directory structure for experiment."""

    @abstractmethod
    def get_experiment_directory(
        self, experiment_id: str, include_archived: bool = False
    ) -> Path:
        """Get path to experiment directory."""

    @abstractmethod
    def experiment_exists(
        self, experiment_id: str, include_archived: bool = False
    ) -> bool:
        """Check if experiment exists."""

    @abstractmethod
    def list_experiments(self, include_archived: bool = False) -> list[str]:
        """List all experiment IDs."""


class MetadataStorage(ABC):
    """Interface for experiment metadata operations."""

    @abstractmethod
    def save_metadata(
        self,
        experiment_id: str,
        metadata: dict[str, Any],
        include_archived: bool = False,
    ) -> None:
        """Save experiment metadata."""

    @abstractmethod
    def load_metadata(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any]:
        """Load experiment metadata."""

    @abstractmethod
    def update_experiment_metadata(
        self,
        experiment_id: str,
        updates: dict[str, Any],
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """Update experiment metadata with new values."""


class ConfigurationStorage(ABC):
    """Interface for experiment configuration operations."""

    @abstractmethod
    def save_config(self, experiment_id: str, config: dict[str, Any]) -> None:
        """Save experiment configuration."""

    @abstractmethod
    def load_config(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any]:
        """Load experiment configuration."""


class ResultsStorage(ABC):
    """Interface for experiment results operations."""

    @abstractmethod
    def save_results(self, experiment_id: str, results: list[dict[str, Any]]) -> None:
        """Save experiment results."""

    @abstractmethod
    def load_results(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """Load experiment results."""

    @abstractmethod
    def add_result_step(
        self,
        experiment_id: str,
        result_data: dict[str, Any],
        step: int | None = None,
    ) -> int:
        """Add a result step to experiment results."""


class ArtifactStorage(ABC):
    """Interface for experiment artifact operations."""

    @abstractmethod
    def save_artifact(
        self,
        experiment_id: str,
        obj: Any,
        filename: str,
        saver: Callable[..., None] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save a Python object to experiment's artifacts directory."""

    @abstractmethod
    def save_text_artifact(
        self, experiment_id: str, artifact_name: str, content: str
    ) -> Path:
        """Save text content as an artifact."""

    @abstractmethod
    def get_log_paths(self, experiment_id: str) -> dict[str, Path]:
        """Get paths for log files."""


class ArchiveStorage(ABC):
    """Interface for experiment archiving operations."""

    @abstractmethod
    def archive_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Archive an experiment by moving it to archive directory."""

    @abstractmethod
    def unarchive_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Unarchive an experiment by moving it back to experiments directory."""

    @abstractmethod
    def delete_experiment(self, experiment_id: str) -> None:
        """Permanently delete an experiment directory."""

    @abstractmethod
    def delete_archived_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> None:
        """Permanently delete an archived experiment directory."""

    @abstractmethod
    def list_archived_experiments(self, archive_dir: Path | None = None) -> list[str]:
        """List all archived experiment IDs."""

    @abstractmethod
    def archived_experiment_exists(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> bool:
        """Check if archived experiment exists."""

    @abstractmethod
    def get_archived_experiment_directory(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Get path to archived experiment directory."""


class ExperimentStorageInterface(
    ExperimentDirectoryManager,
    MetadataStorage,
    ConfigurationStorage,
    ResultsStorage,
    ArtifactStorage,
    ArchiveStorage,
):
    """Combined interface for all experiment storage operations."""

    def get_experiment_dir(
        self, experiment_id: str, include_archived: bool = False
    ) -> Path:
        """Alias for get_experiment_directory to match show command usage."""
        return self.get_experiment_directory(experiment_id, include_archived)
