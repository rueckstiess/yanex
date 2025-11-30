"""Artifact storage for experiments."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..utils.exceptions import StorageError
from .artifact_io import (
    _validate_filename,
    artifact_exists_at_path,
    copy_artifact_to_path,
    list_artifacts_at_path,
    load_artifact_from_path,
    save_artifact_to_path,
)
from .storage_interfaces import ArtifactStorage, ExperimentDirectoryManager


class FileSystemArtifactStorage(ArtifactStorage):
    """File system-based experiment artifact storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize artifact storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def copy_artifact(
        self, experiment_id: str, source_path: Path | str, filename: str | None = None
    ) -> Path:
        """Copy an existing file to experiment's artifacts directory.

        Args:
            experiment_id: Experiment identifier
            source_path: Path to source file
            filename: Optional filename to use (defaults to source filename)

        Returns:
            Path where artifact was saved

        Raises:
            StorageError: If artifact cannot be copied
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        artifacts_dir = exp_dir / "artifacts"

        try:
            return copy_artifact_to_path(source_path, artifacts_dir, filename)
        except Exception as e:
            raise StorageError(f"Failed to copy artifact: {e}") from e

    def save_artifact(
        self,
        experiment_id: str,
        obj: Any,
        filename: str,
        saver: Callable[..., None] | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save a Python object to experiment's artifacts directory.

        Args:
            experiment_id: Experiment identifier
            obj: Python object to save
            filename: Name for saved artifact
            saver: Optional custom saver function (obj, path, **kwargs) -> None
            **kwargs: Additional arguments passed to the underlying save function

        Returns:
            Path where artifact was saved

        Raises:
            StorageError: If artifact cannot be saved
        """
        # Validate filename to prevent path traversal
        try:
            filename = _validate_filename(filename)
        except ValueError as e:
            raise StorageError(f"Invalid artifact filename: {e}") from e

        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        artifacts_dir = exp_dir / "artifacts"
        target_path = artifacts_dir / filename

        try:
            save_artifact_to_path(obj, target_path, saver, **kwargs)
        except Exception as e:
            raise StorageError(f"Failed to save artifact: {e}") from e

        return target_path

    def load_artifact(
        self,
        experiment_id: str,
        filename: str,
        loader: Callable[[Path], Any] | None = None,
        include_archived: bool = False,
        format: str | None = None,
    ) -> Any | None:
        """Load an artifact from experiment's artifacts directory.

        Args:
            experiment_id: Experiment identifier
            filename: Name of artifact to load
            loader: Optional custom loader function (path) -> object
            include_archived: Whether to check archived experiments
            format: Optional format name for explicit format selection

        Returns:
            Loaded object, or None if artifact doesn't exist

        Raises:
            StorageError: If artifact cannot be loaded
        """
        # Validate filename to prevent path traversal
        try:
            filename = _validate_filename(filename)
        except ValueError as e:
            raise StorageError(f"Invalid artifact filename: {e}") from e

        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived=include_archived
        )
        artifacts_dir = exp_dir / "artifacts"
        artifact_path = artifacts_dir / filename

        if not artifact_path.exists():
            return None

        try:
            return load_artifact_from_path(artifact_path, loader, format=format)
        except Exception as e:
            raise StorageError(f"Failed to load artifact: {e}") from e

    def artifact_exists(
        self, experiment_id: str, filename: str, include_archived: bool = False
    ) -> bool:
        """Check if an artifact exists.

        Args:
            experiment_id: Experiment identifier
            filename: Name of artifact to check
            include_archived: Whether to check archived experiments

        Returns:
            True if artifact exists, False otherwise
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived=include_archived
        )
        artifacts_dir = exp_dir / "artifacts"
        return artifact_exists_at_path(artifacts_dir, filename)

    def list_artifacts(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[str]:
        """List all artifacts in experiment's artifacts directory.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to check archived experiments

        Returns:
            List of artifact filenames (sorted)
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived=include_archived
        )
        artifacts_dir = exp_dir / "artifacts"
        return list_artifacts_at_path(artifacts_dir)

    def save_text_artifact(
        self, experiment_id: str, artifact_name: str, content: str
    ) -> Path:
        """Save text content as an artifact (legacy method).

        Args:
            experiment_id: Experiment identifier
            artifact_name: Name for the artifact
            content: Text content to save

        Returns:
            Path where artifact was saved

        Raises:
            StorageError: If artifact cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        artifacts_dir = exp_dir / "artifacts"
        artifact_path = artifacts_dir / artifact_name

        try:
            with artifact_path.open("w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise StorageError(f"Failed to save text artifact: {e}") from e

        return artifact_path

    def get_log_paths(self, experiment_id: str) -> dict[str, Path]:
        """Get paths for log files.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary with log file paths
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)

        return {
            "stdout": exp_dir / "stdout.log",
            "stderr": exp_dir / "stderr.log",
        }
