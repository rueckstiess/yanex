"""Artifact storage for experiments."""

import shutil
from pathlib import Path

from ..utils.exceptions import StorageError
from .storage_interfaces import ArtifactStorage, ExperimentDirectoryManager


class FileSystemArtifactStorage(ArtifactStorage):
    """File system-based experiment artifact storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize artifact storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def save_artifact(
        self, experiment_id: str, artifact_name: str, source_path: Path
    ) -> Path:
        """Save an artifact file.

        Args:
            experiment_id: Experiment identifier
            artifact_name: Name for the artifact
            source_path: Path to source file

        Returns:
            Path where artifact was saved

        Raises:
            StorageError: If artifact cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        artifacts_dir = exp_dir / "artifacts"
        artifact_path = artifacts_dir / artifact_name

        try:
            if source_path.is_file():
                shutil.copy2(source_path, artifact_path)
            else:
                raise StorageError(f"Source path is not a file: {source_path}")
        except Exception as e:
            raise StorageError(f"Failed to save artifact: {e}") from e

        return artifact_path

    def save_text_artifact(
        self, experiment_id: str, artifact_name: str, content: str
    ) -> Path:
        """Save text content as an artifact.

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
