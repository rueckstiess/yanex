"""Archive storage for experiments."""

import shutil
from pathlib import Path

from ..utils.exceptions import StorageError
from .storage_interfaces import ArchiveStorage, ExperimentDirectoryManager


class FileSystemArchiveStorage(ArchiveStorage):
    """File system-based experiment archive storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize archive storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager
        self.experiments_dir = directory_manager.experiments_dir

    def archive_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Archive an experiment by moving it to archive directory.

        Args:
            experiment_id: Experiment identifier
            archive_dir: Archive directory, defaults to ./experiments/archived

        Returns:
            Path where experiment was archived

        Raises:
            StorageError: If archiving fails
        """
        if archive_dir is None:
            archive_dir = self.experiments_dir / "archived"

        archive_dir.mkdir(parents=True, exist_ok=True)

        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        archive_path = archive_dir / experiment_id

        if archive_path.exists():
            raise StorageError(f"Archive path already exists: {archive_path}")

        try:
            shutil.move(str(exp_dir), str(archive_path))
        except Exception as e:
            raise StorageError(f"Failed to archive experiment: {e}") from e

        return archive_path

    def unarchive_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Unarchive an experiment by moving it back to experiments directory.

        Args:
            experiment_id: Experiment identifier
            archive_dir: Archive directory, defaults to ./experiments/archived

        Returns:
            Path where experiment was unarchived

        Raises:
            StorageError: If unarchiving fails
        """
        if archive_dir is None:
            archive_dir = self.experiments_dir / "archived"

        archive_path = archive_dir / experiment_id
        if not archive_path.exists():
            raise StorageError(f"Archived experiment not found: {archive_path}")

        exp_dir = self.experiments_dir / experiment_id
        if exp_dir.exists():
            raise StorageError(f"Experiment directory already exists: {exp_dir}")

        try:
            shutil.move(str(archive_path), str(exp_dir))
        except Exception as e:
            raise StorageError(f"Failed to unarchive experiment: {e}") from e

        return exp_dir

    def delete_experiment(self, experiment_id: str) -> None:
        """Permanently delete an experiment directory.

        Args:
            experiment_id: Experiment identifier

        Raises:
            StorageError: If deletion fails
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)

        try:
            shutil.rmtree(exp_dir)
        except Exception as e:
            raise StorageError(f"Failed to delete experiment: {e}") from e

    def delete_archived_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> None:
        """Permanently delete an archived experiment directory.

        Args:
            experiment_id: Experiment identifier
            archive_dir: Archive directory, defaults to ./experiments/archived

        Raises:
            StorageError: If deletion fails
        """
        if archive_dir is None:
            archive_dir = self.experiments_dir / "archived"

        archive_path = archive_dir / experiment_id
        if not archive_path.exists():
            raise StorageError(f"Archived experiment not found: {archive_path}")

        try:
            shutil.rmtree(archive_path)
        except Exception as e:
            raise StorageError(f"Failed to delete archived experiment: {e}") from e

    def list_archived_experiments(self, archive_dir: Path | None = None) -> list[str]:
        """List all archived experiment IDs.

        Args:
            archive_dir: Archive directory, defaults to ./experiments/archived

        Returns:
            List of archived experiment IDs
        """
        if archive_dir is None:
            archive_dir = self.experiments_dir / "archived"

        if not archive_dir.exists():
            return []

        experiment_ids = []
        for item in archive_dir.iterdir():
            if item.is_dir() and len(item.name) == 8:
                # Basic validation that it looks like an experiment ID
                experiment_ids.append(item.name)

        return sorted(experiment_ids)

    def archived_experiment_exists(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> bool:
        """Check if archived experiment exists.

        Args:
            experiment_id: Experiment identifier
            archive_dir: Archive directory, defaults to ./experiments/archived

        Returns:
            True if archived experiment exists
        """
        if archive_dir is None:
            archive_dir = self.experiments_dir / "archived"

        archive_path = archive_dir / experiment_id
        return archive_path.exists() and archive_path.is_dir()

    def get_archived_experiment_directory(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Get path to archived experiment directory.

        Args:
            experiment_id: Experiment identifier
            archive_dir: Archive directory, defaults to ./experiments/archived

        Returns:
            Path to archived experiment directory

        Raises:
            StorageError: If archived experiment directory doesn't exist
        """
        if archive_dir is None:
            archive_dir = self.experiments_dir / "archived"

        archive_path = archive_dir / experiment_id

        if not archive_path.exists():
            raise StorageError(
                f"Archived experiment directory not found: {archive_path}"
            )

        return archive_path
