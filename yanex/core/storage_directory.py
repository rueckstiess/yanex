"""Directory management for experiment storage."""

from pathlib import Path

from ..utils.exceptions import StorageError
from .storage_interfaces import ExperimentDirectoryManager


class FileSystemDirectoryManager(ExperimentDirectoryManager):
    """File system-based experiment directory management."""

    def __init__(self, experiments_dir: Path = None):
        """Initialize directory manager.

        Args:
            experiments_dir: Base directory for experiments, defaults to ./experiments
        """
        if experiments_dir is None:
            experiments_dir = Path.cwd() / "experiments"

        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_directory(self, experiment_id: str) -> Path:
        """Create directory structure for experiment.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Path to experiment directory

        Raises:
            StorageError: If directory creation fails
        """
        exp_dir = self.experiments_dir / experiment_id

        if exp_dir.exists():
            raise StorageError(f"Experiment directory already exists: {exp_dir}")

        try:
            exp_dir.mkdir(parents=True)
            (exp_dir / "artifacts").mkdir()
        except Exception as e:
            raise StorageError(f"Failed to create experiment directory: {e}") from e

        return exp_dir

    def get_experiment_directory(
        self, experiment_id: str, include_archived: bool = False
    ) -> Path:
        """Get path to experiment directory.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            Path to experiment directory

        Raises:
            StorageError: If experiment directory doesn't exist
        """
        exp_dir = self.experiments_dir / experiment_id

        if exp_dir.exists():
            return exp_dir

        if include_archived:
            archive_dir = self.experiments_dir / "archived"
            archive_path = archive_dir / experiment_id
            if archive_path.exists():
                return archive_path

        # If we get here, experiment not found
        locations = [f"{exp_dir}"]
        if include_archived:
            locations.append(f"{archive_path}")

        raise StorageError(f"Experiment directory not found in: {', '.join(locations)}")

    def experiment_exists(
        self, experiment_id: str, include_archived: bool = False
    ) -> bool:
        """Check if experiment exists.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to check archived experiments too

        Returns:
            True if experiment exists
        """
        exp_dir = self.experiments_dir / experiment_id
        if exp_dir.exists() and exp_dir.is_dir():
            return True

        if include_archived:
            archive_dir = self.experiments_dir / "archived"
            archive_path = archive_dir / experiment_id
            return archive_path.exists() and archive_path.is_dir()

        return False

    def list_experiments(self, include_archived: bool = False) -> list[str]:
        """List all experiment IDs.

        Args:
            include_archived: Whether to include archived experiments

        Returns:
            List of experiment IDs
        """
        experiment_ids = []

        # List regular experiments
        if self.experiments_dir.exists():
            for item in self.experiments_dir.iterdir():
                if (
                    item.is_dir()
                    and item.name != "archived"
                    and (item / "metadata.json").exists()
                ):
                    # Validate by checking for experiment metadata
                    experiment_ids.append(item.name)

        # List archived experiments if requested
        if include_archived:
            archive_dir = self.experiments_dir / "archived"
            if archive_dir.exists():
                for item in archive_dir.iterdir():
                    if item.is_dir() and (item / "metadata.json").exists():
                        # Validate by checking for experiment metadata
                        experiment_ids.append(item.name)

        return sorted(experiment_ids)
