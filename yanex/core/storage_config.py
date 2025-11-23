"""Configuration storage for experiments."""

from typing import Any

from ..utils.exceptions import StorageError
from .config import load_yaml_config, save_yaml_config
from .storage_interfaces import ConfigurationStorage, ExperimentDirectoryManager


class FileSystemConfigurationStorage(ConfigurationStorage):
    """File system-based experiment configuration storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize configuration storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def save_config(self, experiment_id: str, config: dict[str, Any]) -> None:
        """Save experiment configuration.

        Args:
            experiment_id: Experiment identifier
            config: Configuration dictionary to save

        Raises:
            StorageError: If configuration cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        config_path = exp_dir / "params.yaml"

        try:
            save_yaml_config(config, config_path)
        except Exception as e:
            raise StorageError(f"Failed to save config: {e}") from e

    def load_config(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any]:
        """Load experiment configuration.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            Configuration dictionary

        Raises:
            StorageError: If configuration cannot be loaded
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        config_path = exp_dir / "params.yaml"

        if not config_path.exists():
            return {}

        try:
            return load_yaml_config(config_path)
        except Exception as e:
            raise StorageError(f"Failed to load config: {e}") from e
