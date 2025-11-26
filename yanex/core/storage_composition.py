"""Composition class for experiment storage."""

from pathlib import Path
from typing import Any

from .dependencies import DependencyStorage
from .storage_archive import FileSystemArchiveStorage
from .storage_artifacts import FileSystemArtifactStorage
from .storage_config import FileSystemConfigurationStorage
from .storage_directory import FileSystemDirectoryManager
from .storage_executions import FileSystemScriptRunStorage
from .storage_interfaces import ExperimentStorageInterface
from .storage_metadata import FileSystemMetadataStorage
from .storage_metrics import FileSystemMetricsStorage


class CompositeExperimentStorage(ExperimentStorageInterface):
    """Composite storage that combines all storage components."""

    def __init__(self, experiments_dir: Path = None):
        """Initialize composite storage with all components.

        Args:
            experiments_dir: Base directory for experiments, defaults to ./experiments
        """
        # Initialize directory manager first (other components depend on it)
        self.directory_manager = FileSystemDirectoryManager(experiments_dir)

        # Initialize specialized storage components
        self.metadata_storage = FileSystemMetadataStorage(self.directory_manager)
        self.config_storage = FileSystemConfigurationStorage(self.directory_manager)
        self.metrics_storage = FileSystemMetricsStorage(self.directory_manager)
        self.script_run_storage = FileSystemScriptRunStorage(self.directory_manager)
        self.artifact_storage = FileSystemArtifactStorage(self.directory_manager)
        self.archive_storage = FileSystemArchiveStorage(self.directory_manager)
        self.dependency_storage = DependencyStorage(self.directory_manager)

    # Directory management methods
    def create_experiment_directory(self, experiment_id: str) -> Path:
        """Create directory structure for experiment."""
        return self.directory_manager.create_experiment_directory(experiment_id)

    def get_experiment_directory(
        self, experiment_id: str, include_archived: bool = False
    ) -> Path:
        """Get path to experiment directory."""
        return self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )

    def experiment_exists(
        self, experiment_id: str, include_archived: bool = False
    ) -> bool:
        """Check if experiment exists."""
        return self.directory_manager.experiment_exists(experiment_id, include_archived)

    def list_experiments(self, include_archived: bool = False) -> list[str]:
        """List all experiment IDs."""
        return self.directory_manager.list_experiments(include_archived)

    # Metadata methods
    def save_metadata(
        self,
        experiment_id: str,
        metadata: dict[str, Any],
        include_archived: bool = False,
    ) -> None:
        """Save experiment metadata."""
        self.metadata_storage.save_metadata(experiment_id, metadata, include_archived)

    def load_metadata(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any]:
        """Load experiment metadata."""
        return self.metadata_storage.load_metadata(experiment_id, include_archived)

    def update_experiment_metadata(
        self,
        experiment_id: str,
        updates: dict[str, Any],
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """Update experiment metadata with new values."""
        return self.metadata_storage.update_experiment_metadata(
            experiment_id, updates, include_archived
        )

    # Configuration methods
    def save_config(self, experiment_id: str, config: dict[str, Any]) -> None:
        """Save experiment configuration."""
        self.config_storage.save_config(experiment_id, config)

    def load_config(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any]:
        """Load experiment configuration."""
        return self.config_storage.load_config(experiment_id, include_archived)

    # Metrics methods
    def save_results(self, experiment_id: str, results: list[dict[str, Any]]) -> None:
        """Save experiment metrics."""
        self.metrics_storage.save_results(experiment_id, results)

    def load_results(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """Load experiment metrics."""
        return self.metrics_storage.load_results(experiment_id, include_archived)

    def add_result_step(
        self,
        experiment_id: str,
        result_data: dict[str, Any],
        step: int | None = None,
    ) -> int:
        """Add a metrics step to experiment results."""
        return self.metrics_storage.add_result_step(experiment_id, result_data, step)

    # Script run methods
    def add_script_run(
        self,
        experiment_id: str,
        script_run_data: dict[str, Any],
    ) -> None:
        """Add a script run record."""
        self.script_run_storage.add_script_run(experiment_id, script_run_data)

    def load_script_runs(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """Load experiment script runs."""
        return self.script_run_storage.load_script_runs(experiment_id, include_archived)

    # Artifact methods
    def copy_artifact(
        self, experiment_id: str, source_path: Path | str, filename: str | None = None
    ) -> Path:
        """Copy an existing file to experiment's artifacts directory."""
        return self.artifact_storage.copy_artifact(experiment_id, source_path, filename)

    def save_artifact(
        self,
        experiment_id: str,
        obj: Any,
        filename: str,
        saver: Any | None = None,
        **kwargs: Any,
    ) -> Path:
        """Save a Python object to experiment's artifacts directory."""
        return self.artifact_storage.save_artifact(
            experiment_id, obj, filename, saver, **kwargs
        )

    def load_artifact(
        self,
        experiment_id: str,
        filename: str,
        loader: Any | None = None,
        include_archived: bool = False,
        format: str | None = None,
    ) -> Any | None:
        """Load an artifact from experiment's artifacts directory."""
        return self.artifact_storage.load_artifact(
            experiment_id, filename, loader, include_archived, format
        )

    def artifact_exists(
        self, experiment_id: str, filename: str, include_archived: bool = False
    ) -> bool:
        """Check if an artifact exists."""
        return self.artifact_storage.artifact_exists(
            experiment_id, filename, include_archived
        )

    def list_artifacts(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[str]:
        """List all artifacts in experiment's artifacts directory."""
        return self.artifact_storage.list_artifacts(experiment_id, include_archived)

    def save_text_artifact(
        self, experiment_id: str, artifact_name: str, content: str
    ) -> Path:
        """Save text content as an artifact (legacy method)."""
        return self.artifact_storage.save_text_artifact(
            experiment_id, artifact_name, content
        )

    def get_log_paths(self, experiment_id: str) -> dict[str, Path]:
        """Get paths for log files."""
        return self.artifact_storage.get_log_paths(experiment_id)

    # Archive methods
    def archive_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Archive an experiment by moving it to archive directory."""
        return self.archive_storage.archive_experiment(experiment_id, archive_dir)

    def unarchive_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Unarchive an experiment by moving it back to experiments directory."""
        return self.archive_storage.unarchive_experiment(experiment_id, archive_dir)

    def delete_experiment(self, experiment_id: str) -> None:
        """Permanently delete an experiment directory."""
        self.archive_storage.delete_experiment(experiment_id)

    def delete_archived_experiment(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> None:
        """Permanently delete an archived experiment directory."""
        self.archive_storage.delete_archived_experiment(experiment_id, archive_dir)

    def list_archived_experiments(self, archive_dir: Path | None = None) -> list[str]:
        """List all archived experiment IDs."""
        return self.archive_storage.list_archived_experiments(archive_dir)

    def archived_experiment_exists(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> bool:
        """Check if archived experiment exists."""
        return self.archive_storage.archived_experiment_exists(
            experiment_id, archive_dir
        )

    def get_archived_experiment_directory(
        self, experiment_id: str, archive_dir: Path | None = None
    ) -> Path:
        """Get path to archived experiment directory."""
        return self.archive_storage.get_archived_experiment_directory(
            experiment_id, archive_dir
        )
