"""Script run storage for experiments."""

import json
from datetime import datetime
from typing import Any

from ..utils.exceptions import StorageError
from .storage_interfaces import ExperimentDirectoryManager


class FileSystemScriptRunStorage:
    """File system-based experiment script run storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize script run storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def save_script_runs(
        self, experiment_id: str, script_runs: list[dict[str, Any]]
    ) -> None:
        """Save experiment script runs.

        Args:
            experiment_id: Experiment identifier
            script_runs: List of script run dictionaries

        Raises:
            StorageError: If script runs cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        script_runs_path = exp_dir / "script_runs.json"

        try:
            with script_runs_path.open("w", encoding="utf-8") as f:
                json.dump(script_runs, f, indent=2)
        except Exception as e:
            raise StorageError(f"Failed to save script runs: {e}") from e

    def load_script_runs(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """Load experiment script runs.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            List of script run dictionaries

        Raises:
            StorageError: If script runs cannot be loaded
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        script_runs_path = exp_dir / "script_runs.json"

        if not script_runs_path.exists():
            return []

        try:
            with script_runs_path.open("r", encoding="utf-8") as f:
                script_runs = json.load(f)
                return script_runs if isinstance(script_runs, list) else []
        except Exception as e:
            raise StorageError(f"Failed to load script runs: {e}") from e

    def add_script_run(
        self,
        experiment_id: str,
        script_run_data: dict[str, Any],
    ) -> None:
        """Add a script run record to experiment script runs.

        Args:
            experiment_id: Experiment identifier
            script_run_data: Script run data to record

        Raises:
            StorageError: If script run cannot be added
        """
        script_runs = self.load_script_runs(experiment_id)

        # Create script run entry
        script_run_entry = script_run_data.copy()
        script_run_entry["recorded_at"] = datetime.utcnow().isoformat()

        # Add new script run
        script_runs.append(script_run_entry)

        # Save updated script runs
        self.save_script_runs(experiment_id, script_runs)
