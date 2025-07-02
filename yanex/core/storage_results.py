"""Results storage for experiments."""

import json
from datetime import datetime
from typing import Any

from ..utils.exceptions import StorageError
from .storage_interfaces import ExperimentDirectoryManager, ResultsStorage


class FileSystemResultsStorage(ResultsStorage):
    """File system-based experiment results storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize results storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def save_results(self, experiment_id: str, results: list[dict[str, Any]]) -> None:
        """Save experiment results.

        Args:
            experiment_id: Experiment identifier
            results: List of result dictionaries

        Raises:
            StorageError: If results cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        results_path = exp_dir / "results.json"

        try:
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            raise StorageError(f"Failed to save results: {e}") from e

    def load_results(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """Load experiment results.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            List of result dictionaries

        Raises:
            StorageError: If results cannot be loaded
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        results_path = exp_dir / "results.json"

        if not results_path.exists():
            return []

        try:
            with results_path.open("r", encoding="utf-8") as f:
                results = json.load(f)
                return results if isinstance(results, list) else []
        except Exception as e:
            raise StorageError(f"Failed to load results: {e}") from e

    def add_result_step(
        self,
        experiment_id: str,
        result_data: dict[str, Any],
        step: int | None = None,
    ) -> int:
        """Add a result step to experiment results.

        Args:
            experiment_id: Experiment identifier
            result_data: Result data for this step
            step: Step number, auto-incremented if None

        Returns:
            Step number that was used

        Raises:
            StorageError: If result cannot be added
        """
        results = self.load_results(experiment_id)

        # Determine step number
        if step is None:
            # Auto-increment: find highest step number and add 1
            max_step = -1
            for result in results:
                if "step" in result and isinstance(result["step"], int):
                    max_step = max(max_step, result["step"])
            step = max_step + 1

        # Check if step already exists
        existing_index = None
        for i, result in enumerate(results):
            if result.get("step") == step:
                existing_index = i
                break

        # Create result entry
        result_entry = result_data.copy()
        result_entry["step"] = step
        result_entry["timestamp"] = datetime.utcnow().isoformat()

        # Add or replace result
        if existing_index is not None:
            # Replace existing step (with warning - handled by caller)
            results[existing_index] = result_entry
        else:
            # Add new result
            results.append(result_entry)

        # Sort results by step
        results.sort(key=lambda x: x.get("step", 0))

        # Save updated results
        self.save_results(experiment_id, results)

        return step
