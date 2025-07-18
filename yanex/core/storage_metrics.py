"""Metrics storage for experiments."""

import json
from datetime import datetime
from typing import Any

from ..utils.exceptions import StorageError
from .storage_interfaces import ExperimentDirectoryManager, ResultsStorage


class FileSystemMetricsStorage(ResultsStorage):
    """File system-based experiment metrics storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize metrics storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def save_results(self, experiment_id: str, results: list[dict[str, Any]]) -> None:
        """Save experiment metrics.

        Args:
            experiment_id: Experiment identifier
            results: List of metrics dictionaries

        Raises:
            StorageError: If metrics cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        results_path = exp_dir / "metrics.json"

        try:
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            raise StorageError(f"Failed to save metrics: {e}") from e

    def load_results(
        self, experiment_id: str, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """Load experiment metrics.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            List of metrics dictionaries

        Raises:
            StorageError: If metrics cannot be loaded
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        metrics_path = exp_dir / "metrics.json"
        legacy_path = exp_dir / "results.json"

        # Check for legacy results.json file and migrate it
        if not metrics_path.exists() and legacy_path.exists():
            try:
                # Migrate legacy results.json to metrics.json
                with legacy_path.open("r", encoding="utf-8") as f:
                    legacy_data = json.load(f)

                # Save as metrics.json
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(legacy_data, f, indent=2)

                # Keep legacy file for backward compatibility
                print(
                    f"Info: Migrated results.json to metrics.json for experiment {experiment_id}"
                )
            except Exception as e:
                print(f"Warning: Failed to migrate legacy results.json: {e}")
                # Continue with legacy file
                try:
                    with legacy_path.open("r", encoding="utf-8") as f:
                        results = json.load(f)
                        return results if isinstance(results, list) else []
                except Exception as legacy_e:
                    raise StorageError(
                        f"Failed to load legacy metrics: {legacy_e}"
                    ) from legacy_e

        # Use metrics.json if it exists
        if metrics_path.exists():
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    results = json.load(f)
                    return results if isinstance(results, list) else []
            except Exception as e:
                raise StorageError(f"Failed to load metrics: {e}") from e

        # Fall back to legacy results.json if no metrics.json
        if legacy_path.exists():
            try:
                with legacy_path.open("r", encoding="utf-8") as f:
                    results = json.load(f)
                    return results if isinstance(results, list) else []
            except Exception as e:
                raise StorageError(f"Failed to load legacy metrics: {e}") from e

        return []

    def add_result_step(
        self,
        experiment_id: str,
        result_data: dict[str, Any],
        step: int | None = None,
    ) -> int:
        """Add a metrics step to experiment results.

        Args:
            experiment_id: Experiment identifier
            result_data: Metrics data for this step
            step: Step number, auto-incremented if None

        Returns:
            Step number that was used

        Raises:
            StorageError: If metrics cannot be added
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

        # Add or merge result
        if existing_index is not None:
            # Merge with existing step data
            existing_result = results[existing_index]
            # Keep existing timestamp and step, merge data
            merged_entry = existing_result.copy()
            merged_entry.update(result_data)  # New data overwrites existing fields
            merged_entry["step"] = step  # Ensure step stays the same
            merged_entry["last_updated"] = datetime.utcnow().isoformat()
            results[existing_index] = merged_entry
        else:
            # Add new result
            results.append(result_entry)

        # Sort results by step
        results.sort(key=lambda x: x.get("step", 0))

        # Save updated results
        self.save_results(experiment_id, results)

        return step
