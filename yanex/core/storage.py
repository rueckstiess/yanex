"""
Storage management for experiments.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.exceptions import StorageError
from .config import save_yaml_config


class ExperimentStorage:
    """Manages file storage for experiments."""

    def __init__(self, experiments_dir: Path = None):
        """
        Initialize experiment storage.

        Args:
            experiments_dir: Base directory for experiments, defaults to ./experiments
        """
        if experiments_dir is None:
            experiments_dir = Path.cwd() / "experiments"

        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment_directory(self, experiment_id: str) -> Path:
        """
        Create directory structure for experiment.

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
        """
        Get path to experiment directory.

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

    def get_experiment_dir(
        self, experiment_id: str, include_archived: bool = False
    ) -> Path:
        """
        Alias for get_experiment_directory to match show command usage.
        """
        return self.get_experiment_directory(experiment_id, include_archived)

    def save_metadata(
        self,
        experiment_id: str,
        metadata: Dict[str, Any],
        include_archived: bool = False,
    ) -> None:
        """
        Save experiment metadata.

        Args:
            experiment_id: Experiment identifier
            metadata: Metadata dictionary to save
            include_archived: Whether to search archived experiments too

        Raises:
            StorageError: If metadata cannot be saved
        """
        exp_dir = self.get_experiment_directory(experiment_id, include_archived)
        metadata_path = exp_dir / "metadata.json"

        # Add timestamp
        metadata_with_timestamp = metadata.copy()
        metadata_with_timestamp["saved_at"] = datetime.utcnow().isoformat()

        try:
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata_with_timestamp, f, indent=2, sort_keys=True)
        except Exception as e:
            raise StorageError(f"Failed to save metadata: {e}") from e

    def load_metadata(
        self, experiment_id: str, include_archived: bool = False
    ) -> Dict[str, Any]:
        """
        Load experiment metadata.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            Metadata dictionary

        Raises:
            StorageError: If metadata cannot be loaded
        """
        exp_dir = self.get_experiment_directory(experiment_id, include_archived)
        metadata_path = exp_dir / "metadata.json"

        if not metadata_path.exists():
            raise StorageError(f"Metadata file not found: {metadata_path}")

        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(f"Failed to load metadata: {e}") from e

    def save_config(self, experiment_id: str, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration.

        Args:
            experiment_id: Experiment identifier
            config: Configuration dictionary to save

        Raises:
            StorageError: If configuration cannot be saved
        """
        exp_dir = self.get_experiment_directory(experiment_id)
        config_path = exp_dir / "config.yaml"

        try:
            save_yaml_config(config, config_path)
        except Exception as e:
            raise StorageError(f"Failed to save config: {e}") from e

    def load_config(
        self, experiment_id: str, include_archived: bool = False
    ) -> Dict[str, Any]:
        """
        Load experiment configuration.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            Configuration dictionary

        Raises:
            StorageError: If configuration cannot be loaded
        """
        exp_dir = self.get_experiment_directory(experiment_id, include_archived)
        config_path = exp_dir / "config.yaml"

        if not config_path.exists():
            return {}

        try:
            from .config import load_yaml_config

            return load_yaml_config(config_path)
        except Exception as e:
            raise StorageError(f"Failed to load config: {e}") from e

    def save_results(self, experiment_id: str, results: List[Dict[str, Any]]) -> None:
        """
        Save experiment results.

        Args:
            experiment_id: Experiment identifier
            results: List of result dictionaries

        Raises:
            StorageError: If results cannot be saved
        """
        exp_dir = self.get_experiment_directory(experiment_id)
        results_path = exp_dir / "results.json"

        try:
            with results_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            raise StorageError(f"Failed to save results: {e}") from e

    def load_results(
        self, experiment_id: str, include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load experiment results.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            List of result dictionaries

        Raises:
            StorageError: If results cannot be loaded
        """
        exp_dir = self.get_experiment_directory(experiment_id, include_archived)
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
        result_data: Dict[str, Any],
        step: Optional[int] = None,
    ) -> int:
        """
        Add a result step to experiment results.

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

    def save_artifact(
        self, experiment_id: str, artifact_name: str, source_path: Path
    ) -> Path:
        """
        Save an artifact file.

        Args:
            experiment_id: Experiment identifier
            artifact_name: Name for the artifact
            source_path: Path to source file

        Returns:
            Path where artifact was saved

        Raises:
            StorageError: If artifact cannot be saved
        """
        exp_dir = self.get_experiment_directory(experiment_id)
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
        """
        Save text content as an artifact.

        Args:
            experiment_id: Experiment identifier
            artifact_name: Name for the artifact
            content: Text content to save

        Returns:
            Path where artifact was saved

        Raises:
            StorageError: If artifact cannot be saved
        """
        exp_dir = self.get_experiment_directory(experiment_id)
        artifacts_dir = exp_dir / "artifacts"
        artifact_path = artifacts_dir / artifact_name

        try:
            with artifact_path.open("w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise StorageError(f"Failed to save text artifact: {e}") from e

        return artifact_path

    def get_log_paths(self, experiment_id: str) -> Dict[str, Path]:
        """
        Get paths for log files.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Dictionary with log file paths
        """
        exp_dir = self.get_experiment_directory(experiment_id)

        return {
            "stdout": exp_dir / "stdout.log",
            "stderr": exp_dir / "stderr.log",
        }

    def list_experiments(self, include_archived: bool = False) -> List[str]:
        """
        List all experiment IDs.

        Args:
            include_archived: Whether to include archived experiments

        Returns:
            List of experiment IDs
        """
        experiment_ids = []

        # List regular experiments
        if self.experiments_dir.exists():
            for item in self.experiments_dir.iterdir():
                if item.is_dir() and len(item.name) == 8 and item.name != "archived":
                    # Basic validation that it looks like an experiment ID
                    experiment_ids.append(item.name)

        # List archived experiments if requested
        if include_archived:
            archive_dir = self.experiments_dir / "archived"
            if archive_dir.exists():
                for item in archive_dir.iterdir():
                    if item.is_dir() and len(item.name) == 8:
                        # Basic validation that it looks like an experiment ID
                        experiment_ids.append(item.name)

        return sorted(experiment_ids)

    def experiment_exists(
        self, experiment_id: str, include_archived: bool = False
    ) -> bool:
        """
        Check if experiment exists.

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
            return self.archived_experiment_exists(experiment_id)

        return False

    def archive_experiment(
        self, experiment_id: str, archive_dir: Optional[Path] = None
    ) -> Path:
        """
        Archive an experiment by moving it to archive directory.

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

        exp_dir = self.get_experiment_directory(experiment_id)
        archive_path = archive_dir / experiment_id

        if archive_path.exists():
            raise StorageError(f"Archive path already exists: {archive_path}")

        try:
            shutil.move(str(exp_dir), str(archive_path))
        except Exception as e:
            raise StorageError(f"Failed to archive experiment: {e}") from e

        return archive_path

    def unarchive_experiment(
        self, experiment_id: str, archive_dir: Optional[Path] = None
    ) -> Path:
        """
        Unarchive an experiment by moving it back to experiments directory.

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
        """
        Permanently delete an experiment directory.

        Args:
            experiment_id: Experiment identifier

        Raises:
            StorageError: If deletion fails
        """
        exp_dir = self.get_experiment_directory(experiment_id)

        try:
            shutil.rmtree(exp_dir)
        except Exception as e:
            raise StorageError(f"Failed to delete experiment: {e}") from e

    def delete_archived_experiment(
        self, experiment_id: str, archive_dir: Optional[Path] = None
    ) -> None:
        """
        Permanently delete an archived experiment directory.

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

    def list_archived_experiments(
        self, archive_dir: Optional[Path] = None
    ) -> List[str]:
        """
        List all archived experiment IDs.

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
        self, experiment_id: str, archive_dir: Optional[Path] = None
    ) -> bool:
        """
        Check if archived experiment exists.

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
        self, experiment_id: str, archive_dir: Optional[Path] = None
    ) -> Path:
        """
        Get path to archived experiment directory.

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

    def update_experiment_metadata(
        self,
        experiment_id: str,
        updates: Dict[str, Any],
        include_archived: bool = False,
    ) -> Dict[str, Any]:
        """
        Update experiment metadata with new values.

        Args:
            experiment_id: Experiment identifier
            updates: Dictionary of metadata updates to apply
            include_archived: Whether to search archived experiments too

        Returns:
            Updated metadata dictionary

        Raises:
            StorageError: If metadata cannot be updated
        """
        # Load current metadata
        current_metadata = self.load_metadata(experiment_id, include_archived)

        # Apply updates
        updated_metadata = current_metadata.copy()

        # Handle tag operations first (before the main loop)
        if "add_tags" in updates or "remove_tags" in updates:
            current_tags = set(updated_metadata.get("tags", []))

            if "add_tags" in updates:
                current_tags.update(updates["add_tags"])

            if "remove_tags" in updates:
                current_tags.difference_update(updates["remove_tags"])

            updated_metadata["tags"] = sorted(current_tags)

        # Handle other field updates
        for key, value in updates.items():
            if key in ["add_tags", "remove_tags"]:
                # Skip these as they're handled above
                continue
            elif key in ["name", "description", "status"]:
                # Direct field updates
                if value == "":
                    # Empty string means clear the field
                    updated_metadata[key] = None
                else:
                    updated_metadata[key] = value
            else:
                # Other fields - direct assignment
                updated_metadata[key] = value

        # Save updated metadata
        self.save_metadata(experiment_id, updated_metadata, include_archived)

        return updated_metadata
