"""Dependency storage for experiments."""

import json
from datetime import datetime
from typing import Any

from ..utils.exceptions import StorageError
from .storage_interfaces import ExperimentDirectoryManager


class FileSystemDependencyStorage:
    """File system-based experiment dependency storage."""

    def __init__(self, directory_manager: ExperimentDirectoryManager):
        """Initialize dependency storage.

        Args:
            directory_manager: Directory manager for experiment paths
        """
        self.directory_manager = directory_manager

    def save_dependencies(
        self,
        experiment_id: str,
        dependencies: dict[str, Any],
        include_archived: bool = False,
    ) -> None:
        """Save dependency information.

        Args:
            experiment_id: Experiment identifier
            dependencies: Dependency data to save
            include_archived: Whether to search archived experiments too

        Raises:
            StorageError: If dependencies cannot be saved
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        deps_path = exp_dir / "dependencies.json"

        try:
            with deps_path.open("w", encoding="utf-8") as f:
                json.dump(dependencies, f, indent=2, sort_keys=True)
        except Exception as e:
            raise StorageError(f"Failed to save dependencies: {e}") from e

    def load_dependencies(
        self, experiment_id: str, include_archived: bool = False
    ) -> dict[str, Any] | None:
        """Load dependency information.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            Dependency dictionary or None if no dependencies exist

        Raises:
            StorageError: If dependencies cannot be loaded
        """
        exp_dir = self.directory_manager.get_experiment_directory(
            experiment_id, include_archived
        )
        deps_path = exp_dir / "dependencies.json"

        if not deps_path.exists():
            return None

        try:
            with deps_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise StorageError(f"Failed to load dependencies: {e}") from e

    def add_dependent(
        self,
        experiment_id: str,
        dependent_id: str,
        slot_name: str,
        include_archived: bool = False,
    ) -> None:
        """Add a dependent to this experiment's reverse index.

        This updates the depended_by list for the dependency experiment.

        Args:
            experiment_id: The dependency experiment ID
            dependent_id: The experiment that depends on this one
            slot_name: The slot name used by the dependent
            include_archived: Whether to search archived experiments too

        Raises:
            StorageError: If update fails
        """
        # Load existing dependencies or create empty structure
        deps = self.load_dependencies(experiment_id, include_archived) or {
            "version": "1.0",
            "declared_slots": {},
            "resolved_dependencies": {},
            "validation": None,
            "depended_by": [],
        }

        # Add new dependent
        deps["depended_by"].append(
            {
                "experiment_id": dependent_id,
                "slot_name": slot_name,
                "created_at": datetime.utcnow().isoformat(),
            }
        )

        # Save back
        self.save_dependencies(experiment_id, deps, include_archived)

    def remove_dependent(
        self,
        experiment_id: str,
        dependent_id: str,
        include_archived: bool = False,
    ) -> None:
        """Remove a dependent from this experiment's reverse index.

        Args:
            experiment_id: The dependency experiment ID
            dependent_id: The experiment that depends on this one
            include_archived: Whether to search archived experiments too

        Raises:
            StorageError: If update fails
        """
        deps = self.load_dependencies(experiment_id, include_archived)
        if not deps:
            return

        # Filter out the dependent
        deps["depended_by"] = [
            d for d in deps["depended_by"] if d["experiment_id"] != dependent_id
        ]

        # Save back
        self.save_dependencies(experiment_id, deps, include_archived)

    def experiment_has_dependencies(
        self, experiment_id: str, include_archived: bool = False
    ) -> bool:
        """Check if experiment has any dependencies.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            True if experiment has dependencies
        """
        deps = self.load_dependencies(experiment_id, include_archived)
        if not deps:
            return False
        return bool(deps.get("resolved_dependencies"))

    def experiment_is_depended_on(
        self, experiment_id: str, include_archived: bool = False
    ) -> bool:
        """Check if any experiments depend on this one.

        Args:
            experiment_id: Experiment identifier
            include_archived: Whether to search archived experiments too

        Returns:
            True if other experiments depend on this one
        """
        deps = self.load_dependencies(experiment_id, include_archived)
        if not deps:
            return False
        return bool(deps.get("depended_by"))
