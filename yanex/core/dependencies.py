"""Dependency resolution and validation for experiment workflows."""

import json
from datetime import UTC, datetime
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Any

from ..utils.exceptions import (
    CircularDependencyError,
    ExperimentNotFoundError,
    InvalidDependencyError,
    StorageError,
)
from ..utils.id_resolution import resolve_experiment_id

if TYPE_CHECKING:
    from .manager import ExperimentManager
    from .storage_directory import FileSystemDirectoryManager


class DependencyStorage:
    """Handles persistence of dependency data."""

    def __init__(self, directory_manager: "FileSystemDirectoryManager"):
        """Initialize dependency storage.

        Args:
            directory_manager: File system directory manager instance.
        """
        self.directory_manager = directory_manager

    def save_dependencies(
        self,
        experiment_id: str,
        dependencies: dict[str, str],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save dependencies.json file.

        Args:
            experiment_id: Experiment ID to save dependencies for.
            dependencies: Dict mapping slot names to full experiment IDs.
            metadata: Optional metadata about each dependency.
        """
        exp_dir = self.directory_manager.get_experiment_directory(experiment_id)
        dependencies_file = exp_dir / "dependencies.json"

        data = {
            "dependencies": dependencies,
            "created_at": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
        }

        with open(dependencies_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_dependencies(
        self,
        experiment_id: str,
        include_archived: bool = False,
    ) -> dict[str, Any]:
        """Load dependencies.json file.

        Args:
            experiment_id: Experiment ID to load dependencies for.
            include_archived: Whether to search archived experiments.

        Returns:
            Dict with keys: dependencies (dict), created_at, metadata.
            Returns empty dict with empty dependencies dict if file doesn't exist.
        """
        try:
            exp_dir = self.directory_manager.get_experiment_directory(
                experiment_id, include_archived=include_archived
            )
            dependencies_file = exp_dir / "dependencies.json"

            if not dependencies_file.exists():
                return {"dependencies": {}, "created_at": None, "metadata": {}}

            with open(dependencies_file) as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            return {"dependencies": {}, "created_at": None, "metadata": {}}
        except json.JSONDecodeError:
            # Malformed file - return empty
            return {"dependencies": {}, "created_at": None, "metadata": {}}

    def dependency_file_exists(
        self,
        experiment_id: str,
        include_archived: bool = False,
    ) -> bool:
        """Check if dependencies.json exists.

        Args:
            experiment_id: Experiment ID to check.
            include_archived: Whether to search archived experiments.

        Returns:
            True if dependencies.json file exists.
        """
        try:
            exp_dir = self.directory_manager.get_experiment_directory(
                experiment_id, include_archived=include_archived
            )
            dependencies_file = exp_dir / "dependencies.json"
            return dependencies_file.exists()
        except StorageError:
            return False


class DependencyResolver:
    """Handles dependency resolution, validation, and graph operations."""

    def __init__(self, manager: "ExperimentManager"):
        """Initialize dependency resolver.

        Args:
            manager: ExperimentManager instance.
        """
        self.manager = manager

    def resolve_short_id(self, short_id: str, include_archived: bool = True) -> str:
        """Resolve short experiment ID to full ID using existing utilities.

        Args:
            short_id: Partial experiment ID (4+ characters).
            include_archived: Whether to search archived experiments.

        Returns:
            Full 8-character experiment ID.

        Raises:
            ExperimentNotFoundError: No matching experiment.
            AmbiguousIDError: Multiple matches found.
        """
        return resolve_experiment_id(
            short_id, self.manager, include_archived=include_archived
        )

    def validate_dependency(
        self,
        experiment_id: str,
        for_staging: bool = False,
        include_archived: bool = True,
    ) -> None:
        """Validate that experiment can be used as dependency.

        Args:
            experiment_id: Full experiment ID to validate.
            for_staging: If True, allow dependencies with status="staged".
            include_archived: Whether to allow archived experiments as dependencies.

        Raises:
            ExperimentNotFoundError: Experiment doesn't exist.
            InvalidDependencyError: Experiment status is not valid for dependency.
        """
        # Check if experiment exists
        if not self.manager.storage.experiment_exists(
            experiment_id, include_archived=include_archived
        ):
            raise ExperimentNotFoundError(experiment_id)

        # Load metadata to check status
        metadata = self.manager.storage.load_metadata(
            experiment_id, include_archived=include_archived
        )
        status = metadata.get("status")

        # Validate status
        if for_staging:
            # When creating staged experiments, allow staged dependencies
            valid_statuses = ["completed", "staged"]
        else:
            # Normal case: only completed experiments
            valid_statuses = ["completed"]

        if status not in valid_statuses:
            name = metadata.get("name") or "unnamed"
            raise InvalidDependencyError(
                f"Dependency '{experiment_id}' has invalid status '{status}'.\n\n"
                f"Dependencies must have status: {' or '.join(repr(s) for s in valid_statuses)}.\n\n"
                f"Current status: {status}\n"
                f"Experiment: {experiment_id} ({name})\n\n"
                "To fix:\n"
                f"  - Wait for experiment to complete if it's running\n"
                f"  - Use '--stage' flag if creating staged experiments"
            )

    def resolve_and_validate_dependencies(
        self,
        dependencies: dict[str, str],
        for_staging: bool = False,
        include_archived: bool = True,
    ) -> dict[str, str]:
        """Resolve short IDs and validate all dependencies.

        Args:
            dependencies: Dict mapping slot names to experiment IDs (may be short IDs).
            for_staging: If True, allow dependencies with status="staged".
            include_archived: Whether to allow archived experiments as dependencies.

        Returns:
            Dict mapping slot names to full experiment IDs.

        Raises:
            ExperimentNotFoundError: Dependency doesn't exist.
            AmbiguousIDError: Short ID matches multiple experiments.
            InvalidDependencyError: Dependency has invalid status.
        """
        resolved = {}

        for slot, dep_id in dependencies.items():
            # Resolve short ID to full ID
            full_id = self.resolve_short_id(dep_id, include_archived=include_archived)

            # Validate dependency
            self.validate_dependency(
                full_id, for_staging=for_staging, include_archived=include_archived
            )

            resolved[slot] = full_id

        return resolved

    def get_transitive_dependencies(
        self,
        experiment_id: str,
        include_self: bool = False,
        include_archived: bool = True,
    ) -> list[str]:
        """Get all dependencies (direct + transitive) in topological order.

        Uses graphlib.TopologicalSorter from Python standard library for
        graph traversal and cycle detection.

        Args:
            experiment_id: Experiment ID to get dependencies for.
            include_self: If True, include experiment_id in result.
            include_archived: Whether to search archived experiments.

        Returns:
            List of experiment IDs in topological order (dependencies before dependents).

        Raises:
            CircularDependencyError: Circular dependency detected.
        """
        # Build dependency graph
        graph: dict[str, list[str]] = {}
        to_visit = [experiment_id]
        visited = set()

        while to_visit:
            exp_id = to_visit.pop()
            if exp_id in visited:
                continue
            visited.add(exp_id)

            # Load dependencies for this experiment
            storage = self.manager.storage
            if hasattr(storage, "dependency_storage"):
                dep_data = storage.dependency_storage.load_dependencies(
                    exp_id, include_archived=include_archived
                )
            else:
                # Fallback for testing or if storage doesn't have dependency_storage yet
                dep_data = {"dependencies": {}}

            # Extract dependency IDs from dict (values are the exp IDs)
            deps_dict = dep_data.get("dependencies", {})
            dep_ids = list(deps_dict.values())
            graph[exp_id] = dep_ids

            # Add dependencies to visit queue
            to_visit.extend(dep_ids)

        # Use TopologicalSorter for ordering and cycle detection
        try:
            sorter = TopologicalSorter(graph)
            sorted_ids = list(sorter.static_order())
        except CycleError as e:
            raise CircularDependencyError(
                f"Circular dependency detected in experiment '{experiment_id}': {e}"
            ) from e

        # Filter to only include dependencies (not the experiment itself unless requested)
        if include_self:
            return sorted_ids
        else:
            return [exp_id for exp_id in sorted_ids if exp_id != experiment_id]

    def detect_circular_dependency(
        self,
        experiment_id: str,
        new_dependency_id: str,
        include_archived: bool = True,
    ) -> bool:
        """Check if adding dependency would create a cycle.

        Uses graphlib.TopologicalSorter which automatically detects cycles.

        Args:
            experiment_id: Experiment that would get new dependency.
            new_dependency_id: Dependency to be added.
            include_archived: Whether to search archived experiments.

        Returns:
            True if circular dependency would be created.
        """
        try:
            # Get all dependencies of the new dependency
            dep_chain = self.get_transitive_dependencies(
                new_dependency_id,
                include_self=True,
                include_archived=include_archived,
            )

            # If current experiment is in that chain, it's circular
            return experiment_id in dep_chain
        except CircularDependencyError:
            # If the new dependency itself has a cycle, that's also a problem
            return True

    def find_artifact_in_dependencies(
        self,
        experiment_id: str,
        artifact_filename: str,
        include_archived: bool = True,
    ) -> tuple[str | None, list[str]]:
        """Search for artifact in experiment and all dependencies.

        Search order:
        1. Current experiment
        2. Direct dependencies (in declaration order)
        3. Transitive dependencies (topological order)

        Args:
            experiment_id: Current experiment ID.
            artifact_filename: Name of artifact to find.
            include_archived: Whether to search archived experiments.

        Returns:
            Tuple of (experiment_id_with_artifact, all_experiment_ids_with_artifact)
            - If found uniquely: ("abc12345", ["abc12345"])
            - If found in multiple: (None, ["abc12345", "def67890"])
            - If not found: (None, [])
        """
        # Get search order: current + transitive deps
        search_order = [experiment_id] + self.get_transitive_dependencies(
            experiment_id, include_archived=include_archived
        )

        # Search for artifact
        found_in = []
        for exp_id in search_order:
            if self.manager.storage.artifact_exists(
                exp_id, artifact_filename, include_archived=include_archived
            ):
                found_in.append(exp_id)

        if len(found_in) == 0:
            return (None, [])
        elif len(found_in) == 1:
            return (found_in[0], found_in)
        else:
            return (None, found_in)  # Ambiguous
