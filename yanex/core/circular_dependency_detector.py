"""Circular dependency detection for experiments."""

from ..utils.exceptions import CircularDependencyError


class CircularDependencyDetector:
    """Detect circular dependencies using depth-first search."""

    def __init__(self, storage):
        """Initialize detector with storage.

        Args:
            storage: Storage interface for loading dependencies
        """
        self.storage = storage

    def check_for_cycles(
        self, new_experiment_id: str, resolved_deps: dict[str, str]
    ) -> None:
        """Check if adding dependencies would create a cycle.

        Args:
            new_experiment_id: ID of experiment being created
            resolved_deps: Dependencies to add (slot -> experiment_id)

        Raises:
            CircularDependencyError: If a cycle is detected

        Algorithm:
            For each dependency D in resolved_deps:
                Run DFS from D
                If we reach new_experiment_id, cycle detected

        Example cycle:
            Creating exp3 with dependency on exp1
            But exp1 depends on exp2
            And exp2 depends on exp3 (not created yet)

            DFS from exp1:
                exp1 -> exp2 -> (would depend on exp3) -> exp1
                Cycle detected!
        """
        for _slot_name, dep_id in resolved_deps.items():
            # Start DFS from each dependency
            visited: set[str] = set()
            path: list[str] = []

            if self._has_path_to(dep_id, new_experiment_id, visited, path):
                # Cycle detected: dep_id has a path back to new_experiment_id
                cycle_list = path + [new_experiment_id, path[0]]

                raise CircularDependencyError(cycle_list)

    def _has_path_to(
        self, current_id: str, target_id: str, visited: set[str], path: list[str]
    ) -> bool:
        """DFS to check if there's a path from current_id to target_id.

        Args:
            current_id: Current node in DFS
            target_id: Target we're searching for
            visited: Set of visited nodes (prevents infinite loops)
            path: Current path (for error messages)

        Returns:
            True if path exists, False otherwise
        """
        # Base case: reached target
        if current_id == target_id:
            return True

        # Already visited this node
        if current_id in visited:
            return False

        visited.add(current_id)
        path.append(current_id)

        # Load dependencies of current node
        try:
            deps_data = self.storage.load_dependencies(
                current_id, include_archived=True
            )
        except Exception:
            # Experiment doesn't exist or has no dependencies
            path.pop()
            return False

        if not deps_data:
            path.pop()
            return False

        # Check each dependency recursively
        resolved_dependencies = deps_data.get("resolved_dependencies", {})
        for _dep_slot, dep_exp_id in resolved_dependencies.items():
            if self._has_path_to(dep_exp_id, target_id, visited, path):
                return True

        # No path found through this node
        path.pop()
        return False
