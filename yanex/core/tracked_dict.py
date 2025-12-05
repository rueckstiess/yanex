"""Dictionary wrapper that tracks which keys are accessed during script execution.

This module provides the TrackedDict class, which is used to monitor parameter
access patterns in experiment scripts. By wrapping configuration dictionaries,
we can determine which parameters were actually used vs simply present in the
config file.

Additionally, TrackedDict performs lazy conflict detection when dependencies
are provided, ensuring that parameter values are consistent across the
experiment dependency chain.
"""

import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..results.experiment import Experiment


class _MissingSentinel:
    """Sentinel object for detecting missing parameters."""

    pass


_MISSING = _MissingSentinel()


class TrackedDict(dict):
    """Dictionary wrapper that tracks accesses and detects dependency conflicts.

    This class extends dict to monitor access patterns while maintaining complete
    compatibility with standard dict operations. It tracks the full path to nested
    values (e.g., "model.train.learning_rate") and provides thread-safe access
    tracking.

    When dependencies are provided, it also performs lazy conflict detection on
    leaf value access, raising ParameterConflictError if the same parameter has
    different values across dependencies.

    Attributes:
        _accessed_paths: Set of full path strings that have been accessed
        _path: Current path prefix for nested tracking
        _lock: Thread lock for safe concurrent access
        _dependencies: Dict mapping slot names to Experiment objects (root only)
        _root: Reference to root TrackedDict for sharing state

    Example:
        >>> config = {"model": {"lr": 0.01, "layers": 5}, "seed": 42}
        >>> tracked = TrackedDict(config)
        >>> lr = tracked["model"]["lr"]
        >>> tracked.get_accessed_paths()
        {'model', 'model.lr'}
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        path: str = "",
        *,
        dependencies: dict[str, "Experiment"] | None = None,
        _root: "TrackedDict | None" = None,
    ) -> None:
        """Initialize TrackedDict with optional data and path.

        Args:
            data: Dictionary to wrap and track. Defaults to empty dict.
            path: Path prefix for nested tracking (used internally for recursion)
            dependencies: Dict mapping slot names to Experiment objects.
                         Only set on root TrackedDict, nested dicts share via _root.
            _root: Reference to root TrackedDict (used internally for nested dicts)
        """
        super().__init__(data or {})
        self._path = path

        # Root TrackedDict owns the shared state
        if _root is None:
            self._accessed_paths: set[str] = set()
            self._lock = threading.Lock()
            self._dependencies: dict[str, Experiment] = dependencies or {}
            self._root: TrackedDict = self
        else:
            # Nested TrackedDicts share state via root reference
            self._root = _root
            # Access shared state through _root
            self._accessed_paths = _root._accessed_paths
            self._lock = _root._lock
            self._dependencies = _root._dependencies

    def __getitem__(self, key: str) -> Any:
        """Get item by key, mark as accessed, and check for conflicts.

        Args:
            key: Dictionary key to access

        Returns:
            Value associated with key

        Raises:
            KeyError: If key doesn't exist
            ParameterConflictError: If value conflicts with dependency values
        """
        # Build full path for this access
        full_path = f"{self._path}.{key}" if self._path else key

        # Thread-safe tracking
        with self._lock:
            self._accessed_paths.add(full_path)

        # Get the value
        value = super().__getitem__(key)

        # Wrap nested dicts recursively for continued tracking
        if isinstance(value, dict) and not isinstance(value, TrackedDict):
            # Create tracked wrapper with accumulated path, sharing root
            tracked_value = TrackedDict(value, path=full_path, _root=self._root)
            # Cache the wrapped value
            super().__setitem__(key, tracked_value)
            return tracked_value

        # For leaf values (non-dict), check for conflicts with dependencies
        if not isinstance(value, dict) and self._root._dependencies:
            self._check_for_conflicts(full_path, value)

        return value

    def _check_for_conflicts(self, full_path: str, local_value: Any) -> None:
        """Check if parameter conflicts with dependency values.

        Args:
            full_path: Full dotted path to the parameter (e.g., "model.lr")
            local_value: Value from local config

        Raises:
            ParameterConflictError: If conflicting values are found
        """
        from ..utils.exceptions import ParameterConflictError

        # Collect all values: local + dependencies
        # Map: hashable_value -> list of (source_name, experiment_id)
        value_sources: dict[Any, list[tuple[str, str | None]]] = {}

        # Add local value (always present since we're checking a leaf)
        hashable_local = self._make_hashable(local_value)
        value_sources[hashable_local] = [("config", None)]

        # Check each dependency
        for slot_name, dep_experiment in self._root._dependencies.items():
            try:
                dep_value = dep_experiment.get_param(full_path, default=_MISSING)
                if dep_value is not _MISSING:
                    hashable_dep = self._make_hashable(dep_value)
                    if hashable_dep not in value_sources:
                        value_sources[hashable_dep] = []
                    value_sources[hashable_dep].append((slot_name, dep_experiment.id))
            except Exception:
                # If dependency param loading fails, skip it
                continue

        # If more than one unique value exists, it's a conflict
        if len(value_sources) > 1:
            # Build conflicts dict for error message
            # Use repr() for unhashable types (list, dict, set) as dict keys
            conflicts: dict[Any, list[tuple[str, str | None]]] = {}

            for _, sources in value_sources.items():
                # Recover original value for display
                if sources[0][1] is None:
                    # This is the local config value
                    original_val = local_value
                else:
                    # Get from first dependency that has this value
                    slot_name = sources[0][0]
                    dep = self._root._dependencies[slot_name]
                    original_val = dep.get_param(full_path)

                # Use repr() for unhashable types to allow use as dict key
                display_key = (
                    repr(original_val)
                    if isinstance(original_val, (list, dict, set))
                    else original_val
                )
                conflicts[display_key] = sources

            raise ParameterConflictError(full_path, conflicts)

    @staticmethod
    def _make_hashable(value: Any) -> Any:
        """Convert value to hashable type for comparison.

        Args:
            value: Any value to make hashable

        Returns:
            Hashable representation of the value
        """
        if isinstance(value, dict):
            return tuple(
                sorted((k, TrackedDict._make_hashable(v)) for k, v in value.items())
            )
        elif isinstance(value, list):
            return tuple(TrackedDict._make_hashable(item) for item in value)
        elif isinstance(value, set):
            return frozenset(TrackedDict._make_hashable(item) for item in value)
        else:
            return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get item by key with optional default, mark as accessed if key exists.

        Args:
            key: Dictionary key to access
            default: Value to return if key doesn't exist

        Returns:
            Value associated with key, or default if key doesn't exist
        """
        if key in self:
            return self[key]  # This will trigger tracking and conflict check
        return default

    def __iter__(self):
        """Return iterator over keys and mark all as accessed.

        Returns:
            Iterator over dictionary keys
        """
        # Mark all keys as accessed (same as keys())
        self.keys()
        return super().__iter__()

    def keys(self):
        """Return dict keys and mark all as accessed.

        Rationale: Iterating keys typically means all values will be used,
        and we can't track individual accesses within the iteration.

        Returns:
            Dictionary keys view
        """
        # Mark all top-level keys as accessed
        with self._lock:
            for key in super().keys():
                full_path = f"{self._path}.{key}" if self._path else key
                self._accessed_paths.add(full_path)

        return super().keys()

    def values(self):
        """Return dict values and mark all keys as accessed.

        Returns:
            Dictionary values view
        """
        # Mark all keys as accessed (same as keys())
        self.keys()
        return super().values()

    def items(self):
        """Return dict items and mark all keys as accessed.

        Returns:
            Dictionary items view
        """
        # Mark all keys as accessed (same as keys())
        self.keys()
        return super().items()

    def __contains__(self, key: str) -> bool:
        """Check if key exists WITHOUT marking as accessed.

        Rationale: Existence checks are exploration, not usage.

        Args:
            key: Key to check

        Returns:
            True if key exists, False otherwise
        """
        return super().__contains__(key)

    def __len__(self) -> int:
        """Return number of items WITHOUT marking as accessed.

        Returns:
            Number of items in dictionary
        """
        return super().__len__()

    def get_accessed_paths(self) -> set[str]:
        """Get all accessed paths.

        Returns:
            Set of accessed path strings (e.g., {"model", "model.lr"})
        """
        with self._lock:
            return self._accessed_paths.copy()

    def clear_accessed_paths(self) -> None:
        """Clear all tracked accesses (useful for testing)."""
        with self._lock:
            self._accessed_paths.clear()
