"""Dictionary wrapper that tracks which keys are accessed during script execution.

This module provides the TrackedDict class, which is used to monitor parameter
access patterns in experiment scripts. By wrapping configuration dictionaries,
we can determine which parameters were actually used vs simply present in the
config file.
"""

import threading
from typing import Any


class TrackedDict(dict):
    """Dictionary wrapper that tracks which keys are accessed during script execution.

    This class extends dict to monitor access patterns while maintaining complete
    compatibility with standard dict operations. It tracks the full path to nested
    values (e.g., "model.train.learning_rate") and provides thread-safe access
    tracking.

    Attributes:
        _accessed_paths: Set of full path strings that have been accessed
        _path: Current path prefix for nested tracking
        _lock: Thread lock for safe concurrent access

    Example:
        >>> config = {"model": {"lr": 0.01, "layers": 5}, "seed": 42}
        >>> tracked = TrackedDict(config)
        >>> lr = tracked["model"]["lr"]
        >>> tracked.get_accessed_paths()
        {'model', 'model.lr'}
    """

    def __init__(self, data: dict[str, Any] | None = None, path: str = "") -> None:
        """Initialize TrackedDict with optional data and path.

        Args:
            data: Dictionary to wrap and track. Defaults to empty dict.
            path: Path prefix for nested tracking (used internally for recursion)
        """
        super().__init__(data or {})
        self._accessed_paths: set[str] = set()
        self._path = path
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> Any:
        """Get item by key and mark as accessed.

        Args:
            key: Dictionary key to access

        Returns:
            Value associated with key

        Raises:
            KeyError: If key doesn't exist
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
            # Create tracked wrapper with accumulated path
            tracked_value = TrackedDict(value, path=full_path)
            # Share the accessed paths set for unified tracking
            tracked_value._accessed_paths = self._accessed_paths
            tracked_value._lock = self._lock
            # Cache the wrapped value
            super().__setitem__(key, tracked_value)
            return tracked_value

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
            return self[key]  # This will trigger tracking
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
