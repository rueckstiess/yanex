"""Parameter access tracking utilities for experiments.

This module provides functionality to extract accessed parameters from TrackedDict
instances and save them to experiment storage. It handles path reconstruction,
nested dict building, and integration with the storage layer.
"""

from typing import Any

from .tracked_dict import TrackedDict


def extract_accessed_params(
    tracked_dict: TrackedDict,
    original_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract only accessed parameters, using original config for values.

    This function takes the set of accessed paths (e.g., {"model.lr", "model.layers", "seed"})
    and reconstructs a nested dictionary containing only those values. It intelligently handles
    cases where both parent and child paths are accessed by only including leaf values.

    When original_config is provided, values are extracted from it instead of from the
    TrackedDict. This is important because users may mutate the TrackedDict (via pop, del,
    etc.) but we still want to save the original values that were accessed.

    Args:
        tracked_dict: TrackedDict instance with tracked accesses
        original_config: Original configuration dict to get values from.
                        If None, falls back to TrackedDict (legacy/test behavior).

    Returns:
        Dictionary containing only accessed parameters with original values

    Example:
        >>> config = {"model": {"lr": 0.01, "layers": 5, "dropout": 0.1}, "seed": 42}
        >>> tracked = TrackedDict(config)
        >>> _ = tracked["model"]["lr"]
        >>> _ = tracked["seed"]
        >>> extract_accessed_params(tracked, original_config=config)
        {'model': {'lr': 0.01}, 'seed': 42}
    """
    accessed_paths = tracked_dict.get_accessed_paths()

    if not accessed_paths:
        return {}

    # Use original config for values if provided, otherwise fall back to tracked_dict
    source_config = original_config if original_config is not None else tracked_dict

    # Deduplicate to get only leaf paths (deepest accessed values)
    # This removes parent paths when children are also accessed
    leaf_paths = deduplicate_paths(accessed_paths)

    # Build nested dict from leaf paths only
    result: dict[str, Any] = {}

    for path in leaf_paths:
        try:
            # Get the value from the SOURCE config (original or TrackedDict)
            value = _get_value_by_path(source_config, path)

            # Set the value in result dict, creating nested structure as needed
            _set_value_by_path(result, path, value)
        except KeyError:
            # Key was accessed but doesn't exist in source config - skip
            continue

    return result


def _get_value_by_path(data: dict, path: str, separator: str = ".") -> Any:
    """Get value from nested dict using dot-separated path.

    Args:
        data: Dictionary to traverse
        path: Dot-separated path (e.g., "model.train.lr")
        separator: Path separator (default: ".")

    Returns:
        Value at the specified path

    Raises:
        KeyError: If path doesn't exist

    Example:
        >>> data = {"model": {"train": {"lr": 0.01}}}
        >>> _get_value_by_path(data, "model.train.lr")
        0.01
    """
    keys = path.split(separator)
    current = data

    for key in keys:
        current = current[key]

    return current


def _set_value_by_path(data: dict, path: str, value: Any, separator: str = ".") -> None:
    """Set value in nested dict using dot-separated path, creating structure as needed.

    Args:
        data: Dictionary to modify (modified in-place)
        path: Dot-separated path (e.g., "model.train.lr")
        value: Value to set
        separator: Path separator (default: ".")

    Example:
        >>> data = {}
        >>> _set_value_by_path(data, "model.train.lr", 0.01)
        >>> data
        {'model': {'train': {'lr': 0.01}}}
    """
    keys = path.split(separator)
    current = data

    # Navigate to the parent of the target key, creating dicts as needed
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    # Set the final value
    # If value is a TrackedDict, convert to plain dict to avoid serialization issues
    if isinstance(value, TrackedDict):
        current[keys[-1]] = dict(value)
    else:
        current[keys[-1]] = value


def save_accessed_params(experiment_id: str, tracked_dict: TrackedDict) -> None:
    """Save only accessed parameters to experiment storage.

    This function:
    1. Gets the set of accessed paths from TrackedDict
    2. Loads the ORIGINAL config from params.yaml (saved at experiment creation)
    3. Filters original config to only accessed paths
    4. Saves the filtered config back to params.yaml

    This approach ensures that even if the TrackedDict was mutated (via pop, del,
    etc.), the original values are preserved in the final params.yaml.

    If no parameters were accessed, an empty dict is saved.

    Args:
        experiment_id: ID of the experiment
        tracked_dict: TrackedDict instance with tracked accesses

    Example:
        >>> import atexit
        >>> tracked_params = TrackedDict(config)
        >>> atexit.register(save_accessed_params, "abc12345", tracked_params)
    """
    # Import here to avoid circular dependency
    from ..api import _get_experiment_manager

    manager = _get_experiment_manager()

    # Load ORIGINAL config from storage (saved at experiment creation)
    original_config = manager.storage.load_config(experiment_id)

    # Extract only accessed params using ORIGINAL values (not TrackedDict values)
    accessed_params = extract_accessed_params(
        tracked_dict, original_config=original_config
    )

    # Save filtered config back to storage
    manager.storage.save_config(experiment_id, accessed_params)


def deduplicate_paths(paths: set[str], separator: str = ".") -> set[str]:
    """Remove redundant paths where parent paths include child paths.

    If both "model" and "model.lr" are accessed, we only need "model" since
    accessing a dict marks the entire dict. However, we want to preserve the
    granular paths to know which specific leaves were accessed.

    This function is primarily for optimization/cleanup but isn't strictly
    necessary for correctness.

    Args:
        paths: Set of dot-separated paths
        separator: Path separator (default: ".")

    Returns:
        Set of paths with redundant entries removed

    Example:
        >>> paths = {"model", "model.train", "model.train.lr", "seed"}
        >>> deduplicate_paths(paths)
        {'model.train.lr', 'seed'}

    Note:
        Currently not used in the main flow, but available for optimization.
    """
    if not paths:
        return set()

    # Sort paths by depth (deeper first)
    sorted_paths = sorted(paths, key=lambda p: p.count(separator), reverse=True)

    result = set()
    for path in sorted_paths:
        # Check if any existing path in result is a child of this path
        is_parent_of_existing = any(
            existing.startswith(path + separator) for existing in result
        )

        if not is_parent_of_existing:
            # Remove any existing paths that are parents of this path
            result = {p for p in result if not path.startswith(p + separator)}
            result.add(path)

    return result
