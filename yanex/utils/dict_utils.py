"""Utility functions for dictionary manipulation."""

from typing import Any


def flatten_dict(
    nested_dict: dict[str, Any], parent_key: str = "", separator: str = "."
) -> dict[str, Any]:
    """
    Flatten a nested dictionary using dot notation.

    Args:
        nested_dict: The nested dictionary to flatten
        parent_key: The base key for recursion (used internally)
        separator: The separator to use between keys (default: ".")

    Returns:
        A flattened dictionary with dot-notation keys

    Examples:
        >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        {"a.b": 1, "a.c": 2, "d": 3}

        >>> flatten_dict({"model": {"lr": 0.001, "dropout": 0.1}, "epochs": 10})
        {"model.lr": 0.001, "model.dropout": 0.1, "epochs": 10}
    """
    items: list[tuple[str, Any]] = []

    for key, value in nested_dict.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict) and value:
            # Recursively flatten nested dictionaries
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            # Add leaf values (including empty dicts, lists, and primitives)
            items.append((new_key, value))

    return dict(items)


def unflatten_dict(flat_dict: dict[str, Any], separator: str = ".") -> dict[str, Any]:
    """
    Unflatten a dictionary with dot notation keys back to nested structure.

    Args:
        flat_dict: The flattened dictionary with dot-notation keys
        separator: The separator used between keys (default: ".")

    Returns:
        A nested dictionary

    Examples:
        >>> unflatten_dict({"a.b": 1, "a.c": 2, "d": 3})
        {"a": {"b": 1, "c": 2}, "d": 3}

        >>> unflatten_dict({"model.lr": 0.001, "model.dropout": 0.1, "epochs": 10})
        {"model": {"lr": 0.001, "dropout": 0.1}, "epochs": 10}
    """
    result: dict[str, Any] = {}

    for key, value in flat_dict.items():
        parts = key.split(separator)
        current = result

        # Navigate/create nested structure
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = value

    return result


def get_nested_value(
    data: dict[str, Any], path: str, default: Any = None, separator: str = "."
) -> Any:
    """
    Get value from nested dictionary using dot-separated path.

    This function traverses a nested dictionary structure using a dot-notation
    path string and returns the value at that location. If the path doesn't exist,
    returns the default value.

    Args:
        data: The nested dictionary to traverse
        path: Dot-separated path to the value (e.g., "model.train.learning_rate")
        default: Value to return if path doesn't exist (default: None)
        separator: Path separator (default: ".")

    Returns:
        Value at the specified path, or default if path doesn't exist

    Examples:
        >>> data = {"model": {"train": {"lr": 0.01, "epochs": 20}}, "seed": 42}
        >>> get_nested_value(data, "model.train.lr")
        0.01

        >>> get_nested_value(data, "model.train")
        {"lr": 0.01, "epochs": 20}

        >>> get_nested_value(data, "nonexistent", default="missing")
        'missing'

        >>> get_nested_value(data, "seed")
        42
    """
    keys = path.split(separator)
    current = data

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries, with override values taking precedence.

    Recursively merges nested dictionaries. For non-dict values, the override
    value replaces the base value. Neither input dictionary is modified.

    Args:
        base: Base dictionary
        override: Dictionary with values that take precedence

    Returns:
        New merged dictionary

    Examples:
        >>> base = {"a": {"b": 1, "c": 2}, "d": 3}
        >>> override = {"a": {"b": 10}, "e": 5}
        >>> deep_merge(base, override)
        {'a': {'b': 10, 'c': 2}, 'd': 3, 'e': 5}

        >>> base = {"model": {"lr": 0.001, "layers": 3}}
        >>> override = {"model": {"lr": 0.01}}
        >>> deep_merge(base, override)
        {'model': {'lr': 0.01, 'layers': 3}}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result
