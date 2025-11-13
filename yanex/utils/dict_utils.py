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
