"""Experiment data serialization helpers.

This module provides utilities for converting experiment metadata to
serializable formats suitable for JSON, CSV, and markdown output.
"""

from datetime import datetime
from pathlib import Path
from typing import Any


def serialize_value(value: Any) -> Any:
    """Serialize a value for JSON/CSV output.

    Handles datetime, Path, and nested structures recursively.

    Args:
        value: Any value to serialize

    Returns:
        JSON-serializable value
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def experiment_to_dict(
    experiment: dict[str, Any],
    include_fields: list[str] | None = None,
    exclude_fields: list[str] | None = None,
    flatten: bool = False,
) -> dict[str, Any]:
    """Convert experiment metadata to a serializable dictionary.

    Args:
        experiment: Raw experiment metadata dict
        include_fields: If provided, only include these fields
        exclude_fields: If provided, exclude these fields
        flatten: If True, flatten nested dicts (e.g., git.commit -> git_commit)

    Returns:
        Serializable dict suitable for JSON/CSV/markdown output
    """
    # Define the standard fields we want to include by default
    standard_fields = [
        "id",
        "name",
        "status",
        "script_path",
        "tags",
        "description",
        "created_at",
        "started_at",
        "completed_at",
        "failed_at",
        "cancelled_at",
        "duration",
        "error_message",
    ]

    # Determine which fields to include
    if include_fields is not None:
        fields_to_include = include_fields
    else:
        fields_to_include = standard_fields
        # Also include any additional top-level fields not in the exclude list
        for key in experiment.keys():
            if key not in fields_to_include and key not in (exclude_fields or []):
                fields_to_include.append(key)

    # Build the result dict
    result: dict[str, Any] = {}

    for field in fields_to_include:
        if exclude_fields and field in exclude_fields:
            continue

        value = experiment.get(field)
        if value is not None or field in ["name", "description", "tags"]:
            if flatten and isinstance(value, dict):
                # Flatten nested dict with prefix
                for nested_key, nested_value in value.items():
                    flat_key = f"{field}_{nested_key}"
                    result[flat_key] = serialize_value(nested_value)
            else:
                result[field] = serialize_value(value)

    return result


def experiment_to_flat_dict(
    experiment: dict[str, Any],
    include_nested: list[str] | None = None,
) -> dict[str, Any]:
    """Convert experiment to a completely flat dict for CSV output.

    Flattens nested structures like 'git' and 'environment' into
    prefixed keys (e.g., git_commit, git_branch).

    Args:
        experiment: Raw experiment metadata dict
        include_nested: List of nested fields to include (e.g., ['git', 'environment'])

    Returns:
        Flat dict with no nested structures
    """
    if include_nested is None:
        include_nested = []

    result: dict[str, Any] = {}

    # Core fields
    core_fields = [
        "id",
        "name",
        "status",
        "script_path",
        "tags",
        "description",
        "created_at",
        "started_at",
        "completed_at",
        "duration",
        "error_message",
    ]

    for field in core_fields:
        value = experiment.get(field)
        if isinstance(value, list):
            # Convert list to comma-separated string for CSV
            result[field] = ",".join(str(v) for v in value) if value else ""
        else:
            result[field] = serialize_value(value)

    # Handle nested fields if requested
    for nested_field in include_nested:
        nested_data = experiment.get(nested_field, {})
        if isinstance(nested_data, dict):
            for key, value in nested_data.items():
                flat_key = f"{nested_field}_{key}"
                result[flat_key] = serialize_value(value)

    return result


def experiments_to_list(
    experiments: list[dict[str, Any]],
    include_fields: list[str] | None = None,
    exclude_fields: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert list of experiments to serializable list of dicts.

    Args:
        experiments: List of experiment metadata dicts
        include_fields: If provided, only include these fields
        exclude_fields: If provided, exclude these fields

    Returns:
        List of serializable dicts
    """
    return [
        experiment_to_dict(exp, include_fields, exclude_fields) for exp in experiments
    ]


def format_tags_for_output(tags: list[str] | None) -> str:
    """Format tags list for display/output.

    Args:
        tags: List of tag strings

    Returns:
        Comma-separated string of tags, or empty string if no tags
    """
    if not tags:
        return ""
    return ", ".join(tags)


def format_duration_for_output(duration: float | None) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        duration: Duration in seconds

    Returns:
        Formatted string like "2m 34s" or "-" if no duration
    """
    if duration is None:
        return "-"

    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"
