"""
Input validation utilities.
"""

import re
from pathlib import Path
from typing import Any

from .exceptions import ValidationError


def validate_experiment_name(name: str) -> str:
    """
    Validate experiment name.

    Args:
        name: Experiment name to validate

    Returns:
        Validated name

    Raises:
        ValidationError: If name is invalid
    """
    if not name or not name.strip():
        raise ValidationError("Experiment name cannot be empty")

    name = name.strip()

    # Check for invalid characters (basic validation)
    if not re.match(r"^[a-zA-Z0-9_\-\s]+$", name):
        raise ValidationError(
            "Experiment name can only contain letters, numbers, spaces, hyphens, and underscores"
        )

    return name


def validate_experiment_id(experiment_id: str) -> str:
    """
    Validate experiment ID format.

    Args:
        experiment_id: ID to validate

    Returns:
        Validated ID

    Raises:
        ValidationError: If ID format is invalid
    """
    if not re.match(r"^[a-f0-9]{8}$", experiment_id):
        raise ValidationError(
            f"Invalid experiment ID format: {experiment_id}. "
            "Expected 8-character hex string (e.g., 'a1b2c3d4')"
        )

    return experiment_id


def validate_tags(tags: list[str]) -> list[str]:
    """
    Validate experiment tags.

    Args:
        tags: List of tags to validate

    Returns:
        Validated and cleaned tags

    Raises:
        ValidationError: If any tag is invalid
    """
    if not isinstance(tags, list):
        raise ValidationError("Tags must be a list")

    validated_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValidationError(f"Tag must be a string: {tag}")

        tag = tag.strip()
        if not tag:
            continue  # Skip empty tags

        if len(tag) > 50:
            raise ValidationError(f"Tag too long (max 50 chars): {tag}")

        if not re.match(r"^[a-zA-Z0-9_\-]+$", tag):
            raise ValidationError(
                f"Tag contains invalid characters: {tag}. "
                "Only letters, numbers, hyphens, and underscores allowed"
            )

        validated_tags.append(tag)

    return validated_tags


def validate_script_path(script_path: Path) -> Path:
    """
    Validate experiment script path.

    Args:
        script_path: Path to script file

    Returns:
        Validated path

    Raises:
        ValidationError: If script path is invalid
    """
    if not script_path.exists():
        raise ValidationError(f"Script file does not exist: {script_path}")

    if not script_path.is_file():
        raise ValidationError(f"Script path is not a file: {script_path}")

    if script_path.suffix != ".py":
        raise ValidationError(f"Script must be a Python file (.py): {script_path}")

    return script_path


def validate_config_data(config_data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate configuration data structure.

    Args:
        config_data: Configuration dictionary

    Returns:
        Validated configuration

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config_data, dict):
        raise ValidationError("Configuration must be a dictionary")

    # Import SweepParameter to check for sweep instances
    # Import here to avoid circular dependency
    try:
        from ..core.config import SweepParameter
    except ImportError:
        SweepParameter = None  # type: ignore

    # Basic validation - ensure values are JSON serializable types or sweep parameters
    allowed_types = (str, int, float, bool, list, dict, type(None))

    def validate_value(key: str, value: Any) -> None:
        # Allow SweepParameter instances (for parameter sweeps)
        if SweepParameter is not None and isinstance(value, SweepParameter):
            return

        if not isinstance(value, allowed_types):
            raise ValidationError(
                f"Configuration value for '{key}' must be JSON serializable, "
                f"got {type(value).__name__}"
            )

        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                validate_value(f"{key}.{subkey}", subvalue)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                validate_value(f"{key}[{i}]", item)

    for key, value in config_data.items():
        validate_value(key, value)

    return config_data
