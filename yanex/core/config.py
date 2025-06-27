"""
Configuration management for experiments.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..utils.exceptions import ConfigError
from ..utils.validation import validate_config_data


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        ConfigError: If config file cannot be loaded or parsed
    """
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    if not config_path.is_file():
        raise ConfigError(f"Configuration path is not a file: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config: {e}") from e
    except Exception as e:
        raise ConfigError(f"Failed to read config file: {e}") from e

    if not isinstance(config_data, dict):
        raise ConfigError(
            f"Configuration must be a dictionary, got {type(config_data)}"
        )

    return validate_config_data(config_data)


def save_yaml_config(config_data: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config_data: Configuration dictionary to save
        config_path: Path where to save the configuration

    Raises:
        ConfigError: If config cannot be saved
    """
    validate_config_data(config_data)

    try:
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                config_data, f, default_flow_style=False, sort_keys=True, indent=2
            )
    except Exception as e:
        raise ConfigError(f"Failed to save config file: {e}") from e


def parse_param_overrides(param_strings: list[str]) -> Dict[str, Any]:
    """
    Parse parameter override strings from CLI.

    Args:
        param_strings: List of "key=value" strings

    Returns:
        Dictionary of parsed parameters

    Raises:
        ConfigError: If parameter format is invalid
    """
    overrides = {}

    for param_string in param_strings:
        if "=" not in param_string:
            raise ConfigError(
                f"Invalid parameter format: {param_string}. Expected 'key=value'"
            )

        key, value_str = param_string.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        if not key:
            raise ConfigError(f"Empty parameter key in: {param_string}")

        # Try to parse value as different types
        parsed_value = _parse_parameter_value(value_str)

        # Support nested keys like "model.learning_rate=0.01"
        _set_nested_key(overrides, key, parsed_value)

    return overrides


def _parse_parameter_value(value_str: str) -> Any:
    """
    Parse parameter value string to appropriate Python type.

    Args:
        value_str: String value to parse

    Returns:
        Parsed value with appropriate type
    """
    value_str = value_str.strip()

    # Handle empty string
    if not value_str:
        return ""

    # Handle boolean values
    if value_str.lower() in ("true", "yes", "1", "on"):
        return True
    if value_str.lower() in ("false", "no", "0", "off"):
        return False

    # Handle null/none
    if value_str.lower() in ("null", "none", "~"):
        return None

    # Try to parse as number
    try:
        # Try integer first
        if "." not in value_str and "e" not in value_str.lower():
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        pass

    # Try to parse as JSON-like structures
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            # Simple list parsing (comma-separated)
            content = value_str[1:-1].strip()
            if not content:
                return []
            items = [
                _parse_parameter_value(item.strip()) for item in content.split(",")
            ]
            return items
        except Exception:
            pass

    # Return as string
    return value_str


def _set_nested_key(config_dict: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set nested key in configuration dictionary.

    Args:
        config_dict: Configuration dictionary to modify
        key: Potentially nested key (e.g., "model.learning_rate")
        value: Value to set
    """
    keys = key.split(".")
    current = config_dict

    # Navigate to the nested location
    for key_part in keys[:-1]:
        if key_part not in current:
            current[key_part] = {}
        elif not isinstance(current[key_part], dict):
            # If intermediate key exists but is not a dict, override it
            current[key_part] = {}
        current = current[key_part]

    # Set the final value
    current[keys[-1]] = value


def merge_configs(
    base_config: Dict[str, Any], override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.

    Args:
        base_config: Base configuration dictionary
        override_config: Override configuration dictionary

    Returns:
        Merged configuration dictionary

    Note:
        Override config takes precedence. Nested dictionaries are merged recursively.
    """
    result = copy.deepcopy(base_config)

    def merge_recursive(base: Dict[str, Any], override: Dict[str, Any]) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_recursive(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    merge_recursive(result, override_config)
    return result


def resolve_config(
    config_path: Optional[Path] = None,
    param_overrides: Optional[list[str]] = None,
    default_config_name: str = "config.yaml",
) -> Dict[str, Any]:
    """
    Resolve final configuration from file and parameter overrides.

    Args:
        config_path: Path to configuration file
        param_overrides: List of parameter override strings
        default_config_name: Default config filename to look for

    Returns:
        Resolved configuration dictionary

    Raises:
        ConfigError: If configuration cannot be resolved
    """
    # Start with empty config
    config = {}

    # Load from file if specified or if default exists
    if config_path is None:
        default_path = Path.cwd() / default_config_name
        if default_path.exists():
            config_path = default_path

    if config_path is not None:
        config = load_yaml_config(config_path)

    # Apply parameter overrides
    if param_overrides:
        override_config = parse_param_overrides(param_overrides)
        config = merge_configs(config, override_config)

    return config
