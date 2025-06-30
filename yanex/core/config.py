"""
Configuration management for experiments.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from ..utils.exceptions import ConfigError
from ..utils.validation import validate_config_data


def load_yaml_config(config_path: Path) -> dict[str, Any]:
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


def save_yaml_config(config_data: dict[str, Any], config_path: Path) -> None:
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


def parse_param_overrides(param_strings: list[str]) -> dict[str, Any]:
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

    Supports sweep syntax: range(), linspace(), logspace(), list()

    Args:
        value_str: String value to parse

    Returns:
        Parsed value with appropriate type (including SweepParameter instances)
    """
    value_str = value_str.strip()

    # Handle empty string
    if not value_str:
        return ""

    # Handle null/none
    if value_str.lower() in ("null", "none", "~"):
        return None

    # Check for sweep syntax first
    sweep_result = _parse_sweep_syntax(value_str)
    if sweep_result is not None:
        return sweep_result

    # Try to parse as number first (before booleans)
    try:
        # Try integer first
        if "." not in value_str and "e" not in value_str.lower():
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        pass

    # Handle boolean values (after numbers, so "1" and "0" are treated as numbers)
    if value_str.lower() in ("true", "yes", "on"):
        return True
    if value_str.lower() in ("false", "no", "off"):
        return False

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


def _parse_sweep_syntax(value_str: str) -> Optional[SweepParameter]:
    """
    Parse sweep syntax into SweepParameter objects.

    Supported syntax:
    - range(start, stop, step)
    - linspace(start, stop, count)
    - logspace(start, stop, count)
    - list(item1, item2, ...)

    Args:
        value_str: String value to parse

    Returns:
        SweepParameter instance or None if not sweep syntax
    """
    # Regular expressions for sweep function parsing
    range_pattern = r"range\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)"
    linspace_pattern = r"linspace\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)"
    logspace_pattern = r"logspace\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)"
    list_pattern = r"list\(\s*([^)]*)\s*\)"

    # Try range() syntax
    match = re.match(range_pattern, value_str)
    if match:
        try:
            start = _parse_numeric_value(match.group(1))
            stop = _parse_numeric_value(match.group(2))
            step = _parse_numeric_value(match.group(3))
            return RangeSweep(start, stop, step)
        except Exception as e:
            raise ConfigError(f"Invalid range() syntax: {value_str}. Error: {e}")

    # Try linspace() syntax
    match = re.match(linspace_pattern, value_str)
    if match:
        try:
            start = _parse_numeric_value(match.group(1))
            stop = _parse_numeric_value(match.group(2))
            count = int(_parse_numeric_value(match.group(3)))
            return LinspaceSweep(start, stop, count)
        except Exception as e:
            raise ConfigError(f"Invalid linspace() syntax: {value_str}. Error: {e}")

    # Try logspace() syntax
    match = re.match(logspace_pattern, value_str)
    if match:
        try:
            start = _parse_numeric_value(match.group(1))
            stop = _parse_numeric_value(match.group(2))
            count = int(_parse_numeric_value(match.group(3)))
            return LogspaceSweep(start, stop, count)
        except Exception as e:
            raise ConfigError(f"Invalid logspace() syntax: {value_str}. Error: {e}")

    # Try list() syntax
    match = re.match(list_pattern, value_str)
    if match:
        try:
            content = match.group(1).strip()
            if not content:
                raise ConfigError("List sweep cannot be empty")

            # Parse comma-separated items
            items = []
            for item_str in content.split(","):
                item_str = item_str.strip()
                if not item_str:
                    continue
                # Parse each item as a regular parameter value (recursive, but won't match sweep syntax)
                parsed_item = _parse_non_sweep_value(item_str)
                items.append(parsed_item)

            if not items:
                raise ConfigError("List sweep cannot be empty")

            return ListSweep(items)
        except Exception as e:
            raise ConfigError(f"Invalid list() syntax: {value_str}. Error: {e}")

    return None


def _parse_numeric_value(value_str: str) -> Union[int, float]:
    """Parse a string as a numeric value (int or float)."""
    value_str = value_str.strip()

    try:
        # Try integer first
        if "." not in value_str and "e" not in value_str.lower():
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        raise ConfigError(f"Expected numeric value, got: {value_str}")


def _parse_non_sweep_value(value_str: str) -> Any:
    """Parse parameter value without sweep syntax detection."""
    value_str = value_str.strip()

    # Handle quoted strings
    if (value_str.startswith('"') and value_str.endswith('"')) or (
        value_str.startswith("'") and value_str.endswith("'")
    ):
        return value_str[1:-1]

    # Handle null/none
    if value_str.lower() in ("null", "none", "~"):
        return None

    # Try to parse as number
    try:
        if "." not in value_str and "e" not in value_str.lower():
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        pass

    # Handle boolean values
    if value_str.lower() in ("true", "yes", "on"):
        return True
    if value_str.lower() in ("false", "no", "off"):
        return False

    # Return as string
    return value_str


def _set_nested_key(config_dict: dict[str, Any], key: str, value: Any) -> None:
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
    base_config: dict[str, Any], override_config: dict[str, Any]
) -> dict[str, Any]:
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

    def merge_recursive(base: dict[str, Any], override: dict[str, Any]) -> None:
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
) -> dict[str, Any]:
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


# Parameter Sweep Classes and Functions


class SweepParameter:
    """Base class for parameter sweep definitions."""

    def generate_values(self) -> list[Any]:
        """Generate list of values for this sweep parameter."""
        raise NotImplementedError


class RangeSweep(SweepParameter):
    """Range-based parameter sweep: range(start, stop, step)"""

    def __init__(
        self, start: Union[int, float], stop: Union[int, float], step: Union[int, float]
    ):
        self.start = start
        self.stop = stop
        self.step = step

        if step == 0:
            raise ConfigError("Range step cannot be zero")
        if (stop - start) * step < 0:
            raise ConfigError("Range step direction doesn't match start/stop values")

    def generate_values(self) -> list[Union[int, float]]:
        """Generate range values."""
        values = []
        current = self.start

        if self.step > 0:
            while current < self.stop:
                values.append(current)
                current += self.step
        else:
            while current > self.stop:
                values.append(current)
                current += self.step

        return values

    def __repr__(self) -> str:
        return f"RangeSweep({self.start}, {self.stop}, {self.step})"


class LinspaceSweep(SweepParameter):
    """Linear space parameter sweep: linspace(start, stop, count)"""

    def __init__(self, start: Union[int, float], stop: Union[int, float], count: int):
        self.start = start
        self.stop = stop
        self.count = count

        if count <= 0:
            raise ConfigError("Linspace count must be positive")

    def generate_values(self) -> list[float]:
        """Generate linearly spaced values."""
        if self.count == 1:
            return [float(self.start)]

        step = (self.stop - self.start) / (self.count - 1)
        return [self.start + i * step for i in range(self.count)]

    def __repr__(self) -> str:
        return f"LinspaceSweep({self.start}, {self.stop}, {self.count})"


class LogspaceSweep(SweepParameter):
    """Logarithmic space parameter sweep: logspace(start, stop, count)"""

    def __init__(self, start: Union[int, float], stop: Union[int, float], count: int):
        self.start = start
        self.stop = stop
        self.count = count

        if count <= 0:
            raise ConfigError("Logspace count must be positive")

    def generate_values(self) -> list[float]:
        """Generate logarithmically spaced values."""
        if self.count == 1:
            return [10.0**self.start]

        step = (self.stop - self.start) / (self.count - 1)
        return [10.0 ** (self.start + i * step) for i in range(self.count)]

    def __repr__(self) -> str:
        return f"LogspaceSweep({self.start}, {self.stop}, {self.count})"


class ListSweep(SweepParameter):
    """Explicit list parameter sweep: list(item1, item2, ...)"""

    def __init__(self, items: list[Any]):
        if not items:
            raise ConfigError("List sweep cannot be empty")
        self.items = items

    def generate_values(self) -> list[Any]:
        """Return the explicit list of values."""
        return self.items.copy()

    def __repr__(self) -> str:
        return f"ListSweep({self.items})"


def has_sweep_parameters(config: dict[str, Any]) -> bool:
    """
    Check if configuration contains any sweep parameters.

    Args:
        config: Configuration dictionary to check

    Returns:
        True if any values are SweepParameter instances
    """

    def check_dict(d: dict[str, Any]) -> bool:
        for value in d.values():
            if isinstance(value, SweepParameter):
                return True
            elif isinstance(value, dict):
                if check_dict(value):
                    return True
        return False

    return check_dict(config)


def expand_parameter_sweeps(config: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Expand parameter sweeps into individual configurations.

    Generates cross-product of all sweep parameters while keeping regular parameters.

    Args:
        config: Configuration dictionary potentially containing SweepParameter instances

    Returns:
        List of configuration dictionaries with sweep parameters expanded

    Example:
        Input: {"lr": RangeSweep(0.01, 0.03, 0.01), "batch_size": 32}
        Output: [
            {"lr": 0.01, "batch_size": 32},
            {"lr": 0.02, "batch_size": 32}
        ]
    """
    if not has_sweep_parameters(config):
        return [config]

    # Find all sweep parameters and their paths
    sweep_params = []

    def find_sweeps(d: dict[str, Any], path: str = "") -> None:
        for key, value in d.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, SweepParameter):
                sweep_params.append((current_path, value))
            elif isinstance(value, dict):
                find_sweeps(value, current_path)

    find_sweeps(config)

    if not sweep_params:
        return [config]

    # Generate all combinations using itertools.product
    import itertools

    sweep_paths, sweep_objects = zip(*sweep_params)
    sweep_value_lists = [sweep_obj.generate_values() for sweep_obj in sweep_objects]

    # Generate cross-product of all sweep parameter values
    expanded_configs = []
    for value_combination in itertools.product(*sweep_value_lists):
        # Create a deep copy of the original config
        expanded_config = copy.deepcopy(config)

        # Replace sweep parameters with concrete values
        for path, value in zip(sweep_paths, value_combination):
            _set_nested_key(expanded_config, path, value)

        expanded_configs.append(expanded_config)

    return expanded_configs
