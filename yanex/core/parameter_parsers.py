"""Parameter parsing strategies for Yanex configuration system."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from ..utils.exceptions import ConfigError
from .config import LinspaceSweep, ListSweep, LogspaceSweep, RangeSweep, SweepParameter


class ParameterParser(ABC):
    """Abstract base class for parameter value parsers."""

    @abstractmethod
    def can_parse(self, value_str: str) -> bool:
        """Check if this parser can handle the given value string.

        Args:
            value_str: The string value to check.

        Returns:
            True if this parser can handle the value, False otherwise.
        """

    @abstractmethod
    def parse(self, value_str: str) -> Any:
        """Parse the string value into appropriate Python type.

        Args:
            value_str: The string value to parse.

        Returns:
            Parsed value with appropriate type.

        Raises:
            ConfigError: If parsing fails.
        """


class NullParser(ParameterParser):
    """Parser for null/none values."""

    def can_parse(self, value_str: str) -> bool:
        """Check if value represents null/none."""
        return value_str.strip().lower() in ("null", "none", "~", "")

    def parse(self, value_str: str) -> str | None:
        """Parse null/none values."""
        stripped = value_str.strip()
        if not stripped:
            return ""
        return None


class SweepParameterParser(ParameterParser):
    """Parser for sweep syntax: range(), linspace(), logspace(), list()."""

    def can_parse(self, value_str: str) -> bool:
        """Check if value contains sweep syntax."""
        return bool(re.match(r"(range|linspace|logspace|list)\s*\(", value_str.strip()))

    def parse(self, value_str: str) -> SweepParameter:
        """Parse sweep syntax into SweepParameter objects."""
        value_str = value_str.strip()

        # Regular expressions for sweep function parsing
        range_pattern = r"range\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)"
        linspace_pattern = r"linspace\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)"
        logspace_pattern = r"logspace\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)"
        list_pattern = r"list\(\s*([^)]*)\s*\)"

        # Try range() syntax
        match = re.match(range_pattern, value_str)
        if match:
            try:
                start = self._parse_numeric_value(match.group(1))
                stop = self._parse_numeric_value(match.group(2))
                step = self._parse_numeric_value(match.group(3))
                return RangeSweep(start, stop, step)
            except Exception as e:
                raise ConfigError(f"Invalid range() syntax: {value_str}. Error: {e}")

        # Try linspace() syntax
        match = re.match(linspace_pattern, value_str)
        if match:
            try:
                start = self._parse_numeric_value(match.group(1))
                stop = self._parse_numeric_value(match.group(2))
                count = int(self._parse_numeric_value(match.group(3)))
                return LinspaceSweep(start, stop, count)
            except Exception as e:
                raise ConfigError(f"Invalid linspace() syntax: {value_str}. Error: {e}")

        # Try logspace() syntax
        match = re.match(logspace_pattern, value_str)
        if match:
            try:
                start = self._parse_numeric_value(match.group(1))
                stop = self._parse_numeric_value(match.group(2))
                count = int(self._parse_numeric_value(match.group(3)))
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
                    # Use BasicParameterParser for list items to avoid recursion
                    basic_parser = BasicParameterParser()
                    parsed_item = basic_parser.parse(item_str)
                    items.append(parsed_item)

                if not items:
                    raise ConfigError("List sweep cannot be empty")

                return ListSweep(items)
            except Exception as e:
                raise ConfigError(f"Invalid list() syntax: {value_str}. Error: {e}")

        raise ConfigError(f"Invalid sweep syntax: {value_str}")

    def _parse_numeric_value(self, value_str: str) -> int | float:
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


class NumericParser(ParameterParser):
    """Parser for numeric values (int/float)."""

    def can_parse(self, value_str: str) -> bool:
        """Check if value is numeric."""
        try:
            value_str = value_str.strip()
            if "." in value_str or "e" in value_str.lower():
                float(value_str)
            else:
                int(value_str)
            return True
        except ValueError:
            return False

    def parse(self, value_str: str) -> int | float:
        """Parse numeric values."""
        value_str = value_str.strip()
        try:
            # Try integer first
            if "." not in value_str and "e" not in value_str.lower():
                return int(value_str)
            else:
                return float(value_str)
        except ValueError:
            raise ConfigError(f"Invalid numeric value: {value_str}")


class BooleanParser(ParameterParser):
    """Parser for boolean values."""

    def can_parse(self, value_str: str) -> bool:
        """Check if value is boolean."""
        lower_val = value_str.strip().lower()
        return lower_val in ("true", "false", "yes", "no", "on", "off")

    def parse(self, value_str: str) -> bool:
        """Parse boolean values."""
        lower_val = value_str.strip().lower()
        if lower_val in ("true", "yes", "on"):
            return True
        elif lower_val in ("false", "no", "off"):
            return False
        else:
            raise ConfigError(f"Invalid boolean value: {value_str}")


class ListParser(ParameterParser):
    """Parser for list values [item1, item2, ...]."""

    def can_parse(self, value_str: str) -> bool:
        """Check if value is a list."""
        stripped = value_str.strip()
        return stripped.startswith("[") and stripped.endswith("]")

    def parse(self, value_str: str) -> list[Any]:
        """Parse list values."""
        value_str = value_str.strip()
        try:
            # Simple list parsing (comma-separated)
            content = value_str[1:-1].strip()
            if not content:
                return []

            # Use BasicParameterParser to parse each item (avoids circular import)
            basic_parser = BasicParameterParser()

            items = []
            for item in content.split(","):
                item_str = item.strip()
                if item_str:  # Skip empty items
                    parsed_item = basic_parser.parse(item_str)
                    items.append(parsed_item)

            return items
        except Exception as e:
            raise ConfigError(f"Invalid list syntax: {value_str}. Error: {e}")


class QuotedStringParser(ParameterParser):
    """Parser for quoted string values."""

    def can_parse(self, value_str: str) -> bool:
        """Check if value is a quoted string."""
        stripped = value_str.strip()
        return (stripped.startswith('"') and stripped.endswith('"')) or (
            stripped.startswith("'") and stripped.endswith("'")
        )

    def parse(self, value_str: str) -> str:
        """Parse quoted string values."""
        stripped = value_str.strip()
        return stripped[1:-1]  # Remove quotes


class StringParser(ParameterParser):
    """Parser for string values (fallback)."""

    def can_parse(self, value_str: str) -> bool:
        """String parser can handle any value."""
        return True

    def parse(self, value_str: str) -> str:
        """Return value as string."""
        return value_str


class BasicParameterParser:
    """Parser for basic (non-sweep) parameter values."""

    def __init__(self):
        """Initialize with parsers excluding sweep syntax."""
        self.parsers = [
            NullParser(),
            QuotedStringParser(),
            NumericParser(),
            BooleanParser(),
            StringParser(),  # Always last as fallback
        ]

    def parse(self, value_str: str) -> Any:
        """Parse a parameter value without sweep syntax detection."""
        for parser in self.parsers:
            if parser.can_parse(value_str):
                return parser.parse(value_str)

        # Should never reach here since StringParser accepts everything
        return value_str
