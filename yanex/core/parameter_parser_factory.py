"""Factory for parameter parsers in Yanex configuration system."""

from typing import Any

from .parameter_parsers import (
    BooleanParser,
    ListParser,
    NullParser,
    NumericParser,
    ParameterParser,
    QuotedStringParser,
    StringParser,
    SweepParameterParser,
)


class ParameterParserFactory:
    """Factory for creating and coordinating parameter parsers."""

    def __init__(self):
        """Initialize with all available parsers in priority order."""
        self.parsers: list[ParameterParser] = [
            NullParser(),  # Handle empty/null first
            SweepParameterParser(),  # Handle sweep syntax before other parsing
            NumericParser(),  # Parse numbers before booleans (so "1"/"0" are numbers)
            BooleanParser(),  # Parse boolean values
            ListParser(),  # Parse list syntax
            QuotedStringParser(),  # Parse quoted strings
            StringParser(),  # Fallback to string (always last)
        ]

    def parse_value(self, value_str: str) -> Any:
        """Parse a parameter value string using the appropriate parser.

        Args:
            value_str: String value to parse.

        Returns:
            Parsed value with appropriate type.

        Raises:
            ConfigError: If parsing fails.
        """
        for parser in self.parsers:
            if parser.can_parse(value_str):
                return parser.parse(value_str)

        # Should never reach here since StringParser accepts everything
        return value_str

    def get_parser_for_value(self, value_str: str) -> ParameterParser:
        """Get the appropriate parser for a given value string.

        Args:
            value_str: String value to find parser for.

        Returns:
            The first parser that can handle the value.
        """
        for parser in self.parsers:
            if parser.can_parse(value_str):
                return parser

        # Should never reach here since StringParser accepts everything
        return self.parsers[-1]  # Return StringParser as fallback
