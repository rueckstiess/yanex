"""
Console formatting components for yanex CLI commands.

This module provides utilities for consistent output formatting across
multiple output modes (console, JSON, CSV, markdown).
"""

from .console import ExperimentTableFormatter
from .csv_formatter import (
    format_action_result_csv,
    format_csv,
    output_action_result_csv,
    output_csv,
)
from .json_formatter import (
    YanexJSONEncoder,
    format_action_result,
    format_json,
    output_json,
)
from .markdown_formatter import (
    format_action_result_markdown,
    format_markdown_table,
    output_action_result_markdown,
    output_markdown_table,
)
from .output import (
    OutputMode,
    echo_error,
    echo_info,
    get_output_mode,
    is_machine_output,
    output_mode_options,
    validate_output_mode_flags,
)
from .serializers import (
    experiment_to_dict,
    experiment_to_flat_dict,
    experiments_to_list,
    format_duration_for_output,
    format_tags_for_output,
    serialize_value,
)
from .theme import (
    DATA_TABLE_BOX,
    FAILURE_SYMBOL,
    ID_STYLE,
    METRICS_STYLE,
    NAME_STYLE,
    PANEL_BOX,
    PARAMS_STYLE,
    SCRIPT_STYLE,
    STATUS_COLORS,
    STATUS_SYMBOLS,
    SUCCESS_SYMBOL,
    TABLE_HEADER_STYLE,
    TAGS_STYLE,
    TIMESTAMP_STYLE,
    WARNING_SYMBOL,
)

__all__ = [
    # Console formatter
    "ExperimentTableFormatter",
    # Output modes
    "OutputMode",
    "output_mode_options",
    "get_output_mode",
    "is_machine_output",
    "echo_info",
    "echo_error",
    "validate_output_mode_flags",
    # JSON
    "YanexJSONEncoder",
    "output_json",
    "format_json",
    "format_action_result",
    # CSV
    "output_csv",
    "format_csv",
    "format_action_result_csv",
    "output_action_result_csv",
    # Markdown
    "format_markdown_table",
    "output_markdown_table",
    "format_action_result_markdown",
    "output_action_result_markdown",
    # Serializers
    "serialize_value",
    "experiment_to_dict",
    "experiment_to_flat_dict",
    "experiments_to_list",
    "format_tags_for_output",
    "format_duration_for_output",
    # Theme
    "ID_STYLE",
    "SCRIPT_STYLE",
    "NAME_STYLE",
    "TAGS_STYLE",
    "PARAMS_STYLE",
    "METRICS_STYLE",
    "TIMESTAMP_STYLE",
    "STATUS_COLORS",
    "STATUS_SYMBOLS",
    "DATA_TABLE_BOX",
    "PANEL_BOX",
    "TABLE_HEADER_STYLE",
    "SUCCESS_SYMBOL",
    "FAILURE_SYMBOL",
    "WARNING_SYMBOL",
]
