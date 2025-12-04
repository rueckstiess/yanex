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
from .fields import (
    format_cancelled_message,
    format_description,
    format_duration_styled,
    format_error_message,
    format_experiment_duration,
    format_experiment_id,
    format_experiment_name,
    format_script,
    format_slot_name,
    format_status,
    format_status_symbol,
    format_success_message,
    format_tags,
    format_target_marker,
    format_timestamp_absolute,
    format_timestamp_relative,
    format_verbose,
    format_warning_message,
    truncate_middle,
)
from .getter_output import (
    GETTER_TYPES,
    GetterOutput,
    GetterType,
    get_getter_type,
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
    OutputFormat,
    OutputMode,
    echo_error,
    echo_format_info,
    echo_info,
    format_options,
    get_output_mode,
    is_machine_format,
    is_machine_output,
    output_mode_options,
    resolve_output_format,
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
    DESCRIPTION_STYLE,
    DURATION_STYLE,
    ERROR_STYLE,
    FAILURE_SYMBOL,
    ID_STYLE,
    LABEL_STYLE,
    METRICS_STYLE,
    NAME_STYLE,
    PANEL_BOX,
    PARAMS_STYLE,
    SCRIPT_STYLE,
    SLOT_STYLE,
    STATUS_COLORS,
    STATUS_SYMBOLS,
    STEP_STYLE,
    SUCCESS_STYLE,
    SUCCESS_SYMBOL,
    TABLE_HEADER_STYLE,
    TAGS_STYLE,
    TARGET_STYLE,
    TIMESTAMP_STYLE,
    VERBOSE_STYLE,
    WARNING_STYLE,
    WARNING_SYMBOL,
)

__all__ = [
    # Console formatter
    "ExperimentTableFormatter",
    # Getter output (for yanex get command)
    "GetterType",
    "GetterOutput",
    "GETTER_TYPES",
    "get_getter_type",
    # Output formats (new unified approach)
    "OutputFormat",
    "format_options",
    "resolve_output_format",
    "is_machine_format",
    "echo_format_info",
    # Output modes (legacy, for backwards compatibility)
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
    # Theme constants
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
    # New theme constants
    "SLOT_STYLE",
    "TARGET_STYLE",
    "DESCRIPTION_STYLE",
    "STEP_STYLE",
    "DURATION_STYLE",
    "LABEL_STYLE",
    "WARNING_STYLE",
    "ERROR_STYLE",
    "SUCCESS_STYLE",
    "VERBOSE_STYLE",
    # Field formatters
    "format_experiment_id",
    "format_experiment_name",
    "format_script",
    "format_status",
    "format_status_symbol",
    "format_timestamp_relative",
    "format_timestamp_absolute",
    "format_duration_styled",
    "format_experiment_duration",
    "format_tags",
    "format_slot_name",
    "format_target_marker",
    "format_description",
    # Message formatters
    "format_success_message",
    "format_error_message",
    "format_warning_message",
    "format_verbose",
    "format_cancelled_message",
    # Text utilities
    "truncate_middle",
]
