"""
Rich console formatting for experiment data.
"""

from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from yanex.utils.datetime_utils import format_duration, parse_iso_timestamp

from .fields import (
    format_experiment_duration,
    format_script,
    format_status,
    format_tags,
    format_timestamp_relative,
    truncate_middle,
)
from .theme import (
    DATA_TABLE_BOX,
    ID_STYLE,
    METRICS_STYLE,
    PARAMS_STYLE,
    SCRIPT_STYLE,
    TABLE_HEADER_STYLE,
    TAGS_STYLE,
    TIMESTAMP_STYLE,
)


class ExperimentTableFormatter:
    """
    Rich console formatter for experiment tables.

    Provides colored status indicators, formatted durations, and clean table layout.
    Uses centralized theme constants for consistent styling.
    """

    # Standard columns available for experiment tables
    STANDARD_COLUMNS = ["id", "script", "name", "status", "duration", "tags", "started"]

    # Maximum width for param/metric column headers before middle truncation
    MAX_COLUMN_HEADER_WIDTH = 12

    def __init__(self, console: Console = None):
        """
        Initialize formatter.

        Args:
            console: Rich console instance (creates default if None)
        """
        self.console = console or Console()

    def _get_meta_value(self, exp: dict, key: str, default: Any = None) -> Any:
        """Get a metadata value, checking both prefixed and unprefixed keys.

        Comparison data uses 'meta:' prefixed keys, while list data uses unprefixed.
        This method handles both formats.

        Args:
            exp: Experiment dictionary
            key: Unprefixed key name (e.g., 'id', 'name', 'status')
            default: Default value if not found

        Returns:
            The value from 'meta:{key}' or '{key}', or default if neither exists
        """
        # First check meta: prefixed key (comparison data format)
        prefixed_key = f"meta:{key}"
        if prefixed_key in exp:
            return exp[prefixed_key]
        # Fall back to unprefixed key (list data format)
        return exp.get(key, default)

    def format_experiments_table(
        self,
        experiments: list[dict[str, Any]],
        param_columns: list[str] | None = None,
        metric_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> Table:
        """
        Format experiments as a rich table.

        Args:
            experiments: List of experiment metadata dictionaries.
                For compare mode, each dict may contain 'param:name' and 'metric:name' keys.
            param_columns: Optional list of parameter column names to include (without 'param:' prefix)
            metric_columns: Optional list of metric column names to include (without 'metric:' prefix)
            exclude_columns: Optional list of standard columns to exclude.
                Valid values: "id", "script", "name", "status", "duration", "tags", "started"

        Returns:
            Rich Table object ready for console output
        """
        # Determine which standard columns to show
        exclude_set = set(exclude_columns) if exclude_columns else set()

        # Calculate optimal column widths based on content
        script_width = self._calculate_column_width(
            experiments, "script_path", min_width=15, max_width=60
        )
        name_width = self._calculate_column_width(
            experiments, "name", min_width=12, max_width=50
        )

        # Store name_width for use in _format_name
        self._current_name_width = name_width

        # Create table with borderless style and consistent header
        table = Table(
            show_header=True,
            header_style=TABLE_HEADER_STYLE,
            box=DATA_TABLE_BOX,
        )

        # Add standard columns with theme-consistent styles (unless excluded)
        if "id" not in exclude_set:
            table.add_column("ID", style=ID_STYLE, width=8)
        if "script" not in exclude_set:
            table.add_column("Script", style=SCRIPT_STYLE, width=script_width)
        if "name" not in exclude_set:
            table.add_column("Name", min_width=12, max_width=name_width)
        if "status" not in exclude_set:
            table.add_column("Status", width=12)
        if "duration" not in exclude_set:
            table.add_column("Duration", width=10, justify="right")
        if "tags" not in exclude_set:
            table.add_column("Tags", style=TAGS_STYLE, min_width=8, max_width=20)
        if "started" not in exclude_set:
            table.add_column(
                "Started", style=TIMESTAMP_STYLE, width=15, justify="right"
            )

        # Add optional param columns (truncate long names in middle)
        if param_columns:
            for param_name in param_columns:
                header = truncate_middle(param_name, self.MAX_COLUMN_HEADER_WIDTH)
                table.add_column(header, style=PARAMS_STYLE)

        # Add optional metric columns (truncate long names in middle)
        if metric_columns:
            for metric_name in metric_columns:
                header = truncate_middle(metric_name, self.MAX_COLUMN_HEADER_WIDTH)
                table.add_column(header, style=METRICS_STYLE, justify="right")

        # Add rows using shared formatters
        for exp in experiments:
            row_values = []

            # Add standard column values (unless excluded)
            # Use _get_meta_value to handle both prefixed (meta:key) and unprefixed keys
            if "id" not in exclude_set:
                row_values.append(self._format_id(self._get_meta_value(exp, "id", "")))
            if "script" not in exclude_set:
                row_values.append(
                    format_script(self._get_meta_value(exp, "script_path"))
                )
            if "name" not in exclude_set:
                row_values.append(self._format_name(self._get_meta_value(exp, "name")))
            if "status" not in exclude_set:
                row_values.append(
                    format_status(self._get_meta_value(exp, "status", "unknown"))
                )
            if "duration" not in exclude_set:
                row_values.append(format_experiment_duration(exp))
            if "tags" not in exclude_set:
                row_values.append(format_tags(self._get_meta_value(exp, "tags", [])))
            if "started" not in exclude_set:
                row_values.append(
                    format_timestamp_relative(self._get_meta_value(exp, "started_at"))
                )

            # Add param values
            if param_columns:
                for param_name in param_columns:
                    value = exp.get(f"param:{param_name}")
                    row_values.append(self._format_value(value))

            # Add metric values
            if metric_columns:
                for metric_name in metric_columns:
                    value = exp.get(f"metric:{metric_name}")
                    row_values.append(self._format_value(value))

            table.add_row(*row_values)

        return table

    def _format_value(self, value: Any) -> Text:
        """Format a generic value for display."""
        if value is None or value == "":
            return Text("-", style="dim")
        return Text(str(value))

    def print_experiments_table(
        self, experiments: list[dict[str, Any]], title: str = None
    ) -> None:
        """
        Print experiments table to console.

        Args:
            experiments: List of experiment metadata dictionaries
            title: Optional table title
        """
        if not experiments:
            self.console.print("No experiments found.", style="dim")
            return

        table = self.format_experiments_table(experiments)

        if title:
            table.title = title

        self.console.print(table)

    def print_summary(
        self,
        experiments: list[dict[str, Any]],
        total_count: int = None,
        show_help: bool = True,
    ) -> None:
        """
        Print summary information about the experiments.

        Args:
            experiments: Filtered experiment list
            total_count: Total number of experiments before filtering (if different)
            show_help: Whether to show helpful hints about viewing more experiments
        """
        count = len(experiments)

        if total_count is not None and total_count != count:
            summary = f"Showing {count} of {total_count} experiments"
            # Add helpful hint if showing limited results
            if show_help and count < total_count:
                summary += " (use --all to show all experiments)"
        else:
            summary = f"Found {count} experiment{'s' if count != 1 else ''}"

        self.console.print(f"\n{summary}", style="dim")

    def _format_id(self, experiment_id: str) -> str:
        """Format experiment ID."""
        if not experiment_id:
            return "unknown"
        return experiment_id

    def _format_name(self, name: str) -> Text:
        """Format experiment name with fallback for unnamed experiments."""
        if not name:
            return Text("[unnamed]", style="dim italic")

        # Use calculated width, fallback to 50 if not set
        max_width = getattr(self, "_current_name_width", 50)

        # Truncate in the middle if name exceeds max width
        name = truncate_middle(name, max_width)

        return Text(name)

    def _calculate_column_width(
        self,
        experiments: list[dict[str, Any]],
        field: str,
        min_width: int = 12,
        max_width: int = 50,
    ) -> int:
        """
        Calculate optimal width for a column based on content length.

        Args:
            experiments: List of experiment metadata dictionaries
            field: Field name to check ("script_path" or "name")
            min_width: Minimum column width
            max_width: Maximum column width

        Returns:
            Calculated column width between min and max
        """
        from pathlib import Path

        if not experiments:
            return min_width

        max_length = min_width
        for exp in experiments:
            value = exp.get(field)
            if value:
                # For script_path, extract just the filename
                if field == "script_path":
                    value = Path(value).name
                max_length = max(max_length, len(value))

        return min(max_length, max_width)

    def _format_time(self, time_str: str) -> str:
        """Format timestamp for detailed display."""
        if isinstance(time_str, str):
            dt = parse_iso_timestamp(time_str)
            if dt is None:
                return str(time_str)
        else:
            dt = time_str

        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _calculate_duration(self, start_time: str, end_time: str | None = None) -> str:
        """Calculate and format duration between two times."""
        if isinstance(start_time, str):
            start_dt = parse_iso_timestamp(start_time)
            if start_dt is None:
                return "unknown"
        else:
            start_dt = start_time

        end_dt = None
        if end_time:
            if isinstance(end_time, str):
                end_dt = parse_iso_timestamp(end_time)
            else:
                end_dt = end_time

        return format_duration(start_dt, end_dt)

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"

    def _format_timestamp(self, timestamp: float) -> str:
        """Format Unix timestamp for display."""
        try:
            from datetime import datetime

            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "unknown"
