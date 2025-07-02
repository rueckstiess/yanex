"""
Rich console formatting for experiment data.
"""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

from ...utils.datetime_utils import (
    format_duration,
    format_relative_time,
    parse_iso_timestamp,
)


class ExperimentTableFormatter:
    """
    Rich console formatter for experiment tables.

    Provides colored status indicators, formatted durations, and clean table layout.
    """

    # Status color mapping
    STATUS_COLORS = {
        "completed": "green",
        "failed": "red",
        "running": "yellow",
        "created": "white",
        "cancelled": "bright_red",
        "staged": "cyan",
    }

    # Status symbols for better visual distinction
    STATUS_SYMBOLS = {
        "completed": "✓",
        "failed": "✗",
        "running": "⚡",
        "created": "○",
        "cancelled": "✖",
        "staged": "⏲",
    }

    def __init__(self, console: Console = None):
        """
        Initialize formatter.

        Args:
            console: Rich console instance (creates default if None)
        """
        self.console = console or Console()

    def format_experiments_table(self, experiments: list[dict[str, Any]]) -> Table:
        """
        Format experiments as a rich table.

        Args:
            experiments: List of experiment metadata dictionaries

        Returns:
            Rich Table object ready for console output
        """
        # Create table with columns
        table = Table(show_header=True, header_style="bold")

        # Add columns
        table.add_column("ID", style="dim", width=8)
        table.add_column("Name", min_width=12, max_width=25)
        table.add_column("Status", width=12)
        table.add_column("Duration", width=10, justify="right")
        table.add_column("Tags", min_width=8, max_width=20)
        table.add_column("Started", width=15, justify="right")

        # Add rows
        for exp in experiments:
            table.add_row(
                self._format_id(exp.get("id", "")),
                self._format_name(exp.get("name")),
                self._format_status(exp.get("status", "unknown")),
                self._format_duration(exp),
                self._format_tags(exp.get("tags", [])),
                self._format_started_time(exp.get("started_at")),
            )

        return table

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

        # Truncate long names
        if len(name) > 28:
            name = name[:25] + "..."

        return Text(name)

    def _format_status(self, status: str) -> Text:
        """Format status with color and symbol."""
        color = self.STATUS_COLORS.get(status, "white")
        symbol = self.STATUS_SYMBOLS.get(status, "?")

        return Text(f"{symbol} {status}", style=color)

    def _format_duration(self, experiment: dict[str, Any]) -> Text:
        """Format experiment duration."""
        started_at = experiment.get("started_at")
        status = experiment.get("status", "")

        # Try different possible end time fields
        ended_at = (
            experiment.get("ended_at")
            or experiment.get("completed_at")
            or experiment.get("failed_at")
            or experiment.get("cancelled_at")
        )

        if not started_at:
            return Text("-", style="dim")

        try:
            start_time = parse_iso_timestamp(started_at)
            if start_time is None:
                return Text("-", style="dim")

            end_time = None
            if ended_at:
                end_time = parse_iso_timestamp(ended_at)
            elif status == "running":
                # For running experiments, end_time stays None to show "(ongoing)"
                pass
            else:
                # For non-running experiments without end time, use current time as fallback
                from datetime import timezone

                end_time = datetime.now(timezone.utc)

            duration_str = format_duration(start_time, end_time)

            # Color coding based on status
            if status == "running":
                return Text(duration_str, style="yellow")
            elif status == "completed":
                return Text(duration_str, style="green")
            elif status in ("failed", "cancelled"):
                return Text(duration_str, style="red")
            else:
                return Text(duration_str, style="dim")

        except Exception:
            return Text("unknown", style="dim")

    def _format_tags(self, tags: list[str]) -> Text:
        """Format tags list."""
        if not tags:
            return Text("-", style="dim")

        # Limit displayed tags and truncate if necessary
        display_tags = tags[:3]  # Show max 3 tags

        if len(tags) > 3:
            tag_str = ", ".join(display_tags) + f" (+{len(tags) - 3})"
        else:
            tag_str = ", ".join(display_tags)

        # Truncate if still too long
        if len(tag_str) > 23:
            tag_str = tag_str[:20] + "..."

        return Text(tag_str, style="blue")

    def _format_started_time(self, started_at: str) -> Text:
        """Format started time as relative time."""
        if not started_at:
            return Text("-", style="dim")

        start_time = parse_iso_timestamp(started_at)
        if start_time is None:
            return Text("unknown", style="dim")

        relative_str = format_relative_time(start_time)
        return Text(relative_str, style="cyan")

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
