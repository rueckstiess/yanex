"""
Interactive comparison table using Textual DataTable.

This module provides a terminal-based interactive table for comparing experiments,
with sorting, navigation, and export functionality.
"""

import csv
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Static


class HelpScreen(ModalScreen):
    """Modal screen showing keyboard shortcuts and help."""

    def compose(self) -> ComposeResult:
        """Compose the help screen layout."""
        yield Container(
            Label("yanex compare - Keyboard Shortcuts", classes="help-title"),
            Static(
                """
Navigation:
  ↑/↓, j/k     Navigate rows
  ←/→, h/l     Navigate columns
  Home/End     Jump to first/last row
  PgUp/PgDn    Navigate by page

Sorting:
  s            Sort ascending by current column
  S            Sort descending by current column
  1            Numerical sort ascending
  2            Numerical sort descending
  r            Reset to original order
  R            Reverse current sort order

Other Controls:
  e            Export current view to CSV
  ?            Show this help
  q, Ctrl+C    Quit

Column Types:
  Parameters are prefixed with 'param:'
  Metrics are prefixed with 'metric:'
Missing values are shown as '-'
                """,
                classes="help-content",
            ),
            Button("Close", variant="primary", id="close-help"),
            classes="help-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "close-help":
            self.dismiss()


class ExportScreen(ModalScreen):
    """Modal screen for CSV export."""

    def __init__(self, default_path: str = "comparison.csv"):
        super().__init__()
        self.default_path = default_path

    def compose(self) -> ComposeResult:
        """Compose the export screen layout."""
        yield Container(
            Label("Export Comparison Data", classes="export-title"),
            Label("Enter filename for CSV export:"),
            Input(
                value=self.default_path, placeholder="comparison.csv", id="export-path"
            ),
            Horizontal(
                Button("Export", variant="primary", id="export-confirm"),
                Button("Cancel", variant="default", id="export-cancel"),
                classes="export-buttons",
            ),
            classes="export-dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        if event.button.id == "export-confirm":
            path_input = self.query_one("#export-path", Input)
            self.dismiss(path_input.value)
        elif event.button.id == "export-cancel":
            self.dismiss(None)


class ComparisonTableApp(App):
    """Interactive comparison table application."""

    CSS = """
    .help-dialog {
        width: 80;
        height: 30;
        background: $surface;
        border: thick $primary;
        margin: 2;
        padding: 1;
    }

    .help-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    .help-content {
        margin: 1;
        padding: 1;
        background: $surface-darken-1;
    }

    .export-dialog {
        width: 60;
        height: 12;
        background: $surface;
        border: thick $primary;
        margin: 2;
        padding: 1;
    }

    .export-title {
        text-align: center;
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    .export-buttons {
        margin-top: 1;
        align: center middle;
    }

    DataTable {
        height: 1fr;
    }

    #status-bar {
        height: 1;
        background: $surface-darken-1;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help"),
        Binding("s", "sort_asc", "Sort ↑"),
        Binding("S", "sort_desc", "Sort ↓"),
        Binding("1", "sort_numeric_asc", "Numeric ↑"),
        Binding("2", "sort_numeric_desc", "Numeric ↓"),
        Binding("r", "reset_sort", "Reset"),
        Binding("R", "reverse_sort", "Reverse"),
        Binding("e", "export", "Export"),
    ]

    def __init__(
        self,
        comparison_data: dict[str, Any],
        title: str = "yanex compare",
        export_path: str | None = None,
    ):
        """
        Initialize the comparison table app.

        Args:
            comparison_data: Data from ExperimentComparisonData.get_comparison_data()
            title: Application title
            export_path: Default export path for CSV
        """
        super().__init__()
        self.comparison_data = comparison_data
        self.title_text = title
        self.export_path = export_path or "comparison.csv"
        self.original_rows = comparison_data.get("rows", [])
        # Set default sort to "started_at" descending (latest experiments first)
        self.current_sort_key = "started_at"
        self.current_sort_reverse = True

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header(show_clock=True)
        yield DataTable(id="comparison-table")
        yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the table when app mounts."""
        self.title = self.title_text

        # Apply default sort (started_at descending) - this will also populate the table
        self._sort_table("started_at", reverse=True)

    def _setup_table_columns(self, table: DataTable) -> None:
        """Set up table columns."""
        rows = self.comparison_data.get("rows", [])
        if not rows:
            return

        # Get column names from first row
        first_row = rows[0]
        column_keys = list(first_row.keys())

        # Add columns to table
        for key in column_keys:
            header = self._get_column_header(key)
            table.add_column(header, key=key)

    def _populate_table_data(self, table: DataTable) -> None:
        """Populate table with data."""
        # Save cursor position before clearing
        saved_cursor_row = table.cursor_row
        saved_cursor_column = table.cursor_column

        # Clear table completely (removes columns and rows)
        table.clear(columns=True)
        rows = self.comparison_data.get("rows", [])

        if not rows:
            return

        # Get column order from data
        column_keys = self._get_column_keys()

        # Re-add columns with updated headers (including sort indicators)
        for key in column_keys:
            header = self._get_column_header(key)
            table.add_column(header, key=key)

        # Add rows with formatted values
        for row_data in rows:
            row_values = [
                self._format_cell_value(key, row_data.get(key)) for key in column_keys
            ]
            table.add_row(*row_values)

        # Restore cursor position (with bounds checking)
        if rows:
            max_row = len(rows) - 1
            max_column = len(column_keys) - 1

            # Ensure cursor position is within bounds
            new_cursor_row = min(saved_cursor_row, max_row)
            new_cursor_column = min(saved_cursor_column, max_column)

            # Restore cursor position
            table.move_cursor(row=new_cursor_row, column=new_cursor_column)

    def _get_column_keys(self) -> list[str]:
        """Get column keys from data."""
        rows = self.comparison_data.get("rows", [])
        if not rows:
            return []
        return list(rows[0].keys())

    def _get_column_header(self, key: str) -> str:
        """Get formatted column header with optional sort indicator."""
        # Base header without sort indicator
        if key.startswith("param:"):
            base_header = f"📊 {key[6:]}"  # Remove 'param:' prefix
        elif key.startswith("metric:"):
            base_header = f"📈 {key[7:]}"  # Remove 'metric:' prefix
        elif key == "script_path":
            base_header = "Script"
        elif key == "started_at":
            base_header = "Started"
        elif key == "ended_at":
            base_header = "Ended"
        elif key == "tags":
            base_header = "Tags"
        elif key == "id":
            base_header = "ID"
        elif key == "name":
            base_header = "Name"
        elif key == "status":
            base_header = "Status"
        else:
            base_header = key.replace("_", " ").title()

        # Add sort indicator if this column is currently sorted
        if self.current_sort_key == key:
            sort_indicator = " ↓" if self.current_sort_reverse else " ↑"
            return base_header + sort_indicator

        return base_header

    def _format_cell_value(self, key: str, value: Any) -> str:
        """Format a cell value for display.

        Args:
            key: Column key (e.g., 'script_path', 'started_at', 'tags').
            value: Raw value from comparison data.

        Returns:
            Formatted string for display.
        """
        from pathlib import Path

        from yanex.utils.datetime_utils import format_relative_time, parse_iso_timestamp

        if value is None:
            return "-"

        # Format script_path - extract just the filename
        if key == "script_path":
            if not value:
                return "-"
            return Path(value).name

        # Format timestamps as relative time ("4 hours ago")
        if key in ("started_at", "ended_at"):
            if not value:
                return "-"
            dt = parse_iso_timestamp(value)
            if dt:
                return format_relative_time(dt)
            return str(value)

        # Format tags - convert list to comma-separated string
        if key == "tags":
            if not value:
                return "-"
            if isinstance(value, list):
                return ", ".join(str(t) for t in value) if value else "-"
            return str(value)

        # Handle None/empty for other fields
        if value == "" or value is None:
            return "-"

        return str(value)

    def _update_status_bar(self) -> None:
        """Update the status bar with current information."""
        total_experiments = self.comparison_data.get("total_experiments", 0)
        param_count = len(self.comparison_data.get("param_columns", []))
        metric_count = len(self.comparison_data.get("metric_columns", []))

        status_text = (
            f"Experiments: {total_experiments} | "
            f"Parameters: {param_count} | "
            f"Metrics: {metric_count}"
        )

        if self.current_sort_key:
            sort_direction = "↓" if self.current_sort_reverse else "↑"
            status_text += f" | Sorted by: {self.current_sort_key} {sort_direction}"

        status_bar = self.query_one("#status-bar", Static)
        status_bar.update(status_text)

    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())

    def action_sort_asc(self) -> None:
        """Sort by current column ascending."""
        table = self.query_one(DataTable)
        column_keys = self._get_column_keys()
        if table.cursor_column < len(column_keys):
            column_key = column_keys[table.cursor_column]
            self._sort_table(column_key, reverse=False)

    def action_sort_desc(self) -> None:
        """Sort by current column descending."""
        table = self.query_one(DataTable)
        column_keys = self._get_column_keys()
        if table.cursor_column < len(column_keys):
            column_key = column_keys[table.cursor_column]
            self._sort_table(column_key, reverse=True)

    def action_sort_numeric_asc(self) -> None:
        """Sort by current column numerically ascending."""
        table = self.query_one(DataTable)
        column_keys = self._get_column_keys()
        if table.cursor_column < len(column_keys):
            column_key = column_keys[table.cursor_column]
            self._sort_table(column_key, reverse=False, numeric=True)

    def action_sort_numeric_desc(self) -> None:
        """Sort by current column numerically descending."""
        table = self.query_one(DataTable)
        column_keys = self._get_column_keys()
        if table.cursor_column < len(column_keys):
            column_key = column_keys[table.cursor_column]
            self._sort_table(column_key, reverse=True, numeric=True)

    def action_reset_sort(self) -> None:
        """Reset to default sort order (started_at descending)."""
        # Reset to default sort (started_at descending)
        self._sort_table("started_at", reverse=True)

    def action_reverse_sort(self) -> None:
        """Reverse current sort order."""
        if self.current_sort_key:
            # Reverse the current sort
            self._sort_table(
                self.current_sort_key, reverse=not self.current_sort_reverse
            )
        else:
            # If no current sort, just reverse the rows and update state
            self.comparison_data["rows"].reverse()
            self.current_sort_reverse = not self.current_sort_reverse
            table = self.query_one(DataTable)
            self._populate_table_data(table)
            self._update_status_bar()

    def action_export(self) -> None:
        """Export data to CSV."""
        self.push_screen(ExportScreen(self.export_path), self._handle_export)

    def _handle_export(self, export_path: str | None) -> None:
        """Handle export screen result."""
        if export_path:
            try:
                self._export_to_csv(export_path)
                self.notify(f"Exported to {export_path}", severity="information")
            except Exception as e:
                self.notify(f"Export failed: {e}", severity="error")

    def _export_to_csv(self, file_path: str) -> None:
        """Export current table data to CSV."""
        rows = self.comparison_data.get("rows", [])
        if not rows:
            raise ValueError("No data to export")

        # Get column order from data
        column_keys = self._get_column_keys()
        # For headers, we'll use the display headers we set up
        column_headers = []
        for key in column_keys:
            if key.startswith("param:"):
                column_headers.append(f"📊 {key[6:]}")
            elif key.startswith("metric:"):
                column_headers.append(f"📈 {key[7:]}")
            elif key == "script_path":
                column_headers.append("Script")
            elif key == "started_at":
                column_headers.append("Started")
            elif key == "ended_at":
                column_headers.append("Ended")
            elif key == "tags":
                column_headers.append("Tags")
            elif key == "id":
                column_headers.append("ID")
            elif key == "name":
                column_headers.append("Name")
            elif key == "status":
                column_headers.append("Status")
            else:
                column_headers.append(key.title())

        # Write CSV
        path = Path(file_path)
        with path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow(column_headers)

            # Write data rows
            for row_data in rows:
                row_values = [row_data.get(key, "-") for key in column_keys]
                writer.writerow(row_values)

    def _sort_table(
        self, column_key: str, reverse: bool = False, numeric: bool = False
    ) -> None:
        """Sort table by specified column."""
        rows = self.comparison_data.get("rows", [])
        if not rows:
            return

        def sort_key(row_data: dict[str, Any]) -> Any:
            """Get sort key for a row."""
            value = row_data.get(column_key, "-")

            # Handle missing values
            if value == "-":
                return (
                    ""
                    if not numeric
                    else float("-inf")
                    if not reverse
                    else float("inf")
                )

            # Try numeric conversion if requested
            if numeric:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return float("-inf") if not reverse else float("inf")

            # Use string comparison
            return str(value).lower()

        # Sort the rows
        sorted_rows = sorted(rows, key=sort_key, reverse=reverse)
        self.comparison_data["rows"] = sorted_rows

        # Update status tracking BEFORE populating table (so headers show indicators)
        self.current_sort_key = column_key
        self.current_sort_reverse = reverse

        # Update table display
        table = self.query_one(DataTable)
        self._populate_table_data(table)
        self._update_status_bar()


def run_comparison_table(
    comparison_data: dict[str, Any],
    title: str = "yanex compare",
    export_path: str | None = None,
) -> None:
    """
    Run the interactive comparison table.

    Args:
        comparison_data: Data from ExperimentComparisonData.get_comparison_data()
        title: Application title
        export_path: Default export path for CSV
    """
    app = ComparisonTableApp(comparison_data, title, export_path)
    app.run()


if __name__ == "__main__":
    # Example usage with mock data (matching actual comparison data structure)
    mock_data = {
        "rows": [
            {
                "id": "exp00001",
                "script_path": "/path/to/train.py",
                "name": "experiment-1",
                "started_at": "2025-01-01T10:00:00Z",
                "ended_at": "2025-01-01T11:30:45Z",
                "status": "completed",
                "tags": ["ml", "training"],
                "param:learning_rate": "0.01",
                "param:epochs": "10",
                "metric:accuracy": "0.95",
                "metric:loss": "0.05",
            },
            {
                "id": "exp00002",
                "script_path": "/path/to/train.py",
                "name": "experiment-2",
                "started_at": "2025-01-01T12:00:00Z",
                "ended_at": "2025-01-01T12:45:30Z",
                "status": "failed",
                "tags": ["ml"],
                "param:learning_rate": "0.02",
                "param:epochs": "5",
                "metric:accuracy": "0.87",
                "metric:loss": "0.13",
            },
        ],
        "param_columns": ["learning_rate", "epochs"],
        "metric_columns": ["accuracy", "loss"],
        "column_types": {
            "id": "string",
            "script_path": "string",
            "name": "string",
            "started_at": "datetime",
            "ended_at": "datetime",
            "status": "string",
            "tags": "string",
            "param:learning_rate": "numeric",
            "param:epochs": "numeric",
            "metric:accuracy": "numeric",
            "metric:loss": "numeric",
        },
        "total_experiments": 2,
    }

    run_comparison_table(mock_data)
