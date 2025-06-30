"""
Compare experiments - interactive table with parameters and metrics.
"""

from typing import Optional

import click

from ...core.comparison import ExperimentComparisonData
from ...core.constants import EXPERIMENT_STATUSES
from ...ui.compare_table import run_comparison_table
from ..filters import ExperimentFilter, parse_time_spec
from .confirm import find_experiments_by_filters, find_experiments_by_identifiers


@click.command("compare")
@click.argument("experiment_identifiers", nargs=-1)
@click.option(
    "--status",
    type=click.Choice(EXPERIMENT_STATUSES),
    help="Compare experiments with specific status",
)
@click.option(
    "--name",
    "name_pattern",
    help="Compare experiments matching name pattern (glob syntax)",
)
@click.option(
    "--tag", "tags", multiple=True, help="Compare experiments with ALL specified tags"
)
@click.option(
    "--started-after",
    help="Compare experiments started after date/time (e.g., '2025-01-01', 'yesterday', '1 week ago')",
)
@click.option("--started-before", help="Compare experiments started before date/time")
@click.option("--ended-after", help="Compare experiments ended after date/time")
@click.option("--ended-before", help="Compare experiments ended before date/time")
@click.option(
    "--archived",
    is_flag=True,
    help="Include archived experiments (default: only regular experiments)",
)
@click.option(
    "--params",
    help="Show only specified parameters (comma-separated, e.g., 'learning_rate,epochs')",
)
@click.option(
    "--metrics",
    help="Show only specified metrics (comma-separated, e.g., 'accuracy,loss')",
)
@click.option(
    "--only-different",
    is_flag=True,
    help="Show only columns where values differ between experiments",
)
@click.option(
    "--export",
    "export_path",
    help="Export comparison data to CSV file instead of interactive view",
)
@click.option(
    "--no-interactive",
    is_flag=True,
    help="Print static table instead of interactive view",
)
@click.option(
    "--max-rows",
    type=int,
    help="Limit number of experiments displayed",
)
@click.pass_context
def compare_experiments(
    ctx,
    experiment_identifiers: tuple,
    status: Optional[str],
    name_pattern: Optional[str],
    tags: tuple,
    started_after: Optional[str],
    started_before: Optional[str],
    ended_after: Optional[str],
    ended_before: Optional[str],
    archived: bool,
    params: Optional[str],
    metrics: Optional[str],
    only_different: bool,
    export_path: Optional[str],
    no_interactive: bool,
    max_rows: Optional[int],
):
    """
    Compare experiments in an interactive table showing parameters and metrics.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    The interactive table supports:
    - Navigation with arrow keys or hjkl
    - Sorting by any column (s/S for asc/desc, 1/2 for numeric)
    - Export to CSV (press 'e' in interactive mode)
    - Help (press '?' for keyboard shortcuts)

    Examples:
    \\b
        yanex compare                                    # All experiments
        yanex compare exp1 exp2 exp3                    # Specific experiments
        yanex compare --status completed                # Completed experiments
        yanex compare --tag training --only-different  # Training experiments, show differences only
        yanex compare --params learning_rate,epochs    # Show only specified parameters
        yanex compare --export results.csv             # Export to CSV
        yanex compare --no-interactive                  # Static table output
    """
    try:
        filter_obj = ExperimentFilter()

        # Validate mutually exclusive targeting
        has_identifiers = len(experiment_identifiers) > 0
        has_filters = any(
            [
                status,
                name_pattern,
                tags,
                started_after,
                started_before,
                ended_after,
                ended_before,
            ]
        )

        if has_identifiers and has_filters:
            click.echo(
                "Error: Cannot use both experiment identifiers and filter options. Choose one approach.",
                err=True,
            )
            ctx.exit(1)

        if not has_identifiers and not has_filters:
            # No filters specified - use all experiments
            pass

        # Parse time specifications
        started_after_dt = parse_time_spec(started_after) if started_after else None
        started_before_dt = parse_time_spec(started_before) if started_before else None
        ended_after_dt = parse_time_spec(ended_after) if ended_after else None
        ended_before_dt = parse_time_spec(ended_before) if ended_before else None

        # Find experiments to compare
        if experiment_identifiers:
            # Compare specific experiments by ID/name
            experiments = find_experiments_by_identifiers(
                filter_obj, list(experiment_identifiers), include_archived=archived
            )
        else:
            # Compare experiments by filter criteria
            experiments = find_experiments_by_filters(
                filter_obj,
                status=status,
                name_pattern=name_pattern,
                tags=list(tags) if tags else None,
                started_after=started_after_dt,
                started_before=started_before_dt,
                ended_after=ended_after_dt,
                ended_before=ended_before_dt,
                include_archived=archived,
            )

        if not experiments:
            location = "archived" if archived else "regular"
            click.echo(f"No {location} experiments found to compare.")
            return

        # Limit number of experiments if requested
        if max_rows and len(experiments) > max_rows:
            experiments = experiments[:max_rows]
            click.echo(f"Limiting display to first {max_rows} experiments.")

        # Extract experiment IDs from the experiment metadata dictionaries
        experiment_ids = [exp["id"] for exp in experiments]

        # Parse parameter and metric lists
        param_list = None
        if params:
            param_list = [p.strip() for p in params.split(",") if p.strip()]

        metric_list = None
        if metrics:
            metric_list = [m.strip() for m in metrics.split(",") if m.strip()]

        # Get comparison data
        comparison_data_extractor = ExperimentComparisonData()
        comparison_data = comparison_data_extractor.get_comparison_data(
            experiment_ids=experiment_ids,
            params=param_list,
            metrics=metric_list,
            only_different=only_different,
            include_archived=archived,
        )

        if not comparison_data.get("rows"):
            click.echo("No comparison data available.")
            return

        # Handle export mode
        if export_path:
            _export_comparison_data(comparison_data, export_path)
            click.echo(f"Comparison data exported to {export_path}")
            return

        # Handle static table mode
        if no_interactive:
            _print_static_table(comparison_data)
            return

        # Run interactive table
        total_experiments = comparison_data.get("total_experiments", 0)
        param_count = len(comparison_data.get("param_columns", []))
        metric_count = len(comparison_data.get("metric_columns", []))

        title = f"yanex compare - {total_experiments} experiments, {param_count} params, {metric_count} metrics"

        # Set default export path
        default_export = export_path or "yanex_comparison.csv"

        run_comparison_table(comparison_data, title=title, export_path=default_export)

    except click.ClickException:
        raise  # Re-raise ClickException to show proper error message
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _export_comparison_data(comparison_data: dict, file_path: str) -> None:
    """Export comparison data to CSV file."""
    import csv
    from pathlib import Path

    rows = comparison_data.get("rows", [])
    if not rows:
        raise ValueError("No data to export")

    # Get column order - use same order as UI
    first_row = rows[0]
    column_keys = list(first_row.keys())

    # Generate headers
    column_headers = []
    for key in column_keys:
        if key.startswith("param:"):
            column_headers.append(f"param_{key[6:]}")  # Remove emoji for CSV
        elif key.startswith("metric:"):
            column_headers.append(f"metric_{key[7:]}")  # Remove emoji for CSV
        else:
            column_headers.append(key)

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


def _print_static_table(comparison_data: dict) -> None:
    """Print comparison data as a static table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    rows = comparison_data.get("rows", [])

    if not rows:
        console.print("No data to display")
        return

    # Create table
    table = Table(show_header=True, header_style="bold magenta")

    # Get column order
    first_row = rows[0]
    column_keys = list(first_row.keys())

    # Add columns with formatted headers
    for key in column_keys:
        if key.startswith("param:"):
            header = f"📊 {key[6:]}"
        elif key.startswith("metric:"):
            header = f"📈 {key[7:]}"
        elif key == "duration":
            header = "Duration"
        elif key == "tags":
            header = "Tags"
        elif key == "id":
            header = "ID"
        elif key == "name":
            header = "Name"
        else:
            header = key.title()
        table.add_column(header)

    # Add rows
    for row_data in rows:
        row_values = [str(row_data.get(key, "-")) for key in column_keys]
        table.add_row(*row_values)

    # Print table
    console.print(table)

    # Print summary
    param_count = len(comparison_data.get("param_columns", []))
    metric_count = len(comparison_data.get("metric_columns", []))

    console.print(
        f"\n[dim]Showing {len(rows)} experiments with {param_count} parameters and {metric_count} metrics[/dim]"
    )
