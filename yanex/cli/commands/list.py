"""
List command implementation for yanex CLI.
"""

import click

from ..error_handling import CLIErrorHandler
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from ..formatters import (
    ExperimentTableFormatter,
    OutputFormat,
    experiments_to_list,
    format_options,
    output_csv,
    output_json,
    output_markdown_table,
    resolve_output_format,
)

# Standard columns for list output
LIST_COLUMNS = ["id", "name", "status", "script_path", "tags", "created_at", "duration"]

# Human-readable headers for CSV/markdown output
LIST_HEADERS = {
    "id": "ID",
    "name": "Name",
    "status": "Status",
    "script_path": "Script",
    "tags": "Tags",
    "created_at": "Created",
    "duration": "Duration",
}


@click.command()
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Show all experiments (overrides default limit of 10)",
)
@format_options()
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=True)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def list_experiments(
    ctx: click.Context,
    show_all: bool,
    output_format: str | None,
    json_flag: bool,
    csv_flag: bool,
    markdown_flag: bool,
    limit: int | None,
    ids: str | None,
    status: str | None,
    name_pattern: str | None,
    tags: tuple,
    script_pattern: str | None,
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
    archived: bool,
) -> None:
    """
    List experiments with filtering options.

    Shows the last 10 experiments by default. Use --all to show all experiments
    or -l to specify a custom limit.

    Supports multiple output formats:

    \b
      --format json      Output as JSON (for scripting/AI processing)
      --format csv       Output as CSV (for spreadsheets/data analysis)
      --format markdown  Output as GitHub-flavored markdown

    Examples:

    \b
      # Show last 10 experiments
      yanex list

    \b
      # Show all experiments
      yanex list --all

    \b
      # Export as JSON for processing
      yanex list --format json > experiments.json

    \b
      # Export as CSV for spreadsheets
      yanex list --format csv > experiments.csv

    \b
      # Filter by status
      yanex list -s completed

    \b
      # Filter by name pattern
      yanex list -n "*tuning*"

    \b
      # Filter by multiple tags (AND logic)
      yanex list -t hyperopt -t production

    \b
      # Complex filtering with JSON output
      yanex list -s completed -t production --format json
    """
    # Resolve output format from --format option or legacy flags
    fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)

    verbose = ctx.obj.get("verbose", False)

    try:
        # Parse time specifications using centralized error handling
        started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
            CLIErrorHandler.parse_time_filters(
                started_after, started_before, ended_after, ended_before
            )
        )

        # Parse comma-separated IDs into a list
        ids_list = None
        if ids:
            ids_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

        if verbose:
            click.echo("Filtering experiments...")
            if ids_list:
                click.echo(f"  IDs: {', '.join(ids_list)}")
            if status:
                click.echo(f"  Status: {status}")
            if name_pattern:
                click.echo(f"  Name pattern: {name_pattern}")
            if script_pattern:
                click.echo(f"  Script pattern: {script_pattern}")
            if tags:
                click.echo(f"  Tags: {', '.join(tags)}")
            if started_after:
                click.echo(f"  Started after: {started_after}")
            if started_before:
                click.echo(f"  Started before: {started_before}")
            if ended_after:
                click.echo(f"  Ended after: {ended_after}")
            if ended_before:
                click.echo(f"  Ended before: {ended_before}")

        # Create filter and apply criteria
        experiment_filter = ExperimentFilter()

        # When showing archived experiments, we need to get all experiments first
        # to avoid the default limit cutting off archived experiments
        force_all = show_all or archived

        experiments = experiment_filter.filter_experiments(
            ids=ids_list,
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            script_pattern=script_pattern,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            limit=None if force_all else limit,
            include_all=force_all,
            archived=archived,
        )

        # Filter experiments based on archived flag
        if archived:
            experiments = [exp for exp in experiments if exp.get("archived", False)]
        else:
            experiments = [exp for exp in experiments if not exp.get("archived", False)]

        # Apply limit after filtering by archived status if needed
        if not show_all and limit is not None:
            experiments = experiments[:limit]
        elif not show_all and limit is None and not archived:
            # Only apply default limit to regular experiments, not archived
            experiments = experiments[:10]

        if verbose:
            click.echo(f"Found {len(experiments)} matching experiments")

        # Handle empty results
        if not experiments:
            if fmt == OutputFormat.JSON:
                output_json([])
            elif fmt == OutputFormat.CSV:
                # Output empty CSV with just headers
                output_csv([], columns=LIST_COLUMNS)
            elif fmt == OutputFormat.MARKDOWN:
                click.echo("_No experiments found_")
            else:
                click.echo("No experiments found.")
                _show_filter_suggestions(
                    status,
                    name_pattern,
                    tags,
                    script_pattern,
                    started_after,
                    started_before,
                    ended_after,
                    ended_before,
                )
            return

        # Output based on format
        if fmt == OutputFormat.JSON:
            output_json(experiments_to_list(experiments))
            return

        if fmt == OutputFormat.CSV:
            output_csv(
                experiments_to_list(experiments),
                columns=LIST_COLUMNS,
                headers=LIST_HEADERS,
            )
            return

        if fmt == OutputFormat.MARKDOWN:
            table_title = "Archived Experiments" if archived else "Experiments"
            output_markdown_table(
                experiments_to_list(experiments),
                columns=LIST_COLUMNS,
                headers=LIST_HEADERS,
                title=table_title,
            )
            return

        # Default: Rich console output
        formatter = ExperimentTableFormatter()
        formatter.print_experiments_table(experiments)

        # Show summary if filtering was applied or not showing all
        # Note: name_pattern="" is a valid filter for unnamed experiments
        if any(
            [
                status,
                name_pattern is not None,
                tags,
                script_pattern,
                started_after,
                started_before,
                ended_after,
                ended_before,
            ]
        ) or (not show_all and limit != len(experiments)):
            # Get total count for summary
            total_experiments = experiment_filter.filter_experiments(
                include_all=True, archived=archived
            )

            # Filter total based on archived flag too
            if archived:
                total_experiments = [
                    exp for exp in total_experiments if exp.get("archived", False)
                ]
            else:
                total_experiments = [
                    exp for exp in total_experiments if not exp.get("archived", False)
                ]

            formatter.print_summary(experiments, len(total_experiments))

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error listing experiments: {e}", err=True)
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()


def _show_filter_suggestions(
    status: str | None,
    name_pattern: str | None,
    tags: list[str],
    script_pattern: str | None,
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
) -> None:
    """Show helpful suggestions when no experiments are found."""

    # Check if any filters were applied
    # Note: name_pattern="" is a valid filter for unnamed experiments
    has_filters = any(
        [
            status,
            name_pattern is not None,
            tags,
            script_pattern,
            started_after,
            started_before,
            ended_after,
            ended_before,
        ]
    )

    if has_filters:
        click.echo("\nTry adjusting your filters:")
        click.echo("  • Remove some filters to see more results")
        click.echo("  • Use 'yanex list --all' to see all experiments")
        click.echo("  • Check status with: yanex list --status completed")
    else:
        click.echo("\nNo experiments found. To create your first experiment:")
        click.echo("  yanex run your_script.py")
