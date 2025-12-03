"""
List command implementation for yanex CLI.
"""

import click

from ..error_handling import CLIErrorHandler
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from ..formatters import ExperimentTableFormatter


@click.command()
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    help="Show all experiments (overrides default limit of 10)",
)
@experiment_filter_options(include_ids=False, include_archived=True, include_limit=True)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def list_experiments(
    ctx: click.Context,
    show_all: bool,
    limit: int | None,
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

    Examples:

      # Show last 10 experiments
      yanex list

      # Show all experiments
      yanex list --all

      # Show last 5 experiments
      yanex list -l 5

      # Filter by status
      yanex list -s completed

      # Filter by name pattern
      yanex list -n "*tuning*"

      # Filter unnamed experiments
      yanex list -n ""

      # Filter by multiple tags (AND logic)
      yanex list -t hyperopt -t production

      # Filter by time (started after last week)
      yanex list --started-after "last week"

      # Filter by time range
      yanex list --started-after "last month" --started-before "last week"

      # Complex filtering
      yanex list -s completed -t production --started-after "last month" -l 20
    """
    verbose = ctx.obj.get("verbose", False)

    try:
        # Parse time specifications using centralized error handling
        started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
            CLIErrorHandler.parse_time_filters(
                started_after, started_before, ended_after, ended_before
            )
        )

        if verbose:
            click.echo("Filtering experiments...")
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

        # Format and display results
        formatter = ExperimentTableFormatter()

        if not experiments:
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

        # Display table
        table_title = "Yanex Archived Experiments" if archived else "Yanex Experiments"
        formatter.print_experiments_table(experiments, title=table_title)

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
