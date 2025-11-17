"""
ID command implementation for yanex CLI.

Returns experiment IDs based on filter criteria, primarily for bash substitution.
"""

import json

import click

from ..error_handling import CLIErrorHandler
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options


@click.command()
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["csv", "newline", "json"], case_sensitive=False),
    default="csv",
    help="Output format: csv (default, quoted comma-separated), newline (one per line), or json (array)",
)
@experiment_filter_options(include_ids=False, include_archived=True, include_limit=True)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def get_experiment_ids(
    ctx: click.Context,
    output_format: str,
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
    Get experiment IDs matching filter criteria.

    Returns experiment IDs in the specified format, primarily for use in bash
    substitution commands. The default CSV format is wrapped in double quotes
    for easy use in shell commands.

    Examples:

      # Get IDs of all training experiments (CSV format, default)
      yanex id --tag training

      # Use in bash substitution for dependencies
      yanex run evaluate.py --depends-on model=$(yanex id --tag training)

      # Get IDs of staged experiments for a specific script
      yanex run train.py -D dataprep=$(yanex id --status staged --script dataprep.py)

      # Get IDs in newline format
      yanex id --tag production --format newline

      # Get IDs in JSON format
      yanex id --status completed --format json

      # Get last 5 experiment IDs
      yanex id -l 5

      # Get IDs of failed experiments
      yanex id --status failed
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
            click.echo("Filtering experiments...", err=True)
            if status:
                click.echo(f"  Status: {status}", err=True)
            if name_pattern:
                click.echo(f"  Name pattern: {name_pattern}", err=True)
            if script_pattern:
                click.echo(f"  Script pattern: {script_pattern}", err=True)
            if tags:
                click.echo(f"  Tags: {', '.join(tags)}", err=True)
            if started_after:
                click.echo(f"  Started after: {started_after}", err=True)
            if started_before:
                click.echo(f"  Started before: {started_before}", err=True)
            if ended_after:
                click.echo(f"  Ended after: {ended_after}", err=True)
            if ended_before:
                click.echo(f"  Ended before: {ended_before}", err=True)

        # Create filter and apply criteria
        experiment_filter = ExperimentFilter()

        experiments = experiment_filter.filter_experiments(
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            script_pattern=script_pattern,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            limit=limit,
            include_all=limit is None,
            archived=archived,
        )

        # Filter experiments based on archived flag
        if archived:
            experiments = [exp for exp in experiments if exp.get("archived", False)]
        else:
            experiments = [exp for exp in experiments if not exp.get("archived", False)]

        # Apply limit after filtering by archived status if needed
        if limit is not None:
            experiments = experiments[:limit]

        # Extract experiment IDs
        experiment_ids = [exp["experiment_id"] for exp in experiments]

        if verbose:
            click.echo(f"Found {len(experiment_ids)} matching experiments", err=True)

        # Output in requested format
        if output_format == "csv":
            # CSV format with double quotes (for bash substitution)
            output = f'"{",".join(experiment_ids)}"'
            click.echo(output)
        elif output_format == "newline":
            # One ID per line
            for exp_id in experiment_ids:
                click.echo(exp_id)
        elif output_format == "json":
            # JSON array format
            output = json.dumps(experiment_ids)
            click.echo(output)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"Error getting experiment IDs: {e}", err=True)
        if verbose:
            import traceback

            click.echo(traceback.format_exc(), err=True)
        raise click.Abort()
