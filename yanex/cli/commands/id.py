"""
Command to output experiment IDs matching filters.

This command is designed for composition with other commands,
particularly for dependency sweeps using shell substitution.
"""

import click

from yanex.cli.arguments import experiment_filter_options
from yanex.cli.filters import ExperimentFilter


@click.command("id")
@experiment_filter_options
@click.option(
    "--limit",
    type=int,
    help="Limit number of results",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["line", "csv", "json"]),
    default="csv",
    help="Output format (default: csv for easy shell composition)",
)
@click.pass_context
def id_command(
    ctx,
    limit: int | None,
    output_format: str,
    **filter_kwargs,
):
    """
    Output experiment IDs matching filters (for composition with other commands).

    This command is designed to be used with shell substitution for dependency
    sweeps and batch operations.

    Examples:

      \b
      # Get IDs of completed training experiments
      yanex id --script train.py --status completed

      \b
      # Use in dependency sweep (comma-separated format)
      yanex run evaluate.py -d training=$(yanex id --script train.py --limit 3)

      \b
      # Get IDs in JSON format
      yanex id --script train.py --format json

      \b
      # Get most recent 5 experiments
      yanex id --limit 5

      \b
      # Filter by multiple criteria
      yanex id --script train.py --tag experiment --status completed

    Output formats:
      line  : One ID per line (for iteration in shell loops)
      csv   : Comma-separated (for dependency sweeps)
      json  : JSON array (for programmatic use)
    """
    try:
        # Create filter (this creates a manager internally)
        filter_obj = ExperimentFilter()

        # Apply filters
        experiments = filter_obj.filter_experiments(filter_kwargs)

        # Apply limit if specified
        if limit:
            experiments = experiments[:limit]

        # Extract IDs
        experiment_ids = [exp["id"] for exp in experiments]

        # Output in requested format
        if not experiment_ids:
            # No experiments found - output nothing (allows safe shell composition)
            pass
        elif output_format == "csv":
            click.echo(",".join(experiment_ids))
        elif output_format == "json":
            import json

            click.echo(json.dumps(experiment_ids))
        else:  # line format
            for exp_id in experiment_ids:
                click.echo(exp_id)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
