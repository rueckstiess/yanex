"""
Archive experiments - move them to archived directory.
"""

import click

from ...core.constants import EXPERIMENT_STATUSES
from ..error_handling import (
    BulkOperationReporter,
    CLIErrorHandler,
)
from ..filters import ExperimentFilter
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)


@click.command("archive")
@click.argument("experiment_identifiers", nargs=-1)
@click.option(
    "--status",
    type=click.Choice(EXPERIMENT_STATUSES),
    help="Archive experiments with specific status",
)
@click.option(
    "--name",
    "name_pattern",
    help="Archive experiments matching name pattern (glob syntax)",
)
@click.option(
    "--tag", "tags", multiple=True, help="Archive experiments with ALL specified tags"
)
@click.option(
    "--started-after",
    help="Archive experiments started after date/time (e.g., '2025-01-01', 'yesterday', '1 week ago')",
)
@click.option("--started-before", help="Archive experiments started before date/time")
@click.option("--ended-after", help="Archive experiments ended after date/time")
@click.option("--ended-before", help="Archive experiments ended before date/time")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def archive_experiments(
    ctx,
    experiment_identifiers: tuple,
    status: str | None,
    name_pattern: str | None,
    tags: tuple,
    started_after: str | None,
    started_before: str | None,
    ended_after: str | None,
    ended_before: str | None,
    force: bool,
):
    """
    Archive experiments by moving them to archived directory.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    Examples:
    \\b
        yanex archive exp1 exp2              # Archive specific experiments
        yanex archive --status failed        # Archive all failed experiments
        yanex archive --status completed --ended-before "1 month ago"
        yanex archive --name "*training*"    # Archive experiments with "training" in name
        yanex archive --tag experiment-v1   # Archive experiments with specific tag
    """
    filter_obj = ExperimentFilter()

    # Validate mutually exclusive targeting
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

    CLIErrorHandler.validate_targeting_options(
        list(experiment_identifiers), has_filters, "archive"
    )

    # Parse time specifications
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Find experiments to archive
    if experiment_identifiers:
        # Archive specific experiments by ID/name
        experiments = find_experiments_by_identifiers(
            filter_obj,
            list(experiment_identifiers),
            include_archived=False,  # Can't archive already archived experiments
        )
    else:
        # Archive experiments by filter criteria
        experiments = find_experiments_by_filters(
            filter_obj,
            status=status,
            name_pattern=name_pattern,
            tags=list(tags) if tags else None,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            include_archived=False,  # Can't archive already archived experiments
        )

    # Filter out already archived experiments (extra safety)
    experiments = [exp for exp in experiments if not exp.get("archived", False)]

    if not experiments:
        click.echo("No experiments found to archive.")
        return

    # Show experiments and get confirmation
    if not confirm_experiment_operation(
        experiments, "archive", force, default_yes=True
    ):
        click.echo("Archive operation cancelled.")
        return

    # Archive experiments using centralized reporter
    click.echo(f"Archiving {len(experiments)} experiment(s)...")
    reporter = BulkOperationReporter("archive")

    for exp in experiments:
        experiment_id = exp["id"]
        exp_name = exp.get("name", "[unnamed]")

        try:
            filter_obj.manager.storage.archive_experiment(experiment_id)
            reporter.report_success(experiment_id, exp_name)
        except Exception as e:
            reporter.report_failure(experiment_id, e, exp_name)

    # Report summary and exit with appropriate code
    reporter.report_summary()
    if reporter.has_failures():
        ctx.exit(reporter.get_exit_code())
