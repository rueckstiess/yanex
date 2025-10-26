"""
Unarchive experiments - move them back from archived directory.
"""

import click

from ..error_handling import (
    BulkOperationReporter,
    CLIErrorHandler,
)
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)


@click.command("unarchive")
@click.argument("experiment_identifiers", nargs=-1)
@experiment_filter_options(
    include_ids=False, include_archived=False, include_limit=False
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def unarchive_experiments(
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
    Unarchive experiments by moving them back to experiments directory.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    Examples:
    \\b
        yanex unarchive exp1 exp2            # Unarchive specific experiments
        yanex unarchive -s completed         # Unarchive all completed experiments
        yanex unarchive -n "*training*"      # Unarchive experiments with "training" in name
        yanex unarchive -t experiment-v1     # Unarchive experiments with specific tag
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
        list(experiment_identifiers), has_filters, "unarchive"
    )

    # Parse time specifications
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Find experiments to unarchive
    if experiment_identifiers:
        # Unarchive specific experiments by ID/name
        experiments = find_experiments_by_identifiers(
            filter_obj,
            list(experiment_identifiers),
            archived=True,  # Only search archived experiments
        )
    else:
        # Unarchive experiments by filter criteria
        experiments = find_experiments_by_filters(
            filter_obj,
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            archived=True,  # Only search archived experiments
        )

    # Filter to only archived experiments
    experiments = [exp for exp in experiments if exp.get("archived", False)]

    if not experiments:
        click.echo("No archived experiments found to unarchive.")
        return

    # Show experiments and get confirmation
    if not confirm_experiment_operation(
        experiments, "unarchive", force, "unarchived", default_yes=True
    ):
        click.echo("Unarchive operation cancelled.")
        return

    # Unarchive experiments using centralized reporter
    click.echo(f"Unarchiving {len(experiments)} experiment(s)...")
    reporter = BulkOperationReporter("unarchive")

    for exp in experiments:
        experiment_id = exp["id"]
        exp_name = exp.get("name", "[unnamed]")

        try:
            filter_obj.manager.storage.unarchive_experiment(experiment_id)
            reporter.report_success(experiment_id, exp_name)
        except Exception as e:
            reporter.report_failure(experiment_id, e, exp_name)

    # Report summary and exit with appropriate code
    reporter.report_summary()
    if reporter.has_failures():
        ctx.exit(reporter.get_exit_code())
