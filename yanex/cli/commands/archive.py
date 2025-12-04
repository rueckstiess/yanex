"""
Archive experiments - move them to archived directory.
"""

import click

from ..error_handling import (
    BulkOperationReporter,
    CLIErrorHandler,
)
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from ..formatters import (
    echo_info,
    get_output_mode,
    is_machine_output,
    output_mode_options,
    validate_output_mode_flags,
)
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)


@click.command("archive")
@click.argument("experiment_identifiers", nargs=-1)
@output_mode_options
@experiment_filter_options(
    include_ids=False, include_archived=False, include_limit=False
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def archive_experiments(
    ctx,
    experiment_identifiers: tuple,
    json_output: bool,
    csv_output: bool,
    markdown_output: bool,
    status: str | None,
    name_pattern: str | None,
    tags: tuple,
    script_pattern: str | None,
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

    Supports multiple output formats:

    \b
      --json      Output result as JSON (for scripting/AI processing)
      --csv       Output result as CSV (for data analysis)
      --markdown  Output result as markdown

    Examples:

    \b
        yanex archive exp1 exp2              # Archive specific experiments
        yanex archive -s failed              # Archive all failed experiments
        yanex archive -s completed --ended-before "1 month ago"
        yanex archive -s failed --json       # Archive and output result as JSON
    """
    # Validate output mode flags
    validate_output_mode_flags(json_output, csv_output, markdown_output)
    output_mode = get_output_mode(json_output, csv_output, markdown_output)

    filter_obj = ExperimentFilter()

    # Validate mutually exclusive targeting
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
            archived=False,  # Can't archive already archived experiments
        )
    else:
        # Archive experiments by filter criteria
        experiments = find_experiments_by_filters(
            filter_obj,
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            script_pattern=script_pattern,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            archived=False,  # Can't archive already archived experiments
        )

    # Filter out already archived experiments (extra safety)
    experiments = [exp for exp in experiments if not exp.get("archived", False)]

    if not experiments:
        echo_info("No experiments found to archive.", output_mode)
        return

    # For machine-readable output, skip confirmation
    effective_force = force or is_machine_output(output_mode)

    # Show experiments and get confirmation
    if not confirm_experiment_operation(
        experiments, "archive", effective_force, default_yes=True
    ):
        echo_info("Archive operation cancelled.", output_mode)
        return

    # Archive experiments using centralized reporter with output mode
    echo_info(f"Archiving {len(experiments)} experiment(s)...", output_mode)
    reporter = BulkOperationReporter("archive", output_mode=output_mode)

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
