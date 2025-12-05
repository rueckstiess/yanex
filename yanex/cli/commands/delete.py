"""
Delete experiments permanently.
"""

import click

from ..error_handling import (
    BulkOperationReporter,
    CLIErrorHandler,
)
from ..filters import ExperimentFilter
from ..filters.arguments import experiment_filter_options
from ..formatters import (
    echo_format_info,
    format_options,
    is_machine_format,
    resolve_output_format,
)
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)


@click.command("delete")
@click.argument("experiment_identifiers", nargs=-1)
@format_options()
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=False)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def delete_experiments(
    ctx,
    experiment_identifiers: tuple,
    output_format: str | None,
    json_flag: bool,
    csv_flag: bool,
    markdown_flag: bool,
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
    force: bool,
):
    """
    Permanently delete experiments.

    WARNING: This operation cannot be undone!

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    Supports multiple output formats:

    \b
      --format json      Output result as JSON (for scripting/AI processing)
      --format csv       Output result as CSV (for data analysis)
      --format markdown  Output result as markdown

    Examples:

    \b
        yanex delete exp1 exp2               # Delete specific experiments
        yanex delete -s failed               # Delete all failed experiments
        yanex delete -a --ended-before "6 months ago"
        yanex delete -n "*test*"             # Delete experiments with "test" in name
        yanex delete -t temp                 # Delete experiments with "temp" tag
        yanex delete -s failed --format json # Delete and output result as JSON
    """
    # Resolve output format from --format option or legacy flags
    fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)
    filter_obj = ExperimentFilter()

    # Parse comma-separated IDs into a list
    ids_list = None
    if ids:
        ids_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

    # Validate mutually exclusive targeting
    # Note: name_pattern="" is a valid filter for unnamed experiments
    has_filters = any(
        [
            ids_list,
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
        list(experiment_identifiers), has_filters, "delete"
    )

    # Parse time specifications
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Find experiments to delete
    if experiment_identifiers:
        # Delete specific experiments by ID/name
        experiments = find_experiments_by_identifiers(
            filter_obj, list(experiment_identifiers), archived=archived
        )
    else:
        # Delete experiments by filter criteria
        experiments = find_experiments_by_filters(
            filter_obj,
            ids=ids_list,
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            script_pattern=script_pattern,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            archived=archived,
        )

    # Filter experiments based on archived flag
    if archived:
        experiments = [exp for exp in experiments if exp.get("archived", False)]
    else:
        experiments = [exp for exp in experiments if not exp.get("archived", False)]

    if not experiments:
        location = "archived" if archived else "regular"
        echo_format_info(f"No {location} experiments found to delete.", fmt)
        return

    # For machine-readable output, skip confirmation
    effective_force = force or is_machine_format(fmt)

    # Show experiments and get confirmation (always required for deletion)
    operation_verb = "permanently deleted"
    if not confirm_experiment_operation(
        experiments, "delete", effective_force, operation_verb
    ):
        echo_format_info("Delete operation cancelled.", fmt)
        return

    # Additional warning for bulk deletions (only in console mode)
    if len(experiments) > 1 and not effective_force:
        click.echo()
        click.echo("WARNING: You are about to permanently delete multiple experiments.")
        click.echo("   This action cannot be undone!")
        if not click.confirm("Are you absolutely sure?", default=False):
            echo_format_info("Delete operation cancelled.", fmt)
            return

    # Delete experiments using centralized reporter with output format
    echo_format_info(f"Deleting {len(experiments)} experiment(s)...", fmt)
    reporter = BulkOperationReporter("delete", output_format=fmt)

    for exp in experiments:
        experiment_id = exp["id"]
        exp_name = exp.get("name", "[unnamed]")

        try:
            if exp.get("archived", False):
                # Delete from archived directory
                filter_obj.manager.storage.delete_archived_experiment(experiment_id)
            else:
                # Delete from regular directory
                filter_obj.manager.storage.delete_experiment(experiment_id)

            reporter.report_success(experiment_id, exp_name)
        except Exception as e:
            reporter.report_failure(experiment_id, e, exp_name)

    # Report summary and exit with appropriate code
    reporter.report_summary()
    if reporter.has_failures():
        ctx.exit(reporter.get_exit_code())
