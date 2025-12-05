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


@click.command("unarchive")
@click.argument("experiment_identifiers", nargs=-1)
@format_options()
@experiment_filter_options(
    include_ids=True, include_archived=False, include_limit=False
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def unarchive_experiments(
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
    force: bool,
):
    """
    Unarchive experiments by moving them back to experiments directory.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    Supports multiple output formats:

    \b
      --format json      Output result as JSON (for scripting/AI processing)
      --format csv       Output result as CSV (for data analysis)
      --format markdown  Output result as markdown

    Examples:

    \b
        yanex unarchive exp1 exp2            # Unarchive specific experiments
        yanex unarchive -s completed         # Unarchive all completed experiments
        yanex unarchive -n "*training*"      # Unarchive experiments with "training" in name
        yanex unarchive -t experiment-v1     # Unarchive experiments with specific tag
        yanex unarchive -s completed --format json  # Unarchive and output result as JSON
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
            ids=ids_list,
            status=status,
            name=name_pattern,
            tags=list(tags) if tags else None,
            script_pattern=script_pattern,
            started_after=started_after_dt,
            started_before=started_before_dt,
            ended_after=ended_after_dt,
            ended_before=ended_before_dt,
            archived=True,  # Only search archived experiments
        )

    # Filter to only archived experiments
    experiments = [exp for exp in experiments if exp.get("archived", False)]

    if not experiments:
        echo_format_info("No archived experiments found to unarchive.", fmt)
        return

    # For machine-readable output, skip confirmation
    effective_force = force or is_machine_format(fmt)

    # Show experiments and get confirmation
    if not confirm_experiment_operation(
        experiments, "unarchive", effective_force, "unarchived", default_yes=True
    ):
        echo_format_info("Unarchive operation cancelled.", fmt)
        return

    # Unarchive experiments using centralized reporter with output format
    echo_format_info(f"Unarchiving {len(experiments)} experiment(s)...", fmt)
    reporter = BulkOperationReporter("unarchive", output_format=fmt)

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
