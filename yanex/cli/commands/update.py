"""
Update experiment metadata - name, description, status, and tags.
"""

import click

from ...core.constants import EXPERIMENT_STATUSES
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


@click.command("update")
@click.argument("experiment_identifiers", nargs=-1)
@format_options()
@experiment_filter_options(include_ids=True, include_archived=True, include_limit=False)
@click.option(
    "--set-name", "new_name", help="Set experiment name (use empty string to clear)"
)
@click.option(
    "--set-description",
    "new_description",
    help="Set experiment description (use empty string to clear)",
)
@click.option(
    "--set-status",
    "new_status",
    type=click.Choice(EXPERIMENT_STATUSES),
    help="Set experiment status",
)
@click.option(
    "--add-tag", "add_tags", multiple=True, help="Add tag to experiment(s) (repeatable)"
)
@click.option(
    "--remove-tag",
    "remove_tags",
    multiple=True,
    help="Remove tag from experiment(s) (repeatable)",
)
@click.option(
    "--force", is_flag=True, help="Skip confirmation prompt for bulk operations"
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be updated without making changes"
)
@click.pass_context
@CLIErrorHandler.handle_cli_errors
def update_experiments(
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
    new_name: str | None,
    new_description: str | None,
    new_status: str | None,
    add_tags: tuple,
    remove_tags: tuple,
    force: bool,
    dry_run: bool,
):
    """
    Update experiment metadata including name, description, status, and tags.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    For bulk updates, use filter options instead of identifiers.

    Supports multiple output formats:

    \b
      --format json      Output result as JSON (for scripting/AI processing)
      --format csv       Output result as CSV (for data analysis)
      --format markdown  Output result as markdown

    Examples:

    \b
        # Update single experiment
        yanex update exp1 --set-name "New Name" --set-description "New description"
        yanex update "my experiment" --add-tag production --remove-tag testing
        yanex update exp123 --set-status completed

        # Bulk updates with filters
        yanex update -s failed --set-description "Failed batch run"
        yanex update -t experimental --remove-tag experimental --add-tag archived
        yanex update --ended-before "1 week ago" --add-tag old-runs

        # Preview changes without applying
        yanex update exp1 --set-name "New Name" --dry-run

        # Output result as JSON
        yanex update exp1 --add-tag test --format json
    """
    # Resolve output format from --format option or legacy flags
    fmt = resolve_output_format(output_format, json_flag, csv_flag, markdown_flag)
    filter_obj = ExperimentFilter()

    # Parse comma-separated IDs into a list
    ids_list = None
    if ids:
        ids_list = [id_str.strip() for id_str in ids.split(",") if id_str.strip()]

    # Validate that we have something to update
    if not any(
        [
            new_name is not None,
            new_description is not None,
            new_status,
            add_tags,
            remove_tags,
        ]
    ):
        raise click.ClickException(
            "Must specify at least one update option (--set-name, --set-description, --set-status, --add-tag, --remove-tag)"
        )

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
        list(experiment_identifiers), has_filters, "update"
    )

    # Parse time specifications
    started_after_dt, started_before_dt, ended_after_dt, ended_before_dt = (
        CLIErrorHandler.parse_time_filters(
            started_after, started_before, ended_after, ended_before
        )
    )

    # Find experiments to update
    if experiment_identifiers:
        # Update specific experiments by ID/name
        experiments = find_experiments_by_identifiers(
            filter_obj, list(experiment_identifiers), archived=archived
        )
    else:
        # Update experiments by filter criteria
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

    # Filter based on archived flag
    if archived:
        experiments = [exp for exp in experiments if exp.get("archived", False)]
    else:
        experiments = [exp for exp in experiments if not exp.get("archived", False)]

    if not experiments:
        location = "archived" if archived else "regular"
        message = f"No {location} experiments found to update."

        if experiment_identifiers:
            # When using identifiers, not finding experiments is an error
            raise click.ClickException(message)
        else:
            # When using filters, not finding experiments is just informational
            echo_format_info(message, fmt)
            return

    # Prepare update dictionary
    updates = {}

    if new_name is not None:
        updates["name"] = new_name
    if new_description is not None:
        updates["description"] = new_description
    if new_status:
        updates["status"] = new_status
    if add_tags:
        updates["add_tags"] = list(add_tags)
    if remove_tags:
        updates["remove_tags"] = list(remove_tags)

    # Show what will be updated (only for console output)
    if not is_machine_format(fmt):
        click.echo("Updates to apply:")
        for key, value in updates.items():
            if key == "add_tags":
                click.echo(f"  Add tags: {', '.join(value)}")
            elif key == "remove_tags":
                click.echo(f"  Remove tags: {', '.join(value)}")
            elif key in ["name", "description"] and value == "":
                click.echo(f"  Clear {key}")
            else:
                click.echo(f"  Set {key}: {value}")
        click.echo()

    # For machine-readable output, skip confirmation
    effective_force = force or is_machine_format(fmt)

    # Show experiments and get confirmation for bulk operations or dry run
    if len(experiments) > 1 or dry_run:
        if not confirm_experiment_operation(
            experiments, "update", effective_force or dry_run, "updated"
        ):
            echo_format_info("Update operation cancelled.", fmt)
            return

    if dry_run:
        echo_format_info("Dry run completed. No changes were made.", fmt)
        return

    # Update experiments using centralized reporter with output format
    echo_format_info(f"Updating {len(experiments)} experiment(s)...", fmt)
    reporter = BulkOperationReporter("update", output_format=fmt)

    for exp in experiments:
        experiment_id = exp["id"]
        exp_name = exp.get("name", "[unnamed]")

        try:
            filter_obj.manager.storage.update_experiment_metadata(
                experiment_id, updates, include_archived=archived
            )
            reporter.report_success(experiment_id, exp_name)
        except Exception as e:
            reporter.report_failure(experiment_id, e, exp_name)

    # Report summary and exit with appropriate code
    reporter.report_summary()
    if reporter.has_failures():
        ctx.exit(reporter.get_exit_code())
