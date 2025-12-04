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


@click.command("update")
@click.argument("experiment_identifiers", nargs=-1)
@output_mode_options
@experiment_filter_options(
    include_ids=False, include_archived=True, include_limit=False
)
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
      --json      Output result as JSON (for scripting/AI processing)
      --csv       Output result as CSV (for data analysis)
      --markdown  Output result as markdown

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
        yanex update exp1 --add-tag test --json
    """
    # Validate output mode flags
    validate_output_mode_flags(json_output, csv_output, markdown_output)
    output_mode = get_output_mode(json_output, csv_output, markdown_output)
    filter_obj = ExperimentFilter()

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
            echo_info(message, output_mode)
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
    if not is_machine_output(output_mode):
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
    effective_force = force or is_machine_output(output_mode)

    # Show experiments and get confirmation for bulk operations or dry run
    if len(experiments) > 1 or dry_run:
        if not confirm_experiment_operation(
            experiments, "update", effective_force or dry_run, "updated"
        ):
            echo_info("Update operation cancelled.", output_mode)
            return

    if dry_run:
        echo_info("Dry run completed. No changes were made.", output_mode)
        return

    # Update experiments using centralized reporter with output mode
    echo_info(f"Updating {len(experiments)} experiment(s)...", output_mode)
    reporter = BulkOperationReporter("update", output_mode=output_mode)

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
