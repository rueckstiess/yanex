"""
Update experiment metadata - name, description, status, and tags.
"""

from typing import Optional

import click

from ...core.constants import EXPERIMENT_STATUSES
from ..filters import ExperimentFilter, parse_time_spec
from .confirm import (
    confirm_experiment_operation,
    find_experiments_by_filters,
    find_experiments_by_identifiers,
)


@click.command("update")
@click.argument("experiment_identifiers", nargs=-1)
@click.option(
    "--status",
    "filter_status",
    type=click.Choice(EXPERIMENT_STATUSES),
    help="Filter experiments by status for bulk updates",
)
@click.option(
    "--name",
    "filter_name_pattern",
    help="Filter experiments by name pattern for bulk updates (glob syntax)",
)
@click.option(
    "--tag",
    "filter_tags",
    multiple=True,
    help="Filter experiments by tag for bulk updates (experiments must have ALL specified tags)",
)
@click.option(
    "--started-after",
    help="Filter experiments started after date/time for bulk updates",
)
@click.option(
    "--started-before",
    help="Filter experiments started before date/time for bulk updates",
)
@click.option(
    "--ended-after", help="Filter experiments ended after date/time for bulk updates"
)
@click.option(
    "--ended-before", help="Filter experiments ended before date/time for bulk updates"
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
@click.option("--archived", is_flag=True, help="Update archived experiments")
@click.option(
    "--force", is_flag=True, help="Skip confirmation prompt for bulk operations"
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be updated without making changes"
)
@click.pass_context
def update_experiments(
    ctx,
    experiment_identifiers: tuple,
    filter_status: Optional[str],
    filter_name_pattern: Optional[str],
    filter_tags: tuple,
    started_after: Optional[str],
    started_before: Optional[str],
    ended_after: Optional[str],
    ended_before: Optional[str],
    new_name: Optional[str],
    new_description: Optional[str],
    new_status: Optional[str],
    add_tags: tuple,
    remove_tags: tuple,
    archived: bool,
    force: bool,
    dry_run: bool,
):
    """
    Update experiment metadata including name, description, status, and tags.

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    For bulk updates, use filter options instead of identifiers.

    Examples:
    \\b
        # Update single experiment
        yanex update exp1 --set-name "New Name" --set-description "New description"
        yanex update "my experiment" --add-tag production --remove-tag testing
        yanex update exp123 --set-status completed

        # Bulk updates with filters
        yanex update --status failed --set-description "Failed batch run"
        yanex update --tag experimental --remove-tag experimental --add-tag archived
        yanex update --ended-before "1 week ago" --add-tag old-runs

        # Preview changes without applying
        yanex update exp1 --set-name "New Name" --dry-run
    """
    try:
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
            click.echo(
                "Error: Must specify at least one update option (--set-name, --set-description, --set-status, --add-tag, --remove-tag)",
                err=True,
            )
            ctx.exit(1)

        # Validate mutually exclusive targeting
        has_identifiers = len(experiment_identifiers) > 0
        has_filters = any(
            [
                filter_status,
                filter_name_pattern,
                filter_tags,
                started_after,
                started_before,
                ended_after,
                ended_before,
            ]
        )

        if has_identifiers and has_filters:
            click.echo(
                "Error: Cannot use both experiment identifiers and filter options. Choose one approach.",
                err=True,
            )
            ctx.exit(1)

        if not has_identifiers and not has_filters:
            click.echo(
                "Error: Must specify either experiment identifiers or filter options",
                err=True,
            )
            ctx.exit(1)

        # Parse time specifications
        started_after_dt = parse_time_spec(started_after) if started_after else None
        started_before_dt = parse_time_spec(started_before) if started_before else None
        ended_after_dt = parse_time_spec(ended_after) if ended_after else None
        ended_before_dt = parse_time_spec(ended_before) if ended_before else None

        # Find experiments to update
        if experiment_identifiers:
            # Update specific experiments by ID/name
            experiments = find_experiments_by_identifiers(
                filter_obj, list(experiment_identifiers), include_archived=archived
            )
        else:
            # Update experiments by filter criteria
            experiments = find_experiments_by_filters(
                filter_obj,
                status=filter_status,
                name_pattern=filter_name_pattern,
                tags=list(filter_tags) if filter_tags else None,
                started_after=started_after_dt,
                started_before=started_before_dt,
                ended_after=ended_after_dt,
                ended_before=ended_before_dt,
                include_archived=archived,
            )

        # Filter based on archived flag
        if archived:
            experiments = [exp for exp in experiments if exp.get("archived", False)]
        else:
            experiments = [exp for exp in experiments if not exp.get("archived", False)]

        if not experiments:
            location = "archived" if archived else "regular"
            click.echo(f"No {location} experiments found to update.")
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

        # Show what will be updated
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

        # Show experiments and get confirmation for bulk operations or dry run
        if len(experiments) > 1 or dry_run:
            if not confirm_experiment_operation(
                experiments, "update", force or dry_run, "updated"
            ):
                click.echo("Update operation cancelled.")
                return

        if dry_run:
            click.echo("Dry run completed. No changes were made.")
            return

        # Update experiments
        click.echo(f"Updating {len(experiments)} experiment(s)...")

        success_count = 0
        for exp in experiments:
            try:
                experiment_id = exp["id"]
                filter_obj.manager.storage.update_experiment_metadata(
                    experiment_id, updates, include_archived=archived
                )

                # Show progress
                exp_name = exp.get("name", "[unnamed]")
                click.echo(f"  ✓ Updated {experiment_id} ({exp_name})")
                success_count += 1

            except Exception as e:
                exp_name = exp.get("name", "[unnamed]")
                click.echo(
                    f"  ✗ Failed to update {experiment_id} ({exp_name}): {e}", err=True
                )

        # Summary
        if success_count == len(experiments):
            click.echo(f"Successfully updated {success_count} experiment(s).")
        else:
            failed_count = len(experiments) - success_count
            click.echo(
                f"Updated {success_count} experiment(s), {failed_count} failed.",
                err=True,
            )
            ctx.exit(1)

    except click.ClickException:
        raise  # Re-raise ClickException to show proper error message
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
