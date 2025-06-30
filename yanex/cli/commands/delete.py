"""
Delete experiments permanently.
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


@click.command("delete")
@click.argument("experiment_identifiers", nargs=-1)
@click.option(
    "--status",
    type=click.Choice(EXPERIMENT_STATUSES),
    help="Delete experiments with specific status",
)
@click.option(
    "--name",
    "name_pattern",
    help="Delete experiments matching name pattern (glob syntax)",
)
@click.option(
    "--tag", "tags", multiple=True, help="Delete experiments with ALL specified tags"
)
@click.option(
    "--started-after",
    help="Delete experiments started after date/time (e.g., '2025-01-01', 'yesterday', '1 week ago')",
)
@click.option("--started-before", help="Delete experiments started before date/time")
@click.option("--ended-after", help="Delete experiments ended after date/time")
@click.option("--ended-before", help="Delete experiments ended before date/time")
@click.option(
    "--archived",
    is_flag=True,
    help="Delete from archived experiments (default: delete from regular experiments)",
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def delete_experiments(
    ctx,
    experiment_identifiers: tuple,
    status: Optional[str],
    name_pattern: Optional[str],
    tags: tuple,
    started_after: Optional[str],
    started_before: Optional[str],
    ended_after: Optional[str],
    ended_before: Optional[str],
    archived: bool,
    force: bool,
):
    """
    Permanently delete experiments.

    ⚠️  WARNING: This operation cannot be undone!

    EXPERIMENT_IDENTIFIERS can be experiment IDs or names.
    If no identifiers provided, experiments are filtered by options.

    Examples:
    \\b
        yanex delete exp1 exp2               # Delete specific experiments
        yanex delete --status failed         # Delete all failed experiments
        yanex delete --archived --ended-before "6 months ago"
        yanex delete --name "*test*"         # Delete experiments with "test" in name
        yanex delete --tag temp              # Delete experiments with "temp" tag
    """
    try:
        filter_obj = ExperimentFilter()

        # Validate mutually exclusive targeting
        has_identifiers = len(experiment_identifiers) > 0
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

        # Find experiments to delete
        if experiment_identifiers:
            # Delete specific experiments by ID/name
            experiments = find_experiments_by_identifiers(
                filter_obj, list(experiment_identifiers), include_archived=archived
            )
        else:
            # Delete experiments by filter criteria

            experiments = find_experiments_by_filters(
                filter_obj,
                status=status,
                name_pattern=name_pattern,
                tags=list(tags) if tags else None,
                started_after=started_after_dt,
                started_before=started_before_dt,
                ended_after=ended_after_dt,
                ended_before=ended_before_dt,
                include_archived=archived,
            )

        # Filter experiments based on archived flag
        if archived:
            experiments = [exp for exp in experiments if exp.get("archived", False)]
        else:
            experiments = [exp for exp in experiments if not exp.get("archived", False)]

        if not experiments:
            location = "archived" if archived else "regular"
            click.echo(f"No {location} experiments found to delete.")
            return

        # Show experiments and get confirmation (always required for deletion)
        operation_verb = "permanently deleted"
        if not confirm_experiment_operation(
            experiments, "delete", force, operation_verb
        ):
            click.echo("Delete operation cancelled.")
            return

        # Additional warning for bulk deletions
        if len(experiments) > 1 and not force:
            click.echo()
            click.echo("⚠️  You are about to permanently delete multiple experiments.")
            click.echo("   This action cannot be undone!")
            if not click.confirm("Are you absolutely sure?", default=False):
                click.echo("Delete operation cancelled.")
                return

        # Delete experiments
        click.echo(f"Deleting {len(experiments)} experiment(s)...")

        success_count = 0
        for exp in experiments:
            try:
                experiment_id = exp["id"]

                if exp.get("archived", False):
                    # Delete from archived directory
                    filter_obj.manager.storage.delete_archived_experiment(experiment_id)
                else:
                    # Delete from regular directory
                    filter_obj.manager.storage.delete_experiment(experiment_id)

                # Show progress
                exp_name = exp.get("name", "[unnamed]")
                click.echo(f"  ✓ Deleted {experiment_id} ({exp_name})")
                success_count += 1

            except Exception as e:
                exp_name = exp.get("name", "[unnamed]")
                click.echo(
                    f"  ✗ Failed to delete {experiment_id} ({exp_name}): {e}", err=True
                )

        # Summary
        if success_count == len(experiments):
            click.echo(f"Successfully deleted {success_count} experiment(s).")
        else:
            failed_count = len(experiments) - success_count
            click.echo(
                f"Deleted {success_count} experiment(s), {failed_count} failed.",
                err=True,
            )
            ctx.exit(1)

    except click.ClickException:
        raise  # Re-raise ClickException to show proper error message
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
