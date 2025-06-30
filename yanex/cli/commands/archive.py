"""
Archive experiments - move them to archived directory.
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
def archive_experiments(
    ctx,
    experiment_identifiers: tuple,
    status: Optional[str],
    name_pattern: Optional[str],
    tags: tuple,
    started_after: Optional[str],
    started_before: Optional[str],
    ended_after: Optional[str],
    ended_before: Optional[str],
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

        # Archive experiments
        click.echo(f"Archiving {len(experiments)} experiment(s)...")

        success_count = 0
        for exp in experiments:
            try:
                experiment_id = exp["id"]
                filter_obj.manager.storage.archive_experiment(experiment_id)

                # Show progress
                exp_name = exp.get("name", "[unnamed]")
                click.echo(f"  ✓ Archived {experiment_id} ({exp_name})")
                success_count += 1

            except Exception as e:
                exp_name = exp.get("name", "[unnamed]")
                click.echo(
                    f"  ✗ Failed to archive {experiment_id} ({exp_name}): {e}", err=True
                )

        # Summary
        if success_count == len(experiments):
            click.echo(f"Successfully archived {success_count} experiment(s).")
        else:
            failed_count = len(experiments) - success_count
            click.echo(
                f"Archived {success_count} experiment(s), {failed_count} failed.",
                err=True,
            )
            ctx.exit(1)

    except click.ClickException:
        raise  # Re-raise ClickException to show proper error message
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)
