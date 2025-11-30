"""
Main CLI entry point for yanex.
"""

import click

from .commands.archive import archive_experiments
from .commands.compare import compare_experiments
from .commands.delete import delete_experiments
from .commands.get import get_field
from .commands.list import list_experiments
from .commands.migrate import migrate_experiments
from .commands.open import open_experiment
from .commands.run import run
from .commands.show import show_experiment
from .commands.ui import ui
from .commands.unarchive import unarchive_experiments
from .commands.update import update_experiments


@click.group()
@click.version_option(version="0.1.0", prog_name="yanex")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    Yet Another Experiment Tracker - A lightweight experiment tracking harness.

    Use yanex to track experiments with automatic versioning, parameter management
    and artifact storage.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# Register commands
cli.add_command(run)
cli.add_command(list_experiments, name="list")
cli.add_command(show_experiment, name="show")
cli.add_command(get_field, name="get")
cli.add_command(open_experiment, name="open")
cli.add_command(archive_experiments, name="archive")
cli.add_command(delete_experiments, name="delete")
cli.add_command(migrate_experiments, name="migrate")
cli.add_command(unarchive_experiments, name="unarchive")
cli.add_command(update_experiments, name="update")
cli.add_command(compare_experiments, name="compare")
cli.add_command(ui, name="ui")


if __name__ == "__main__":
    cli()
