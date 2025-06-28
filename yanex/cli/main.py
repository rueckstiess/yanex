"""
Main CLI entry point for yanex.
"""

import click

from .commands.run import run
from .commands.list import list_experiments
from .commands.show import show_experiment


@click.group()
@click.version_option(version="0.1.0", prog_name="yanex")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    Yet Another Experiment Tracker - A lightweight experiment tracking harness.
    
    Use yanex to track machine learning experiments with automatic versioning,
    parameter management, and artifact storage.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# Register commands
cli.add_command(run)
cli.add_command(list_experiments, name="list")
cli.add_command(show_experiment, name="show")


if __name__ == "__main__":
    cli()