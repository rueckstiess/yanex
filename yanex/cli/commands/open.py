"""
Open experiment directory in file explorer.
"""

import platform
import subprocess

import click

from yanex.cli.filters import ExperimentFilter

from .confirm import find_experiment


@click.command("open")
@click.argument("experiment_identifier", required=True)
@click.option(
    "--archived",
    "-a",
    is_flag=True,
    help="Search archived experiments instead of active ones",
)
@click.pass_context
def open_experiment(ctx, experiment_identifier: str, archived: bool):
    """
    Open the experiment directory in the system file explorer.

    EXPERIMENT_IDENTIFIER can be:
    - An experiment ID (8-character string)
    - A prefix of an experiment ID
    - An experiment name

    If multiple experiments have the same name, a list will be shown
    and you'll need to use the unique experiment ID instead.

    Examples:
    \b
        yanex open abc12345        # Open experiment by ID
        yanex open abc             # Open experiment by ID prefix
        yanex open my-experiment   # Open experiment by name
        yanex open abc12345 -a     # Open archived experiment
    """
    try:
        # Create filter (creates default manager)
        filter_obj = ExperimentFilter()

        # Try to find the experiment
        experiment = find_experiment(filter_obj, experiment_identifier, archived)

        if experiment is None:
            click.echo(
                f"Error: No experiment found with ID or name '{experiment_identifier}'",
                err=True,
            )
            ctx.exit(1)

        # Check if we got multiple experiments (name collision or prefix ambiguity)
        if isinstance(experiment, list):
            click.echo(
                f"Multiple experiments found matching '{experiment_identifier}':"
            )
            click.echo()

            from ..formatters.console import ExperimentTableFormatter

            formatter = ExperimentTableFormatter()
            formatter.print_experiments_table(experiment)
            click.echo()
            click.echo(
                "Please use a more specific experiment ID or full name with 'yanex open <id>'."
            )
            ctx.exit(1)

        # Get the experiment directory
        experiment_id = experiment["id"]
        try:
            exp_dir = filter_obj.manager.storage.get_experiment_dir(
                experiment_id, archived
            )
        except Exception as e:
            click.echo(
                f"Error: Could not locate experiment directory: {e}",
                err=True,
            )
            ctx.exit(1)

        # Check if directory exists
        if not exp_dir.exists():
            click.echo(
                f"Error: Experiment directory does not exist: {exp_dir}",
                err=True,
            )
            ctx.exit(1)

        # Print the full path to stdout
        click.echo(f"Opening: {exp_dir}")

        # Open the directory in the system file explorer
        try:
            _open_in_file_explorer(str(exp_dir))
        except Exception as e:
            click.echo(
                f"Error: Could not open file explorer: {e}",
                err=True,
            )
            ctx.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        ctx.exit(1)


def _open_in_file_explorer(path: str):
    """
    Open a folder in the system file explorer.

    Security Notes:
    - Paths are passed as list arguments to subprocess.run (not shell=True),
      which prevents shell injection attacks
    - Paths come from internal storage system, not arbitrary user input
    - No shell metacharacter interpretation occurs with list-based arguments

    Args:
        path: Path to the folder to open

    Raises:
        Exception: If the file explorer could not be opened
    """
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            # Using list arguments (not shell=True) prevents command injection
            subprocess.run(["open", path], check=True, capture_output=True, text=True)
        elif system == "Windows":
            # Use os.startfile on Windows (more reliable)
            import os

            # os.startfile is not available on all platforms, only Windows
            if not hasattr(os, "startfile"):
                raise Exception("os.startfile is not available on this platform")

            os.startfile(path)  # type: ignore
        else:  # Linux and other Unix-like systems
            # Using list arguments (not shell=True) prevents command injection
            subprocess.run(
                ["xdg-open", path], check=True, capture_output=True, text=True
            )
    except subprocess.CalledProcessError as e:
        # Include stderr output if available for better debugging
        error_msg = f"Failed to open file explorer: {e}"
        if e.stderr:
            error_msg += f"\n{e.stderr}"
        raise Exception(error_msg) from e
    except Exception as e:
        raise Exception(f"Failed to open file explorer: {e}") from e
