"""
Utility functions for yanex CLI.
"""

from pathlib import Path
from typing import Any

import click

from ..core.config import resolve_config


def load_and_merge_config(
    config_path: Path | None, param_overrides: list[str], verbose: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load and merge configuration from various sources.

    Args:
        config_path: Optional explicit config file path
        param_overrides: Parameter override strings from CLI
        verbose: Whether to enable verbose output

    Returns:
        Tuple of (experiment_config, cli_defaults)
        - experiment_config: Configuration for experiment parameters
        - cli_defaults: CLI parameter defaults from 'yanex' section
    """
    try:
        # Use existing resolve_config function from core.config
        experiment_config, cli_defaults = resolve_config(
            config_path=config_path,
            param_overrides=param_overrides,
        )

        if verbose:
            if config_path:
                click.echo(f"Loaded config from: {config_path}")
            else:
                # Check if default config was loaded
                default_config = Path.cwd() / "config.yaml"
                if default_config.exists():
                    click.echo(f"Loaded default config: {default_config}")
                else:
                    click.echo("No configuration file found, using defaults")

            if cli_defaults:
                click.echo(f"Loaded CLI defaults: {cli_defaults}")

        return experiment_config, cli_defaults

    except Exception as e:
        raise click.ClickException(f"Failed to load configuration: {e}") from e


def validate_experiment_config(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
) -> None:
    """
    Validate experiment configuration before execution.

    Args:
        script: Script path
        name: Experiment name
        tags: List of tags
        description: Experiment description
        config: Configuration dictionary

    Raises:
        click.ClickException: If validation fails
    """
    # Validate script
    if not script.exists():
        raise click.ClickException(f"Script file does not exist: {script}")

    if not script.suffix == ".py":
        raise click.ClickException(f"Script must be a Python file: {script}")

    # Validate name if provided
    if name is not None:
        if not name.strip():
            raise click.ClickException("Experiment name cannot be empty")

        # Basic name validation (more detailed validation will be in core.validation)

    # Validate tags if provided
    for tag in tags:
        if not tag.strip():
            raise click.ClickException("Tags cannot be empty")
        if " " in tag:
            raise click.ClickException(f"Tag '{tag}' cannot contain spaces")

    # Validate description length
    if description is not None and len(description) > 1000:
        raise click.ClickException("Description too long (max 1000 characters)")


def validate_sweep_requirements(config: dict[str, Any], stage_flag: bool) -> None:
    """
    Validate parameter sweep requirements.

    As of v0.6.0, parameter sweeps can be executed directly without --stage flag.
    This function is kept for backward compatibility but no longer enforces restrictions.

    Args:
        config: Configuration dictionary to check
        stage_flag: Whether --stage flag was provided
    """
    # No validation needed - sweeps are now allowed in all modes
    pass
