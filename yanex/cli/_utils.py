"""
Utility functions for yanex CLI.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ..core.config import has_sweep_parameters, resolve_config


def load_and_merge_config(
    config_path: Optional[Path], param_overrides: List[str], verbose: bool = False
) -> Dict[str, Any]:
    """
    Load and merge configuration from various sources.

    Args:
        config_path: Optional explicit config file path
        param_overrides: Parameter override strings from CLI
        verbose: Whether to enable verbose output

    Returns:
        Merged configuration dictionary
    """
    try:
        # Use existing resolve_config function from core.config
        merged_config = resolve_config(
            config_path=config_path,
            param_overrides=param_overrides,
        )

        if verbose:
            if config_path:
                click.echo(f"Loaded config from: {config_path}")
            else:
                # Check if default config was loaded
                default_config = Path.cwd() / "yanex.yaml"
                if default_config.exists():
                    click.echo(f"Loaded default config: {default_config}")
                else:
                    click.echo("No configuration file found, using defaults")

        return merged_config

    except Exception as e:
        raise click.ClickException(f"Failed to load configuration: {e}") from e


def validate_experiment_config(
    script: Path,
    name: Optional[str],
    tags: List[str],
    description: Optional[str],
    config: Dict[str, Any],
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
        if len(name) > 100:
            raise click.ClickException("Experiment name too long (max 100 characters)")

    # Validate tags if provided
    for tag in tags:
        if not tag.strip():
            raise click.ClickException("Tags cannot be empty")
        if " " in tag:
            raise click.ClickException(f"Tag '{tag}' cannot contain spaces")

    # Validate description length
    if description is not None and len(description) > 1000:
        raise click.ClickException("Description too long (max 1000 characters)")


def validate_sweep_requirements(config: Dict[str, Any], stage_flag: bool) -> None:
    """
    Validate that parameter sweeps are used with --stage flag.

    Args:
        config: Configuration dictionary to check
        stage_flag: Whether --stage flag was provided

    Raises:
        click.ClickException: If sweep parameters used without --stage
    """
    if has_sweep_parameters(config) and not stage_flag:
        raise click.ClickException(
            "Parameter sweeps require --stage flag to avoid accidental batch execution.\n"
            'Use: yanex run script.py --param "lr=range(0.01, 0.1, 0.01)" --stage'
        )
