"""
Utility functions for yanex CLI.
"""

from pathlib import Path
from typing import Any

import click

from ..core.config import merge_configs, resolve_config
from ..core.manager import ExperimentManager


def load_and_merge_config(
    config_paths: tuple[Path, ...] | None,
    clone_from_id: str | None,
    param_overrides: list[str],
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load and merge configuration from various sources.

    Merge precedence (highest to lowest):
    1. CLI parameter overrides (--param)
    2. Config file parameters (--config, merged in order if multiple)
    3. Cloned experiment parameters (--clone-from)

    When multiple config files are provided, they are merged in order (left to right),
    with later configs taking precedence over earlier ones.

    Args:
        config_paths: Tuple of config file paths (can be empty or None)
        clone_from_id: Optional experiment ID to clone parameters from (can be shortened)
        param_overrides: Parameter override strings from CLI
        verbose: Whether to enable verbose output

    Returns:
        Tuple of (experiment_config, cli_defaults)
        - experiment_config: Configuration for experiment parameters
        - cli_defaults: CLI parameter defaults from 'yanex' section

    Raises:
        click.ClickException: If configuration cannot be loaded or clone experiment not found
    """
    try:
        # Start with base config (either empty or from cloned experiment)
        base_config = {}

        if clone_from_id:
            # Load config from existing experiment using the same logic as show command
            from ..cli.commands.confirm import find_experiment
            from ..cli.filters import ExperimentFilter

            filter_obj = ExperimentFilter()
            experiment = find_experiment(
                filter_obj, clone_from_id, include_archived=False
            )

            if experiment is None:
                raise click.ClickException(
                    f"No experiment found with ID or name '{clone_from_id}'"
                )

            if isinstance(experiment, list):
                # Multiple experiments found (ambiguous ID prefix or name)
                click.echo(
                    f"Error: Multiple experiments match '{clone_from_id}'. "
                    "Please use a more specific ID prefix."
                )
                raise click.ClickException(
                    f"Ambiguous experiment identifier: '{clone_from_id}'"
                )

            # Load config from the found experiment
            manager = ExperimentManager()
            full_id = experiment["id"]
            cloned_config = manager.storage.load_config(full_id)
            base_config = cloned_config

            if verbose:
                click.echo(f"Cloned config from experiment: {full_id}")
                if cloned_config:
                    click.echo(f"  Parameters: {cloned_config}")

        # Load and merge config files
        file_config, file_cli_defaults = resolve_config(
            config_paths=config_paths,
            param_overrides=param_overrides,
        )

        if verbose:
            if config_paths:
                for config_path in config_paths:
                    click.echo(f"Loaded config from: {config_path}")
            else:
                # Check if default config was loaded
                default_config = Path.cwd() / "config.yaml"
                if default_config.exists():
                    click.echo(f"Loaded default config: {default_config}")
                else:
                    click.echo("No configuration file found, using defaults")

        # Merge: base (cloned) + file config + param overrides
        # Note: resolve_config already merges file configs + param overrides,
        # so we just need to merge base with the result
        if base_config:
            # Merge base config with file config (file config takes precedence)
            experiment_config = merge_configs(base_config, file_config)
            # CLI defaults come from the file config
            cli_defaults = file_cli_defaults
        else:
            # No cloning, use file config as-is
            experiment_config = file_config
            cli_defaults = file_cli_defaults

        if verbose and cli_defaults:
            click.echo(f"CLI defaults: {cli_defaults}")

        return experiment_config, cli_defaults

    except click.ClickException:
        raise
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
