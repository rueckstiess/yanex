"""
Run command implementation for yanex CLI.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ...core.config import expand_parameter_sweeps, has_sweep_parameters
from ...core.manager import ExperimentManager
from ...core.script_executor import ScriptExecutor


@click.command()
@click.argument("script", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file (YAML/JSON)",
)
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Parameter override in format key=value (repeatable)",
)
@click.option("--name", "-n", help="Experiment name")
@click.option("--tag", "-t", multiple=True, help="Experiment tag (repeatable)")
@click.option("--description", "-d", help="Experiment description")
@click.option("--dry-run", is_flag=True, help="Validate configuration without running")
@click.option(
    "--ignore-dirty",
    is_flag=True,
    help="Allow running with uncommitted changes (bypasses git cleanliness check)",
)
@click.option(
    "--stage",
    is_flag=True,
    help="Stage experiment for later execution instead of running immediately",
)
@click.option(
    "--staged",
    is_flag=True,
    help="Execute staged experiments",
)
@click.pass_context
def run(
    ctx: click.Context,
    script: Optional[Path],
    config: Optional[Path],
    param: List[str],
    name: Optional[str],
    tag: List[str],
    description: Optional[str],
    dry_run: bool,
    ignore_dirty: bool,
    stage: bool,
    staged: bool,
) -> None:
    """
    Run a script as a tracked experiment.

    SCRIPT is the path to the Python script to execute.

    Examples:

      # Basic run
      yanex run train.py

      # With configuration file
      yanex run train.py --config config.yaml

      # With parameter overrides
      yanex run train.py --param learning_rate=0.01 --param epochs=100

      # Parameter sweeps (requires --stage)
      yanex run train.py --param "lr=range(0.01, 0.1, 0.01)" --stage
      yanex run train.py --param "lr=linspace(0.001, 0.1, 5)" --stage
      yanex run train.py --param "lr=logspace(-3, -1, 3)" --stage
      yanex run train.py --param "batch_size=list(16, 32, 64)" --stage

      # Multi-parameter sweep (cross-product)
      yanex run train.py \\
        --param "lr=range(0.01, 0.1, 0.01)" \\
        --param "batch_size=list(16, 32, 64)" \\
        --stage

      # Execute staged experiments
      yanex run --staged

      # Full experiment setup
      yanex run train.py \\
        --config config.yaml \\
        --param learning_rate=0.01 \\
        --name "lr-tuning" \\
        --tag "hyperopt" \\
        --description "Learning rate optimization"
    """
    from .._utils import (
        load_and_merge_config,
        validate_experiment_config,
        validate_sweep_requirements,
    )

    verbose = ctx.obj.get("verbose", False)

    # Handle mutually exclusive flags
    if stage and staged:
        click.echo("Error: Cannot use both --stage and --staged flags", err=True)
        raise click.Abort()

    if staged:
        # Execute staged experiments
        _execute_staged_experiments(verbose)
        return

    # Validate script is provided when not using --staged
    if script is None:
        click.echo("Error: Missing argument 'SCRIPT'", err=True)
        click.echo("Try 'yanex run --help' for help.", err=True)
        raise click.Abort()

    if verbose:
        click.echo(f"Running script: {script}")
        if config:
            click.echo(f"Using config: {config}")
        if param:
            click.echo(f"Parameter overrides: {param}")

    try:
        # Load and merge configuration
        merged_config = load_and_merge_config(
            config_path=config, param_overrides=list(param), verbose=verbose
        )

        if verbose:
            click.echo(f"Merged configuration: {merged_config}")

        # Validate configuration
        validate_experiment_config(
            script=script,
            name=name,
            tags=list(tag),
            description=description,
            config=merged_config,
        )

        # Validate sweep requirements
        validate_sweep_requirements(merged_config, stage)

        if dry_run:
            click.echo("✓ Configuration validation passed")
            click.echo("Dry run completed - experiment would be created with:")
            click.echo(f"  Script: {script}")
            click.echo(f"  Name: {name}")
            click.echo(f"  Tags: {list(tag)}")
            click.echo(f"  Description: {description}")
            click.echo(f"  Config: {merged_config}")
            return

        # Phase 3: Execute or stage experiment
        if stage:
            _stage_experiment(
                script=script,
                name=name,
                tags=list(tag),
                description=description,
                config=merged_config,
                verbose=verbose,
                ignore_dirty=ignore_dirty,
            )
        else:
            _execute_experiment(
                script=script,
                name=name,
                tags=list(tag),
                description=description,
                config=merged_config,
                verbose=verbose,
                ignore_dirty=ignore_dirty,
            )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


def _execute_experiment(
    script: Path,
    name: Optional[str],
    tags: List[str],
    description: Optional[str],
    config: Dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False,
) -> None:
    """Execute script as an experiment with proper lifecycle management."""

    # Create experiment
    manager = ExperimentManager()
    experiment_id = manager.create_experiment(
        script_path=script,
        name=name,
        config=config,
        tags=tags,
        description=description,
        allow_dirty=ignore_dirty,
    )

    if verbose:
        click.echo(f"Created experiment: {experiment_id}")

    # Start experiment
    manager.start_experiment(experiment_id)

    # Execute script using ScriptExecutor
    executor = ScriptExecutor(manager)
    executor.execute_script(experiment_id, script, config, verbose)


def _generate_sweep_experiment_name(
    base_name: Optional[str], config: Dict[str, Any]
) -> str:
    """
    Generate a descriptive name for a sweep experiment based on its parameters.

    Args:
        base_name: Base experiment name (can be None)
        config: Configuration dictionary with parameter values

    Returns:
        Generated experiment name with parameter suffixes
    """
    # Start with base name or default
    if base_name:
        name_parts = [base_name]
    else:
        name_parts = ["sweep"]

    # Extract parameter name-value pairs
    param_parts = []

    def extract_params(d: Dict[str, Any], prefix: str = "") -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                # Handle nested parameters
                new_prefix = f"{prefix}_{key}" if prefix else key
                extract_params(value, new_prefix)
            else:
                # Format parameter name
                param_name = f"{prefix}_{key}" if prefix else key

                # Format parameter value
                if isinstance(value, bool):
                    param_value = str(value).lower()
                elif isinstance(value, (int, float)):
                    # Format numbers with reasonable precision
                    if isinstance(value, float):
                        # Remove trailing zeros and unnecessary decimal point
                        if value == int(value):
                            param_value = str(int(value))
                        else:
                            formatted = f"{value:.6g}"  # Up to 6 significant digits
                            # Replace dots with 'p' and handle scientific notation
                            param_value = (
                                formatted.replace(".", "p")
                                .replace("e", "e")
                                .replace("+", "")
                                .replace("-", "m")
                            )
                    else:
                        param_value = str(value)
                else:
                    # String values
                    param_value = str(value)

                param_parts.append(f"{param_name}_{param_value}")

    extract_params(config)

    # Combine name parts
    if param_parts:
        name_parts.extend(param_parts)

    result = "-".join(name_parts)

    # Ensure name isn't too long (limit to 100 characters)
    if len(result) > 100:
        # Truncate but keep the base name and at least one parameter
        if base_name:
            base_len = len(base_name)
            remaining = 97 - base_len  # Leave room for "-..."
            if param_parts:
                truncated_params = param_parts[0][:remaining]
                result = f"{base_name}-{truncated_params}..."
            else:
                result = base_name[:97] + "..."
        else:
            result = result[:97] + "..."

    return result


def _stage_experiment(
    script: Path,
    name: Optional[str],
    tags: List[str],
    description: Optional[str],
    config: Dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False,
) -> None:
    """Stage experiment(s) for later execution, expanding parameter sweeps."""

    manager = ExperimentManager()

    # Check if this is a parameter sweep
    if has_sweep_parameters(config):
        # Expand parameter sweeps into individual configurations
        expanded_configs = expand_parameter_sweeps(config)

        click.echo(
            f"✓ Parameter sweep detected: expanding into {len(expanded_configs)} experiments"
        )

        experiment_ids = []
        for i, expanded_config in enumerate(expanded_configs):
            # Generate descriptive name for each sweep experiment
            sweep_name = _generate_sweep_experiment_name(name, expanded_config)

            experiment_id = manager.create_experiment(
                script_path=script,
                name=sweep_name,
                config=expanded_config,
                tags=tags,
                description=description,
                allow_dirty=ignore_dirty,
                stage_only=True,
            )

            experiment_ids.append(experiment_id)

            if verbose:
                click.echo(
                    f"  Staged sweep experiment {i + 1}/{len(expanded_configs)}: {experiment_id}"
                )
                click.echo(f"    Config: {expanded_config}")

        # Show summary
        click.echo(f"✓ Staged {len(experiment_ids)} sweep experiments")
        click.echo(f"  IDs: {', '.join(experiment_ids)}")
        click.echo("  Use 'yanex run --staged' to execute all staged experiments")

    else:
        # Single experiment (no sweeps)
        experiment_id = manager.create_experiment(
            script_path=script,
            name=name,
            config=config,
            tags=tags,
            description=description,
            allow_dirty=ignore_dirty,
            stage_only=True,
        )

        if verbose:
            click.echo(f"Staged experiment: {experiment_id}")

        exp_dir = manager.storage.get_experiment_directory(experiment_id)
        click.echo(f"✓ Experiment staged: {experiment_id}")
        click.echo(f"  Directory: {exp_dir}")
        click.echo("  Use 'yanex run --staged' to execute staged experiments")


def _execute_staged_experiments(verbose: bool = False) -> None:
    """Execute all staged experiments."""

    manager = ExperimentManager()
    staged_experiments = manager.get_staged_experiments()

    if not staged_experiments:
        click.echo("No staged experiments found")
        return

    if verbose:
        click.echo(f"Found {len(staged_experiments)} staged experiments")

    for experiment_id in staged_experiments:
        try:
            if verbose:
                click.echo(f"Executing staged experiment: {experiment_id}")

            # Load experiment metadata to get script path and config
            metadata = manager.storage.load_metadata(experiment_id)
            config = manager.storage.load_config(experiment_id)
            script_path = Path(metadata["script_path"])

            # Transition to running state
            manager.execute_staged_experiment(experiment_id)

            # Execute the script using the same logic as _execute_experiment
            _execute_staged_script(
                experiment_id=experiment_id,
                script_path=script_path,
                config=config,
                manager=manager,
                verbose=verbose,
            )

        except Exception as e:
            click.echo(
                f"✗ Failed to execute staged experiment {experiment_id}: {e}", err=True
            )
            try:
                manager.fail_experiment(
                    experiment_id, f"Staged execution failed: {str(e)}"
                )
            except Exception:
                pass  # Best effort to record failure


def _execute_staged_script(
    experiment_id: str,
    script_path: Path,
    config: Dict[str, Any],
    manager: ExperimentManager,
    verbose: bool = False,
) -> None:
    """Execute the script for a staged experiment."""
    
    # Execute script using ScriptExecutor
    executor = ScriptExecutor(manager)
    executor.execute_script(experiment_id, script_path, config, verbose)
