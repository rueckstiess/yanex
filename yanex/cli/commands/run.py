"""
Run command implementation for yanex CLI.
"""

from pathlib import Path
from typing import Any

import click
from rich.console import Console

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
@click.option(
    "--parallel",
    "-j",
    type=int,
    metavar="N",
    help="Execute N experiments in parallel (only valid with --staged). Use 0 for auto (number of CPUs).",
)
@click.pass_context
def run(
    ctx: click.Context,
    script: Path | None,
    config: Path | None,
    param: list[str],
    name: str | None,
    tag: list[str],
    description: str | None,
    dry_run: bool,
    ignore_dirty: bool,
    stage: bool,
    staged: bool,
    parallel: int | None,
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
    console = Console()  # Use stdout with colors

    # Handle mutually exclusive flags
    if stage and staged:
        click.echo("Error: Cannot use both --stage and --staged flags", err=True)
        raise click.Abort()

    # Validate parallel flag
    if parallel is not None and parallel < 0:
        click.echo("Error: --parallel must be 0 (auto) or positive integer", err=True)
        raise click.Abort()

    # Validate stage + parallel combination (incompatible)
    if stage and parallel is not None:
        click.echo(
            "Error: --parallel cannot be used with --stage. "
            "Stage experiments first, then run with: yanex run --staged --parallel N",
            err=True,
        )
        raise click.Abort()

    if staged:
        # Execute staged experiments
        _execute_staged_experiments(verbose, console, max_workers=parallel)
        return

    # Validate script is provided when not using --staged
    if script is None:
        click.echo("Error: Missing argument 'SCRIPT'", err=True)
        click.echo("Try 'yanex run --help' for help.", err=True)
        raise click.Abort()

    if verbose:
        console.print(f"[dim]Running script: {script}[/]")
        if config:
            console.print(f"[dim]Using config: {config}[/]")
        if param:
            console.print(f"[dim]Parameter overrides: {param}[/]")

    try:
        # Load and merge configuration
        experiment_config, cli_defaults = load_and_merge_config(
            config_path=config, param_overrides=list(param), verbose=verbose
        )

        # Resolve CLI parameters with config defaults and CLI overrides
        resolved_name = name if name is not None else cli_defaults.get("name")
        resolved_tags = (
            list(tag) if tag else _normalize_tags(cli_defaults.get("tag", []))
        )
        resolved_description = (
            description if description is not None else cli_defaults.get("description")
        )
        resolved_ignore_dirty = ignore_dirty or cli_defaults.get("ignore_dirty", False)
        resolved_dry_run = dry_run or cli_defaults.get("dry_run", False)
        resolved_stage = stage or cli_defaults.get("stage", False)

        if verbose:
            console.print(f"[dim]Experiment configuration: {experiment_config}[/]")
            if cli_defaults:
                console.print(f"[dim]CLI defaults from config: {cli_defaults}[/]")
            console.print(
                f"[dim]Resolved CLI parameters: name={resolved_name}, tags={resolved_tags}, description={resolved_description}, ignore_dirty={resolved_ignore_dirty}, dry_run={resolved_dry_run}, stage={resolved_stage}[/]"
            )

        # Validate configuration
        validate_experiment_config(
            script=script,
            name=resolved_name,
            tags=resolved_tags,
            description=resolved_description,
            config=experiment_config,
        )

        # Validate sweep requirements
        validate_sweep_requirements(experiment_config, resolved_stage)

        # Validate parallel flag with non-sweep single experiments
        if (
            parallel is not None
            and not staged
            and not resolved_stage
            and not has_sweep_parameters(experiment_config)
        ):
            click.echo(
                "Error: --parallel can only be used with parameter sweeps or --staged",
                err=True,
            )
            raise click.Abort()

        if resolved_dry_run:
            click.echo("✓ Configuration validation passed")
            click.echo("Dry run completed - experiment would be created with:")
            click.echo(f"  Script: {script}")
            click.echo(f"  Name: {resolved_name}")
            click.echo(f"  Tags: {resolved_tags}")
            click.echo(f"  Description: {resolved_description}")
            click.echo(f"  Config: {experiment_config}")
            return

        # Phase 3: Execute, stage, or execute sweep
        if resolved_stage:
            # Stage for later execution
            _stage_experiment(
                script=script,
                name=resolved_name,
                tags=resolved_tags,
                description=resolved_description,
                config=experiment_config,
                verbose=verbose,
                ignore_dirty=resolved_ignore_dirty,
            )
        elif has_sweep_parameters(experiment_config):
            # Direct sweep execution (NEW in v0.6.0)
            _execute_sweep_experiments(
                script=script,
                name=resolved_name,
                tags=resolved_tags,
                description=resolved_description,
                config=experiment_config,
                verbose=verbose,
                ignore_dirty=resolved_ignore_dirty,
                max_workers=parallel,  # None=sequential, N=parallel
            )
        else:
            # Single experiment execution
            _execute_experiment(
                script=script,
                name=resolved_name,
                tags=resolved_tags,
                description=resolved_description,
                config=experiment_config,
                verbose=verbose,
                ignore_dirty=resolved_ignore_dirty,
            )

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort() from e


def _execute_experiment(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False,
) -> None:
    """Execute script as an experiment with proper lifecycle management."""
    console = Console()  # Use stdout with colors

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
        console.print(f"[dim]Created experiment: {experiment_id}[/]")

    # Start experiment
    manager.start_experiment(experiment_id)

    # Execute script using ScriptExecutor
    executor = ScriptExecutor(manager)
    executor.execute_script(experiment_id, script, config, verbose)


def _generate_sweep_experiment_name(
    base_name: str | None,
    config: dict[str, Any],
    sweep_param_paths: list[str] | None = None,
) -> str:
    """
    Generate a descriptive name for a sweep experiment based on its parameters.

    Args:
        base_name: Base experiment name (can be None)
        config: Configuration dictionary with parameter values
        sweep_param_paths: List of parameter paths that were sweep parameters (only these will be included in name)

    Returns:
        Generated experiment name with parameter suffixes for sweep parameters only
    """
    # Start with base name or default
    if base_name:
        name_parts = [base_name]
    else:
        name_parts = ["sweep"]

    # Extract parameter name-value pairs
    param_parts = []

    def extract_params(d: dict[str, Any], prefix: str = "") -> None:
        for key, value in d.items():
            if isinstance(value, dict):
                # Handle nested parameters
                new_prefix = f"{prefix}.{key}" if prefix else key
                extract_params(value, new_prefix)
            else:
                # Format parameter path (using dots to match sweep_param_paths format)
                param_path = f"{prefix}.{key}" if prefix else key

                # Only include this parameter if it's in the sweep paths or if no sweep paths specified
                if sweep_param_paths is None or param_path in sweep_param_paths:
                    # Format parameter name for display (using underscores)
                    param_name = param_path.replace(".", "_")

                    # Format parameter value
                    if isinstance(value, bool):
                        param_value = str(value).lower()
                    elif isinstance(value, int | float):
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
    return result


def _stage_experiment(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False,
) -> None:
    """Stage experiment(s) for later execution, expanding parameter sweeps."""

    manager = ExperimentManager()

    # Check if this is a parameter sweep
    if has_sweep_parameters(config):
        # Expand parameter sweeps into individual configurations
        expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

        click.echo(
            f"✓ Parameter sweep detected: expanding into {len(expanded_configs)} experiments"
        )

        experiment_ids = []
        for i, expanded_config in enumerate(expanded_configs):
            # Generate descriptive name for each sweep experiment
            sweep_name = _generate_sweep_experiment_name(
                name, expanded_config, sweep_param_paths
            )

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


def _execute_staged_experiments(
    verbose: bool = False,
    console: Console = None,
    max_workers: int | None = None,
) -> None:
    """Execute all staged experiments, optionally in parallel.

    Args:
        verbose: Show verbose output
        console: Rich console for output
        max_workers: Maximum parallel workers. None=sequential, 0=auto (CPU count)
    """
    import multiprocessing

    if console is None:
        console = Console()

    manager = ExperimentManager()
    staged_experiments = manager.get_staged_experiments()

    if not staged_experiments:
        console.print("[dim]No staged experiments found[/]")
        return

    # Determine execution mode
    if max_workers is None:
        # Sequential execution (backward compatible)
        _execute_staged_sequential(staged_experiments, manager, verbose, console)
    else:
        # Parallel execution
        if max_workers == 0:
            max_workers = multiprocessing.cpu_count()

        _execute_staged_parallel(
            staged_experiments, manager, verbose, console, max_workers
        )


def _execute_staged_sequential(
    staged_experiments: list[str],
    manager: ExperimentManager,
    verbose: bool,
    console: Console,
) -> None:
    """Execute staged experiments sequentially (original behavior)."""
    if verbose:
        console.print(f"[dim]Found {len(staged_experiments)} staged experiments[/]")

    for experiment_id in staged_experiments:
        try:
            if verbose:
                console.print(f"[dim]Executing staged experiment: {experiment_id}[/]")

            # Load experiment metadata
            metadata = manager.storage.load_metadata(experiment_id)
            config = manager.storage.load_config(experiment_id)
            script_path = Path(metadata["script_path"])

            # Transition to running state
            manager.execute_staged_experiment(experiment_id)

            # Execute the script
            _execute_staged_script(
                experiment_id=experiment_id,
                script_path=script_path,
                config=config,
                manager=manager,
                verbose=verbose,
            )

        except Exception as e:
            console.print(
                f"[red]✗ Failed to execute staged experiment {experiment_id}: {e}[/]"
            )
            try:
                manager.fail_experiment(
                    experiment_id, f"Staged execution failed: {str(e)}"
                )
            except Exception:
                pass  # Best effort to record failure


def _execute_staged_parallel(
    staged_experiments: list[str],
    manager: ExperimentManager,
    verbose: bool,
    console: Console,
    max_workers: int,
) -> None:
    """Execute staged experiments in parallel using multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    console.print(f"[dim]Found {len(staged_experiments)} staged experiments[/]")
    console.print(f"[dim]Executing with {max_workers} parallel workers[/]")

    # Pre-load all experiment data before forking
    experiment_data = []
    for experiment_id in staged_experiments:
        try:
            metadata = manager.storage.load_metadata(experiment_id)
            config = manager.storage.load_config(experiment_id)
            experiment_data.append(
                {
                    "experiment_id": experiment_id,
                    "script_path": Path(metadata["script_path"]),
                    "config": config,
                }
            )
        except Exception as e:
            console.print(f"[red]✗ Failed to load experiment {experiment_id}: {e}[/]")

    # Track results
    completed = 0
    failed = 0

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(
                _execute_single_experiment_worker,
                exp_data["experiment_id"],
                exp_data["script_path"],
                exp_data["config"],
                verbose,
            ): exp_data["experiment_id"]
            for exp_data in experiment_data
        }

        # Process results as they complete
        for future in as_completed(future_to_exp):
            experiment_id = future_to_exp[future]
            try:
                success = future.result()
                if success:
                    completed += 1
                    console.print(f"[green]✓ Experiment completed: {experiment_id}[/]")
                else:
                    failed += 1
                    console.print(f"[red]✗ Experiment failed: {experiment_id}[/]")
            except Exception as e:
                failed += 1
                console.print(f"[red]✗ Experiment error: {experiment_id}: {e}[/]")

    # Summary
    console.print("\n[bold]Execution Summary:[/]")
    console.print(f"  Total: {len(experiment_data)}")
    console.print(f"  [green]Completed: {completed}[/]")
    console.print(f"  [red]Failed: {failed}[/]")


def _execute_single_experiment_worker(
    experiment_id: str,
    script_path: Path,
    config: dict[str, Any],
    verbose: bool,
) -> bool:
    """Worker function for parallel experiment execution.

    This runs in a separate process, so it needs to create its own manager.

    Returns:
        True if experiment succeeded, False otherwise
    """
    # Create fresh manager in this process
    manager = ExperimentManager()

    try:
        # Transition to running state
        manager.execute_staged_experiment(experiment_id)

        # Execute the script
        _execute_staged_script(
            experiment_id=experiment_id,
            script_path=script_path,
            config=config,
            manager=manager,
            verbose=verbose,
        )

        return True

    except Exception as e:
        # Record failure
        try:
            manager.fail_experiment(
                experiment_id, f"Parallel execution failed: {str(e)}"
            )
        except Exception:
            pass  # Best effort

        return False


def _execute_staged_script(
    experiment_id: str,
    script_path: Path,
    config: dict[str, Any],
    manager: ExperimentManager,
    verbose: bool = False,
) -> None:
    """Execute the script for a staged experiment."""

    # Execute script using ScriptExecutor
    executor = ScriptExecutor(manager)
    executor.execute_script(experiment_id, script_path, config, verbose)


def _execute_sweep_experiments(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False,
    max_workers: int | None = None,
) -> None:
    """Execute parameter sweep directly (sequential or parallel).

    This creates and executes experiments on-the-fly without using
    the "staged" status, avoiding interference with existing staged experiments.

    Args:
        script: Path to the Python script
        name: Base experiment name
        tags: List of experiment tags
        description: Experiment description
        config: Configuration with sweep parameters
        verbose: Show verbose output
        ignore_dirty: Allow running with uncommitted changes
        max_workers: Maximum parallel workers. None=sequential, N=parallel
    """
    manager = ExperimentManager()

    # Expand parameter sweeps into individual configurations
    expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

    click.echo(
        f"✓ Parameter sweep detected: running {len(expanded_configs)} experiments"
    )

    if max_workers is None:
        # Sequential execution
        _execute_sweep_sequential(
            script,
            name,
            tags,
            description,
            expanded_configs,
            sweep_param_paths,
            manager,
            verbose,
            ignore_dirty,
        )
    else:
        # Parallel execution
        _execute_sweep_parallel(
            script,
            name,
            tags,
            description,
            expanded_configs,
            sweep_param_paths,
            manager,
            verbose,
            ignore_dirty,
            max_workers,
        )


def _execute_sweep_sequential(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    expanded_configs: list[dict[str, Any]],
    sweep_param_paths: list[str],
    manager: ExperimentManager,
    verbose: bool,
    ignore_dirty: bool,
) -> None:
    """Execute sweep experiments sequentially."""
    console = Console()
    completed = 0
    failed = 0

    for i, expanded_config in enumerate(expanded_configs):
        # Generate descriptive name for each sweep experiment
        sweep_name = _generate_sweep_experiment_name(
            name, expanded_config, sweep_param_paths
        )

        try:
            if verbose:
                console.print(
                    f"[dim][{i + 1}/{len(expanded_configs)}] Starting: {sweep_name}[/]"
                )

            # Create and execute immediately (NOT staged)
            experiment_id = manager.create_experiment(
                script_path=script,
                name=sweep_name,
                config=expanded_config,
                tags=tags,
                description=description,
                allow_dirty=ignore_dirty,
                stage_only=False,  # Create as "created", not "staged"
                allow_parallel=True,  # Allow sweep experiments to run sequentially
            )

            # Start experiment
            manager.start_experiment(experiment_id)

            # Execute script
            executor = ScriptExecutor(manager)
            executor.execute_script(experiment_id, script, expanded_config, verbose)

            completed += 1
            console.print(f"  [green]✓ Completed: {experiment_id}[/]")

        except Exception:
            failed += 1
            # ScriptExecutor already prints detailed error info, just note the failure
            console.print(f"  [red]✗ Failed: {sweep_name}[/]")
            # Continue with next experiment

    # Summary
    click.echo("\n✓ Sweep execution completed")
    click.echo(f"  Total: {len(expanded_configs)}")
    click.echo(f"  Completed: {completed}")
    click.echo(f"  Failed: {failed}")


def _execute_sweep_parallel(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    expanded_configs: list[dict[str, Any]],
    sweep_param_paths: list[str],
    manager: ExperimentManager,
    verbose: bool,
    ignore_dirty: bool,
    max_workers: int,
) -> None:
    """Execute sweep experiments in parallel."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    console = Console()

    if max_workers == 0:
        max_workers = multiprocessing.cpu_count()

    click.echo(f"  Executing with {max_workers} parallel workers")

    # Pre-generate experiment data (names, configs)
    experiment_data = []
    for expanded_config in expanded_configs:
        sweep_name = _generate_sweep_experiment_name(
            name, expanded_config, sweep_param_paths
        )
        experiment_data.append(
            {
                "name": sweep_name,
                "config": expanded_config,
                "script": script,
                "tags": tags,
                "description": description,
                "ignore_dirty": ignore_dirty,
            }
        )

    # Track results
    completed = 0
    failed = 0

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments
        future_to_exp = {
            executor.submit(
                _execute_single_sweep_experiment,
                exp_data["script"],
                exp_data["name"],
                exp_data["tags"],
                exp_data["description"],
                exp_data["config"],
                verbose,
                exp_data["ignore_dirty"],
            ): exp_data["name"]
            for exp_data in experiment_data
        }

        # Process results as they complete
        for future in as_completed(future_to_exp):
            exp_name = future_to_exp[future]
            try:
                success, experiment_id, error_msg = future.result()
                if success:
                    completed += 1
                    console.print(
                        f"  [green]✓ Completed: {experiment_id} ({exp_name})[/]"
                    )
                else:
                    failed += 1
                    if error_msg:
                        console.print(f"  [red]✗ Failed: {exp_name}[/]")
                        console.print(f"    [red]Error: {error_msg}[/]")
                    else:
                        console.print(f"  [red]✗ Failed: {exp_name}[/]")
            except Exception as e:
                failed += 1
                console.print(f"  [red]✗ Error: {exp_name}: {e}[/]")

    # Summary
    click.echo("\n✓ Sweep execution completed")
    click.echo(f"  Total: {len(experiment_data)}")
    click.echo(f"  Completed: {completed}")
    click.echo(f"  Failed: {failed}")


def _execute_single_sweep_experiment(
    script: Path,
    name: str,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool,
    ignore_dirty: bool,
) -> tuple[bool, str, str | None]:
    """Worker function for parallel sweep experiment execution.

    This runs in a separate process, so it needs to create its own manager.

    Returns:
        (success, experiment_id, error_message) tuple
    """
    manager = ExperimentManager()

    try:
        # Create experiment
        experiment_id = manager.create_experiment(
            script_path=script,
            name=name,
            config=config,
            tags=tags,
            description=description,
            allow_dirty=ignore_dirty,
            stage_only=False,  # NOT staged
            allow_parallel=True,  # Allow parallel sweep execution
        )

        # Start experiment
        manager.start_experiment(experiment_id)

        # Execute script
        executor = ScriptExecutor(manager)
        executor.execute_script(experiment_id, script, config, verbose)

        return (True, experiment_id, None)

    except Exception as e:
        # Try to mark as failed if experiment was created
        error_msg = str(e)
        try:
            if "experiment_id" in locals():
                manager.fail_experiment(
                    experiment_id, f"Sweep execution failed: {error_msg}"
                )
        except Exception:
            pass

        return (False, "", error_msg)


def _normalize_tags(tag_value: Any) -> list[str]:
    """Convert config tag value to list format matching CLI --tag behavior."""
    if isinstance(tag_value, str):
        return [tag_value]
    elif isinstance(tag_value, list):
        return [str(t) for t in tag_value]
    else:
        return []
