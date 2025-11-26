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


@click.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@click.argument("script", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Configuration file (YAML/JSON, repeatable)",
)
@click.option(
    "--clone-from",
    type=str,
    metavar="ID",
    help="Clone parameters from existing experiment (ID can be shortened)",
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
@click.option(
    "--depends-on",
    "-D",
    multiple=True,
    metavar="ID",
    help="Experiment ID this run depends on (repeatable, supports short IDs). Can specify multiple dependencies or use comma-separated IDs.",
)
@click.option("--dry-run", is_flag=True, help="Validate configuration without running")
@click.option(
    "--ignore-dirty",
    is_flag=True,
    help="[DEPRECATED] This flag is no longer necessary. Git patches are captured automatically.",
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
    config: tuple[Path, ...],
    clone_from: str | None,
    param: list[str],
    name: str | None,
    tag: list[str],
    description: str | None,
    depends_on: tuple[str, ...],
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

      # With multiple configuration files (merged in order)
      yanex run train.py --config data.yaml --config model.yaml

      # With parameter overrides
      yanex run train.py --param learning_rate=0.01 --param epochs=100

      # Clone parameters from existing experiment
      yanex run train.py --clone-from abc123

      # Clone and override specific parameters
      yanex run train.py --clone-from abc123 --param lr=0.05

      # Pass script-specific arguments (forwarded to script)
      yanex run train.py -p learning_rate=0.01 --data-exp abc123 --fold 0

      # Script args are passed to your script via sys.argv
      # In your script, use argparse to parse them:
      # parser.add_argument('--data-exp', required=True)
      # parser.add_argument('--fold', type=int, default=0)

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

      # Dependencies: Use artifacts from previous experiment
      yanex run train.py -D abc12345

      # Dependencies: Multiple dependencies
      yanex run train.py -D prep1 -D prep2

      # Dependencies: Comma-separated
      yanex run train.py -D prep1,prep2,prep3

      # Dependency sweep: Create one experiment per dependency
      yanex run train.py -D prep1,prep2
      # Creates 2 experiments: one depending on prep1, one on prep2
    """
    from .._utils import (
        load_and_merge_config,
        validate_experiment_config,
        validate_sweep_requirements,
    )

    verbose = ctx.obj.get("verbose", False)
    console = Console()  # Use stdout with colors

    # Show deprecation warning for --ignore-dirty flag
    if ignore_dirty:
        console.print(
            "[yellow]Warning: --ignore-dirty flag is deprecated and no longer necessary. "
            "Git patches are now captured automatically.[/yellow]"
        )

    # Capture script-specific arguments (unknown to yanex)
    script_args = list(ctx.args) if ctx.args else []

    # Parse dependencies (handle comma-separated IDs), preserving groups
    dependency_groups = _parse_dependencies(depends_on)

    # Build parsed CLI arguments dictionary for yanex.get_cli_args()
    # Flatten dependency groups for cli_args (stores what was passed on CLI)
    flat_dependency_ids = [dep for group in dependency_groups for dep in group]
    cli_args = {
        "script": str(script) if script else None,
        "config": [str(c) for c in config] if config else [],
        "clone_from": clone_from,
        "param": list(param),
        "name": name,
        "tag": list(tag),
        "description": description,
        "depends_on": flat_dependency_ids,
        "dry_run": dry_run,
        "stage": stage,
        "staged": staged,
        "parallel": parallel,
    }

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
            for cfg in config:
                console.print(f"[dim]Using config: {cfg}[/]")
        if clone_from:
            console.print(f"[dim]Cloning from experiment: {clone_from}[/]")
        if param:
            console.print(f"[dim]Parameter overrides: {param}[/]")
        if script_args:
            console.print(f"[dim]Script arguments: {script_args}[/]")

    try:
        # Load and merge configuration
        experiment_config, cli_defaults = load_and_merge_config(
            config_paths=config,
            clone_from_id=clone_from,
            param_overrides=list(param),
            verbose=verbose,
        )

        # Resolve CLI parameters with config defaults and CLI overrides
        resolved_name = name if name is not None else cli_defaults.get("name")
        resolved_tags = (
            list(tag) if tag else _normalize_tags(cli_defaults.get("tag", []))
        )
        resolved_description = (
            description if description is not None else cli_defaults.get("description")
        )
        resolved_dry_run = dry_run or cli_defaults.get("dry_run", False)
        resolved_stage = stage or cli_defaults.get("stage", False)

        if verbose:
            console.print(f"[dim]Experiment configuration: {experiment_config}[/]")
            if cli_defaults:
                console.print(f"[dim]CLI defaults from config: {cli_defaults}[/]")
            console.print(
                f"[dim]Resolved CLI parameters: name={resolved_name}, tags={resolved_tags}, description={resolved_description}, dry_run={resolved_dry_run}, stage={resolved_stage}[/]"
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
        # Check if we have dependency sweep (any group has >1 item) or parameter sweep
        has_dep_sweep = _has_dependency_sweep(dependency_groups)
        has_param_sweep = has_sweep_parameters(experiment_config)
        has_any_sweep = has_dep_sweep or has_param_sweep

        if resolved_stage:
            # Stage for later execution
            _stage_experiment(
                script=script,
                name=resolved_name,
                tags=resolved_tags,
                description=resolved_description,
                config=experiment_config,
                dependency_groups=dependency_groups,
                verbose=verbose,
                script_args=script_args,
                cli_args=cli_args,
            )
        elif has_any_sweep:
            # Execute any type of sweep using unified executor
            _execute_sweep(
                script=script,
                name=resolved_name,
                tags=resolved_tags,
                description=resolved_description,
                config=experiment_config,
                dependency_groups=dependency_groups,
                verbose=verbose,
                max_workers=parallel,
                script_args=script_args,
                cli_args=cli_args,
            )
        else:
            # Single experiment execution
            _execute_experiment(
                script=script,
                name=resolved_name,
                tags=resolved_tags,
                description=resolved_description,
                config=experiment_config,
                dependency_groups=dependency_groups,
                verbose=verbose,
                script_args=script_args,
                cli_args=cli_args,
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
    dependency_groups: list[list[str]] | None = None,
    verbose: bool = False,
    script_args: list[str] | None = None,
    cli_args: list[str] | None = None,
) -> None:
    """Execute script as an experiment with proper lifecycle management."""
    console = Console()  # Use stdout with colors

    if script_args is None:
        script_args = []
    if cli_args is None:
        cli_args = []
    if dependency_groups is None:
        dependency_groups = []

    # For single experiment, expand dependency groups to get the single combination
    # (one dep from each group). Returns [[]] if no deps, so [0] gives [] or [dep1, dep2, ...]
    dependency_ids = expand_dependency_groups(dependency_groups)[0]

    # Create experiment
    manager = ExperimentManager()
    experiment_id = manager.create_experiment(
        script_path=script,
        name=name,
        config=config,
        tags=tags,
        description=description,
        dependency_ids=dependency_ids,
        script_args=script_args,
        cli_args=cli_args,
    )

    if verbose:
        console.print(f"[dim]Created experiment: {experiment_id}[/]")

    # Start experiment
    manager.start_experiment(experiment_id)

    # Execute script using ScriptExecutor
    executor = ScriptExecutor(manager)
    executor.execute_script(experiment_id, script, config, verbose, script_args)


def _print_sweep_summary(results: list, total: int) -> None:
    """Print CLI-friendly summary of sweep execution.

    Args:
        results: List of ExperimentResult objects
        total: Total number of experiments attempted
    """
    completed = [r for r in results if r.status == "completed"]
    failed = [r for r in results if r.status == "failed"]

    click.echo("\n✓ Sweep execution completed")
    click.echo(f"  Total: {total}")
    click.echo(f"  Completed: {len(completed)}")
    if failed:
        click.echo(f"  Failed: {len(failed)}")


def _format_value_for_name(value: Any) -> str:
    """
    Format a parameter value for use in experiment name.

    Args:
        value: Parameter value (primitive, list, dict, or other)

    Returns:
        Formatted string representation of the value
    """
    if isinstance(value, list):
        # Join list elements with dash
        formatted_items = [_format_value_for_name(item) for item in value]
        result = "-".join(formatted_items)
    elif isinstance(value, dict):
        # Interleave keys and values: {a: 1, b: 2} -> "a-1-b-2"
        parts = []
        for k, v in value.items():
            parts.append(_format_value_for_name(k))
            parts.append(_format_value_for_name(v))
        result = "-".join(parts)
    else:
        # Convert to string
        result = str(value)

    # Sanitize: replace whitespace and special chars with dash, lowercase
    import re

    result = result.lower()
    result = re.sub(r"[^a-z0-9]+", "-", result)
    # Remove leading/trailing dashes
    result = result.strip("-")

    return result


def _extract_value_from_config(config: dict[str, Any], param_path: str) -> Any:
    """
    Extract a value from nested config dict using dotted path.

    Args:
        config: Configuration dictionary
        param_path: Dotted path to parameter (e.g., "model.dropout")

    Returns:
        The value at the specified path
    """
    parts = param_path.split(".")
    value = config
    for part in parts:
        value = value[part]
    return value


def _generate_sweep_experiment_name(
    base_name: str | None,
    config: dict[str, Any],
    sweep_param_paths: list[str] | None = None,
    dependency_ids: list[str] | None = None,
) -> str:
    """
    Generate experiment name for a sweep experiment.

    Appends dependency IDs and sweep parameter values to base name.
    Format: <base_name>-<dep_id1>-<dep_id2>-<val1>-<val2>

    Args:
        base_name: Base experiment name (can be None)
        config: Configuration dictionary with parameter values
        sweep_param_paths: List of parameter paths that were sweep parameters
        dependency_ids: List of dependency IDs (for dependency sweeps)

    Returns:
        Generated experiment name
    """
    # Start with base name or "sweep" if not provided
    name_parts = [base_name if base_name else "sweep"]

    # Add dependency IDs if present
    if dependency_ids:
        name_parts.extend(dependency_ids)

    # Add sweep parameter values if present
    if sweep_param_paths:
        for param_path in sweep_param_paths:
            value = _extract_value_from_config(config, param_path)
            formatted_value = _format_value_for_name(value)
            name_parts.append(formatted_value)

    # Join all parts with dash
    return "-".join(name_parts)


def _build_parameter_sweep_specs(
    script: Path,
    config: dict[str, Any],
    name: str | None,
    tags: list[str],
    description: str | None,
    dependency_groups: list[list[str]],
    script_args: list[str],
    cli_args: list[str],
) -> list:
    """Build ExperimentSpec objects for parameter sweep.

    Args:
        script: Path to the Python script
        config: Configuration with sweep parameters
        name: Base experiment name
        tags: List of experiment tags
        description: Experiment description
        dependency_groups: List of dependency groups (no sweep, just pass-through)
        script_args: Arguments to pass through to the script
        cli_args: Complete CLI arguments used to run the experiment

    Returns:
        List of ExperimentSpec objects
    """
    from ...executor import ExperimentSpec

    # Expand parameter sweeps into individual configurations
    expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

    # For parameter-only sweep, expand dependency groups to get the single combination
    # (all experiments share the same dependencies)
    dependency_ids = expand_dependency_groups(dependency_groups)[0]

    # Build ExperimentSpec objects for each configuration
    experiments = [
        ExperimentSpec(
            script_path=script,
            config=expanded_config,
            name=_generate_sweep_experiment_name(
                name,
                expanded_config,
                sweep_param_paths,
                dependency_ids=None,  # No dependency sweep in this function
            ),
            tags=tags,
            description=description,
            dependency_ids=dependency_ids,
            script_args=script_args,
            cli_args=cli_args,
        )
        for expanded_config in expanded_configs
    ]

    return experiments


def _build_dependency_sweep_specs(
    script: Path,
    config: dict[str, Any],
    name: str | None,
    tags: list[str],
    description: str | None,
    dependency_groups: list[list[str]],
    script_args: list[str],
    cli_args: list[str],
) -> list:
    """Build ExperimentSpec objects for dependency sweep.

    Creates cross-product of dependency groups. Each experiment gets one
    dependency from each group.

    Args:
        script: Path to the Python script
        config: Configuration (no parameter sweeps)
        name: Base experiment name
        tags: List of experiment tags
        description: Experiment description
        dependency_groups: List of dependency groups (each from one -D flag)
        script_args: Arguments to pass through to the script
        cli_args: Complete CLI arguments used to run the experiment

    Returns:
        List of ExperimentSpec objects
    """
    from ...executor import ExperimentSpec

    # Expand dependency groups into all combinations (cross-product)
    dep_combinations = expand_dependency_groups(dependency_groups)

    # Build ExperimentSpec objects (one per combination)
    experiments = [
        ExperimentSpec(
            script_path=script,
            config=config,
            name=_generate_sweep_experiment_name(
                name, config, sweep_param_paths=None, dependency_ids=dep_combo
            ),
            tags=tags,
            description=description,
            dependency_ids=dep_combo,  # Full combination (one from each group)
            script_args=script_args,
            cli_args=cli_args,
        )
        for dep_combo in dep_combinations
    ]

    return experiments


def _build_cartesian_sweep_specs(
    script: Path,
    config: dict[str, Any],
    name: str | None,
    tags: list[str],
    description: str | None,
    dependency_groups: list[list[str]],
    script_args: list[str],
    cli_args: list[str],
) -> list:
    """Build ExperimentSpec objects for Cartesian product sweep.

    Creates full Cartesian product: dependency_combinations × parameter_combinations.

    Args:
        script: Path to the Python script
        config: Configuration with sweep parameters
        name: Base experiment name
        tags: List of experiment tags
        description: Experiment description
        dependency_groups: List of dependency groups (each from one -D flag)
        script_args: Arguments to pass through to the script
        cli_args: Complete CLI arguments used to run the experiment

    Returns:
        List of ExperimentSpec objects
    """
    from ...executor import ExperimentSpec

    # Expand both dimensions
    dep_combinations = expand_dependency_groups(dependency_groups)
    expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

    # Build ExperimentSpec objects (full Cartesian product)
    experiments = []
    for dep_combo in dep_combinations:
        for expanded_config in expanded_configs:
            sweep_name = _generate_sweep_experiment_name(
                name, expanded_config, sweep_param_paths, dependency_ids=dep_combo
            )
            experiments.append(
                ExperimentSpec(
                    script_path=script,
                    config=expanded_config,
                    name=sweep_name,
                    tags=tags,
                    description=description,
                    dependency_ids=dep_combo,  # Full combination (one from each group)
                    script_args=script_args,
                    cli_args=cli_args,
                )
            )

    return experiments


def _build_sweep_experiment_specs(
    script: Path,
    config: dict[str, Any],
    dependency_groups: list[list[str]],
    name: str | None,
    tags: list[str],
    description: str | None,
    script_args: list[str],
    cli_args: list[str],
) -> list:
    """Build ExperimentSpec objects for sweep execution or staging.

    Unified spec builder that handles all sweep types:
    - Parameter sweeps
    - Dependency sweeps (cross-product of dependency groups)
    - Cartesian product (dependency combinations × parameter combinations)

    Args:
        script: Path to the Python script
        config: Configuration dictionary
        dependency_groups: List of dependency groups (each group from one -D flag)
        name: Base experiment name
        tags: List of experiment tags (without "sweep" tag)
        description: Experiment description
        script_args: Arguments to pass through to the script
        cli_args: Complete CLI arguments used to run the experiment

    Returns:
        List of ExperimentSpec objects, or empty list if no sweep detected
    """
    has_dep_sweep = _has_dependency_sweep(dependency_groups)
    has_param_sweep = has_sweep_parameters(config)

    # Add "sweep" tag to all sweep experiments
    sweep_tags = list(tags) if tags else []
    if "sweep" not in sweep_tags:
        sweep_tags.append("sweep")

    if has_dep_sweep and has_param_sweep:
        # Cartesian product: dependency combinations × parameter combinations
        return _build_cartesian_sweep_specs(
            script=script,
            config=config,
            name=name,
            tags=sweep_tags,
            description=description,
            dependency_groups=dependency_groups,
            script_args=script_args,
            cli_args=cli_args,
        )
    elif has_dep_sweep:
        # Dependency sweep only
        return _build_dependency_sweep_specs(
            script=script,
            config=config,
            name=name,
            tags=sweep_tags,
            description=description,
            dependency_groups=dependency_groups,
            script_args=script_args,
            cli_args=cli_args,
        )
    elif has_param_sweep:
        # Parameter sweep only
        return _build_parameter_sweep_specs(
            script=script,
            config=config,
            name=name,
            tags=sweep_tags,
            description=description,
            dependency_groups=dependency_groups,
            script_args=script_args,
            cli_args=cli_args,
        )
    else:
        # No sweep detected
        return []


def _stage_experiment(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    dependency_groups: list[list[str]] | None = None,
    verbose: bool = False,
    script_args: list[str] | None = None,
    cli_args: list[str] | None = None,
) -> None:
    """Stage experiment(s) for later execution, expanding all sweep types."""

    manager = ExperimentManager()

    if script_args is None:
        script_args = []
    if cli_args is None:
        cli_args = []
    if dependency_groups is None:
        dependency_groups = []

    # Check if this is any type of sweep
    has_dep_sweep = _has_dependency_sweep(dependency_groups)
    has_param_sweep = has_sweep_parameters(config)

    if has_dep_sweep or has_param_sweep:
        # Use shared spec builder to handle all sweep types
        sweep_specs = _build_sweep_experiment_specs(
            script=script,
            config=config,
            dependency_groups=dependency_groups,
            name=name,
            tags=tags,
            description=description,
            script_args=script_args,
            cli_args=cli_args,
        )

        click.echo(f"✓ Sweep detected: expanding into {len(sweep_specs)} experiments")

        experiment_ids = []
        for i, spec in enumerate(sweep_specs):
            experiment_id = manager.create_experiment(
                script_path=spec.script_path,
                name=spec.name,
                config=spec.config,
                tags=spec.tags,
                description=spec.description,
                dependency_ids=spec.dependency_ids,
                stage_only=True,
                script_args=spec.script_args,
                cli_args=spec.cli_args,
            )

            experiment_ids.append(experiment_id)

            if verbose:
                click.echo(
                    f"  Staged sweep experiment {i + 1}/{len(sweep_specs)}: {experiment_id}"
                )
                click.echo(f"    Config: {spec.config}")

        # Show summary
        click.echo(f"✓ Staged {len(experiment_ids)} sweep experiments")
        click.echo(f"  IDs: {', '.join(experiment_ids)}")
        click.echo("  Use 'yanex run --staged' to execute all staged experiments")

    else:
        # Single experiment (no sweeps)
        # Expand dependency groups to get flat list (one from each group)
        dependency_ids = expand_dependency_groups(dependency_groups)[0]
        experiment_id = manager.create_experiment(
            script_path=script,
            name=name,
            config=config,
            tags=tags,
            description=description,
            dependency_ids=dependency_ids,
            stage_only=True,
            script_args=script_args,
            cli_args=cli_args,
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

    # Load script_args from metadata (if present)
    metadata = manager.storage.load_metadata(experiment_id)
    script_args = metadata.get("script_args", [])

    # Execute script using ScriptExecutor
    executor = ScriptExecutor(manager)
    executor.execute_script(experiment_id, script_path, config, verbose, script_args)


def _execute_sweep(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    dependency_groups: list[list[str]] | None = None,
    verbose: bool = False,
    max_workers: int | None = None,
    script_args: list[str] | None = None,
    cli_args: list[str] | None = None,
) -> None:
    """Execute any type of sweep directly using unified spec builder.

    Handles parameter sweeps, dependency sweeps, and Cartesian products.
    Creates and executes experiments on-the-fly without using
    the "staged" status, avoiding interference with existing staged experiments.

    Args:
        script: Path to the Python script
        name: Base experiment name
        tags: List of experiment tags
        description: Experiment description
        config: Configuration (may have sweep parameters)
        dependency_groups: List of dependency groups (each group from one -D flag)
        verbose: Show verbose output
        max_workers: Maximum parallel workers. None=sequential, N=parallel
        script_args: Arguments to pass through to the script
        cli_args: Complete CLI arguments used to run the experiment
    """
    from ...executor import run_multiple

    if script_args is None:
        script_args = []
    if cli_args is None:
        cli_args = []
    if dependency_groups is None:
        dependency_groups = []

    # Use shared spec builder for all sweep types
    sweep_specs = _build_sweep_experiment_specs(
        script=script,
        config=config,
        dependency_groups=dependency_groups,
        name=name,
        tags=tags,
        description=description,
        script_args=script_args,
        cli_args=cli_args,
    )

    # Print generic sweep message
    click.echo(f"✓ Sweep detected: running {len(sweep_specs)} experiments")

    # Use shared executor for sequential or parallel execution
    results = run_multiple(sweep_specs, parallel=max_workers, verbose=verbose)

    # Print CLI-specific summary
    _print_sweep_summary(results, len(sweep_specs))


def _normalize_tags(tag_value: Any) -> list[str]:
    """Convert config tag value to list format matching CLI --tag behavior."""
    if isinstance(tag_value, str):
        return [tag_value]
    elif isinstance(tag_value, list):
        return [str(t) for t in tag_value]
    else:
        return []


def _parse_dependencies(depends_on: tuple[str, ...]) -> list[list[str]]:
    """Parse dependency IDs from CLI arguments, preserving groups.

    Each -D flag becomes a group. Comma-separated IDs within a flag
    are sweep values for that group.

    Args:
        depends_on: Tuple of dependency strings from CLI.

    Returns:
        List of dependency groups. Each group contains IDs from one -D flag.

    Example:
        _parse_dependencies(("abc1,def2", "ghi3"))
        # Returns: [["abc1", "def2"], ["ghi3"]]

        This means: sweep over abc1/def2 for first dep, ghi3 for second dep.
        Creates 2 experiments: [abc1, ghi3] and [def2, ghi3]
    """
    dependency_groups = []
    for dep_arg in depends_on:
        # Split by comma to handle comma-separated IDs within this group
        ids = [dep_id.strip() for dep_id in dep_arg.split(",")]
        ids = [dep_id for dep_id in ids if dep_id]  # Filter empty strings
        if ids:
            dependency_groups.append(ids)
    return dependency_groups


def expand_dependency_groups(dependency_groups: list[list[str]]) -> list[list[str]]:
    """Expand dependency groups into combinations via cross-product.

    Each -D flag represents a dependency "slot". This function generates
    all combinations, taking one dependency from each slot.

    Args:
        dependency_groups: List of groups, each group is a list of dep IDs.
                          E.g., [["exp1", "exp2"], ["exp3", "exp4"]]

    Returns:
        List of dependency combinations, each is a list of dep IDs.
        E.g., [["exp1", "exp3"], ["exp1", "exp4"], ["exp2", "exp3"], ["exp2", "exp4"]]

    Example:
        expand_dependency_groups([["a", "b"], ["c"]])
        # Returns: [["a", "c"], ["b", "c"]]
    """
    if not dependency_groups:
        return [[]]  # Single experiment with no dependencies

    import itertools

    return [list(combo) for combo in itertools.product(*dependency_groups)]


def _has_dependency_sweep(dependency_groups: list[list[str]]) -> bool:
    """Check if dependency groups contain a sweep (any group has >1 item).

    Args:
        dependency_groups: List of dependency groups from _parse_dependencies().

    Returns:
        True if any group has more than one dependency ID (indicating a sweep).
    """
    return any(len(group) > 1 for group in dependency_groups)
