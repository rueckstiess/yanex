"""
Run command implementation for yanex CLI.
"""

from pathlib import Path
from typing import Any

import click
from rich.console import Console

from ...core.config import (
    expand_parameter_sweeps,
    has_sweep_parameters,
    load_yaml_config,
)
from ...core.dependency_validator import DependencyValidator
from ...core.manager import ExperimentManager
from ...core.script_executor import ScriptExecutor
from ...utils.exceptions import DependencyError


def parse_dependency_args(depends_on: list[str]) -> dict[str, list[str]]:
    """Parse --depends-on arguments into structured format.

    Args:
        depends_on: List of "slot=id" or "slot=id1,id2,id3" strings

    Returns:
        Dictionary mapping slot names to lists of experiment IDs
        Example: {"dataprep": ["dp1"], "training": ["tr1", "tr2", "tr3"]}

    Raises:
        click.ClickException: If format is invalid
    """
    parsed = {}

    for arg in depends_on:
        if "=" not in arg:
            raise click.ClickException(
                f"Invalid dependency format: '{arg}'\n"
                f"Expected format: slot=experiment_id or slot=id1,id2,id3"
            )

        slot_name, ids_str = arg.split("=", 1)
        slot_name = slot_name.strip()
        ids_str = ids_str.strip()

        if not slot_name:
            raise click.ClickException("Dependency slot name cannot be empty")

        if not ids_str:
            raise click.ClickException(
                f"No experiment ID provided for slot '{slot_name}'"
            )

        # Split on comma for multiple IDs (dependency sweep)
        exp_ids = [id.strip() for id in ids_str.split(",")]

        # Validate each ID is hex and reasonable length
        for exp_id in exp_ids:
            if not exp_id:
                raise click.ClickException(
                    f"Empty experiment ID in dependency '{slot_name}'"
                )
            # Basic hex validation (allow shortened IDs)
            if not all(c in "0123456789abcdef" for c in exp_id.lower()):
                raise click.ClickException(
                    f"Invalid experiment ID '{exp_id}' for slot '{slot_name}'. "
                    f"IDs must be hexadecimal."
                )

        parsed[slot_name] = exp_ids

    return parsed


def resolve_dependency_declaration(
    script_path: Path, config_path: Path | None
) -> dict[str, Any]:
    """Resolve dependency declaration from config file.

    Looks up the script in yanex.scripts[] array and extracts dependencies.

    Args:
        script_path: Path to the script being run
        config_path: Path to config file (None if no config)

    Returns:
        Declared dependencies in normalized format:
        {"slot": {"script": "script.py", "required": true}}
        Returns {} if no config or script not found in config.
    """
    if not config_path:
        return {}

    try:
        config_data = load_yaml_config(config_path)
    except Exception as e:
        raise click.ClickException(f"Failed to load config: {e}")

    yanex_section = config_data.get("yanex", {})
    scripts = yanex_section.get("scripts", [])

    # Find this script in the array
    script_name = script_path.name
    for script_entry in scripts:
        if script_entry.get("name") == script_name:
            config_dependencies = script_entry.get("dependencies", {})
            if not config_dependencies:
                return {}

            # Normalize to full format
            normalized = {}
            for slot, value in config_dependencies.items():
                if isinstance(value, str):
                    # Shorthand: "dataprep": "dataprep.py"
                    normalized[slot] = {"script": value, "required": True}
                elif isinstance(value, dict):
                    # Full format already
                    normalized[slot] = {
                        "script": value.get("script"),
                        "required": value.get("required", True),
                    }
                else:
                    raise click.ClickException(
                        f"Invalid dependency format for slot '{slot}' in config"
                    )

            return normalized

    # Script not found in config
    return {}


@click.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@click.argument("script", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file (YAML/JSON)",
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
@click.option("--description", help="Experiment description")
@click.option(
    "--depends-on",
    "-d",
    multiple=True,
    help="Dependency on another experiment (format: slot=experiment_id or slot=id1,id2,id3)",
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
    config: Path | None,
    clone_from: str | None,
    param: list[str],
    name: str | None,
    tag: list[str],
    description: str | None,
    depends_on: list[str],
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

    # Build parsed CLI arguments dictionary for yanex.get_cli_args()
    cli_args = {
        "script": str(script) if script else None,
        "config": str(config) if config else None,
        "clone_from": clone_from,
        "param": list(param),
        "name": name,
        "tag": list(tag),
        "description": description,
        "depends_on": list(depends_on),
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
            console.print(f"[dim]Using config: {config}[/]")
        if clone_from:
            console.print(f"[dim]Cloning from experiment: {clone_from}[/]")
        if param:
            console.print(f"[dim]Parameter overrides: {param}[/]")
        if script_args:
            console.print(f"[dim]Script arguments: {script_args}[/]")

    try:
        # Load and merge configuration
        experiment_config, cli_defaults = load_and_merge_config(
            config_path=config,
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

        # Parse and validate dependencies
        parsed_dependencies = None
        declared_dependencies = None
        resolved_dependencies = None

        if depends_on:
            # Parse --depends-on arguments
            parsed_dependencies = parse_dependency_args(depends_on)

            # Resolve dependency declaration from config
            declared_dependencies = resolve_dependency_declaration(script, config)

            # Check if this is a dependency sweep (multiple IDs for any slot)
            has_dependency_sweep = any(
                len(ids) > 1 for ids in parsed_dependencies.values()
            )

            if has_dependency_sweep and not (resolved_stage or parallel is not None):
                # Dependency sweep without staging or parallel - inform user
                total_combinations = 1
                for ids in parsed_dependencies.values():
                    total_combinations *= len(ids)

                click.echo(
                    f"Note: Dependency sweep will create {total_combinations} experiments "
                    f"(run sequentially). Use --parallel N for parallel execution."
                )

            # For single experiment (not a sweep), validate dependencies now
            if not has_dependency_sweep:
                # Flatten to single values for validation
                resolved_dependencies = {
                    slot: ids[0] for slot, ids in parsed_dependencies.items()
                }

                # Validate dependencies before creation
                manager = ExperimentManager()
                validator = DependencyValidator(manager.storage, manager)

                try:
                    validator.validate_dependencies(
                        experiment_id=None,  # Not yet created
                        declared_slots=declared_dependencies,
                        resolved_deps=resolved_dependencies,
                        check_cycles=False,  # Can't check cycles without experiment_id yet
                    )
                except DependencyError as e:
                    click.echo(f"Dependency validation failed: {e}", err=True)
                    raise click.Abort() from e

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
                script_args=script_args,
                cli_args=cli_args,
                dependencies=(declared_dependencies, resolved_dependencies),
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
                max_workers=parallel,  # None=sequential, N=parallel
                script_args=script_args,
                cli_args=cli_args,
                dependencies=(declared_dependencies, parsed_dependencies),
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
                script_args=script_args,
                cli_args=cli_args,
                dependencies=(declared_dependencies, resolved_dependencies),
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
    script_args: list[str] | None = None,
    cli_args: list[str] | None = None,
    dependencies: tuple[dict[str, Any] | None, dict[str, str] | None] | None = None,
) -> None:
    """Execute script as an experiment with proper lifecycle management."""
    console = Console()  # Use stdout with colors

    if script_args is None:
        script_args = []
    if cli_args is None:
        cli_args = []

    # Create experiment
    manager = ExperimentManager()
    experiment_id = manager.create_experiment(
        script_path=script,
        name=name,
        config=config,
        tags=tags,
        description=description,
        script_args=script_args,
        cli_args=cli_args,
    )

    if verbose:
        console.print(f"[dim]Created experiment: {experiment_id}[/]")

    # Save dependencies if provided
    if dependencies and dependencies[0] and dependencies[1]:
        declared_deps, resolved_deps = dependencies

        # Validate and save dependencies
        validator = DependencyValidator(manager.storage, manager)
        try:
            validation_result = validator.validate_dependencies(
                experiment_id=experiment_id,
                declared_slots=declared_deps,
                resolved_deps=resolved_deps,
                check_cycles=True,  # Now we have experiment_id
            )
        except DependencyError as e:
            # Dependency validation failed - clean up experiment
            manager.fail_experiment(experiment_id)
            raise click.ClickException(f"Dependency validation failed: {e}")

        # Build and save dependencies.json
        deps_data = {
            "version": "1.0",
            "declared_slots": declared_deps,
            "resolved_dependencies": resolved_deps,
            "validation": validation_result,
            "depended_by": [],
        }
        manager.storage.save_dependencies(experiment_id, deps_data)

        # Update reverse indexes (add this experiment to each dependency's depended_by list)
        for slot_name, dep_id in resolved_deps.items():
            try:
                manager.storage.add_dependent(
                    dep_id, experiment_id, slot_name, include_archived=True
                )
            except Exception as e:
                # Log warning but don't fail - reverse index is for convenience
                if verbose:
                    console.print(
                        f"[yellow]Warning: Could not update reverse index for {dep_id}: {e}[/yellow]"
                    )

        # Update metadata with dependencies summary
        metadata = manager.storage.load_metadata(experiment_id)
        metadata["dependencies_summary"] = {
            "has_dependencies": True,
            "dependency_count": len(resolved_deps),
            "dependency_slots": list(resolved_deps.keys()),
        }
        manager.storage.save_metadata(experiment_id, metadata)

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


def _generate_sweep_experiment_name(
    base_name: str | None,
    config: dict[str, Any],
    sweep_param_paths: list[str] | None = None,
) -> str:
    """
    Generate experiment name for a sweep experiment.

    Args:
        base_name: Base experiment name (can be None)
        config: Configuration dictionary with parameter values (unused, kept for compatibility)
        sweep_param_paths: List of parameter paths that were sweep parameters (unused, kept for compatibility)

    Returns:
        Base experiment name or "sweep" if no name provided
    """
    # Return the base name as-is, without appending parameter values
    # Parameter values are now tracked via the "sweep" tag instead
    if base_name:
        return base_name
    else:
        return "sweep"


def _stage_experiment(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool = False,
    script_args: list[str] | None = None,
    cli_args: list[str] | None = None,
    dependencies: tuple[dict[str, Any] | None, dict[str, str] | None] | None = None,
) -> None:
    """Stage experiment(s) for later execution, expanding parameter sweeps."""

    # TODO: Implement dependency staging in Phase 2
    if dependencies and dependencies[0]:
        click.echo(
            "[yellow]Warning: Dependencies with --stage not fully implemented yet. "
            "Dependencies will be ignored for staged experiments.[/yellow]"
        )

    manager = ExperimentManager()

    if script_args is None:
        script_args = []
    if cli_args is None:
        cli_args = []

    # Check if this is a parameter sweep
    if has_sweep_parameters(config):
        # Expand parameter sweeps into individual configurations
        expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

        click.echo(
            f"✓ Parameter sweep detected: expanding into {len(expanded_configs)} experiments"
        )

        # Add "sweep" tag to all sweep experiments
        sweep_tags = list(tags) if tags else []
        if "sweep" not in sweep_tags:
            sweep_tags.append("sweep")

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
                tags=sweep_tags,
                description=description,
                stage_only=True,
                script_args=script_args,
                cli_args=cli_args,
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


def _execute_sweep_experiments(
    script: Path,
    name: str | None,
    tags: list[str],
    description: str | None,
    config: dict[str, Any],
    verbose: bool = False,
    max_workers: int | None = None,
    script_args: list[str] | None = None,
    cli_args: list[str] | None = None,
    dependencies: tuple[dict[str, Any] | None, dict[str, list[str]] | None]
    | None = None,
) -> None:
    """Execute parameter sweep directly using shared executor.

    This creates and executes experiments on-the-fly without using
    the "staged" status, avoiding interference with existing staged experiments.

    Args:
        script: Path to the Python script
        name: Base experiment name
        tags: List of experiment tags
        description: Experiment description
        config: Configuration with sweep parameters
        verbose: Show verbose output
        max_workers: Maximum parallel workers. None=sequential, N=parallel
        script_args: Arguments to pass through to the script
        cli_args: Complete CLI arguments used to run the experiment
        dependencies: Tuple of (declared_dependencies, parsed_dependencies) or None
    """
    from ...executor import ExperimentSpec, run_multiple

    # TODO: Implement dependency sweep support in Phase 2
    if dependencies and dependencies[0]:
        click.echo(
            "[yellow]Warning: Dependencies with parameter sweeps not fully implemented yet. "
            "Dependencies will be ignored for sweep experiments.[/yellow]"
        )

    if script_args is None:
        script_args = []
    if cli_args is None:
        cli_args = []

    # Expand parameter sweeps into individual configurations
    expanded_configs, sweep_param_paths = expand_parameter_sweeps(config)

    click.echo(
        f"✓ Parameter sweep detected: running {len(expanded_configs)} experiments"
    )

    # Add "sweep" tag to all sweep experiments (CLI convention)
    sweep_tags = list(tags) if tags else []
    if "sweep" not in sweep_tags:
        sweep_tags.append("sweep")

    # Build ExperimentSpec objects for each configuration
    experiments = [
        ExperimentSpec(
            script_path=script,
            config=expanded_config,
            name=_generate_sweep_experiment_name(
                name, expanded_config, sweep_param_paths
            ),
            tags=sweep_tags,
            description=description,
            script_args=script_args,
            cli_args=cli_args,
        )
        for expanded_config in expanded_configs
    ]

    # Use shared executor for sequential or parallel execution
    results = run_multiple(experiments, parallel=max_workers, verbose=verbose)

    # Print CLI-specific summary
    _print_sweep_summary(results, len(expanded_configs))


def _normalize_tags(tag_value: Any) -> list[str]:
    """Convert config tag value to list format matching CLI --tag behavior."""
    if isinstance(tag_value, str):
        return [tag_value]
    elif isinstance(tag_value, list):
        return [str(t) for t in tag_value]
    else:
        return []
