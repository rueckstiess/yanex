"""
Run command implementation for yanex CLI.
"""

import json
import os
import subprocess
import sys
import click
from pathlib import Path
from typing import Dict, Any, List, Optional

from ...core.config import resolve_config
from ...core.manager import ExperimentManager


@click.command()
@click.argument("script", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config", 
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Configuration file (YAML/JSON)"
)
@click.option(
    "--param",
    "-p", 
    multiple=True,
    help="Parameter override in format key=value (repeatable)"
)
@click.option(
    "--name",
    "-n",
    help="Experiment name"
)
@click.option(
    "--tag",
    "-t",
    multiple=True,
    help="Experiment tag (repeatable)"
)
@click.option(
    "--description",
    "-d",
    help="Experiment description"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without running"
)
@click.option(
    "--ignore-dirty",
    is_flag=True,
    help="Allow running with uncommitted changes (bypasses git cleanliness check)"
)
@click.pass_context
def run(
    ctx: click.Context,
    script: Path,
    config: Optional[Path],
    param: List[str],
    name: Optional[str],
    tag: List[str],
    description: Optional[str],
    dry_run: bool,
    ignore_dirty: bool,
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
      
      # Full experiment setup
      yanex run train.py \\
        --config config.yaml \\
        --param learning_rate=0.01 \\
        --name "lr-tuning" \\
        --tag "hyperopt" \\
        --description "Learning rate optimization"
    """
    from .._utils import parse_param_overrides, load_and_merge_config, validate_experiment_config
    
    verbose = ctx.obj.get("verbose", False)
    
    if verbose:
        click.echo(f"Running script: {script}")
        if config:
            click.echo(f"Using config: {config}")
        if param:
            click.echo(f"Parameter overrides: {param}")
    
    try:
        # Parse parameter overrides
        param_overrides = parse_param_overrides(param)
        
        # Load and merge configuration
        merged_config = load_and_merge_config(
            config_path=config,
            param_overrides=param_overrides,
            verbose=verbose
        )
        
        if verbose:
            click.echo(f"Merged configuration: {merged_config}")
        
        # Validate configuration
        validate_experiment_config(
            script=script,
            name=name,
            tags=list(tag),
            description=description,
            config=merged_config
        )
        
        if dry_run:
            click.echo("✓ Configuration validation passed")
            click.echo("Dry run completed - experiment would be created with:")
            click.echo(f"  Script: {script}")
            click.echo(f"  Name: {name}")
            click.echo(f"  Tags: {list(tag)}")
            click.echo(f"  Description: {description}")
            click.echo(f"  Config: {merged_config}")
            return
        
        # Phase 3: Execute experiment
        _execute_experiment(
            script=script,
            name=name,
            tags=list(tag),
            description=description,
            config=merged_config,
            verbose=verbose,
            ignore_dirty=ignore_dirty
        )
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


def _execute_experiment(
    script: Path,
    name: Optional[str],
    tags: List[str],
    description: Optional[str],
    config: Dict[str, Any],
    verbose: bool = False,
    ignore_dirty: bool = False
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
        allow_dirty=ignore_dirty
    )
    
    if verbose:
        click.echo(f"Created experiment: {experiment_id}")
    
    # Start experiment
    manager.start_experiment(experiment_id)
    
    try:
        # Prepare environment for subprocess
        env = os.environ.copy()
        env["YANEX_EXPERIMENT_ID"] = experiment_id
        
        # Add parameters as environment variables
        for key, value in config.items():
            env[f"YANEX_PARAM_{key}"] = json.dumps(value) if not isinstance(value, str) else value
        
        if verbose:
            click.echo(f"Starting script execution: {script}")
        
        # Execute script in subprocess
        result = subprocess.run(
            [sys.executable, str(script.resolve())],
            env=env,
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Log script output as artifacts
        if result.stdout:
            manager.storage.save_text_artifact(experiment_id, "stdout.txt", result.stdout)
        if result.stderr:
            manager.storage.save_text_artifact(experiment_id, "stderr.txt", result.stderr)
        
        # Handle experiment result based on exit code
        if result.returncode == 0:
            manager.complete_experiment(experiment_id)
            click.echo(f"✓ Experiment completed successfully: {experiment_id}")
            if verbose and result.stdout:
                click.echo("Script output:")
                click.echo(result.stdout)
        else:
            error_msg = f"Script exited with code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr.strip()}"
            manager.fail_experiment(experiment_id, error_msg)
            click.echo(f"✗ Experiment failed: {experiment_id}")
            click.echo(f"Error: {error_msg}")
            if result.stderr:
                click.echo("Error output:")
                click.echo(result.stderr)
            raise click.Abort()
            
    except subprocess.TimeoutExpired:
        manager.fail_experiment(experiment_id, "Script execution timed out")
        click.echo(f"✗ Experiment timed out: {experiment_id}")
        raise click.Abort()
        
    except KeyboardInterrupt:
        manager.cancel_experiment(experiment_id, "Interrupted by user (Ctrl+C)")
        click.echo(f"✗ Experiment cancelled: {experiment_id}")
        raise
        
    except Exception as e:
        manager.fail_experiment(experiment_id, f"Unexpected error: {str(e)}")
        click.echo(f"✗ Experiment failed: {experiment_id}")
        click.echo(f"Error: {e}")
        raise click.Abort()