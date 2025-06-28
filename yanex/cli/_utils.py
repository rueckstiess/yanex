"""
Utility functions for yanex CLI.
"""

import click
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..core.config import resolve_config


def parse_param_overrides(param_list: List[str]) -> Dict[str, str]:
    """
    Parse parameter overrides from --param options.
    
    Args:
        param_list: List of "key=value" strings
        
    Returns:
        Dictionary of parameter overrides
        
    Raises:
        click.BadParameter: If parameter format is invalid
    """
    overrides = {}
    
    for param_str in param_list:
        if "=" not in param_str:
            raise click.BadParameter(
                f"Parameter '{param_str}' must be in format 'key=value'"
            )
        
        key, value = param_str.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        if not key:
            raise click.BadParameter(
                f"Parameter key cannot be empty in '{param_str}'"
            )
        
        overrides[key] = value
    
    return overrides


def load_and_merge_config(
    config_path: Optional[Path],
    param_overrides: Dict[str, str],
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Load and merge configuration from various sources.
    
    Args:
        config_path: Optional explicit config file path
        param_overrides: Parameter overrides from CLI
        verbose: Whether to enable verbose output
        
    Returns:
        Merged configuration dictionary
    """
    try:
        # Use existing resolve_config function from core.config
        merged_config = resolve_config(
            config_path=config_path,
            param_overrides=[f"{k}={v}" for k, v in param_overrides.items()]
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
        raise click.ClickException(f"Failed to load configuration: {e}")


def validate_experiment_config(
    script: Path,
    name: Optional[str],
    tags: List[str],
    description: Optional[str],
    config: Dict[str, Any]
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