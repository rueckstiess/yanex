"""
Command to display all scripts in a yanex module (shared config file).

A yanex module is a config file that defines multiple scripts with their
dependencies, allowing them to share parameter definitions while declaring
their dependency relationships.
"""

from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from yanex.core.config import load_yaml_config


@click.command("module")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to yanex module config file",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Display as dependency tree visualization",
)
def module_command(config: Path, visualize: bool):
    """
    Display all scripts defined in a yanex module (shared config file).

    A yanex module is a config file that contains a yanex.scripts[] array
    defining multiple scripts with their dependencies. This command shows
    all scripts and their relationships.

    Examples:

      \b
      # Show flat list of scripts
      yanex module --config experiments.yaml

      \b
      # Show dependency tree visualization
      yanex module --config experiments.yaml --visualize
    """
    console = Console()

    try:
        # Load config file
        config_data = load_yaml_config(config)

        # Extract scripts array
        scripts = config_data.get("yanex", {}).get("scripts", [])

        if not scripts:
            console.print(
                f"[yellow]No scripts defined in {config.name}[/yellow]\n"
                "Add a yanex.scripts[] array to define multiple scripts."
            )
            return

        # Display based on mode
        if visualize:
            _display_module_tree(console, config.name, scripts)
        else:
            _display_module_list(console, config.name, scripts)

    except Exception as e:
        console.print(f"[red]Error loading module config:[/red] {e}")
        raise click.Abort()


def _display_module_list(console: Console, config_name: str, scripts: list[dict]):
    """Display scripts as a flat table."""
    console.print(f"\n[bold]Yanex Module:[/bold] {config_name}\n")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Script", style="cyan", no_wrap=True)
    table.add_column("Dependencies", style="yellow")
    table.add_column("Description", style="dim")

    for script_entry in scripts:
        script_name = script_entry.get("name", "?")
        description = script_entry.get("description", "")

        # Extract dependency slots
        dependencies = script_entry.get("dependencies", {})
        if dependencies:
            # Normalize dependencies (handle both string and dict formats)
            dep_slots = []
            for slot_name, slot_config in dependencies.items():
                if isinstance(slot_config, str):
                    # Shorthand format: "dataprep": "dataprep.py"
                    dep_slots.append(f"{slot_name} → {slot_config}")
                elif isinstance(slot_config, dict):
                    # Full format: {"script": "dataprep.py", "required": true}
                    script = slot_config.get("script", "?")
                    required = slot_config.get("required", True)
                    slot_str = f"{slot_name} → {script}"
                    if not required:
                        slot_str += " (optional)"
                    dep_slots.append(slot_str)
            deps_str = "\n".join(dep_slots)
        else:
            deps_str = "[dim]none[/dim]"

        table.add_row(script_name, deps_str, description)

    console.print(table)
    console.print()


def _display_module_tree(console: Console, config_name: str, scripts: list[dict]):
    """Display scripts as a dependency tree."""
    console.print(f"\n[bold]Yanex Module:[/bold] {config_name}\n")

    # Build dependency graph
    script_map = {s.get("name"): s for s in scripts if s.get("name")}

    # Find root scripts (no dependencies or all optional dependencies)
    roots = []
    for script_name, script_entry in script_map.items():
        dependencies = script_entry.get("dependencies", {})
        if not dependencies:
            roots.append(script_name)
        else:
            # Check if all dependencies are optional
            normalized_deps = _normalize_dependencies(dependencies)
            all_optional = all(
                not d.get("required", True) for d in normalized_deps.values()
            )
            if all_optional:
                roots.append(script_name)

    # Create tree
    tree = Tree(f"📦 [bold]{config_name}[/bold]")

    if not roots:
        # No clear roots - show all scripts at top level
        console.print(
            "[yellow]Note: No root scripts found (all have dependencies)[/yellow]\n"
        )
        for script_name in script_map.keys():
            _add_script_node(tree, script_name, script_map, set())
    else:
        # Build tree from roots
        for root_name in roots:
            _add_script_node(tree, root_name, script_map, set())

    console.print(tree)
    console.print()


def _add_script_node(
    parent_node, script_name: str, script_map: dict, visited: set[str]
) -> None:
    """Recursively add script and its dependencies to tree."""
    # Prevent cycles in visualization
    if script_name in visited:
        parent_node.add(f"[dim]{script_name} (already shown)[/dim]")
        return

    visited.add(script_name)

    script_entry = script_map.get(script_name)
    if not script_entry:
        parent_node.add(f"[red]{script_name} (not found)[/red]")
        return

    # Create node label
    description = script_entry.get("description", "")
    if description:
        label = f"[cyan]{script_name}[/cyan] [dim]- {description}[/dim]"
    else:
        label = f"[cyan]{script_name}[/cyan]"

    node = parent_node.add(label)

    # Add dependencies as children
    dependencies = script_entry.get("dependencies", {})
    if dependencies:
        normalized_deps = _normalize_dependencies(dependencies)
        for slot_name, slot_config in normalized_deps.items():
            dep_script = slot_config.get("script", "?")
            required = slot_config.get("required", True)

            # Find the script entry that matches this dependency
            matching_script = None
            for other_name, other_entry in script_map.items():
                if other_entry.get("name") == dep_script:
                    matching_script = other_name
                    break

            if matching_script:
                dep_label = f"[yellow]{slot_name}[/yellow] → "
                if not required:
                    dep_label += "[dim](optional)[/dim] "
                dep_node = node.add(dep_label)
                _add_script_node(dep_node, matching_script, script_map, visited.copy())
            else:
                # Dependency script not in module
                req_str = "" if required else " [dim](optional)[/dim]"
                node.add(
                    f"[yellow]{slot_name}[/yellow] → [dim]{dep_script}{req_str} (external)[/dim]"
                )


def _normalize_dependencies(dependencies: dict) -> dict:
    """Normalize dependencies to full format."""
    normalized = {}
    for slot_name, slot_value in dependencies.items():
        if isinstance(slot_value, str):
            # Shorthand: "dataprep": "dataprep.py"
            normalized[slot_name] = {"script": slot_value, "required": True}
        elif isinstance(slot_value, dict):
            # Full format already
            normalized[slot_name] = {
                "script": slot_value.get("script", "?"),
                "required": slot_value.get("required", True),
            }
    return normalized
