"""Lineage visualization formatter for experiment dependency graphs.

This module renders experiment dependency graphs using ASCII DAG visualization
with git-style output format.
"""

import re

import networkx as nx
from dagviz import visualize_dag

from .theme import (
    ID_STYLE,
    NAME_STYLE,
    SCRIPT_STYLE,
    SLOT_STYLE,
    STATUS_COLORS,
    STATUS_SYMBOLS,
    TARGET_STYLE,
)


def format_node_label(
    exp_id: str,
    name: str,
    status: str,
    slots: list[str] | None = None,
    is_target: bool = False,
    script: str = "",
) -> str:
    """Format a node label for DAG visualization.

    Args:
        exp_id: Experiment ID (8-char hex).
        name: Experiment name.
        status: Experiment status (completed, failed, running, etc.).
        slots: List of slot names from incoming edges (e.g., ["data", "model"]).
        is_target: Whether this is the queried experiment.
        script: Script filename (e.g., "train.py").

    Returns:
        Formatted label like "<data> abc12345 train-model (train.py) ✓"
    """
    # Get status symbol
    symbol = STATUS_SYMBOLS.get(status, "?")

    # Build slot prefix
    slot_prefix = ""
    if is_target:
        # Target experiment: show <*> marker instead of slot names
        slot_prefix = "<*> "
    elif slots:
        # Show all slots if multiple incoming edges
        slot_prefix = " ".join(f"<{s}>" for s in sorted(slots)) + " "

    # Build script suffix
    script_suffix = ""
    if script:
        script_suffix = f" ({script})"

    # Build label with slot prefix and script suffix
    # Use Rich markup: \\[ escapes bracket, [italic] styles text, ]] escapes closing bracket
    display_name = name if name else "\\[[italic]unnamed[/italic]]"
    label = f"{slot_prefix}{exp_id} {display_name}{script_suffix} {symbol}"

    return label


def render_lineage_graph(
    graph: nx.DiGraph,
    target_ids: set[str],
    use_color: bool = True,
) -> str:
    """Render lineage graph as git-style ASCII DAG.

    Uses py-dagviz to produce topologically-sorted ASCII art with
    target experiments highlighted.

    Args:
        graph: NetworkX DiGraph with edges in dependency direction
               (dependency -> dependent, i.e., data flow direction).
               Nodes should have 'name' and 'status' attributes.
        target_ids: Set of experiment IDs to highlight in output.
        use_color: Whether to use ANSI colors in output.

    Returns:
        Multi-line string with ASCII DAG visualization.
    """
    if graph.number_of_nodes() == 0:
        # Empty graph - just show the targets
        targets_str = ", ".join(sorted(target_ids)) if target_ids else "none"
        return f"(no lineage data for {targets_str})"

    # Collect slots for each node
    # Strategy: prefer outgoing edge slots (what role this node plays),
    # fall back to incoming edge slots for leaf nodes (what role they use)
    outgoing_slots: dict[str, list[str]] = {}
    incoming_slots: dict[str, list[str]] = {}
    has_outgoing: set[str] = set()

    for u, v, data in graph.edges(data=True):
        slot = data.get("slot", "")
        has_outgoing.add(u)
        if slot:
            # Outgoing: u provides slot to v
            if u not in outgoing_slots:
                outgoing_slots[u] = []
            if slot not in outgoing_slots[u]:
                outgoing_slots[u].append(slot)
            # Incoming: v uses slot from u
            if v not in incoming_slots:
                incoming_slots[v] = []
            if slot not in incoming_slots[v]:
                incoming_slots[v].append(slot)

    # Build final slot mapping: prefer outgoing, use incoming for leaf nodes
    node_slots: dict[str, list[str]] = {}
    for node in graph.nodes():
        if node in outgoing_slots:
            # Node has outgoing edges - show what role it plays
            node_slots[node] = outgoing_slots[node]
        elif node in incoming_slots and node not in has_outgoing:
            # Leaf node (no outgoing edges) - show what role it uses
            node_slots[node] = incoming_slots[node]

    # Create a new graph with formatted labels as node IDs
    # py-dagviz uses node IDs as labels directly
    label_graph = nx.DiGraph()
    id_to_label: dict[str, str] = {}

    for node in graph.nodes():
        attrs = graph.nodes[node]
        name = attrs.get("name", "")
        status = attrs.get("status", "unknown")
        script = attrs.get("script", "")
        is_target = node in target_ids
        slots = node_slots.get(node, [])

        label = format_node_label(node, name, status, slots, is_target, script)
        id_to_label[node] = label

    # Add edges with formatted labels
    for u, v in graph.edges():
        label_graph.add_edge(id_to_label[u], id_to_label[v])

    # Ensure all nodes are present even if no edges
    for node in graph.nodes():
        if id_to_label[node] not in label_graph:
            label_graph.add_node(id_to_label[node])

    # Render using py-dagviz
    # Note: visualize_dag returns string with ASCII art
    try:
        output = visualize_dag(label_graph)
    except Exception as e:
        # Fallback to simple list if rendering fails
        lines = []
        for node in graph.nodes():
            attrs = graph.nodes[node]
            name = attrs.get("name", "")
            status = attrs.get("status", "unknown")
            script = attrs.get("script", "")
            is_target = node in target_ids
            slots = node_slots.get(node, [])
            label = format_node_label(node, name, status, slots, is_target, script)
            lines.append(label)
        output = "\n".join(lines)
        output += f"\n(DAG rendering failed: {e})"

    # Apply colors if requested
    if use_color:
        output = _apply_colors(output)

    return output


def render_lineage_components(
    graph: nx.DiGraph,
    target_ids: set[str],
    use_color: bool = True,
) -> list[str]:
    """Render lineage graph, splitting into components if disconnected.

    Args:
        graph: NetworkX DiGraph with experiment metadata.
        target_ids: Set of experiment IDs to highlight.
        use_color: Whether to use ANSI colors.

    Returns:
        List of rendered strings, one per connected component.
        Components are sorted by: (1) contains target, (2) size descending.
    """
    if graph.number_of_nodes() == 0:
        targets_str = ", ".join(sorted(target_ids)) if target_ids else "none"
        return [f"(no lineage data for {targets_str})"]

    # Get weakly connected components
    components = list(nx.weakly_connected_components(graph))

    if len(components) == 1:
        # Single connected component - render normally
        return [render_lineage_graph(graph, target_ids, use_color)]

    # Multiple components - render each separately
    results = []

    # Sort components: those with targets first, then by size descending
    def component_sort_key(component: set[str]) -> tuple[int, int]:
        has_target = 1 if (component & target_ids) else 0
        return (-has_target, -len(component))

    sorted_components = sorted(components, key=component_sort_key)

    for component in sorted_components:
        subgraph = graph.subgraph(component).copy()
        targets_in_component = target_ids & component

        # Render this component (may have no targets if exploring full lineage)
        rendered = render_lineage_graph(subgraph, targets_in_component, use_color)
        results.append(rendered)

    return results


def _apply_colors(text: str) -> str:
    """Apply Rich-compatible colors to lineage output.

    Applies consistent styling using theme constants:
    - Target marker <*> → TARGET_STYLE (magenta)
    - Slot names <data> → SLOT_STYLE (cyan)
    - Script names (script.py) → SCRIPT_STYLE (dim cyan)
    - Experiment IDs → ID_STYLE (dim)
    - Experiment names → NAME_STYLE (white)
    - Status symbols → colored by STATUS_COLORS

    Args:
        text: Plain text output from dagviz.

    Returns:
        Text with Rich markup for terminal display.
    """
    lines = text.split("\n")
    colored_lines = []

    for line in lines:
        colored_line = _color_line(line)
        colored_lines.append(colored_line)

    return "\n".join(colored_lines)


def _color_line(line: str) -> str:
    """Apply colors to a single line of lineage output.

    Args:
        line: Single line from dagviz output.

    Returns:
        Line with Rich markup applied.
    """
    # Skip empty lines or lines that are just tree characters
    if not line.strip() or not re.search(r"[0-9a-f]{8}", line):
        return line

    # Apply slot colors using theme constants
    # Special case: <*> gets TARGET_STYLE (magenta)
    line = re.sub(
        r"(<\*>)",
        rf"[{TARGET_STYLE}]\1[/{TARGET_STYLE}]",
        line,
    )
    # Apply SLOT_STYLE (cyan) to other slots (exclude <*> which is already styled)
    line = re.sub(
        r"(<(?!\*)([a-zA-Z_][a-zA-Z0-9_]*)>)",
        rf"[{SLOT_STYLE}]\1[/{SLOT_STYLE}]",
        line,
    )

    # Apply SCRIPT_STYLE to script names in parentheses: (script.py)
    line = re.sub(
        r"\(([a-zA-Z0-9_.-]+\.py)\)",
        rf"[{SCRIPT_STYLE}](\1)[/{SCRIPT_STYLE}]",
        line,
    )

    # Apply ID_STYLE to experiment IDs: 8-char hex
    line = re.sub(
        r"\b([0-9a-f]{8})\b",
        rf"[{ID_STYLE}]\1[/{ID_STYLE}]",
        line,
    )

    # Apply name styling to all experiments using NAME_STYLE
    # Names appear after the ID and before the status symbol
    # NOTE: Must apply before status symbol colors so the regex can match plain symbols
    # Target experiments are distinguished by their <*> marker, not their name color
    line = _style_experiment_name(line, NAME_STYLE)

    # Apply status symbol colors (last, after name styling)
    for status, symbol in STATUS_SYMBOLS.items():
        if symbol in line:
            color = STATUS_COLORS.get(status, "white")
            line = line.replace(symbol, f"[{color}]{symbol}[/]")

    return line


def _style_experiment_name(line: str, style: str) -> str:
    """Apply style to the experiment name in a line.

    The name appears after the ID closing tag and before the status symbol.

    Args:
        line: Line with ID already styled using ID_STYLE.
        style: Rich style to apply to the name.

    Returns:
        Line with name styled.
    """
    # Pattern to find: [/ID_STYLE] name status_symbol
    # Name can contain letters, numbers, hyphens, underscores
    # We need to match the text between the ID close tag and the status symbol

    # Build pattern for status symbols
    status_symbols_escaped = "|".join(re.escape(s) for s in STATUS_SYMBOLS.values())

    # Match: [/ID_STYLE] followed by space, then name (non-greedy), then status symbol
    # Use the actual ID_STYLE value in the pattern
    pattern = rf"(\[/{ID_STYLE}\]) ([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*) ({status_symbols_escaped})"

    def replace_name(match: re.Match) -> str:
        id_close = match.group(1)
        name = match.group(2)
        symbol = match.group(3)
        return f"{id_close} [{style}]{name}[/] {symbol}"

    return re.sub(pattern, replace_name, line)


def get_lineage_ids(graph: nx.DiGraph) -> list[str]:
    """Extract experiment IDs from a lineage graph.

    Args:
        graph: NetworkX DiGraph with experiment IDs as nodes.

    Returns:
        List of experiment IDs in topological order.
    """
    try:
        # Return in topological order (dependencies first)
        return list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        # Graph has cycles (shouldn't happen) - return arbitrary order
        return list(graph.nodes())


def lineage_to_json(graph: nx.DiGraph, target_ids: set[str]) -> dict:
    """Convert lineage graph to JSON-serializable dict.

    Args:
        graph: NetworkX DiGraph with experiment metadata.
        target_ids: Set of queried experiment IDs.

    Returns:
        Dict with nodes, edges, and targets information.
    """
    nodes = []
    for node in graph.nodes():
        attrs = graph.nodes[node]
        nodes.append(
            {
                "id": node,
                "name": attrs.get("name", ""),
                "status": attrs.get("status", "unknown"),
            }
        )

    edges = []
    for u, v, data in graph.edges(data=True):
        edges.append(
            {
                "from": u,
                "to": v,
                "slot": data.get("slot", ""),
            }
        )

    return {
        "targets": sorted(target_ids),
        "nodes": nodes,
        "edges": edges,
    }


def lineage_to_csv(graph: nx.DiGraph) -> str:
    """Convert lineage graph to CSV edge list.

    Args:
        graph: NetworkX DiGraph with experiment metadata.

    Returns:
        CSV string with header and edge rows.
    """
    lines = ["from,to,slot"]
    for u, v, data in graph.edges(data=True):
        slot = data.get("slot", "")
        lines.append(f"{u},{v},{slot}")
    return "\n".join(lines)
