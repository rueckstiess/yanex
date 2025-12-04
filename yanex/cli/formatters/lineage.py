"""Lineage visualization formatter for experiment dependency graphs.

This module renders experiment dependency graphs using ASCII DAG visualization
with git-style output format.
"""

import re

import networkx as nx
from dagviz import visualize_dag

from .theme import STATUS_COLORS, STATUS_SYMBOLS


def format_node_label(
    exp_id: str,
    name: str,
    status: str,
    slots: list[str] | None = None,
    is_target: bool = False,
) -> str:
    """Format a node label for DAG visualization.

    Args:
        exp_id: Experiment ID (8-char hex).
        name: Experiment name.
        status: Experiment status (completed, failed, running, etc.).
        slots: List of slot names from incoming edges (e.g., ["data", "model"]).
        is_target: Whether this is the queried experiment.

    Returns:
        Formatted label like "[data] abc12345 train-model ✓ <-"
    """
    # Get status symbol
    symbol = STATUS_SYMBOLS.get(status, "?")

    # Build slot prefix
    # Use escaped brackets to avoid Rich markup interpretation
    slot_prefix = ""
    if is_target:
        # Target experiment: show [this] marker instead of slot names
        slot_prefix = "\\[this] "
    elif slots:
        # Show all slots if multiple incoming edges
        # Escape brackets for Rich: \[ renders as literal [
        slot_prefix = " ".join(f"\\[{s}]" for s in sorted(slots)) + " "

    # Build label with slot prefix
    if name:
        label = f"{slot_prefix}{exp_id} {name} {symbol}"
    else:
        label = f"{slot_prefix}{exp_id} {symbol}"

    return label


def render_lineage_graph(
    graph: nx.DiGraph,
    target_id: str,
    use_color: bool = True,
) -> str:
    """Render lineage graph as git-style ASCII DAG.

    Uses py-dagviz to produce topologically-sorted ASCII art with
    the target experiment highlighted.

    Args:
        graph: NetworkX DiGraph with edges in dependency direction
               (dependency -> dependent, i.e., data flow direction).
               Nodes should have 'name' and 'status' attributes.
        target_id: Experiment ID to highlight in output.
        use_color: Whether to use ANSI colors in output.

    Returns:
        Multi-line string with ASCII DAG visualization.
    """
    if graph.number_of_nodes() == 0:
        # Empty graph - just show the target
        return f"(no lineage data for {target_id})"

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
        is_target = node == target_id
        slots = node_slots.get(node, [])

        label = format_node_label(node, name, status, slots, is_target)
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
            is_target = node == target_id
            slots = node_slots.get(node, [])
            label = format_node_label(node, name, status, slots, is_target)
            lines.append(label)
        output = "\n".join(lines)
        output += f"\n(DAG rendering failed: {e})"

    # Apply colors if requested
    if use_color:
        output = _apply_colors(output, target_id)

    return output


def _apply_colors(text: str, target_id: str) -> str:
    """Apply Rich-compatible colors to lineage output.

    Applies consistent styling:
    - Slot names [data] → cyan
    - Experiment IDs → dim (gray)
    - Target experiment name → bold white
    - Other experiment names → dim white
    - Status symbols → colored by status

    Args:
        text: Plain text output from dagviz.
        target_id: The queried experiment ID (for bold styling).

    Returns:
        Text with Rich markup for terminal display.
    """
    lines = text.split("\n")
    colored_lines = []

    for line in lines:
        colored_line = _color_line(line, target_id)
        colored_lines.append(colored_line)

    return "\n".join(colored_lines)


def _color_line(line: str, target_id: str) -> str:
    """Apply colors to a single line of lineage output.

    Args:
        line: Single line from dagviz output.
        target_id: The queried experiment ID.

    Returns:
        Line with Rich markup applied.
    """
    # Skip empty lines or lines that are just tree characters
    if not line.strip() or not re.search(r"[0-9a-f]{8}", line):
        return line

    # Check if this line contains the target experiment
    is_target_line = target_id in line

    # Apply slot colors: \[word] → [cyan]\[word][/cyan]
    # Special case: \[this] gets yellow (target marker)
    # The brackets are escaped for Rich (\[), so match that pattern
    line = re.sub(
        r"\\(\[this\])",
        r"[yellow]\\\1[/yellow]",
        line,
    )
    # Apply cyan to other slots (exclude [this] which is already yellow)
    line = re.sub(
        r"\\(\[(?!this\])([a-zA-Z_][a-zA-Z0-9_]*)\])",
        r"[cyan]\\\1[/cyan]",
        line,
    )

    # Apply ID colors: 8-char hex → [dim]id[/dim]
    line = re.sub(
        r"\b([0-9a-f]{8})\b",
        r"[dim]\1[/dim]",
        line,
    )

    # Apply name styling to all experiments
    # Names appear after the ID and before the status symbol
    # NOTE: Must apply before status symbol colors so the regex can match plain symbols
    if is_target_line:
        # Target experiment: yellow name (stands out from regular white), no bold
        line = _style_experiment_name(line, "yellow not bold")
    else:
        # Non-target names: explicit white without bold to ensure consistent styling
        line = _style_experiment_name(line, "white not bold")

    # Apply status symbol colors (last, after name styling)
    for status, symbol in STATUS_SYMBOLS.items():
        if symbol in line:
            color = STATUS_COLORS.get(status, "white")
            line = line.replace(symbol, f"[{color}]{symbol}[/]")

    return line


def _style_experiment_name(line: str, style: str) -> str:
    """Apply style to the experiment name in a line.

    The name appears after the ID (and [/dim]) and before the status symbol.

    Args:
        line: Line with ID already styled as [dim]id[/dim].
        style: Rich style to apply to the name.

    Returns:
        Line with name styled.
    """
    # Pattern to find: [/dim] name status_symbol
    # Name can contain letters, numbers, hyphens, underscores
    # We need to match the text between [/dim] and the status symbol

    # Build pattern for status symbols
    status_symbols_escaped = "|".join(re.escape(s) for s in STATUS_SYMBOLS.values())

    # Match: [/dim] followed by space, then name (non-greedy), then status symbol
    pattern = (
        rf"(\[/dim\]) ([a-zA-Z0-9_-]+(?:\s+[a-zA-Z0-9_-]+)*) ({status_symbols_escaped})"
    )

    def replace_name(match: re.Match) -> str:
        dim_close = match.group(1)
        name = match.group(2)
        symbol = match.group(3)
        return f"{dim_close} [{style}]{name}[/] {symbol}"

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


def lineage_to_json(graph: nx.DiGraph, target_id: str) -> dict:
    """Convert lineage graph to JSON-serializable dict.

    Args:
        graph: NetworkX DiGraph with experiment metadata.
        target_id: The queried experiment ID.

    Returns:
        Dict with nodes, edges, and target information.
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
        "target": target_id,
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
