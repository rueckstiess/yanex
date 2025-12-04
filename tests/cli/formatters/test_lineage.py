"""Tests for lineage visualization formatter."""

import networkx as nx

from yanex.cli.formatters.lineage import (
    format_node_label,
    get_lineage_ids,
    lineage_to_csv,
    lineage_to_json,
    render_lineage_graph,
)


class TestFormatNodeLabel:
    """Tests for format_node_label function."""

    def test_basic_label_with_name(self):
        """Test basic label formatting with experiment name."""
        label = format_node_label("abc12345", "train-model", "completed")
        assert "abc12345" in label
        assert "train-model" in label
        assert "✓" in label  # completed symbol

    def test_basic_label_without_name(self):
        """Test label formatting without experiment name."""
        label = format_node_label("abc12345", "", "completed")
        assert "abc12345" in label
        assert "✓" in label

    def test_target_experiment_marker(self):
        """Test that target experiment gets [this] marker."""
        label = format_node_label("abc12345", "train", "completed", is_target=True)
        assert "\\[this]" in label
        # Target should NOT have slot names even if provided
        label_with_slots = format_node_label(
            "abc12345", "train", "completed", slots=["data"], is_target=True
        )
        assert "\\[this]" in label_with_slots
        assert "\\[data]" not in label_with_slots

    def test_slot_prefix_for_non_target(self):
        """Test slot prefix for non-target experiments."""
        label = format_node_label(
            "abc12345", "train", "completed", slots=["data"], is_target=False
        )
        assert "\\[data]" in label
        assert "\\[this]" not in label

    def test_multiple_slots(self):
        """Test multiple slots are sorted and displayed."""
        label = format_node_label(
            "abc12345", "train", "completed", slots=["model", "data"], is_target=False
        )
        # Slots should be sorted alphabetically
        assert "\\[data]" in label
        assert "\\[model]" in label
        # data should come before model (alphabetical)
        data_pos = label.find("\\[data]")
        model_pos = label.find("\\[model]")
        assert data_pos < model_pos

    def test_status_symbols(self):
        """Test different status symbols are shown."""
        assert "✓" in format_node_label("id", "name", "completed")
        assert "✗" in format_node_label("id", "name", "failed")
        assert "⚡" in format_node_label("id", "name", "running")
        assert "○" in format_node_label("id", "name", "created")
        assert "?" in format_node_label("id", "name", "unknown_status")


class TestRenderLineageGraph:
    """Tests for render_lineage_graph function."""

    def test_empty_graph(self):
        """Test rendering empty graph."""
        graph = nx.DiGraph()
        output = render_lineage_graph(graph, "abc12345", use_color=False)
        assert "no lineage data" in output

    def test_single_node(self):
        """Test rendering graph with single node."""
        graph = nx.DiGraph()
        graph.add_node("abc12345", name="experiment", status="completed")
        output = render_lineage_graph(graph, "abc12345", use_color=False)
        assert "abc12345" in output
        assert "experiment" in output

    def test_simple_chain(self):
        """Test rendering simple dependency chain."""
        graph = nx.DiGraph()
        graph.add_node("parent", name="parent-exp", status="completed")
        graph.add_node("child", name="child-exp", status="completed")
        graph.add_edge("parent", "child", slot="data")

        output = render_lineage_graph(graph, "child", use_color=False)
        assert "parent" in output
        assert "child" in output
        assert "[this]" in output  # target marker for child
        assert "[data]" in output  # slot name

    def test_target_highlighted(self):
        """Test that target experiment is highlighted with [this]."""
        graph = nx.DiGraph()
        graph.add_node("abc12345", name="target", status="completed")
        output = render_lineage_graph(graph, "abc12345", use_color=False)
        assert "[this]" in output

    def test_color_disabled(self):
        """Test that colors can be disabled."""
        graph = nx.DiGraph()
        graph.add_node("abc12345", name="test", status="completed")
        output = render_lineage_graph(graph, "abc12345", use_color=False)
        # Should not contain Rich markup
        assert "[dim]" not in output
        assert "[yellow]" not in output

    def test_color_enabled(self):
        """Test that colors are applied when enabled."""
        graph = nx.DiGraph()
        graph.add_node("abc12345", name="test", status="completed")
        output = render_lineage_graph(graph, "abc12345", use_color=True)
        # Should contain Rich markup
        assert "[dim]" in output or "[yellow]" in output


class TestGetLineageIds:
    """Tests for get_lineage_ids function."""

    def test_empty_graph(self):
        """Test extracting IDs from empty graph."""
        graph = nx.DiGraph()
        ids = get_lineage_ids(graph)
        assert ids == []

    def test_single_node(self):
        """Test extracting single ID."""
        graph = nx.DiGraph()
        graph.add_node("abc12345")
        ids = get_lineage_ids(graph)
        assert ids == ["abc12345"]

    def test_topological_order(self):
        """Test IDs are returned in topological order."""
        graph = nx.DiGraph()
        graph.add_edge("parent", "child")
        graph.add_edge("child", "grandchild")
        ids = get_lineage_ids(graph)
        # Parent should come before child, child before grandchild
        assert ids.index("parent") < ids.index("child")
        assert ids.index("child") < ids.index("grandchild")

    def test_diamond_pattern(self):
        """Test topological order with diamond pattern."""
        graph = nx.DiGraph()
        graph.add_edge("root", "left")
        graph.add_edge("root", "right")
        graph.add_edge("left", "bottom")
        graph.add_edge("right", "bottom")
        ids = get_lineage_ids(graph)
        # Root should come first, bottom should come last
        assert ids[0] == "root"
        assert ids[-1] == "bottom"


class TestLineageToJson:
    """Tests for lineage_to_json function."""

    def test_empty_graph(self):
        """Test JSON output for empty graph."""
        graph = nx.DiGraph()
        result = lineage_to_json(graph, "target")
        assert result["target"] == "target"
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_nodes_include_metadata(self):
        """Test nodes include id, name, and status."""
        graph = nx.DiGraph()
        graph.add_node("abc12345", name="test-exp", status="completed")
        result = lineage_to_json(graph, "abc12345")

        assert len(result["nodes"]) == 1
        node = result["nodes"][0]
        assert node["id"] == "abc12345"
        assert node["name"] == "test-exp"
        assert node["status"] == "completed"

    def test_edges_include_slot(self):
        """Test edges include from, to, and slot."""
        graph = nx.DiGraph()
        graph.add_node("parent", name="", status="completed")
        graph.add_node("child", name="", status="completed")
        graph.add_edge("parent", "child", slot="data")
        result = lineage_to_json(graph, "child")

        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert edge["from"] == "parent"
        assert edge["to"] == "child"
        assert edge["slot"] == "data"


class TestLineageToCsv:
    """Tests for lineage_to_csv function."""

    def test_empty_graph(self):
        """Test CSV output for empty graph."""
        graph = nx.DiGraph()
        result = lineage_to_csv(graph)
        assert result == "from,to,slot"

    def test_single_edge(self):
        """Test CSV with single edge."""
        graph = nx.DiGraph()
        graph.add_edge("parent", "child", slot="data")
        result = lineage_to_csv(graph)
        lines = result.split("\n")
        assert lines[0] == "from,to,slot"
        assert lines[1] == "parent,child,data"

    def test_multiple_edges(self):
        """Test CSV with multiple edges."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b", slot="data")
        graph.add_edge("b", "c", slot="model")
        result = lineage_to_csv(graph)
        lines = result.split("\n")
        assert len(lines) == 3  # header + 2 edges

    def test_edge_without_slot(self):
        """Test CSV handles edges without slot attribute."""
        graph = nx.DiGraph()
        graph.add_edge("a", "b")  # No slot
        result = lineage_to_csv(graph)
        lines = result.split("\n")
        assert lines[1] == "a,b,"  # Empty slot


class TestSlotAssignment:
    """Tests for slot assignment logic in render_lineage_graph."""

    def test_outgoing_slots_preferred(self):
        """Test that outgoing edge slots are shown for nodes with outgoing edges."""
        graph = nx.DiGraph()
        graph.add_node("parent", name="parent", status="completed")
        graph.add_node("child", name="child", status="completed")
        graph.add_edge("parent", "child", slot="data")

        output = render_lineage_graph(graph, "child", use_color=False)
        # Parent provides "data" slot to child, so parent should show [data]
        assert "[data]" in output

    def test_incoming_slots_for_leaf_nodes(self):
        """Test that leaf nodes show incoming edge slots."""
        graph = nx.DiGraph()
        graph.add_node("parent", name="parent", status="completed")
        graph.add_node("leaf", name="leaf", status="completed")
        graph.add_edge("parent", "leaf", slot="output")

        # Render from parent's perspective (leaf is downstream)
        output = render_lineage_graph(graph, "parent", use_color=False)
        # Leaf node should show [output] since it's what it receives
        assert "[output]" in output

    def test_target_always_shows_this(self):
        """Test target always shows [this] regardless of slots."""
        graph = nx.DiGraph()
        graph.add_node("parent", name="parent", status="completed")
        graph.add_node("target", name="target", status="completed")
        graph.add_edge("parent", "target", slot="data")

        output = render_lineage_graph(graph, "target", use_color=False)
        # Target should show [this], not [data]
        assert "[this]" in output
        # The slot info should still appear on parent
        lines = output.split("\n")
        parent_line = [line for line in lines if "parent" in line][0]
        assert "[data]" in parent_line
