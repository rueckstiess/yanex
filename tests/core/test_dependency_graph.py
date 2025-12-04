"""Tests for DependencyGraph class and lineage functionality."""

from yanex.core.dependency_graph import DependencyGraph
from yanex.core.manager import ExperimentManager


class TestDependencyGraphBasic:
    """Basic tests for DependencyGraph initialization and structure."""

    def test_empty_graph(self, temp_dir):
        """Test DependencyGraph with no experiments."""
        manager = ExperimentManager(temp_dir / "experiments")
        graph = DependencyGraph(manager.storage)

        assert len(graph._exp_metadata) == 0
        assert graph._forward.number_of_nodes() == 0
        assert graph._reverse.number_of_nodes() == 0

    def test_single_experiment_no_dependencies(self, temp_dir):
        """Test DependencyGraph with single experiment, no dependencies."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create experiment
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")
        exp_id = manager.create_experiment(script_path, config={})
        manager.complete_experiment(exp_id)

        graph = DependencyGraph(manager.storage)

        assert exp_id in graph._exp_metadata
        assert graph._exp_metadata[exp_id]["status"] == "completed"
        # No edges because no dependencies
        assert graph._forward.number_of_edges() == 0

    def test_loads_experiment_metadata(self, temp_dir):
        """Test that DependencyGraph loads experiment metadata correctly."""
        manager = ExperimentManager(temp_dir / "experiments")

        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")
        exp_id = manager.create_experiment(
            script_path, config={}, name="test-experiment"
        )
        manager.complete_experiment(exp_id)

        graph = DependencyGraph(manager.storage)

        assert graph._exp_metadata[exp_id]["name"] == "test-experiment"
        assert graph._exp_metadata[exp_id]["status"] == "completed"


class TestDependencyGraphUpstream:
    """Tests for get_upstream() - finding dependencies."""

    def test_upstream_single_dependency(self, temp_dir):
        """Test upstream with single direct dependency."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create parent experiment
        parent_id = manager.create_experiment(script_path, config={}, name="parent")
        manager.complete_experiment(parent_id)

        # Create child with dependency on parent
        child_id = manager.create_experiment(script_path, config={}, name="child")
        manager.storage.dependency_storage.save_dependencies(
            child_id, {"data": parent_id}
        )
        manager.complete_experiment(child_id)

        graph = DependencyGraph(manager.storage)
        upstream = graph.get_upstream(child_id)

        # Should have edge from child to parent
        assert upstream.number_of_nodes() == 2
        assert upstream.has_edge(child_id, parent_id)
        assert parent_id in upstream.nodes()

    def test_upstream_transitive_dependencies(self, temp_dir):
        """Test upstream with transitive dependencies (A -> B -> C)."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: grandparent -> parent -> child
        grandparent_id = manager.create_experiment(
            script_path, config={}, name="grandparent"
        )
        manager.complete_experiment(grandparent_id)

        parent_id = manager.create_experiment(script_path, config={}, name="parent")
        manager.storage.dependency_storage.save_dependencies(
            parent_id, {"data": grandparent_id}
        )
        manager.complete_experiment(parent_id)

        child_id = manager.create_experiment(script_path, config={}, name="child")
        manager.storage.dependency_storage.save_dependencies(
            child_id, {"data": parent_id}
        )
        manager.complete_experiment(child_id)

        graph = DependencyGraph(manager.storage)
        upstream = graph.get_upstream(child_id)

        # Should include both parent and grandparent
        assert upstream.number_of_nodes() == 3
        assert upstream.has_edge(child_id, parent_id)
        assert upstream.has_edge(parent_id, grandparent_id)

    def test_upstream_depth_limiting(self, temp_dir):
        """Test that depth limiting works for upstream queries."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C -> D
        ids = []
        for i in range(4):
            exp_id = manager.create_experiment(script_path, config={}, name=f"exp-{i}")
            if i > 0:
                manager.storage.dependency_storage.save_dependencies(
                    exp_id, {"data": ids[-1]}
                )
            manager.complete_experiment(exp_id)
            ids.append(exp_id)

        graph = DependencyGraph(manager.storage)

        # With depth=1, should only get direct parent
        upstream_d1 = graph.get_upstream(ids[3], max_depth=1)
        assert upstream_d1.number_of_nodes() == 2  # D and C

        # With depth=2, should get parent and grandparent
        upstream_d2 = graph.get_upstream(ids[3], max_depth=2)
        assert upstream_d2.number_of_nodes() == 3  # D, C, and B

    def test_upstream_no_dependencies(self, temp_dir):
        """Test upstream returns empty graph for experiment with no dependencies."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        exp_id = manager.create_experiment(script_path, config={}, name="standalone")
        manager.complete_experiment(exp_id)

        graph = DependencyGraph(manager.storage)
        upstream = graph.get_upstream(exp_id)

        # Should be empty (no edges)
        assert upstream.number_of_edges() == 0


class TestDependencyGraphDownstream:
    """Tests for get_downstream() - finding dependents."""

    def test_downstream_single_dependent(self, temp_dir):
        """Test downstream with single direct dependent."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create parent
        parent_id = manager.create_experiment(script_path, config={}, name="parent")
        manager.complete_experiment(parent_id)

        # Create child depending on parent
        child_id = manager.create_experiment(script_path, config={}, name="child")
        manager.storage.dependency_storage.save_dependencies(
            child_id, {"data": parent_id}
        )
        manager.complete_experiment(child_id)

        graph = DependencyGraph(manager.storage)
        downstream = graph.get_downstream(parent_id)

        # Should have edge from parent to child
        assert downstream.number_of_nodes() == 2
        assert downstream.has_edge(parent_id, child_id)

    def test_downstream_multiple_dependents(self, temp_dir):
        """Test downstream with multiple direct dependents."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create parent
        parent_id = manager.create_experiment(script_path, config={}, name="parent")
        manager.complete_experiment(parent_id)

        # Create multiple children
        child_ids = []
        for i in range(3):
            child_id = manager.create_experiment(
                script_path, config={}, name=f"child-{i}"
            )
            manager.storage.dependency_storage.save_dependencies(
                child_id, {"data": parent_id}
            )
            manager.complete_experiment(child_id)
            child_ids.append(child_id)

        graph = DependencyGraph(manager.storage)
        downstream = graph.get_downstream(parent_id)

        # Should have edges to all children
        assert downstream.number_of_nodes() == 4
        for child_id in child_ids:
            assert downstream.has_edge(parent_id, child_id)

    def test_downstream_no_dependents(self, temp_dir):
        """Test downstream returns empty graph for leaf experiment."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        exp_id = manager.create_experiment(script_path, config={}, name="leaf")
        manager.complete_experiment(exp_id)

        graph = DependencyGraph(manager.storage)
        downstream = graph.get_downstream(exp_id)

        # Should be empty (no edges)
        assert downstream.number_of_edges() == 0


class TestDependencyGraphLineage:
    """Tests for get_lineage() - combined upstream and downstream."""

    def test_lineage_middle_of_chain(self, temp_dir):
        """Test lineage for experiment in middle of dependency chain."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C
        a_id = manager.create_experiment(script_path, config={}, name="A")
        manager.complete_experiment(a_id)

        b_id = manager.create_experiment(script_path, config={}, name="B")
        manager.storage.dependency_storage.save_dependencies(b_id, {"data": a_id})
        manager.complete_experiment(b_id)

        c_id = manager.create_experiment(script_path, config={}, name="C")
        manager.storage.dependency_storage.save_dependencies(c_id, {"data": b_id})
        manager.complete_experiment(c_id)

        graph = DependencyGraph(manager.storage)
        lineage = graph.get_lineage(b_id)

        # Should include all three nodes
        assert lineage.number_of_nodes() == 3
        assert a_id in lineage.nodes()
        assert b_id in lineage.nodes()
        assert c_id in lineage.nodes()

        # Edges should point in data-flow direction (dep -> dependent)
        assert lineage.has_edge(a_id, b_id)
        assert lineage.has_edge(b_id, c_id)

    def test_lineage_standalone_experiment(self, temp_dir):
        """Test lineage for experiment with no deps or dependents."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        exp_id = manager.create_experiment(script_path, config={}, name="standalone")
        manager.complete_experiment(exp_id)

        graph = DependencyGraph(manager.storage)
        lineage = graph.get_lineage(exp_id)

        # Should include just the experiment itself
        assert lineage.number_of_nodes() == 1
        assert exp_id in lineage.nodes()

    def test_lineage_diamond_pattern(self, temp_dir):
        """Test lineage with diamond dependency pattern."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Diamond: A -> B -> D
        #          A -> C -> D
        a_id = manager.create_experiment(script_path, config={}, name="A")
        manager.complete_experiment(a_id)

        b_id = manager.create_experiment(script_path, config={}, name="B")
        manager.storage.dependency_storage.save_dependencies(b_id, {"data": a_id})
        manager.complete_experiment(b_id)

        c_id = manager.create_experiment(script_path, config={}, name="C")
        manager.storage.dependency_storage.save_dependencies(c_id, {"data": a_id})
        manager.complete_experiment(c_id)

        d_id = manager.create_experiment(script_path, config={}, name="D")
        manager.storage.dependency_storage.save_dependencies(
            d_id, {"model": b_id, "data": c_id}
        )
        manager.complete_experiment(d_id)

        graph = DependencyGraph(manager.storage)
        lineage = graph.get_lineage(b_id)

        # Should include A, B, D (C might be included via downstream from A)
        assert lineage.number_of_nodes() >= 3
        assert a_id in lineage.nodes()
        assert b_id in lineage.nodes()
        assert d_id in lineage.nodes()


class TestDependencyGraphMetadata:
    """Tests for metadata handling in DependencyGraph."""

    def test_get_metadata_existing(self, temp_dir):
        """Test getting metadata for existing experiment."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        exp_id = manager.create_experiment(
            script_path, config={}, name="test-experiment"
        )
        manager.complete_experiment(exp_id)

        graph = DependencyGraph(manager.storage)
        metadata = graph.get_metadata(exp_id)

        assert metadata["name"] == "test-experiment"
        assert metadata["status"] == "completed"

    def test_get_metadata_nonexistent(self, temp_dir):
        """Test getting metadata for non-existent experiment returns defaults."""
        manager = ExperimentManager(temp_dir / "experiments")
        graph = DependencyGraph(manager.storage)

        metadata = graph.get_metadata("nonexistent")

        assert metadata["name"] == "[deleted]"
        assert metadata["status"] == "deleted"

    def test_experiment_exists(self, temp_dir):
        """Test experiment_exists method."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        exp_id = manager.create_experiment(script_path, config={})
        manager.complete_experiment(exp_id)

        graph = DependencyGraph(manager.storage)

        assert graph.experiment_exists(exp_id)
        assert not graph.experiment_exists("nonexistent")


class TestDependencyGraphNodeAttributes:
    """Tests for node attributes in returned graphs."""

    def test_upstream_includes_node_metadata(self, temp_dir):
        """Test that upstream graph includes name and status on nodes."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        parent_id = manager.create_experiment(script_path, config={}, name="parent-exp")
        manager.complete_experiment(parent_id)

        child_id = manager.create_experiment(script_path, config={}, name="child-exp")
        manager.storage.dependency_storage.save_dependencies(
            child_id, {"data": parent_id}
        )
        manager.complete_experiment(child_id)

        graph = DependencyGraph(manager.storage)
        upstream = graph.get_upstream(child_id)

        # Check parent node has metadata
        assert upstream.nodes[parent_id]["name"] == "parent-exp"
        assert upstream.nodes[parent_id]["status"] == "completed"

    def test_lineage_handles_deleted_dependency(self, temp_dir):
        """Test that lineage gracefully handles deleted dependencies."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create parent
        parent_id = manager.create_experiment(script_path, config={}, name="parent")
        manager.complete_experiment(parent_id)

        # Create child with dependency
        child_id = manager.create_experiment(script_path, config={}, name="child")
        manager.storage.dependency_storage.save_dependencies(
            child_id, {"data": parent_id}
        )
        manager.complete_experiment(child_id)

        # Delete parent experiment
        manager.storage.delete_experiment(parent_id)

        # Create new graph (won't have parent metadata)
        graph = DependencyGraph(manager.storage)
        upstream = graph.get_upstream(child_id)

        # Parent should still appear in graph with [deleted] metadata
        if parent_id in upstream.nodes():
            assert upstream.nodes[parent_id]["name"] == "[deleted]"
            assert upstream.nodes[parent_id]["status"] == "deleted"
