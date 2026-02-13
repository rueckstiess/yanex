"""Tests for yanex.get_graph() Run API function."""

import pytest

import yanex
from yanex.core.manager import ExperimentManager
from yanex.results.graph import ExperimentGraph
from yanex.utils.exceptions import ExperimentContextError


def _create_experiment(manager, temp_dir, name, script_name="test.py", config=None):
    """Helper to create an experiment with a given name and script."""
    script_path = temp_dir / script_name
    if not script_path.exists():
        script_path.write_text("print('test')")
    exp_id = manager.create_experiment(script_path, config=config or {}, name=name)
    manager.complete_experiment(exp_id)
    return exp_id


def _add_dependency(manager, child_id, parent_id, slot="data"):
    """Helper to add a dependency relationship, merging with existing deps."""
    existing = manager.storage.dependency_storage.load_dependencies(child_id)
    deps = existing.get("dependencies", {})
    deps[slot] = parent_id
    manager.storage.dependency_storage.save_dependencies(child_id, deps)


class TestGetGraphRunAPI:
    """Tests for yanex.get_graph() during experiment execution."""

    def test_get_graph_standalone_raises(self, monkeypatch):
        """get_graph() without experiment context raises ExperimentContextError."""
        monkeypatch.delenv("YANEX_EXPERIMENT_ID", raising=False)
        monkeypatch.delenv("YANEX_CLI_ACTIVE", raising=False)

        with pytest.raises(ExperimentContextError, match="get_graph"):
            yanex.get_graph()

    def test_get_graph_in_experiment_context(self, temp_dir, monkeypatch):
        """get_graph() during experiment execution returns the graph."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b = _create_experiment(manager, temp_dir, "train")
        _add_dependency(manager, b, a, "data")

        # Simulate being inside experiment b
        monkeypatch.setenv("YANEX_EXPERIMENT_ID", b)
        monkeypatch.setenv("YANEX_CLI_ACTIVE", "1")
        monkeypatch.setenv("YANEX_EXPERIMENTS_DIR", str(temp_dir / "experiments"))

        # Reset cached state
        yanex.api._experiment_manager = None

        graph = yanex.get_graph()

        assert isinstance(graph, ExperimentGraph)
        assert len(graph) == 2
        assert a in graph
        assert b in graph

    def test_get_graph_excludes_siblings_by_default(self, temp_dir, monkeypatch):
        """get_graph() returns only lineage (upstream + downstream), not siblings."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b1 = _create_experiment(manager, temp_dir, "train-1")
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(manager, temp_dir, "train-2")
        _add_dependency(manager, b2, a, "data")

        # Simulate being inside experiment b1
        monkeypatch.setenv("YANEX_EXPERIMENT_ID", b1)
        monkeypatch.setenv("YANEX_CLI_ACTIVE", "1")
        monkeypatch.setenv("YANEX_EXPERIMENTS_DIR", str(temp_dir / "experiments"))

        yanex.api._experiment_manager = None

        graph = yanex.get_graph()

        # Lineage from b1: upstream={a}, downstream={}. Sibling b2 excluded.
        assert len(graph) == 2
        assert a in graph
        assert b1 in graph
        assert b2 not in graph

    def test_get_graph_weakly_connected_includes_siblings(self, temp_dir, monkeypatch):
        """get_graph(weakly_connected=True) includes siblings."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b1 = _create_experiment(manager, temp_dir, "train-1")
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(manager, temp_dir, "train-2")
        _add_dependency(manager, b2, a, "data")

        # Simulate being inside experiment b1
        monkeypatch.setenv("YANEX_EXPERIMENT_ID", b1)
        monkeypatch.setenv("YANEX_CLI_ACTIVE", "1")
        monkeypatch.setenv("YANEX_EXPERIMENTS_DIR", str(temp_dir / "experiments"))

        yanex.api._experiment_manager = None

        graph = yanex.get_graph(weakly_connected=True)

        # Full component includes sibling b2
        assert len(graph) == 3
        assert b2 in graph
