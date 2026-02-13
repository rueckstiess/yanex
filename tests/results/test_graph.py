"""Tests for ExperimentGraph class."""

import json

import pytest

from yanex.core.manager import ExperimentManager
from yanex.results.graph import ExperimentGraph
from yanex.utils.exceptions import (
    AmbiguousArtifactError,
    AmbiguousKeyError,
    ExperimentNotFoundError,
    KeyNotFoundError,
)


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


def _add_artifact(manager, exp_id, filename, content="test data"):
    """Helper to add an artifact to an experiment."""
    exp_dir = manager.storage.get_experiment_directory(exp_id)
    artifacts_dir = exp_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / filename).write_text(content)


def _add_metrics(manager, exp_id, metrics_list):
    """Helper to add metrics to an experiment."""
    exp_dir = manager.storage.get_experiment_directory(exp_id)
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics_list, f)


class TestExperimentGraphBuild:
    """Tests for ExperimentGraph.build() factory method."""

    def test_build_single_experiment(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "solo")

        graph = ExperimentGraph.build(manager, exp_id)

        assert len(graph) == 1
        assert exp_id in graph

    def test_build_nonexistent_experiment(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        with pytest.raises(ExperimentNotFoundError):
            ExperimentGraph.build(manager, "nonexistent")

    def test_build_linear_chain(self, temp_dir):
        """A -> B -> C: all three should be in the graph."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep", "dataprep.py")
        b = _create_experiment(manager, temp_dir, "train", "train.py")
        _add_dependency(manager, b, a, "data")
        c = _create_experiment(manager, temp_dir, "eval", "eval.py")
        _add_dependency(manager, c, b, "model")

        graph = ExperimentGraph.build(manager, c)

        assert len(graph) == 3
        assert a in graph
        assert b in graph
        assert c in graph

    def test_build_from_middle_of_chain(self, temp_dir):
        """Building from B in A -> B -> C should include all three."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b = _create_experiment(manager, temp_dir, "train")
        _add_dependency(manager, b, a, "data")
        c = _create_experiment(manager, temp_dir, "eval")
        _add_dependency(manager, c, b, "model")

        graph = ExperimentGraph.build(manager, b)

        assert len(graph) == 3

    def test_build_excludes_siblings_by_default(self, temp_dir):
        """A -> B1, A -> B2: building from B1 should NOT include sibling B2."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b1 = _create_experiment(manager, temp_dir, "train-1")
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(manager, temp_dir, "train-2")
        _add_dependency(manager, b2, a, "data")

        graph = ExperimentGraph.build(manager, b1)

        assert len(graph) == 2
        assert a in graph
        assert b1 in graph
        assert b2 not in graph

    def test_build_includes_siblings_when_weakly_connected(self, temp_dir):
        """A -> B1, A -> B2: weakly_connected=True includes B2 via A."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b1 = _create_experiment(manager, temp_dir, "train-1")
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(manager, temp_dir, "train-2")
        _add_dependency(manager, b2, a, "data")

        graph = ExperimentGraph.build(manager, b1, weakly_connected=True)

        assert len(graph) == 3
        assert b2 in graph

    def test_build_diamond(self, temp_dir):
        """A -> B, A -> C, B -> D, C -> D: all four in the graph."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b = _create_experiment(manager, temp_dir, "branch-1")
        _add_dependency(manager, b, a, "data")
        c = _create_experiment(manager, temp_dir, "branch-2")
        _add_dependency(manager, c, a, "data")
        d = _create_experiment(manager, temp_dir, "merge")
        _add_dependency(manager, d, b, "model-1")
        _add_dependency(manager, d, c, "model-2")

        graph = ExperimentGraph.build(manager, d)

        assert len(graph) == 4


class TestExperimentGraphNavigation:
    """Tests for roots, leaves, and graph properties."""

    def test_roots_linear_chain(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b = _create_experiment(manager, temp_dir, "middle")
        _add_dependency(manager, b, a, "data")
        c = _create_experiment(manager, temp_dir, "leaf")
        _add_dependency(manager, c, b, "data")

        graph = ExperimentGraph.build(manager, c)

        root_ids = [e.id for e in graph.roots]
        assert root_ids == [a]

    def test_leaves_linear_chain(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b = _create_experiment(manager, temp_dir, "middle")
        _add_dependency(manager, b, a, "data")
        c = _create_experiment(manager, temp_dir, "leaf")
        _add_dependency(manager, c, b, "data")

        graph = ExperimentGraph.build(manager, c)

        leaf_ids = [e.id for e in graph.leaves]
        assert leaf_ids == [c]

    def test_roots_fan_out(self, temp_dir):
        """A -> B1, B2, B3: A is the only root, all B's are leaves."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b_ids = []
        for i in range(3):
            b = _create_experiment(manager, temp_dir, f"leaf-{i}")
            _add_dependency(manager, b, a, "data")
            b_ids.append(b)

        graph = ExperimentGraph.build(manager, a)

        root_ids = {e.id for e in graph.roots}
        leaf_ids = {e.id for e in graph.leaves}
        assert root_ids == {a}
        assert leaf_ids == set(b_ids)

    def test_isolated_experiment_is_both_root_and_leaf(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "solo")

        graph = ExperimentGraph.build(manager, exp_id)

        assert len(graph.roots) == 1
        assert len(graph.leaves) == 1
        assert graph.roots[0].id == exp_id
        assert graph.leaves[0].id == exp_id

    def test_digraph_property(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b = _create_experiment(manager, temp_dir, "child")
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, b)

        # Digraph edges should be in data-flow direction (root -> child)
        assert graph.digraph.has_edge(a, b)
        assert not graph.digraph.has_edge(b, a)


class TestExperimentGraphContainer:
    """Tests for container protocol (__getitem__, __len__, __iter__, __contains__)."""

    def test_getitem(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)

        exp = graph[exp_id]
        assert exp.id == exp_id
        assert exp.name == "test"

    def test_getitem_missing(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)

        with pytest.raises(KeyError):
            graph["nonexistent"]

    def test_len(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "a")
        b = _create_experiment(manager, temp_dir, "b")
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)

        assert len(graph) == 2

    def test_iter(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "a")
        b = _create_experiment(manager, temp_dir, "b")
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)

        ids = {e.id for e in graph}
        assert ids == {a, b}

    def test_contains(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)

        assert exp_id in graph
        assert "nonexistent" not in graph

    def test_repr(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "a")
        b = _create_experiment(manager, temp_dir, "b")
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)

        r = repr(graph)
        assert "ExperimentGraph" in r
        assert "experiments=2" in r


class TestExperimentGraphFilter:
    """Tests for graph.filter() using ExperimentFilter."""

    def test_filter_by_status(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b_id = manager.create_experiment(
            temp_dir / "test.py", config={}, name="running"
        )
        _add_dependency(manager, b_id, a, "data")
        # b_id is still "running" (not completed)

        graph = ExperimentGraph.build(manager, a)

        completed = graph.filter(status="completed")
        assert len(completed) == 1
        assert completed[0].id == a

    def test_filter_by_script_pattern(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep", "dataprep.py")
        b = _create_experiment(manager, temp_dir, "train", "train.py")
        _add_dependency(manager, b, a, "data")
        c = _create_experiment(manager, temp_dir, "eval", "eval.py")
        _add_dependency(manager, c, b, "model")

        graph = ExperimentGraph.build(manager, c)

        train_runs = graph.filter(script_pattern="train.py")
        assert len(train_runs) == 1
        assert train_runs[0].id == b

    def test_filter_by_tags(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b = _create_experiment(manager, temp_dir, "tagged")
        _add_dependency(manager, b, a, "data")

        # Add tags to b
        from yanex.results.experiment import Experiment

        exp_b = Experiment(b, manager=manager)
        exp_b.add_tags(["sweep"])

        graph = ExperimentGraph.build(manager, a)

        tagged = graph.filter(tags=["sweep"])
        assert len(tagged) == 1
        assert tagged[0].id == b

    def test_filter_by_name(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep-v1")
        b = _create_experiment(manager, temp_dir, "train-fold-1")
        _add_dependency(manager, b, a, "data")
        c = _create_experiment(manager, temp_dir, "train-fold-2")
        _add_dependency(manager, c, a, "data")

        graph = ExperimentGraph.build(manager, a)

        folds = graph.filter(name="train-fold-*")
        assert len(folds) == 2
        fold_ids = {e.id for e in folds}
        assert fold_ids == {b, c}

    def test_filter_combined(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep", "dataprep.py")
        b = _create_experiment(manager, temp_dir, "train", "train.py")
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)

        result = graph.filter(status="completed", script_pattern="train.py")
        assert len(result) == 1
        assert result[0].id == b

    def test_filter_no_matches(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)

        result = graph.filter(status="failed")
        assert result == []


class TestExperimentGraphLoadArtifact:
    """Tests for graph.load_artifact()."""

    def test_load_artifact_from_root(self, temp_dir):
        """Load artifact that exists only in the root experiment."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        _add_artifact(manager, a, "dataset.json", '{"data": [1, 2, 3]}')
        b = _create_experiment(manager, temp_dir, "train")
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, b)

        result = graph.load_artifact("dataset.json")
        assert result == {"data": [1, 2, 3]}

    def test_load_artifact_from_leaf(self, temp_dir):
        """Load artifact that exists only in the leaf experiment."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b = _create_experiment(manager, temp_dir, "eval")
        _add_dependency(manager, b, a, "data")
        _add_artifact(manager, b, "results.json", '{"accuracy": 0.95}')

        graph = ExperimentGraph.build(manager, a)

        result = graph.load_artifact("results.json")
        assert result == {"accuracy": 0.95}

    def test_load_artifact_not_found(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)

        result = graph.load_artifact("nonexistent.json")
        assert result is None

    def test_load_artifact_ambiguous(self, temp_dir):
        """Same filename in multiple experiments raises AmbiguousArtifactError."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b1 = _create_experiment(manager, temp_dir, "train-1")
        _add_dependency(manager, b1, a, "data")
        _add_artifact(manager, b1, "model.pt", "model 1")
        b2 = _create_experiment(manager, temp_dir, "train-2")
        _add_dependency(manager, b2, a, "data")
        _add_artifact(manager, b2, "model.pt", "model 2")

        graph = ExperimentGraph.build(manager, a)

        with pytest.raises(AmbiguousArtifactError) as exc_info:
            graph.load_artifact("model.pt")
        assert b1 in exc_info.value.experiment_ids
        assert b2 in exc_info.value.experiment_ids

    def test_load_artifact_unique_despite_fan_out(self, temp_dir):
        """Artifact in root is unique even with fan-out downstream."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        _add_artifact(manager, a, "dataset.json", '{"data": true}')
        b1 = _create_experiment(manager, temp_dir, "train-1")
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(manager, temp_dir, "train-2")
        _add_dependency(manager, b2, a, "data")

        graph = ExperimentGraph.build(manager, a)

        # dataset.json only in root — no ambiguity
        result = graph.load_artifact("dataset.json")
        assert result == {"data": True}


class TestExperimentGraphGetParam:
    """Tests for graph.get_param()."""

    def test_get_param_unique(self, temp_dir):
        """Param in only one experiment returns its value."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "dataprep", config={"dataset_path": "/data/train.csv"}
        )
        b = _create_experiment(
            manager, temp_dir, "train", config={"learning_rate": 0.01}
        )
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)

        assert graph.get_param("learning_rate") == 0.01
        assert graph.get_param("dataset_path") == "/data/train.csv"

    def test_get_param_not_found_raises(self, temp_dir):
        """Missing param raises KeyNotFoundError."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test", config={"lr": 0.01})

        graph = ExperimentGraph.build(manager, exp_id)

        with pytest.raises(KeyNotFoundError):
            graph.get_param("nonexistent")

    def test_get_param_same_value_not_ambiguous(self, temp_dir):
        """Same param with same value across experiments is not ambiguous."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "a", config={"epochs": 100})
        b = _create_experiment(manager, temp_dir, "b", config={"epochs": 100})
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)

        assert graph.get_param("epochs") == 100

    def test_get_param_different_values_ambiguous(self, temp_dir):
        """Same param with different values raises ValueError."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b1 = _create_experiment(
            manager, temp_dir, "train-1", config={"learning_rate": 0.01}
        )
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(
            manager, temp_dir, "train-2", config={"learning_rate": 0.001}
        )
        _add_dependency(manager, b2, a, "data")

        graph = ExperimentGraph.build(manager, a)

        with pytest.raises(ValueError, match="learning_rate"):
            graph.get_param("learning_rate")

    def test_get_param_nested(self, temp_dir):
        """Nested params accessible via dot notation."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(
            manager, temp_dir, "test", config={"model": {"n_layers": 6}}
        )

        graph = ExperimentGraph.build(manager, exp_id)

        assert graph.get_param("model.n_layers") == 6


class TestExperimentGraphGetMetric:
    """Tests for graph.get_metric()."""

    def test_get_metric_unique(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b = _create_experiment(manager, temp_dir, "train")
        _add_dependency(manager, b, a, "data")
        _add_metrics(
            manager, b, [{"step": 1, "accuracy": 0.85}, {"step": 2, "accuracy": 0.95}]
        )

        graph = ExperimentGraph.build(manager, a)

        # get_metric returns list for multi-step metrics
        result = graph.get_metric("accuracy")
        assert result == [0.85, 0.95]

    def test_get_metric_not_found_raises(self, temp_dir):
        """Missing metric raises KeyNotFoundError."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)

        with pytest.raises(KeyNotFoundError):
            graph.get_metric("nonexistent")

    def test_get_metric_different_values_ambiguous(self, temp_dir):
        """Same metric in multiple experiments with different values raises ValueError."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b1 = _create_experiment(manager, temp_dir, "train-1")
        _add_dependency(manager, b1, a, "data")
        _add_metrics(manager, b1, [{"step": 1, "accuracy": 0.90}])
        b2 = _create_experiment(manager, temp_dir, "train-2")
        _add_dependency(manager, b2, a, "data")
        _add_metrics(manager, b2, [{"step": 1, "accuracy": 0.85}])

        graph = ExperimentGraph.build(manager, a)

        with pytest.raises(ValueError, match="accuracy"):
            graph.get_metric("accuracy")


class TestExperimentGraphFromExperiment:
    """Tests for Experiment.get_graph()."""

    def test_experiment_get_graph(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b = _create_experiment(manager, temp_dir, "train")
        _add_dependency(manager, b, a, "data")

        from yanex.results.experiment import Experiment

        exp = Experiment(b, manager=manager)
        graph = exp.get_graph()

        assert len(graph) == 2
        assert a in graph
        assert b in graph


class TestExperimentGraphFromResultsManager:
    """Tests for ResultsManager.get_graph()."""

    def test_manager_get_graph(self, temp_dir):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b = _create_experiment(manager, temp_dir, "train")
        _add_dependency(manager, b, a, "data")

        from yanex.results.manager import ResultsManager

        results_mgr = ResultsManager(temp_dir / "experiments")
        graph = results_mgr.get_graph(b)

        assert len(graph) == 2
        assert a in graph


class TestExperimentGraphGetParams:
    """Tests for graph.get_params()."""

    def test_get_params_linear_chain(self, temp_dir):
        """Distinct params across experiments returns merged nested dict."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "dataprep", config={"dataset": "mnist", "samples": 1000}
        )
        b = _create_experiment(
            manager,
            temp_dir,
            "train",
            config={"model": {"lr": 0.01, "layers": 3}},
        )
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)
        params = graph.get_params()

        # Returns nested structure
        assert params["dataset"] == "mnist"
        assert params["samples"] == 1000
        assert params["model"]["lr"] == 0.01
        assert params["model"]["layers"] == 3

    def test_get_params_conflict_raises(self, temp_dir):
        """Same key with different values raises ValueError."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b1 = _create_experiment(manager, temp_dir, "train-1", config={"lr": 0.01})
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(manager, temp_dir, "train-2", config={"lr": 0.1})
        _add_dependency(manager, b2, a, "data")

        graph = ExperimentGraph.build(manager, a)

        with pytest.raises(ValueError, match="lr"):
            graph.get_params()

    def test_get_params_same_value_no_conflict(self, temp_dir):
        """Same key with same value across experiments is OK."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "a", config={"epochs": 100, "dataset": "mnist"}
        )
        b = _create_experiment(
            manager, temp_dir, "b", config={"epochs": 100, "lr": 0.01}
        )
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)
        params = graph.get_params()

        assert params["epochs"] == 100
        assert params["dataset"] == "mnist"
        assert params["lr"] == 0.01

    def test_get_params_single_experiment(self, temp_dir):
        """Single-node graph returns that experiment's params."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(
            manager, temp_dir, "solo", config={"lr": 0.01, "epochs": 50}
        )

        graph = ExperimentGraph.build(manager, exp_id)
        params = graph.get_params()

        assert params == {"lr": 0.01, "epochs": 50}

    def test_get_params_empty_config(self, temp_dir):
        """Experiment with no config contributes nothing."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "a", config={})
        b = _create_experiment(manager, temp_dir, "b", config={"lr": 0.01})
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)
        params = graph.get_params()

        assert params == {"lr": 0.01}


class TestExperimentGraphAccessResolver:
    """Tests for get_param() and get_metric() with AccessResolver integration."""

    def test_get_param_subpath_resolution(self, temp_dir):
        """get_param('lr') resolves to 'model.lr' if unambiguous."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(
            manager, temp_dir, "test", config={"model": {"lr": 0.001, "layers": 3}}
        )

        graph = ExperimentGraph.build(manager, exp_id)

        assert graph.get_param("lr") == 0.001
        assert graph.get_param("layers") == 3

    def test_get_param_exact_match(self, temp_dir):
        """get_param('model.lr') exact match still works."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(
            manager, temp_dir, "test", config={"model": {"lr": 0.001}}
        )

        graph = ExperimentGraph.build(manager, exp_id)

        assert graph.get_param("model.lr") == 0.001

    def test_get_param_ambiguous_subpath_raises(self, temp_dir):
        """get_param('lr') raises AmbiguousKeyError if it matches multiple paths."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Two experiments with different nested paths ending in 'lr'
        a = _create_experiment(manager, temp_dir, "a", config={"model": {"lr": 0.01}})
        b = _create_experiment(
            manager, temp_dir, "b", config={"optimizer": {"lr": 0.001}}
        )
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)

        with pytest.raises(AmbiguousKeyError):
            graph.get_param("lr")

    def test_get_param_missing_key_raises(self, temp_dir):
        """get_param() raises KeyNotFoundError for missing key."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test", config={"lr": 0.01})

        graph = ExperimentGraph.build(manager, exp_id)

        with pytest.raises(KeyNotFoundError):
            graph.get_param("nonexistent_param")

    def test_get_param_conflicting_values_raises(self, temp_dir):
        """Same resolved param with different values raises ValueError."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "root")
        b1 = _create_experiment(manager, temp_dir, "train-1", config={"lr": 0.01})
        _add_dependency(manager, b1, a, "data")
        b2 = _create_experiment(manager, temp_dir, "train-2", config={"lr": 0.1})
        _add_dependency(manager, b2, a, "data")

        graph = ExperimentGraph.build(manager, a)

        with pytest.raises(ValueError, match="lr"):
            graph.get_param("lr")

    def test_get_metric_missing_raises(self, temp_dir):
        """get_metric() raises KeyNotFoundError for missing metric."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)

        with pytest.raises(KeyNotFoundError):
            graph.get_metric("nonexistent_metric")

    def test_get_metric_same_value_across_experiments(self, temp_dir):
        """Metric with same value in multiple experiments returns value."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "a")
        _add_metrics(manager, a, [{"step": 1, "loss": 0.5}])
        b = _create_experiment(manager, temp_dir, "b")
        _add_dependency(manager, b, a, "data")
        _add_metrics(manager, b, [{"step": 1, "loss": 0.5}])

        graph = ExperimentGraph.build(manager, a)

        assert graph.get_metric("loss") == 0.5


class TestExperimentGraphCompare:
    """Tests for graph.compare()."""

    def test_compare_returns_dataframe(self, temp_dir):
        """Basic comparison returns a pandas DataFrame."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "dataprep", config={"dataset": "mnist"}
        )
        b = _create_experiment(manager, temp_dir, "train", config={"lr": 0.01})
        _add_dependency(manager, b, a, "data")
        _add_metrics(manager, b, [{"step": 1, "accuracy": 0.95}])

        graph = ExperimentGraph.build(manager, a)
        df = graph.compare()

        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # both experiments

    def test_compare_with_script_filter(self, temp_dir):
        """compare(script_pattern=...) filters to matching experiments."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "dataprep", "dataprep.py", config={"dataset": "mnist"}
        )
        b = _create_experiment(
            manager, temp_dir, "train", "train.py", config={"lr": 0.01}
        )
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)
        df = graph.compare(script_pattern="train.py")

        assert len(df) == 1

    def test_compare_with_include_dep_params(self, temp_dir):
        """include_dep_params=True pulls upstream params into each row."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "dataprep", "dataprep.py", config={"dataset": "mnist"}
        )
        b = _create_experiment(
            manager, temp_dir, "train", "train.py", config={"lr": 0.01}
        )
        _add_dependency(manager, b, a, "data")

        graph = ExperimentGraph.build(manager, a)
        df = graph.compare(
            script_pattern="train.py",
            include_dep_params=True,
            params="all",
        )

        # Train row should have both its own lr AND upstream dataset param
        assert len(df) == 1
        assert "param:lr" in df.columns
        assert "param:dataset" in df.columns

    def test_compare_empty_filter_returns_empty_df(self, temp_dir):
        """Filter that matches nothing returns empty DataFrame."""
        manager = ExperimentManager(temp_dir / "experiments")
        exp_id = _create_experiment(manager, temp_dir, "test")

        graph = ExperimentGraph.build(manager, exp_id)
        df = graph.compare(status="failed")

        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


class TestExperimentGraphGetMetrics:
    """Tests for graph.get_metrics()."""

    def test_get_metrics_returns_dataframe(self, temp_dir):
        """Returns long-format DataFrame with expected columns."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "train", config={"lr": 0.01})
        _add_metrics(manager, a, [{"step": 1, "loss": 2.5}, {"step": 2, "loss": 1.5}])

        graph = ExperimentGraph.build(manager, a)
        df = graph.get_metrics()

        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert "experiment_id" in df.columns
        assert "step" in df.columns
        assert "metric_name" in df.columns
        assert "value" in df.columns

    def test_get_metrics_with_script_filter(self, temp_dir):
        """Filters to matching experiments."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "dataprep", "dataprep.py", config={"dataset": "mnist"}
        )
        b = _create_experiment(
            manager, temp_dir, "train", "train.py", config={"lr": 0.01}
        )
        _add_dependency(manager, b, a, "data")
        _add_metrics(manager, b, [{"step": 1, "loss": 2.0}])

        graph = ExperimentGraph.build(manager, a)
        df = graph.get_metrics(script_pattern="train.py")

        # Only train experiment's metrics
        assert all(df["experiment_id"] == b)

    def test_get_metrics_with_include_dep_params(self, temp_dir):
        """include_dep_params=True adds upstream params as columns."""
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(
            manager, temp_dir, "dataprep", "dataprep.py", config={"dataset": "mnist"}
        )
        b = _create_experiment(
            manager, temp_dir, "train", "train.py", config={"lr": 0.01}
        )
        _add_dependency(manager, b, a, "data")
        _add_metrics(manager, b, [{"step": 1, "loss": 2.0}])

        graph = ExperimentGraph.build(manager, a)
        df = graph.get_metrics(
            script_pattern="train.py",
            include_dep_params=True,
            params="all",
        )

        # Should have upstream param as column
        assert "dataset" in df.columns


class TestExperimentGraphModuleAPI:
    """Tests for yanex.results.get_graph() module-level function."""

    def test_module_get_graph(self, temp_dir, monkeypatch):
        manager = ExperimentManager(temp_dir / "experiments")

        a = _create_experiment(manager, temp_dir, "dataprep")
        b = _create_experiment(manager, temp_dir, "train")
        _add_dependency(manager, b, a, "data")

        import yanex.results as yr

        # Reset the default manager and point to our temp dir
        monkeypatch.setattr(yr, "_default_manager", None)
        monkeypatch.setenv("YANEX_EXPERIMENTS_DIR", str(temp_dir / "experiments"))

        graph = yr.get_graph(b)

        assert len(graph) == 2
        assert a in graph
