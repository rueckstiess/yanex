"""Tests for edge cases and error handling in dependency tracking."""

import pytest

from yanex.core.dependencies import DependencyResolver
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import CircularDependencyError, ExperimentNotFoundError


class TestDependencyStorageErrorHandling:
    """Test error handling in DependencyStorage."""

    def test_load_dependencies_malformed_json(self, temp_dir):
        """Test loading dependencies with malformed JSON file."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment
        exp_id = manager.create_experiment(script_path, config={}, stage_only=True)

        # Manually corrupt the dependencies file
        dep_file = (
            manager.storage.get_experiment_directory(exp_id) / "dependencies.json"
        )
        # First save some dependencies to create the file
        manager.storage.dependency_storage.save_dependencies(
            exp_id, {"dep1": "abc12345"}, {}
        )

        # Now corrupt it with malformed JSON
        dep_file.write_text("{malformed json")

        # Should return empty dependencies instead of crashing
        loaded = manager.storage.dependency_storage.load_dependencies(exp_id)
        assert loaded["dependencies"] == {}

    def test_load_dependencies_missing_dependencies_file(self, temp_dir):
        """Test loading dependencies when dependencies.json doesn't exist."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment without dependencies (no dependencies.json file)
        exp_id = manager.create_experiment(script_path, config={}, stage_only=True)

        # Ensure no dependencies file exists
        dep_file = (
            manager.storage.get_experiment_directory(exp_id) / "dependencies.json"
        )
        if dep_file.exists():
            dep_file.unlink()

        # Should return empty dependencies
        loaded = manager.storage.dependency_storage.load_dependencies(exp_id)
        assert loaded["dependencies"] == {}

    def test_dependency_file_exists_with_exception(self, temp_dir):
        """Test dependency_file_exists error handling."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Try with non-existent experiment (should return False, not crash)
        exists = manager.storage.dependency_storage.dependency_file_exists("notfound")
        assert exists is False


class TestDependencyResolverErrorHandling:
    """Test error handling in DependencyResolver."""

    def test_validate_dependency_nonexistent_experiment(self, temp_dir):
        """Test validating non-existent experiment raises error."""
        manager = ExperimentManager(temp_dir / "experiments")
        resolver = DependencyResolver(manager)

        with pytest.raises(ExperimentNotFoundError):
            resolver.validate_dependency("notfound12")

    def test_circular_dependency_detection(self, temp_dir):
        """Test circular dependency detection with topological sort."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create two experiments
        exp_a = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-a"
        )
        exp_b = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-b"
        )

        # Manually create circular dependency: A -> B, B -> A
        manager.storage.dependency_storage.save_dependencies(exp_a, {"dep1": exp_b}, {})
        manager.storage.dependency_storage.save_dependencies(exp_b, {"dep1": exp_a}, {})

        # Try to get transitive dependencies (should raise CircularDependencyError)
        resolver = DependencyResolver(manager)
        with pytest.raises(CircularDependencyError):
            resolver.get_transitive_dependencies(exp_a)

    def test_detect_circular_dependency_method(self, temp_dir):
        """Test detect_circular_dependency method."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B
        exp_b = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-b"
        )
        exp_a = manager.create_experiment(
            script_path,
            config={},
            dependencies={"dep1": exp_b},
            stage_only=True,
            name="exp-a",
        )

        # Try to make B depend on A (would create cycle)
        resolver = DependencyResolver(manager)
        is_circular = resolver.detect_circular_dependency(exp_b, exp_a)
        assert is_circular is True

        # Non-circular case
        exp_c = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-c"
        )
        is_circular = resolver.detect_circular_dependency(exp_a, exp_c)
        assert is_circular is False


class TestArtifactFindingEdgeCases:
    """Test edge cases in artifact finding."""

    def test_artifact_not_found_returns_none(self, temp_dir):
        """Test finding non-existent artifact."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment without artifacts
        exp_id = manager.create_experiment(script_path, config={}, stage_only=True)

        # Find non-existent artifact
        resolver = DependencyResolver(manager)
        found_in, all_matches = resolver.find_artifact_in_dependencies(
            exp_id, "nonexistent.pkl"
        )

        assert found_in is None
        assert all_matches == []

    def test_artifact_ambiguous_multiple_matches(self, temp_dir):
        """Test finding artifact that exists in multiple dependencies."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create two dependencies with same artifact
        dep1_id = manager.create_experiment(script_path, config={}, stage_only=True)
        artifact1_path = (
            manager.storage.get_experiment_directory(dep1_id)
            / "artifacts"
            / "model.pkl"
        )
        artifact1_path.parent.mkdir(exist_ok=True)
        artifact1_path.write_text("artifact data 1")

        dep2_id = manager.create_experiment(script_path, config={}, stage_only=True)
        artifact2_path = (
            manager.storage.get_experiment_directory(dep2_id)
            / "artifacts"
            / "model.pkl"
        )
        artifact2_path.parent.mkdir(exist_ok=True)
        artifact2_path.write_text("artifact data 2")

        # Create experiment depending on both
        exp_id = manager.create_experiment(
            script_path,
            config={},
            dependencies={"data": dep1_id, "model": dep2_id},
            stage_only=True,
        )

        # Find artifact (should be ambiguous)
        resolver = DependencyResolver(manager)
        found_in, all_matches = resolver.find_artifact_in_dependencies(
            exp_id, "model.pkl"
        )

        assert found_in is None  # Ambiguous
        assert len(all_matches) == 2
        assert dep1_id in all_matches
        assert dep2_id in all_matches


class TestTransitiveDependenciesEdgeCases:
    """Test edge cases in transitive dependency resolution."""

    def test_transitive_dependencies_with_include_self(self, temp_dir):
        """Test transitive dependencies with include_self=True."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C
        exp_c = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-c"
        )
        exp_b = manager.create_experiment(
            script_path,
            config={},
            dependencies={"dep1": exp_c},
            stage_only=True,
            name="exp-b",
        )
        exp_a = manager.create_experiment(
            script_path,
            config={},
            dependencies={"dep1": exp_b},
            stage_only=True,
            name="exp-a",
        )

        # Get transitive dependencies with include_self
        resolver = DependencyResolver(manager)
        deps = resolver.get_transitive_dependencies(exp_a, include_self=True)

        # Should include A, B, C in topological order
        assert exp_a in deps
        assert exp_b in deps
        assert exp_c in deps

        # C should come before B, B should come before A
        assert deps.index(exp_c) < deps.index(exp_b)
        assert deps.index(exp_b) < deps.index(exp_a)

    def test_transitive_dependencies_without_include_self(self, temp_dir):
        """Test transitive dependencies with include_self=False (default)."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C
        exp_c = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-c"
        )
        exp_b = manager.create_experiment(
            script_path,
            config={},
            dependencies={"dep1": exp_c},
            stage_only=True,
            name="exp-b",
        )
        exp_a = manager.create_experiment(
            script_path,
            config={},
            dependencies={"dep1": exp_b},
            stage_only=True,
            name="exp-a",
        )

        # Get transitive dependencies without include_self
        resolver = DependencyResolver(manager)
        deps = resolver.get_transitive_dependencies(exp_a, include_self=False)

        # Should NOT include A
        assert exp_a not in deps
        assert exp_b in deps
        assert exp_c in deps
