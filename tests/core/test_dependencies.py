"""
Tests for dependency tracking functionality.

Tests DependencyStorage, DependencyResolver, and dependency integration with ExperimentManager.
"""

import pytest

from yanex.core.dependencies import DependencyResolver
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import (
    ExperimentNotFoundError,
    InvalidDependencyError,
)


class TestDependencyStorage:
    """Test DependencyStorage for persistence of dependency data."""

    def test_save_and_load_dependencies(self, temp_dir):
        """Test saving and loading dependencies."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create experiment
        exp_id = manager.generate_experiment_id()
        manager.storage.create_experiment_directory(exp_id)

        # Save dependencies
        dep_ids = ["abc12345", "def67890"]
        metadata = {"dep1": "info"}
        manager.storage.dependency_storage.save_dependencies(exp_id, dep_ids, metadata)

        # Load dependencies
        loaded = manager.storage.dependency_storage.load_dependencies(exp_id)

        assert loaded["dependency_ids"] == dep_ids
        assert loaded["metadata"] == metadata
        assert "created_at" in loaded

    def test_load_nonexistent_dependencies(self, temp_dir):
        """Test loading dependencies that don't exist returns empty."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create experiment without dependencies file
        exp_id = manager.generate_experiment_id()
        manager.storage.create_experiment_directory(exp_id)

        # Load should return empty
        loaded = manager.storage.dependency_storage.load_dependencies(exp_id)

        assert loaded["dependency_ids"] == []
        assert loaded["created_at"] is None
        assert loaded["metadata"] == {}

    def test_dependency_file_exists(self, temp_dir):
        """Test checking if dependencies file exists."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create experiment
        exp_id = manager.generate_experiment_id()
        manager.storage.create_experiment_directory(exp_id)

        # Should not exist initially
        assert not manager.storage.dependency_storage.dependency_file_exists(exp_id)

        # Save dependencies
        manager.storage.dependency_storage.save_dependencies(exp_id, ["abc12345"])

        # Should exist now
        assert manager.storage.dependency_storage.dependency_file_exists(exp_id)


class TestDependencyResolver:
    """Test DependencyResolver for resolution and validation."""

    def test_resolve_short_id(self, temp_dir):
        """Test resolving short experiment IDs."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create experiment
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")
        exp_id = manager.create_experiment(script_path, config={})

        # Resolve using full ID
        resolver = DependencyResolver(manager)
        assert resolver.resolve_short_id(exp_id) == exp_id

        # Resolve using short ID prefix
        short_id = exp_id[:4]
        assert resolver.resolve_short_id(short_id) == exp_id

    def test_resolve_short_id_not_found(self, temp_dir):
        """Test resolving non-existent ID raises error."""
        manager = ExperimentManager(temp_dir / "experiments")
        resolver = DependencyResolver(manager)

        with pytest.raises(ExperimentNotFoundError):
            resolver.resolve_short_id("notfound")

    def test_resolve_short_id_ambiguous(self, temp_dir):
        """Test resolving ambiguous short ID raises error."""
        # This test is skipped because it's difficult to create ambiguous IDs
        # without mocking. The functionality is tested in test_id_resolution.py
        pytest.skip("Ambiguous ID testing requires mocking ID generation")

    def test_validate_dependency_completed(self, temp_dir):
        """Test validating completed dependency."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create and complete experiment
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")
        exp_id = manager.create_experiment(script_path, config={})
        manager.start_experiment(exp_id)
        manager.complete_experiment(exp_id)

        # Validate should pass
        resolver = DependencyResolver(manager)
        resolver.validate_dependency(exp_id, for_staging=False)  # Should not raise

    def test_validate_dependency_staged(self, temp_dir):
        """Test validating staged dependency."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create staged experiment
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")
        exp_id = manager.create_experiment(script_path, config={}, stage_only=True)

        # Validation should fail without for_staging
        resolver = DependencyResolver(manager)
        with pytest.raises(InvalidDependencyError):
            resolver.validate_dependency(exp_id, for_staging=False)

        # Validation should pass with for_staging=True
        resolver.validate_dependency(exp_id, for_staging=True)  # Should not raise

    def test_validate_dependency_running(self, temp_dir):
        """Test validating running dependency fails."""
        manager = ExperimentManager(temp_dir / "experiments")

        # Create and start experiment
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")
        exp_id = manager.create_experiment(script_path, config={})
        manager.start_experiment(exp_id)

        # Validation should fail
        resolver = DependencyResolver(manager)
        with pytest.raises(InvalidDependencyError) as exc_info:
            resolver.validate_dependency(exp_id)

        assert "invalid status" in str(exc_info.value).lower()

    def test_validate_dependency_not_found(self, temp_dir):
        """Test validating non-existent dependency fails."""
        manager = ExperimentManager(temp_dir / "experiments")
        resolver = DependencyResolver(manager)

        with pytest.raises(ExperimentNotFoundError):
            resolver.validate_dependency("notfound")

    def test_get_transitive_dependencies_linear(self, temp_dir):
        """Test getting transitive dependencies in linear chain."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C
        exp_c = manager.create_experiment(script_path, config={}, stage_only=True)
        exp_b = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_c], stage_only=True
        )
        exp_a = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_b], stage_only=True
        )

        # Get transitive dependencies of A
        resolver = DependencyResolver(manager)
        deps = resolver.get_transitive_dependencies(exp_a)

        # Should return [C, B] in topological order
        assert len(deps) == 2
        assert exp_c in deps
        assert exp_b in deps
        # C should come before B (dependencies before dependents)
        assert deps.index(exp_c) < deps.index(exp_b)

    def test_get_transitive_dependencies_diamond(self, temp_dir):
        """Test getting transitive dependencies in diamond pattern."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create diamond: D -> B,C -> A
        exp_a = manager.create_experiment(script_path, config={}, stage_only=True)
        exp_b = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_a], stage_only=True
        )
        exp_c = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_a], stage_only=True
        )
        exp_d = manager.create_experiment(
            script_path,
            config={},
            dependency_ids=[exp_b, exp_c],
            stage_only=True,
        )

        # Get transitive dependencies of D
        resolver = DependencyResolver(manager)
        deps = resolver.get_transitive_dependencies(exp_d)

        # Should return [A, B, C] in topological order
        assert len(deps) == 3
        assert exp_a in deps
        assert exp_b in deps
        assert exp_c in deps
        # A should come before B and C
        assert deps.index(exp_a) < deps.index(exp_b)
        assert deps.index(exp_a) < deps.index(exp_c)

    def test_get_transitive_dependencies_include_self(self, temp_dir):
        """Test getting transitive dependencies with include_self=True."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B
        exp_b = manager.create_experiment(script_path, config={}, stage_only=True)
        exp_a = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_b], stage_only=True
        )

        # Get transitive dependencies with include_self
        resolver = DependencyResolver(manager)
        deps = resolver.get_transitive_dependencies(exp_a, include_self=True)

        # Should return [B, A]
        assert len(deps) == 2
        assert exp_b in deps
        assert exp_a in deps
        assert deps.index(exp_b) < deps.index(exp_a)

    def test_detect_circular_dependency_direct(self, temp_dir):
        """Test detecting direct circular dependency."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment A
        exp_a = manager.create_experiment(script_path, config={}, stage_only=True)

        # Try to make A depend on itself
        resolver = DependencyResolver(manager)
        is_circular = resolver.detect_circular_dependency(exp_a, exp_a)

        assert is_circular

    def test_detect_circular_dependency_indirect(self, temp_dir):
        """Test detecting indirect circular dependency."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B
        exp_b = manager.create_experiment(script_path, config={}, stage_only=True)
        exp_a = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_b], stage_only=True
        )

        # Try to make B depend on A (creates cycle)
        resolver = DependencyResolver(manager)
        is_circular = resolver.detect_circular_dependency(exp_b, exp_a)

        assert is_circular

    def test_find_artifact_in_dependencies_current(self, temp_dir):
        """Test finding artifact in current experiment."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment with artifact
        exp_id = manager.create_experiment(script_path, config={})
        artifact_path = (
            manager.storage.get_experiment_directory(exp_id) / "artifacts" / "model.pkl"
        )
        artifact_path.parent.mkdir(exist_ok=True)
        artifact_path.write_text("artifact data")

        # Find artifact
        resolver = DependencyResolver(manager)
        found_in, all_matches = resolver.find_artifact_in_dependencies(
            exp_id, "model.pkl"
        )

        assert found_in == exp_id
        assert all_matches == [exp_id]

    def test_find_artifact_in_dependencies_dep(self, temp_dir):
        """Test finding artifact in dependency."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create dependency with artifact (staged so it can be a dependency)
        dep_id = manager.create_experiment(script_path, config={}, stage_only=True)
        artifact_path = (
            manager.storage.get_experiment_directory(dep_id) / "artifacts" / "model.pkl"
        )
        artifact_path.parent.mkdir(exist_ok=True)
        artifact_path.write_text("artifact data")

        # Create experiment depending on it
        exp_id = manager.create_experiment(
            script_path, config={}, dependency_ids=[dep_id], stage_only=True
        )

        # Find artifact
        resolver = DependencyResolver(manager)
        found_in, all_matches = resolver.find_artifact_in_dependencies(
            exp_id, "model.pkl"
        )

        assert found_in == dep_id
        assert all_matches == [dep_id]

    def test_find_artifact_in_dependencies_ambiguous(self, temp_dir):
        """Test finding artifact in multiple dependencies."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create two dependencies with same artifact (staged so they can be dependencies)
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
            dependency_ids=[dep1_id, dep2_id],
            stage_only=True,
        )

        # Find artifact (should be ambiguous)
        resolver = DependencyResolver(manager)
        found_in, all_matches = resolver.find_artifact_in_dependencies(
            exp_id, "model.pkl"
        )

        assert found_in is None  # Ambiguous
        assert set(all_matches) == {dep1_id, dep2_id}

    def test_find_artifact_in_dependencies_not_found(self, temp_dir):
        """Test finding artifact that doesn't exist."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment without artifact
        exp_id = manager.create_experiment(script_path, config={})

        # Find artifact
        resolver = DependencyResolver(manager)
        found_in, all_matches = resolver.find_artifact_in_dependencies(
            exp_id, "model.pkl"
        )

        assert found_in is None
        assert all_matches == []


class TestExperimentManagerDependencies:
    """Test ExperimentManager integration with dependencies."""

    def test_create_experiment_with_dependencies(self, temp_dir):
        """Test creating experiment with dependencies."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create dependency
        dep_id = manager.create_experiment(script_path, config={}, stage_only=True)

        # Create experiment with dependency
        exp_id = manager.create_experiment(
            script_path, config={}, dependency_ids=[dep_id], stage_only=True
        )

        # Check dependencies were saved
        deps = manager.storage.dependency_storage.load_dependencies(exp_id)
        assert deps["dependency_ids"] == [dep_id]

    def test_create_experiment_with_short_dependency_ids(self, temp_dir):
        """Test creating experiment with short dependency IDs."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create dependency
        dep_id = manager.create_experiment(script_path, config={}, stage_only=True)

        # Create experiment with short dependency ID
        short_id = dep_id[:4]
        exp_id = manager.create_experiment(
            script_path, config={}, dependency_ids=[short_id], stage_only=True
        )

        # Check full ID was resolved and saved
        deps = manager.storage.dependency_storage.load_dependencies(exp_id)
        assert deps["dependency_ids"] == [dep_id]

    def test_create_experiment_circular_dependency_error(self, temp_dir):
        """Test creating experiment with circular dependency fails."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment chain: A -> B
        exp_b = manager.create_experiment(script_path, config={}, stage_only=True)
        exp_a = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_b], stage_only=True
        )

        # Try to make B depend on A (creates cycle: A -> B -> A)
        # This requires manually creating the circular dependency since the
        # create_experiment validation prevents it
        # We test the detection in DependencyResolver tests instead
        resolver = DependencyResolver(manager)
        is_circular = resolver.detect_circular_dependency(exp_b, exp_a)
        assert is_circular  # Should detect the cycle

    def test_create_experiment_invalid_dependency_status(self, temp_dir):
        """Test creating experiment with invalid dependency status fails."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create running experiment
        dep_id = manager.create_experiment(script_path, config={})
        manager.start_experiment(dep_id)

        # Try to use as dependency (should fail - not completed)
        with pytest.raises(InvalidDependencyError):
            manager.create_experiment(script_path, config={}, dependency_ids=[dep_id])

    def test_create_experiment_nonexistent_dependency(self, temp_dir):
        """Test creating experiment with non-existent dependency fails."""
        manager = ExperimentManager(temp_dir / "experiments")
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Try to create with non-existent dependency
        with pytest.raises(ExperimentNotFoundError):
            manager.create_experiment(
                script_path, config={}, dependency_ids=["notfound"]
            )
