"""
Tests for dependency tracking API functions.

Tests yanex.get_dependencies() and yanex.load_artifact() with dependency search.
"""

import pytest

import yanex
from yanex.core.manager import ExperimentManager
from yanex.utils.exceptions import AmbiguousArtifactError


class TestGetDependencies:
    """Test yanex.get_dependencies() API function."""

    def test_get_dependencies_no_context(self):
        """Test get_dependencies in standalone mode returns empty list."""
        # Clear any existing context
        yanex._clear_current_experiment_id()

        deps = yanex.get_dependencies()
        assert deps == []

    def test_get_dependencies_no_dependencies(self, temp_dir):
        """Test get_dependencies with experiment that has no dependencies."""
        # Use default ExperimentManager (session-wide isolated directory)
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment without dependencies
        exp_id = manager.create_experiment(script_path, config={})

        # Set experiment context
        yanex._set_current_experiment_id(exp_id)

        try:
            deps = yanex.get_dependencies()
            assert deps == []
        finally:
            yanex._clear_current_experiment_id()

    def test_get_dependencies_direct_only(self, temp_dir):
        """Test get_dependencies returns only direct dependencies."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C
        exp_c = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-c"
        )
        exp_b = manager.create_experiment(
            script_path,
            config={},
            dependency_ids=[exp_c],
            stage_only=True,
            name="exp-b",
        )
        exp_a = manager.create_experiment(
            script_path,
            config={},
            dependency_ids=[exp_b],
            stage_only=True,
            name="exp-a",
        )

        # Set context to A
        yanex._set_current_experiment_id(exp_a)

        try:
            deps = yanex.get_dependencies(transitive=False)

            # Should return only B (direct dependency)
            assert len(deps) == 1
            assert deps[0].id == exp_b
            assert deps[0].name == "exp-b"
            assert deps[0].status == "staged"
        finally:
            yanex._clear_current_experiment_id()

    def test_get_dependencies_transitive(self, temp_dir):
        """Test get_dependencies with transitive=True returns all dependencies."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C
        exp_c = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-c"
        )
        exp_b = manager.create_experiment(
            script_path,
            config={},
            dependency_ids=[exp_c],
            stage_only=True,
            name="exp-b",
        )
        exp_a = manager.create_experiment(
            script_path,
            config={},
            dependency_ids=[exp_b],
            stage_only=True,
            name="exp-a",
        )

        # Set context to A
        yanex._set_current_experiment_id(exp_a)

        try:
            deps = yanex.get_dependencies(transitive=True)

            # Should return B and C in topological order
            assert len(deps) == 2
            dep_ids = [d.id for d in deps]
            assert exp_b in dep_ids
            assert exp_c in dep_ids
            # C should come before B
            assert dep_ids.index(exp_c) < dep_ids.index(exp_b)
        finally:
            yanex._clear_current_experiment_id()

    def test_get_dependencies_include_self(self, temp_dir):
        """Test get_dependencies with include_self=True."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B
        exp_b = manager.create_experiment(
            script_path, config={}, stage_only=True, name="exp-b"
        )
        exp_a = manager.create_experiment(
            script_path,
            config={},
            dependency_ids=[exp_b],
            stage_only=True,
            name="exp-a",
        )

        # Set context to A
        yanex._set_current_experiment_id(exp_a)

        try:
            deps = yanex.get_dependencies(transitive=True, include_self=True)

            # Should return B and A
            assert len(deps) == 2
            dep_ids = [d.id for d in deps]
            assert exp_b in dep_ids
            assert exp_a in dep_ids
            # B should come before A
            assert dep_ids.index(exp_b) < dep_ids.index(exp_a)
        finally:
            yanex._clear_current_experiment_id()

    def test_dependency_object_methods(self, temp_dir):
        """Test Dependency object methods."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create dependency with artifacts
        dep_id = manager.create_experiment(
            script_path, config={}, stage_only=True, name="dep"
        )
        artifacts_dir = manager.storage.get_experiment_directory(dep_id) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / "model.txt").write_text("model data")
        (artifacts_dir / "data.txt").write_text('{"key": "value"}')

        # Create experiment depending on it
        exp_id = manager.create_experiment(
            script_path, config={}, dependency_ids=[dep_id], stage_only=True
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            deps = yanex.get_dependencies()
            dep = deps[0]

            # Test list_artifacts()
            artifacts = dep.list_artifacts()
            assert "model.txt" in artifacts
            assert "data.txt" in artifacts

            # Test load_artifact()
            model_data = dep.load_artifact("model.txt")
            assert model_data == "model data"

        finally:
            yanex._clear_current_experiment_id()


class TestLoadArtifactWithDependencies:
    """Test yanex.load_artifact() with dependency search."""

    def test_load_artifact_from_current(self, temp_dir):
        """Test loading artifact from current experiment."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment with artifact
        exp_id = manager.create_experiment(script_path, config={})
        artifacts_dir = manager.storage.get_experiment_directory(exp_id) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        (artifacts_dir / "model.txt").write_text("model data")

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Load artifact
            data = yanex.load_artifact("model.txt")
            assert data == "model data"
        finally:
            yanex._clear_current_experiment_id()

    def test_load_artifact_from_dependency(self, temp_dir):
        """Test loading artifact from dependency."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create dependency with artifact
        dep_id = manager.create_experiment(script_path, config={}, stage_only=True)
        dep_artifacts_dir = (
            manager.storage.get_experiment_directory(dep_id) / "artifacts"
        )
        dep_artifacts_dir.mkdir(exist_ok=True)
        (dep_artifacts_dir / "model.txt").write_text("model data")

        # Create experiment depending on it (staged for testing)
        exp_id = manager.create_experiment(
            script_path, config={}, dependency_ids=[dep_id], stage_only=True
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Load artifact (should find in dependency)
            data = yanex.load_artifact("model.txt")
            assert data == "model data"
        finally:
            yanex._clear_current_experiment_id()

    def test_load_artifact_from_transitive_dependency(self, temp_dir):
        """Test loading artifact from transitive dependency."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create chain: A -> B -> C, artifact in C
        exp_c = manager.create_experiment(script_path, config={}, stage_only=True)
        c_artifacts_dir = manager.storage.get_experiment_directory(exp_c) / "artifacts"
        c_artifacts_dir.mkdir(exist_ok=True)
        (c_artifacts_dir / "model.txt").write_text("model data")

        exp_b = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_c], stage_only=True
        )
        exp_a = manager.create_experiment(
            script_path, config={}, dependency_ids=[exp_b], stage_only=True
        )

        # Set context to A
        yanex._set_current_experiment_id(exp_a)

        try:
            # Load artifact (should find in transitive dependency C)
            data = yanex.load_artifact("model.txt")
            assert data == "model data"
        finally:
            yanex._clear_current_experiment_id()

    def test_load_artifact_not_found(self, temp_dir):
        """Test loading non-existent artifact returns None."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment without artifact
        exp_id = manager.create_experiment(script_path, config={})

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Load non-existent artifact
            data = yanex.load_artifact("nonexistent.txt")
            assert data is None
        finally:
            yanex._clear_current_experiment_id()

    def test_load_artifact_ambiguous_error(self, temp_dir):
        """Test loading artifact found in multiple dependencies raises error."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create two dependencies with same artifact
        dep1_id = manager.create_experiment(script_path, config={}, stage_only=True)
        dep1_artifacts_dir = (
            manager.storage.get_experiment_directory(dep1_id) / "artifacts"
        )
        dep1_artifacts_dir.mkdir(exist_ok=True)
        (dep1_artifacts_dir / "model.txt").write_text("model data 1")

        dep2_id = manager.create_experiment(script_path, config={}, stage_only=True)
        dep2_artifacts_dir = (
            manager.storage.get_experiment_directory(dep2_id) / "artifacts"
        )
        dep2_artifacts_dir.mkdir(exist_ok=True)
        (dep2_artifacts_dir / "model.txt").write_text("model data 2")

        # Create experiment depending on both
        exp_id = manager.create_experiment(
            script_path,
            config={},
            dependency_ids=[dep1_id, dep2_id],
            stage_only=True,
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Load artifact (should raise ambiguity error)
            with pytest.raises(AmbiguousArtifactError) as exc_info:
                yanex.load_artifact("model.txt")

            # Check error message
            assert "model.txt" in str(exc_info.value)
            assert dep1_id in str(exc_info.value)
            assert dep2_id in str(exc_info.value)
        finally:
            yanex._clear_current_experiment_id()

    def test_load_artifact_prefers_current_over_dependency(self, temp_dir):
        """Test that load_artifact prefers current experiment over dependency."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create dependency with artifact
        dep_id = manager.create_experiment(script_path, config={}, stage_only=True)
        dep_artifacts_dir = (
            manager.storage.get_experiment_directory(dep_id) / "artifacts"
        )
        dep_artifacts_dir.mkdir(exist_ok=True)
        (dep_artifacts_dir / "model.txt").write_text("dep model data")

        # Create experiment with same artifact (staged for testing)
        exp_id = manager.create_experiment(
            script_path, config={}, dependency_ids=[dep_id], stage_only=True
        )
        exp_artifacts_dir = (
            manager.storage.get_experiment_directory(exp_id) / "artifacts"
        )
        exp_artifacts_dir.mkdir(exist_ok=True)
        (exp_artifacts_dir / "model.txt").write_text("exp model data")

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Load artifact (should find in current experiment, not dependency)
            data = yanex.load_artifact("model.txt")
            assert data == "exp model data"
        finally:
            yanex._clear_current_experiment_id()

    def test_load_artifact_standalone_mode(self, temp_dir):
        """Test load_artifact in standalone mode doesn't search dependencies."""
        # Clear any existing context
        yanex._clear_current_experiment_id()

        # Ensure we're in standalone mode
        assert yanex.is_standalone()

        # Load should return None (no artifacts in standalone mode)
        data = yanex.load_artifact("model.txt")
        assert data is None
