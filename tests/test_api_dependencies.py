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
        """Test get_dependencies in standalone mode returns empty dict."""
        # Clear any existing context
        yanex._clear_current_experiment_id()

        deps = yanex.get_dependencies()
        assert deps == {}

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
            assert deps == {}
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

        # Set context to A
        yanex._set_current_experiment_id(exp_a)

        try:
            deps = yanex.get_dependencies(transitive=False)

            # Should return dict with slot -> Experiment (only B)
            assert len(deps) == 1
            assert "dep1" in deps
            assert deps["dep1"].id == exp_b
            assert deps["dep1"].name == "exp-b"
            assert deps["dep1"].status == "staged"
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
            dependencies={"dep1": exp_b},
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
            script_path, config={}, dependencies={"dep1": dep_id}, stage_only=True
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            deps = yanex.get_dependencies()
            dep = deps["dep1"]

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
            script_path, config={}, dependencies={"dep1": dep_id}, stage_only=True
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
            script_path, config={}, dependencies={"dep1": exp_c}, stage_only=True
        )
        exp_a = manager.create_experiment(
            script_path, config={}, dependencies={"dep1": exp_b}, stage_only=True
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
            dependencies={"dep1": dep1_id, "dep2": dep2_id},
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
            script_path, config={}, dependencies={"dep1": dep_id}, stage_only=True
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


class TestAssertDependency:
    """Test yanex.assert_dependency() API function."""

    def test_assert_dependency_success(self, temp_dir):
        """Test assert_dependency succeeds when dependency matches."""
        manager = ExperimentManager()
        prep_script = temp_dir / "prepare_data.py"
        prep_script.write_text("print('prepare')")
        train_script = temp_dir / "train_model.py"
        train_script.write_text("print('train')")

        # Create preprocessing dependency
        dep_id = manager.create_experiment(
            prep_script, config={}, stage_only=True, name="prep"
        )

        # Create training experiment depending on it
        exp_id = manager.create_experiment(
            train_script,
            config={},
            dependencies={"dep1": dep_id},
            stage_only=True,
            name="train",
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Should not raise - dependency from prepare_data.py exists
            yanex.assert_dependency("prepare_data.py")
        finally:
            yanex._clear_current_experiment_id()

    def test_assert_dependency_no_dependencies(self, temp_dir):
        """Test assert_dependency fails when experiment has no dependencies."""
        manager = ExperimentManager()
        script_path = temp_dir / "test.py"
        script_path.write_text("print('test')")

        # Create experiment without dependencies
        exp_id = manager.create_experiment(script_path, config={})

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Should call fail() which raises _ExperimentFailedException
            with pytest.raises(yanex._ExperimentFailedException) as exc_info:
                yanex.assert_dependency("prepare_data.py")

            # Check that the failure message mentions the missing dependency
            assert "prepare_data.py" in str(exc_info.value)
        finally:
            yanex._clear_current_experiment_id()

    def test_assert_dependency_wrong_script(self, temp_dir):
        """Test assert_dependency fails when dependency script doesn't match."""
        manager = ExperimentManager()
        prep_script = temp_dir / "prepare_data.py"
        prep_script.write_text("print('prepare')")
        train_script = temp_dir / "train_model.py"
        train_script.write_text("print('train')")

        # Create preprocessing dependency
        dep_id = manager.create_experiment(
            prep_script, config={}, stage_only=True, name="prep"
        )

        # Create training experiment depending on it
        exp_id = manager.create_experiment(
            train_script,
            config={},
            dependencies={"dep1": dep_id},
            stage_only=True,
            name="train",
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Should call fail() which raises _ExperimentFailedException
            with pytest.raises(yanex._ExperimentFailedException) as exc_info:
                yanex.assert_dependency("different_script.py")

            # Error should mention the missing dependency
            assert "different_script.py" in str(exc_info.value)
        finally:
            yanex._clear_current_experiment_id()

    def test_assert_dependency_standalone_mode(self):
        """Test assert_dependency is a no-op in standalone mode."""
        # Clear any existing context
        yanex._clear_current_experiment_id()

        # Ensure we're in standalone mode
        assert yanex.is_standalone()

        # Should be a no-op - doesn't raise, just returns
        yanex.assert_dependency("prepare_data.py")
        # If we get here, test passes

    def test_assert_dependency_multiple_dependencies(self, temp_dir):
        """Test assert_dependency with multiple dependencies finds match."""
        manager = ExperimentManager()
        prep_script = temp_dir / "prepare_data.py"
        prep_script.write_text("print('prepare')")
        load_script = temp_dir / "load_model.py"
        load_script.write_text("print('load')")
        eval_script = temp_dir / "evaluate.py"
        eval_script.write_text("print('evaluate')")

        # Create two dependencies
        dep1_id = manager.create_experiment(
            prep_script, config={}, stage_only=True, name="prep"
        )
        dep2_id = manager.create_experiment(
            load_script, config={}, stage_only=True, name="load"
        )

        # Create experiment depending on both
        exp_id = manager.create_experiment(
            eval_script,
            config={},
            dependencies={"dep1": dep1_id, "dep2": dep2_id},
            stage_only=True,
            name="eval",
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Should find prepare_data.py in dependencies
            yanex.assert_dependency("prepare_data.py")

            # Should find load_model.py in dependencies
            yanex.assert_dependency("load_model.py")

            # Should fail for non-existent dependency
            with pytest.raises(yanex._ExperimentFailedException) as exc_info:
                yanex.assert_dependency("missing.py")

            # Error should mention the missing dependency
            assert "missing.py" in str(exc_info.value)
        finally:
            yanex._clear_current_experiment_id()

    def test_assert_dependency_with_path_components(self, temp_dir):
        """Test assert_dependency matches just the filename, not full path."""
        manager = ExperimentManager()
        # Create script with subdirectory
        subdir = temp_dir / "scripts"
        subdir.mkdir(exist_ok=True)
        prep_script = subdir / "prepare_data.py"
        prep_script.write_text("print('prepare')")
        train_script = temp_dir / "train_model.py"
        train_script.write_text("print('train')")

        # Create preprocessing dependency
        dep_id = manager.create_experiment(
            prep_script, config={}, stage_only=True, name="prep"
        )

        # Create training experiment depending on it
        exp_id = manager.create_experiment(
            train_script,
            config={},
            dependencies={"dep1": dep_id},
            stage_only=True,
            name="train",
        )

        # Set context
        yanex._set_current_experiment_id(exp_id)

        try:
            # Should match using just the filename
            yanex.assert_dependency("prepare_data.py")
        finally:
            yanex._clear_current_experiment_id()
