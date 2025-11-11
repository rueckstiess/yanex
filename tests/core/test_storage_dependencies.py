"""Tests for dependency storage."""

import json
from pathlib import Path

import pytest

from yanex.core.storage_dependencies import FileSystemDependencyStorage
from yanex.core.storage_interfaces import ExperimentDirectoryManager
from yanex.utils.exceptions import StorageError


class MockDirectoryManager:
    """Mock directory manager for testing."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.experiments_dir = base_dir / "experiments"
        self.archived_dir = base_dir / "archived"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.archived_dir.mkdir(parents=True, exist_ok=True)

    def get_experiment_directory(
        self, experiment_id: str, include_archived: bool = False
    ) -> Path:
        """Get experiment directory path."""
        if include_archived:
            archived_path = self.archived_dir / experiment_id
            if archived_path.exists():
                return archived_path

        exp_path = self.experiments_dir / experiment_id
        exp_path.mkdir(parents=True, exist_ok=True)
        return exp_path


class TestFileSystemDependencyStorage:
    """Test the file system dependency storage."""

    @pytest.fixture
    def storage(self, temp_dir):
        """Create storage instance with temp directory."""
        dir_manager = MockDirectoryManager(temp_dir)
        return FileSystemDependencyStorage(dir_manager)

    @pytest.fixture
    def sample_dependencies(self):
        """Sample dependency data."""
        return {
            "version": "1.0",
            "declared_slots": {"dataprep": "dataprep.py"},
            "resolved_dependencies": {"dataprep": "dep123"},
            "validation": {
                "validated_at": "2024-01-01T00:00:00",
                "status": "valid",
                "checks": [],
            },
            "depended_by": [],
        }

    def test_save_dependencies(self, storage, temp_dir, sample_dependencies):
        """Test saving dependencies."""
        experiment_id = "exp123"

        storage.save_dependencies(experiment_id, sample_dependencies)

        deps_file = temp_dir / "experiments" / experiment_id / "dependencies.json"
        assert deps_file.exists()

        with deps_file.open("r") as f:
            loaded = json.load(f)

        assert loaded["version"] == "1.0"
        assert loaded["declared_slots"] == {"dataprep": "dataprep.py"}
        assert loaded["resolved_dependencies"] == {"dataprep": "dep123"}

    def test_save_dependencies_sorted_keys(
        self, storage, temp_dir, sample_dependencies
    ):
        """Test that saved dependencies have sorted keys."""
        experiment_id = "exp123"

        storage.save_dependencies(experiment_id, sample_dependencies)

        deps_file = temp_dir / "experiments" / experiment_id / "dependencies.json"
        content = deps_file.read_text()

        # Check that keys are sorted (declared_slots before resolved_dependencies, etc.)
        assert content.index("declared_slots") < content.index("resolved_dependencies")
        assert content.index("resolved_dependencies") < content.index("validation")

    def test_save_dependencies_with_archived(self, storage, temp_dir):
        """Test saving dependencies to archived experiment."""
        experiment_id = "exp123"

        # Create archived directory
        archived_dir = temp_dir / "archived" / experiment_id
        archived_dir.mkdir(parents=True, exist_ok=True)

        deps_data = {"version": "1.0", "depended_by": []}

        storage.save_dependencies(experiment_id, deps_data, include_archived=True)

        deps_file = archived_dir / "dependencies.json"
        assert deps_file.exists()

    def test_save_dependencies_failure(self, storage, temp_dir, monkeypatch):
        """Test save failure handling."""
        experiment_id = "exp123"

        # Mock json.dump to raise an exception
        def mock_dump(*args, **kwargs):
            raise IOError("Mock write failure")

        monkeypatch.setattr("json.dump", mock_dump)

        with pytest.raises(StorageError) as exc_info:
            storage.save_dependencies(experiment_id, {"version": "1.0"})

        assert "Failed to save dependencies" in str(exc_info.value)

    def test_load_dependencies(self, storage, temp_dir, sample_dependencies):
        """Test loading dependencies."""
        experiment_id = "exp123"

        # First save
        storage.save_dependencies(experiment_id, sample_dependencies)

        # Then load
        loaded = storage.load_dependencies(experiment_id)

        assert loaded is not None
        assert loaded["version"] == "1.0"
        assert loaded["declared_slots"] == {"dataprep": "dataprep.py"}
        assert loaded["resolved_dependencies"] == {"dataprep": "dep123"}

    def test_load_dependencies_not_exists(self, storage):
        """Test loading dependencies when file doesn't exist."""
        result = storage.load_dependencies("nonexistent")

        assert result is None

    def test_load_dependencies_with_archived(self, storage, temp_dir):
        """Test loading dependencies from archived experiment."""
        experiment_id = "exp123"

        # Create in archived directory
        archived_dir = temp_dir / "archived" / experiment_id
        archived_dir.mkdir(parents=True, exist_ok=True)
        deps_file = archived_dir / "dependencies.json"

        deps_data = {"version": "1.0", "archived": True}
        with deps_file.open("w") as f:
            json.dump(deps_data, f)

        # Load with include_archived
        loaded = storage.load_dependencies(experiment_id, include_archived=True)

        assert loaded is not None
        assert loaded["archived"] is True

    def test_load_dependencies_invalid_json(self, storage, temp_dir):
        """Test load failure with invalid JSON."""
        experiment_id = "exp123"

        # Create invalid JSON file
        exp_dir = temp_dir / "experiments" / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        deps_file = exp_dir / "dependencies.json"
        deps_file.write_text("{ invalid json }")

        with pytest.raises(StorageError) as exc_info:
            storage.load_dependencies(experiment_id)

        assert "Failed to load dependencies" in str(exc_info.value)

    def test_add_dependent(self, storage, temp_dir):
        """Test adding a dependent to experiment."""
        dependency_id = "dep123"
        dependent_id = "exp456"
        slot_name = "dataprep"

        # Add dependent
        storage.add_dependent(dependency_id, dependent_id, slot_name)

        # Load and check
        deps = storage.load_dependencies(dependency_id)

        assert deps is not None
        assert len(deps["depended_by"]) == 1
        assert deps["depended_by"][0]["experiment_id"] == dependent_id
        assert deps["depended_by"][0]["slot_name"] == slot_name
        assert "created_at" in deps["depended_by"][0]

    def test_add_dependent_to_existing(self, storage, temp_dir, sample_dependencies):
        """Test adding dependent to experiment that already has dependencies."""
        dependency_id = "dep123"
        dependent_id = "exp456"
        slot_name = "training"

        # Save existing dependencies first
        storage.save_dependencies(dependency_id, sample_dependencies)

        # Add dependent
        storage.add_dependent(dependency_id, dependent_id, slot_name)

        # Check it was added
        deps = storage.load_dependencies(dependency_id)
        assert len(deps["depended_by"]) == 1
        assert deps["depended_by"][0]["experiment_id"] == dependent_id

        # Check existing data preserved
        assert deps["declared_slots"] == sample_dependencies["declared_slots"]
        assert (
            deps["resolved_dependencies"]
            == sample_dependencies["resolved_dependencies"]
        )

    def test_add_multiple_dependents(self, storage):
        """Test adding multiple dependents."""
        dependency_id = "dep123"

        storage.add_dependent(dependency_id, "exp1", "slot1")
        storage.add_dependent(dependency_id, "exp2", "slot2")
        storage.add_dependent(dependency_id, "exp3", "slot1")

        deps = storage.load_dependencies(dependency_id)
        assert len(deps["depended_by"]) == 3

    def test_remove_dependent(self, storage):
        """Test removing a dependent."""
        dependency_id = "dep123"

        # Add two dependents
        storage.add_dependent(dependency_id, "exp1", "slot1")
        storage.add_dependent(dependency_id, "exp2", "slot2")

        # Remove one
        storage.remove_dependent(dependency_id, "exp1")

        # Check only exp2 remains
        deps = storage.load_dependencies(dependency_id)
        assert len(deps["depended_by"]) == 1
        assert deps["depended_by"][0]["experiment_id"] == "exp2"

    def test_remove_dependent_not_exists(self, storage):
        """Test removing dependent from experiment without dependencies."""
        # Should not raise error
        storage.remove_dependent("nonexistent", "exp1")

    def test_remove_dependent_from_empty_list(self, storage, sample_dependencies):
        """Test removing dependent when depended_by is empty."""
        dependency_id = "dep123"

        # Save with empty depended_by
        storage.save_dependencies(dependency_id, sample_dependencies)

        # Remove (shouldn't error)
        storage.remove_dependent(dependency_id, "exp1")

        # Check depended_by still empty
        deps = storage.load_dependencies(dependency_id)
        assert len(deps["depended_by"]) == 0

    def test_experiment_has_dependencies_true(self, storage, sample_dependencies):
        """Test checking if experiment has dependencies (true case)."""
        experiment_id = "exp123"

        storage.save_dependencies(experiment_id, sample_dependencies)

        assert storage.experiment_has_dependencies(experiment_id) is True

    def test_experiment_has_dependencies_false_no_file(self, storage):
        """Test checking dependencies when file doesn't exist."""
        assert storage.experiment_has_dependencies("nonexistent") is False

    def test_experiment_has_dependencies_false_empty(self, storage):
        """Test checking dependencies when resolved_dependencies is empty."""
        experiment_id = "exp123"

        deps_data = {
            "version": "1.0",
            "resolved_dependencies": {},  # Empty
            "depended_by": [],
        }

        storage.save_dependencies(experiment_id, deps_data)

        assert storage.experiment_has_dependencies(experiment_id) is False

    def test_experiment_has_dependencies_with_archived(self, storage, temp_dir):
        """Test checking dependencies in archived experiment."""
        experiment_id = "exp123"

        # Create in archived dir
        archived_dir = temp_dir / "archived" / experiment_id
        archived_dir.mkdir(parents=True, exist_ok=True)
        deps_file = archived_dir / "dependencies.json"

        deps_data = {
            "version": "1.0",
            "resolved_dependencies": {"slot": "dep456"},
        }

        with deps_file.open("w") as f:
            json.dump(deps_data, f)

        # Check with include_archived
        assert (
            storage.experiment_has_dependencies(experiment_id, include_archived=True)
            is True
        )

    def test_experiment_is_depended_on_true(self, storage):
        """Test checking if experiment is depended on (true case)."""
        dependency_id = "dep123"

        # Add a dependent
        storage.add_dependent(dependency_id, "exp456", "slot1")

        assert storage.experiment_is_depended_on(dependency_id) is True

    def test_experiment_is_depended_on_false_no_file(self, storage):
        """Test checking depended_on when file doesn't exist."""
        assert storage.experiment_is_depended_on("nonexistent") is False

    def test_experiment_is_depended_on_false_empty(self, storage):
        """Test checking depended_on when list is empty."""
        experiment_id = "exp123"

        deps_data = {"version": "1.0", "depended_by": []}  # Empty

        storage.save_dependencies(experiment_id, deps_data)

        assert storage.experiment_is_depended_on(experiment_id) is False

    def test_experiment_is_depended_on_with_archived(self, storage, temp_dir):
        """Test checking depended_on in archived experiment."""
        experiment_id = "exp123"

        # Create in archived dir
        archived_dir = temp_dir / "archived" / experiment_id
        archived_dir.mkdir(parents=True, exist_ok=True)
        deps_file = archived_dir / "dependencies.json"

        deps_data = {
            "version": "1.0",
            "depended_by": [{"experiment_id": "exp456", "slot_name": "slot1"}],
        }

        with deps_file.open("w") as f:
            json.dump(deps_data, f)

        # Check with include_archived
        assert (
            storage.experiment_is_depended_on(experiment_id, include_archived=True)
            is True
        )

    def test_round_trip_save_and_load(self, storage, sample_dependencies):
        """Test complete round-trip: save and load."""
        experiment_id = "exp123"

        # Save
        storage.save_dependencies(experiment_id, sample_dependencies)

        # Load
        loaded = storage.load_dependencies(experiment_id)

        # Should match exactly
        assert loaded == sample_dependencies

    def test_dependent_timestamps_unique(self, storage):
        """Test that dependent timestamps are created."""
        dependency_id = "dep123"

        storage.add_dependent(dependency_id, "exp1", "slot1")

        deps = storage.load_dependencies(dependency_id)
        dependent = deps["depended_by"][0]

        assert "created_at" in dependent
        assert "T" in dependent["created_at"]  # ISO format
