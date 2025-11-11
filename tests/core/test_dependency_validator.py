"""Tests for dependency validation."""

import pytest

from yanex.core.dependency_validator import DependencyValidator
from yanex.utils.exceptions import (
    CircularDependencyError,
    DependencyError,
    InvalidDependencyError,
)


class MockStorage:
    """Mock storage for testing validation."""

    def __init__(self):
        self.experiments = {}
        self.dependencies = {}

    def add_experiment(self, exp_id: str, script_path: str, status: str):
        """Add an experiment."""
        self.experiments[exp_id] = {
            "script_path": script_path,
            "status": status,
        }

    def add_dependencies(self, exp_id: str, resolved_deps: dict[str, str]):
        """Add dependencies for an experiment."""
        self.dependencies[exp_id] = {
            "resolved_dependencies": resolved_deps,
        }

    def experiment_exists(self, exp_id: str, include_archived: bool = False):
        """Check if experiment exists."""
        return exp_id in self.experiments

    def load_metadata(self, exp_id: str, include_archived: bool = False):
        """Load experiment metadata."""
        if exp_id not in self.experiments:
            raise FileNotFoundError(f"Experiment {exp_id} not found")
        return self.experiments[exp_id]

    def load_dependencies(self, exp_id: str, include_archived: bool = False):
        """Load dependencies."""
        return self.dependencies.get(exp_id)


class TestDependencyValidator:
    """Test the dependency validator."""

    def test_valid_single_dependency(self):
        """Test validation with one valid dependency."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")

        validator = DependencyValidator(storage)
        result = validator.validate_dependencies(
            experiment_id="exp123",
            declared_slots={"dataprep": "dataprep.py"},
            resolved_deps={"dataprep": "dep1"},
            check_cycles=False,  # No cycle check needed for single new experiment
        )

        assert result["status"] == "valid"
        assert len(result["checks"]) == 1
        check = result["checks"][0]
        assert check["slot"] == "dataprep"
        assert check["experiment_id"] == "dep1"
        assert check["script_match"] is True
        assert check["experiment_status"] == "completed"
        assert check["valid"] is True

    def test_valid_multiple_dependencies(self):
        """Test validation with multiple valid dependencies."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")
        storage.add_experiment("dep2", "/path/to/training.py", "completed")

        validator = DependencyValidator(storage)
        result = validator.validate_dependencies(
            experiment_id="exp123",
            declared_slots={"dataprep": "dataprep.py", "training": "training.py"},
            resolved_deps={"dataprep": "dep1", "training": "dep2"},
            check_cycles=False,
        )

        assert result["status"] == "valid"
        assert len(result["checks"]) == 2

    def test_missing_required_dependency(self):
        """Test that missing required dependency raises error."""
        storage = MockStorage()

        validator = DependencyValidator(storage)
        with pytest.raises(DependencyError) as exc_info:
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={"dataprep": "dataprep.py", "training": "training.py"},
                resolved_deps={"dataprep": "dep1"},  # Missing training
                check_cycles=False,
            )

        assert "Missing required dependency 'training'" in str(exc_info.value)

    def test_unknown_dependency_slot(self):
        """Test that unknown slot raises error."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")
        storage.add_experiment("dep2", "/path/to/unknown.py", "completed")

        validator = DependencyValidator(storage)
        with pytest.raises(InvalidDependencyError) as exc_info:
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={"dataprep": "dataprep.py"},
                resolved_deps={
                    "dataprep": "dep1",  # Required slot provided
                    "unknown_slot": "dep2",  # Unknown slot
                },
                check_cycles=False,
            )

        assert "Unknown dependency slot 'unknown_slot'" in str(exc_info.value)
        assert "Expected slots: dataprep" in str(exc_info.value)

    def test_unknown_slot_no_declared_slots(self):
        """Test unknown slot when no slots declared."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")

        validator = DependencyValidator(storage)
        with pytest.raises(InvalidDependencyError) as exc_info:
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={},  # No declared slots
                resolved_deps={"some_slot": "dep1"},
                check_cycles=False,
            )

        assert "Unknown dependency slot 'some_slot'" in str(exc_info.value)
        assert "No dependencies declared in config" in str(exc_info.value)

    def test_dependency_not_found(self):
        """Test that non-existent dependency raises error."""
        storage = MockStorage()

        validator = DependencyValidator(storage)
        with pytest.raises(DependencyError) as exc_info:
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={"dataprep": "dataprep.py"},
                resolved_deps={"dataprep": "nonexistent"},
                check_cycles=False,
            )

        assert "Dependency experiment 'nonexistent' not found" in str(exc_info.value)

    def test_script_mismatch(self):
        """Test that script mismatch raises error."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/wrong_script.py", "completed")

        validator = DependencyValidator(storage)
        with pytest.raises(DependencyError) as exc_info:
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={"dataprep": "dataprep.py"},
                resolved_deps={"dataprep": "dep1"},
                check_cycles=False,
            )

        assert "requires script 'dataprep.py'" in str(exc_info.value)
        assert "ran 'wrong_script.py'" in str(exc_info.value)

    def test_invalid_status_running(self):
        """Test that running dependency raises error."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "running")

        validator = DependencyValidator(storage)
        with pytest.raises(DependencyError) as exc_info:
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={"dataprep": "dataprep.py"},
                resolved_deps={"dataprep": "dep1"},
                check_cycles=False,
            )

        assert "has status 'running'" in str(exc_info.value)
        assert "Only 'completed' experiments" in str(exc_info.value)

    def test_invalid_status_failed(self):
        """Test that failed dependency raises error."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "failed")

        validator = DependencyValidator(storage)
        with pytest.raises(DependencyError) as exc_info:
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={"dataprep": "dataprep.py"},
                resolved_deps={"dataprep": "dep1"},
                check_cycles=False,
            )

        assert "has status 'failed'" in str(exc_info.value)

    def test_circular_dependency_detected(self):
        """Test that circular dependency is detected."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")
        # dep1 depends on exp123 (would create cycle)
        storage.add_dependencies("dep1", {"dep": "exp123"})

        validator = DependencyValidator(storage)
        with pytest.raises(CircularDependencyError):
            validator.validate_dependencies(
                experiment_id="exp123",
                declared_slots={"dataprep": "dataprep.py"},
                resolved_deps={"dataprep": "dep1"},
                check_cycles=True,  # Enable cycle check
            )

    def test_skip_circular_dependency_check(self):
        """Test that circular check can be skipped."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")
        storage.add_dependencies("dep1", {"dep": "exp123"})

        validator = DependencyValidator(storage)
        # Should not raise CircularDependencyError with check_cycles=False
        result = validator.validate_dependencies(
            experiment_id="exp123",
            declared_slots={"dataprep": "dataprep.py"},
            resolved_deps={"dataprep": "dep1"},
            check_cycles=False,
        )

        assert result["status"] == "valid"

    def test_no_experiment_id_skips_cycle_check(self):
        """Test that cycle check is skipped when experiment_id is None."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")

        validator = DependencyValidator(storage)
        # Should not raise even if cycle would exist
        result = validator.validate_dependencies(
            experiment_id=None,  # Not yet created
            declared_slots={"dataprep": "dataprep.py"},
            resolved_deps={"dataprep": "dep1"},
            check_cycles=True,
        )

        assert result["status"] == "valid"

    def test_normalize_shorthand_format(self):
        """Test normalization of shorthand slot format."""
        validator = DependencyValidator(None)

        normalized = validator._normalize_declared_slots(
            {"dataprep": "dataprep.py", "training": "training.py"}
        )

        assert normalized == {
            "dataprep": {"script": "dataprep.py", "required": True},
            "training": {"script": "training.py", "required": True},
        }

    def test_normalize_full_format(self):
        """Test normalization of full slot format."""
        validator = DependencyValidator(None)

        normalized = validator._normalize_declared_slots(
            {
                "dataprep": {"script": "dataprep.py", "required": True},
                "optional": {"script": "optional.py", "required": False},
            }
        )

        assert normalized["dataprep"]["required"] is True
        assert normalized["optional"]["required"] is False

    def test_normalize_mixed_format(self):
        """Test normalization with mixed formats."""
        validator = DependencyValidator(None)

        normalized = validator._normalize_declared_slots(
            {
                "dataprep": "dataprep.py",  # Shorthand
                "training": {"script": "training.py", "required": True},  # Full
            }
        )

        assert normalized["dataprep"] == {"script": "dataprep.py", "required": True}
        assert normalized["training"] == {"script": "training.py", "required": True}

    def test_normalize_missing_script_field(self):
        """Test that missing script field raises error."""
        validator = DependencyValidator(None)

        with pytest.raises(ValueError) as exc_info:
            validator._normalize_declared_slots(
                {
                    "dataprep": {"required": True}  # Missing script
                }
            )

        assert "missing 'script' field" in str(exc_info.value)

    def test_normalize_invalid_type(self):
        """Test that invalid slot type raises error."""
        validator = DependencyValidator(None)

        with pytest.raises(ValueError) as exc_info:
            validator._normalize_declared_slots(
                {
                    "dataprep": 123  # Invalid type (not string or dict)
                }
            )

        assert "Invalid slot configuration" in str(exc_info.value)

    def test_optional_dependency_not_provided(self):
        """Test that optional dependencies don't need to be provided."""
        storage = MockStorage()

        validator = DependencyValidator(storage)
        # Should not raise for missing optional dependency
        result = validator.validate_dependencies(
            experiment_id="exp123",
            declared_slots={"dataprep": {"script": "dataprep.py", "required": False}},
            resolved_deps={},  # Not provided
            check_cycles=False,
        )

        assert result["status"] == "valid"
        assert len(result["checks"]) == 0

    def test_validation_result_structure(self):
        """Test that validation result has correct structure."""
        storage = MockStorage()
        storage.add_experiment("dep1", "/path/to/dataprep.py", "completed")

        validator = DependencyValidator(storage)
        result = validator.validate_dependencies(
            experiment_id="exp123",
            declared_slots={"dataprep": "dataprep.py"},
            resolved_deps={"dataprep": "dep1"},
            check_cycles=False,
        )

        # Check structure
        assert "validated_at" in result
        assert "status" in result
        assert "checks" in result
        assert isinstance(result["checks"], list)

        # Check timestamp format
        assert "T" in result["validated_at"]  # ISO format

    def test_empty_dependencies(self):
        """Test validation with no dependencies."""
        storage = MockStorage()

        validator = DependencyValidator(storage)
        result = validator.validate_dependencies(
            experiment_id="exp123",
            declared_slots={},
            resolved_deps={},
            check_cycles=False,
        )

        assert result["status"] == "valid"
        assert len(result["checks"]) == 0

    def test_script_path_extraction(self):
        """Test that script name is correctly extracted from full path."""
        storage = MockStorage()
        # Store with full path
        storage.add_experiment(
            "dep1", "/very/long/path/to/scripts/dataprep.py", "completed"
        )

        validator = DependencyValidator(storage)
        # Should match just the filename
        result = validator.validate_dependencies(
            experiment_id="exp123",
            declared_slots={"dataprep": "dataprep.py"},
            resolved_deps={"dataprep": "dep1"},
            check_cycles=False,
        )

        assert result["status"] == "valid"
        check = result["checks"][0]
        assert check["script_actual"] == "dataprep.py"
