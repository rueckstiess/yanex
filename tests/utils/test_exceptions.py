"""
Tests for yanex.utils.exceptions module.
"""

from yanex.utils.exceptions import (
    ConfigError,
    DirtyWorkingDirectoryError,
    ExperimentAlreadyRunningError,
    ExperimentContextError,
    ExperimentError,
    ExperimentNotFoundError,
    GitError,
    StorageError,
    ValidationError,
    YanexError,
)


class TestBaseExceptions:
    """Test base exception classes."""

    def test_yanex_error_inheritance(self):
        """Test that YanexError inherits from Exception."""
        error = YanexError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_all_exceptions_inherit_from_yanex_error(self):
        """Test that all custom exceptions inherit from YanexError."""
        exception_classes = [
            ExperimentError,
            GitError,
            ConfigError,
            StorageError,
            ValidationError,
            ExperimentNotFoundError,
            ExperimentAlreadyRunningError,
            DirtyWorkingDirectoryError,
            ExperimentContextError,
        ]

        for exc_class in exception_classes:
            error = exc_class("test")
            assert isinstance(error, YanexError)
            assert isinstance(error, Exception)


class TestSpecificExceptions:
    """Test specific exception classes with custom behavior."""

    def test_experiment_not_found_error(self):
        """Test ExperimentNotFoundError with identifier."""
        identifier = "test_exp_123"
        error = ExperimentNotFoundError(identifier)

        assert error.identifier == identifier
        assert str(error) == f"Experiment not found: {identifier}"
        assert isinstance(error, ExperimentError)

    def test_experiment_already_running_error(self):
        """Test ExperimentAlreadyRunningError with running ID."""
        running_id = "abc12345"
        error = ExperimentAlreadyRunningError(running_id)

        assert error.running_id == running_id
        assert str(error) == f"Another experiment is already running: {running_id}"
        assert isinstance(error, ExperimentError)

    def test_dirty_working_directory_error(self):
        """Test DirtyWorkingDirectoryError with changes list."""
        changes = ["Modified: file1.py", "Untracked: file2.py", "Staged: file3.py"]
        error = DirtyWorkingDirectoryError(changes)

        assert error.changes == changes
        assert "Working directory is not clean:" in str(error)
        assert "Modified: file1.py" in str(error)
        assert "Untracked: file2.py" in str(error)
        assert "Staged: file3.py" in str(error)
        assert isinstance(error, GitError)

    def test_dirty_working_directory_error_empty_changes(self):
        """Test DirtyWorkingDirectoryError with empty changes list."""
        changes = []
        error = DirtyWorkingDirectoryError(changes)

        assert error.changes == []
        assert "Working directory is not clean:" in str(error)
