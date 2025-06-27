"""
Custom exceptions for yanex.
"""


class YanexError(Exception):
    """Base exception for all yanex errors."""

    pass


class ExperimentError(YanexError):
    """Errors related to experiment management."""

    pass


class GitError(YanexError):
    """Errors related to git operations."""

    pass


class ConfigError(YanexError):
    """Errors related to configuration loading and validation."""

    pass


class StorageError(YanexError):
    """Errors related to file storage operations."""

    pass


class ValidationError(YanexError):
    """Errors related to input validation."""

    pass


class ExperimentNotFoundError(ExperimentError):
    """Raised when an experiment cannot be found by ID or name."""

    def __init__(self, identifier: str) -> None:
        super().__init__(f"Experiment not found: {identifier}")
        self.identifier = identifier


class ExperimentAlreadyRunningError(ExperimentError):
    """Raised when trying to start an experiment while another is running."""

    def __init__(self, running_id: str) -> None:
        super().__init__(f"Another experiment is already running: {running_id}")
        self.running_id = running_id


class DirtyWorkingDirectoryError(GitError):
    """Raised when git working directory is not clean."""

    def __init__(self, changes: list[str]) -> None:
        change_list = "\n".join(f"  - {change}" for change in changes)
        super().__init__(f"Working directory is not clean:\n{change_list}")
        self.changes = changes


class ExperimentContextError(ExperimentError):
    """Raised when experiment context is used incorrectly."""

    pass
