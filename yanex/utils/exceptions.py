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


class AmbiguousIDError(ExperimentError):
    """Raised when a short ID matches multiple experiments."""

    def __init__(self, short_id: str, matches: list[str]) -> None:
        match_list = "\n".join(f"  - {match}" for match in matches)
        super().__init__(
            f"Short ID '{short_id}' matches multiple experiments:\n{match_list}\n\n"
            "Use a longer ID to disambiguate."
        )
        self.short_id = short_id
        self.matches = matches


class CircularDependencyError(ExperimentError):
    """Raised when a circular dependency is detected."""

    pass


class InvalidDependencyError(ExperimentError):
    """Raised when a dependency validation fails."""

    pass


class AmbiguousArtifactError(ExperimentError):
    """Raised when an artifact is found in multiple dependency experiments."""

    def __init__(self, filename: str, experiment_ids: list[str]) -> None:
        exp_list = "\n".join(
            f"  {i + 1}. {exp_id}" for i, exp_id in enumerate(experiment_ids)
        )
        super().__init__(
            f"Artifact '{filename}' found in multiple experiments:\n{exp_list}\n\n"
            "Load explicitly from specific dependency:\n"
            "  deps = yanex.get_dependencies(transitive=True)\n"
            f"  artifact = deps[0].load_artifact('{filename}')"
        )
        self.filename = filename
        self.experiment_ids = experiment_ids


class ParameterConflictError(ExperimentError):
    """Raised when a parameter has conflicting values across dependencies.

    This error is raised when yanex.get_param() detects that a parameter
    has different values in the current experiment's config vs its dependencies.
    This prevents silent parameter mismatches in experiment pipelines.
    """

    def __init__(
        self,
        param_key: str,
        conflicts: dict,
    ) -> None:
        """Initialize ParameterConflictError.

        Args:
            param_key: The conflicting parameter key (e.g., "lr" or "model.lr")
            conflicts: Dict mapping values to list of (source, experiment_id) tuples.
                      source is "config" for local config or slot name for dependencies.
                      experiment_id is None for local config.
        """
        lines = [f"Parameter '{param_key}' has conflicting values:\n"]

        for value, sources in conflicts.items():
            lines.append(f"  Value: {value!r}")
            for source, exp_id in sources:
                if exp_id is None:
                    lines.append(f"    - {source}")
                else:
                    lines.append(f"    - dependency '{source}' ({exp_id})")
            lines.append("")

        lines.append("To resolve:")
        lines.append(
            f'  - Use specific dependency:  yanex.get_param("{param_key}", '
            'from_dependency="<slot>")'
        )
        lines.append(
            f'  - Use config only:          yanex.get_param("{param_key}", '
            "ignore_dependencies=True)"
        )

        super().__init__("\n".join(lines))
        self.param_key = param_key
        self.conflicts = conflicts


class AmbiguousKeyError(YanexError):
    """Raised when a key matches multiple canonical keys during resolution."""

    def __init__(self, key: str, matches: list[str]) -> None:
        match_list = "\n".join(f"  - {match}" for match in matches)
        super().__init__(
            f"'{key}' matches multiple keys:\n{match_list}\n\n"
            "Use a more specific key or the full qualified name."
        )
        self.key = key
        self.matches = matches


class InvalidGroupError(YanexError):
    """Raised when using wrong group prefix (e.g., --params metric:foo)."""

    def __init__(self, key: str, expected_group: str, actual_group: str) -> None:
        super().__init__(
            f"'{key}' belongs to '{actual_group}' group, "
            f"but was used with '{expected_group}' scope."
        )
        self.key = key
        self.expected_group = expected_group
        self.actual_group = actual_group


class KeyNotFoundError(YanexError):
    """Raised when a key cannot be resolved to any canonical key."""

    def __init__(self, key: str, scope: str | None = None) -> None:
        if scope:
            super().__init__(f"Key '{key}' not found in '{scope}' group.")
        else:
            super().__init__(f"Key '{key}' not found in any group.")
        self.key = key
        self.scope = scope
