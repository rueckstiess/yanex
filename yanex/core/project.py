"""Project detection and resolution utilities.

A "project" in yanex is identified by the last path component of the git
repository root (e.g., /Users/thomas/code/myproject → "myproject").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def detect_project_from_cwd() -> str | None:
    """Detect project name from the current working directory's git repo.

    Finds the closest git repository root by searching parent directories,
    then returns the last path component as the project name.

    Returns:
        Project name (last component of repo root path), or None if not
        in a git repository.
    """
    try:
        from .git_utils import get_git_repo

        repo = get_git_repo(Path.cwd())
        return Path(repo.working_dir).name
    except Exception:
        return None


def derive_project_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Derive project name from existing experiment metadata.

    Extracts the project from the stored git repository path
    (environment.git.repository.repo_path) for backward compatibility
    with experiments created before the project feature.

    Args:
        metadata: Experiment metadata dictionary.

    Returns:
        Project name derived from repo_path, or None if not available.
    """
    try:
        repo_path = (
            metadata.get("environment", {})
            .get("git", {})
            .get("repository", {})
            .get("repo_path")
        )
        if repo_path:
            return Path(repo_path).name
    except Exception:
        pass
    return None


def resolve_project_for_run(
    cli_project: str | None,
    config_project: str | None,
) -> str | None:
    """Resolve project name for the ``yanex run`` command.

    Priority (highest to lowest):
        1. Explicit CLI ``--project`` argument
        2. Config file ``yanex.project`` key
        3. Auto-detect from current working directory's git repo

    Args:
        cli_project: Value from ``--project`` CLI flag (or None).
        config_project: Value from ``yanex.project`` config key (or None).

    Returns:
        Resolved project name, or None if detection fails.
    """
    if cli_project is not None:
        return cli_project
    if config_project is not None:
        return config_project
    return detect_project_from_cwd()
