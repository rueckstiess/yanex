"""
Environment capture utilities.
"""

import platform
import sys
from pathlib import Path
from typing import Any

import git

from ..utils.exceptions import GitError
from .git_utils import get_current_commit_info, get_git_repo, get_repository_info


def capture_python_environment() -> dict[str, Any]:
    """
    Capture Python environment information.

    Returns:
        Dictionary with Python environment details
    """
    return {
        "python_version": sys.version,
        "python_version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "python_executable": sys.executable,
        "python_path": sys.path.copy(),
        "platform": platform.platform(),
    }


def capture_system_environment() -> dict[str, Any]:
    """
    Capture system environment information.

    Returns:
        Dictionary with system details
    """
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
        },
        "hostname": platform.node(),
        "working_directory": str(Path.cwd()),
    }


def capture_git_environment(repo_path: Path | None = None) -> dict[str, Any]:
    """
    Capture git environment information.

    Args:
        repo_path: Path to git repository, defaults to current directory

    Returns:
        Dictionary with git environment details
    """
    try:
        repo = get_git_repo(repo_path)

        git_info = {
            "repository": get_repository_info(repo),
            "commit": get_current_commit_info(repo),
            "git_version": git.Git().version(),
        }

        return git_info

    except GitError:
        # If we can't get git info, return minimal info
        return {
            "repository": None,
            "commit": None,
            "git_version": None,
            "error": "Git repository not found or not accessible",
        }


def capture_dependencies() -> dict[str, Any]:
    """
    Capture dependency information.

    Returns:
        Dictionary with dependency details
    """
    deps_info = {
        "requirements_txt": None,
        "environment_yml": None,
        "pyproject_toml": None,
    }

    cwd = Path.cwd()

    # Check for requirements.txt
    requirements_path = cwd / "requirements.txt"
    if requirements_path.exists():
        try:
            deps_info["requirements_txt"] = requirements_path.read_text().strip()
        except Exception:
            deps_info["requirements_txt"] = "Error reading requirements.txt"

    # Check for environment.yml (conda)
    env_yml_path = cwd / "environment.yml"
    if env_yml_path.exists():
        try:
            deps_info["environment_yml"] = env_yml_path.read_text().strip()
        except Exception:
            deps_info["environment_yml"] = "Error reading environment.yml"

    # Check for pyproject.toml
    pyproject_path = cwd / "pyproject.toml"
    if pyproject_path.exists():
        try:
            deps_info["pyproject_toml"] = pyproject_path.read_text().strip()
        except Exception:
            deps_info["pyproject_toml"] = "Error reading pyproject.toml"

    return deps_info


def capture_full_environment(repo_path: Path | None = None) -> dict[str, Any]:
    """
    Capture complete environment information.

    Args:
        repo_path: Path to git repository, defaults to current directory

    Returns:
        Dictionary with complete environment details
    """
    return {
        "python": capture_python_environment(),
        "system": capture_system_environment(),
        "git": capture_git_environment(repo_path),
        "dependencies": capture_dependencies(),
        "capture_timestamp": None,  # Will be set by storage layer
    }
