"""
Git integration utilities.
"""

from pathlib import Path
from typing import Optional

import git
from git import Repo

from ..utils.exceptions import DirtyWorkingDirectoryError, GitError


def get_git_repo(path: Optional[Path] = None) -> Repo:
    """
    Get git repository instance.

    Args:
        path: Path to search for git repo, defaults to current directory

    Returns:
        Git repository instance

    Raises:
        GitError: If no git repository found
    """
    if path is None:
        path = Path.cwd()

    try:
        return Repo(path, search_parent_directories=True)
    except git.InvalidGitRepositoryError as e:
        raise GitError(f"No git repository found at {path}") from e


def validate_clean_working_directory(repo: Optional[Repo] = None) -> None:
    """
    Validate that git working directory is clean.

    Args:
        repo: Git repository instance, defaults to current directory

    Raises:
        DirtyWorkingDirectoryError: If working directory has uncommitted changes
        GitError: If git operations fail
    """
    if repo is None:
        repo = get_git_repo()

    try:
        if repo.is_dirty():
            # Get list of changed files
            changes = []

            # Modified files
            for item in repo.index.diff(None):
                changes.append(f"Modified: {item.a_path}")

            # Staged files
            for item in repo.index.diff("HEAD"):
                changes.append(f"Staged: {item.a_path}")

            # Untracked files
            for file_path in repo.untracked_files:
                changes.append(f"Untracked: {file_path}")

            raise DirtyWorkingDirectoryError(changes)

    except git.GitError as e:
        raise GitError(f"Git operation failed: {e}") from e


def get_current_commit_info(repo: Optional[Repo] = None) -> dict[str, str]:
    """
    Get current git commit information.

    Args:
        repo: Git repository instance, defaults to current directory

    Returns:
        Dictionary with commit information

    Raises:
        GitError: If git operations fail
    """
    if repo is None:
        repo = get_git_repo()

    try:
        commit = repo.head.commit

        return {
            "commit_hash": commit.hexsha,
            "commit_hash_short": commit.hexsha[:8],
            "branch": repo.active_branch.name,
            "author": str(commit.author),
            "message": commit.message.strip(),
            "committed_date": commit.committed_datetime.isoformat(),
        }

    except git.GitError as e:
        raise GitError(f"Failed to get commit info: {e}") from e


def get_repository_info(repo: Optional[Repo] = None) -> dict[str, str]:
    """
    Get git repository information.

    Args:
        repo: Git repository instance, defaults to current directory

    Returns:
        Dictionary with repository information

    Raises:
        GitError: If git operations fail
    """
    if repo is None:
        repo = get_git_repo()

    try:
        info = {
            "repo_path": str(repo.working_dir),
            "git_dir": str(repo.git_dir),
        }

        # Get remote URL if available
        try:
            if repo.remotes:
                origin = repo.remotes.origin
                info["remote_url"] = origin.url
            else:
                info["remote_url"] = None
        except (AttributeError, IndexError):
            info["remote_url"] = None

        return info

    except git.GitError as e:
        raise GitError(f"Failed to get repository info: {e}") from e


def ensure_git_available() -> None:
    """
    Ensure git is available in the system.

    Raises:
        GitError: If git command is not available
    """
    try:
        git.Git().version()
    except git.GitCommandError as e:
        raise GitError("Git command not found. Please install Git.") from e
