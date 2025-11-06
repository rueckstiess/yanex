"""
Git integration utilities.
"""

from pathlib import Path

import git
from git import Repo

from ..utils.exceptions import DirtyWorkingDirectoryError, GitError


def get_git_repo(path: Path | None = None) -> Repo:
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


def validate_clean_working_directory(repo: Repo | None = None) -> None:
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


def has_uncommitted_changes(repo: Repo | None = None) -> bool:
    """Check if working directory has uncommitted changes.

    Checks for both staged and unstaged changes in tracked files only.
    Untracked files are not considered uncommitted changes.

    Args:
        repo: Git repository instance. If None, detects from cwd.

    Returns:
        True if uncommitted changes exist, False if clean.

    Raises:
        GitError: If git operations fail
    """
    if repo is None:
        repo = get_git_repo()

    try:
        # Check if any tracked files have changes (staged or unstaged)
        # This excludes untracked files
        return repo.is_dirty()
    except git.GitError as e:
        raise GitError(f"Failed to check git status: {e}") from e


def generate_git_patch(repo: Repo | None = None) -> str | None:
    """Generate patch of all uncommitted changes (staged + unstaged).

    Captures differences between HEAD and working directory, including
    both staged and unstaged changes for tracked files only. Untracked
    files are excluded.

    Args:
        repo: Git repository instance. If None, detects from cwd.

    Returns:
        Patch string if changes exist, None if working directory is clean.
        Returns None (not empty string) for clean state.

    Raises:
        GitError: If git operations fail
    """
    if repo is None:
        repo = get_git_repo()

    try:
        # Check if any changes exist (optimize for common clean case)
        if not has_uncommitted_changes(repo):
            return None

        # Generate diff between HEAD and working directory
        # This captures both staged and unstaged changes
        # Binary files are handled automatically by git ("Binary files differ")
        patch = repo.git.diff("HEAD")

        # Return None for clean state (shouldn't happen due to check above, but be safe)
        if not patch or not patch.strip():
            return None

        return patch

    except git.GitError as e:
        raise GitError(f"Failed to generate git patch: {e}") from e


def get_current_commit_info(repo: Repo | None = None) -> dict[str, str]:
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


def get_repository_info(repo: Repo | None = None) -> dict[str, str]:
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
