"""
Git integration utilities.
"""

import logging
from pathlib import Path

import git
from git import Repo

from ..utils.exceptions import DirtyWorkingDirectoryError, GitError

logger = logging.getLogger(__name__)


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
        # Exclude .ipynb files - they're large (especially with images), cause
        # false positives in secret scanning, and rarely contain reproducibility-
        # critical code (that's in .py and config files)
        patch = repo.git.diff("HEAD", "--", ":(exclude)*.ipynb")

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


def check_patch_size(patch: str, max_size_mb: float = 1.0) -> dict[str, bool | float]:
    """Check if git patch exceeds recommended size limit.

    Large patches can indicate performance issues and increase storage costs.
    This function validates patch size and provides size information.

    Args:
        patch: Git patch content as string
        max_size_mb: Maximum recommended patch size in megabytes (default: 1.0 MB)

    Returns:
        Dictionary with:
            - exceeds_limit (bool): True if patch exceeds max_size_mb
            - size_mb (float): Actual patch size in megabytes
            - size_bytes (int): Actual patch size in bytes
    """
    size_bytes = len(patch.encode("utf-8"))
    size_mb = size_bytes / (1024 * 1024)

    return {
        "exceeds_limit": size_mb > max_size_mb,
        "size_mb": round(size_mb, 2),
        "size_bytes": size_bytes,
    }


def scan_patch_for_secrets(patch: str) -> dict[str, bool | list[dict[str, str]]]:
    """Scan git patch for potential secrets using detect-secrets.

    Scans patch content for sensitive information like API keys, tokens,
    credentials, and other secrets using the detect-secrets library.

    This function scans the actual modified files in the working directory
    (not the patch itself) because detect-secrets is designed to scan source
    files, not git patch format.

    Args:
        patch: Git patch content as string

    Returns:
        Dictionary with:
            - has_secrets (bool): True if potential secrets detected
            - findings (list): List of secret findings, each with:
                - type (str): Type of secret detected
                - line (str): Line number in source file
                - filename (str): Filename where secret was found

    Note:
        This function may produce false positives. Review findings before
        taking action. Returns has_secrets=False if detect-secrets is not
        available or scanning fails.
    """
    try:
        from detect_secrets import SecretsCollection
        from detect_secrets.settings import default_settings
    except ImportError:
        logger.warning(
            "detect-secrets not installed. Skipping secret scanning. "
            "Install with: pip install detect-secrets"
        )
        return {"has_secrets": False, "findings": []}

    findings = []

    try:
        # Parse patch to find modified files and their changed line ranges
        # Git patch format:
        # diff --git a/filename b/filename
        # @@ -old_start,old_count +new_start,new_count @@
        import re
        from pathlib import Path as PathlibPath

        # Track which files were modified and which lines changed
        modified_files = {}  # filename -> set of modified line numbers
        current_file = None
        current_source_line = None

        patch_lines = patch.split("\n")
        for line in patch_lines:
            if line.startswith("diff --git"):
                # Extract filename from "diff --git a/path b/path"
                parts = line.split()
                if len(parts) >= 4:
                    # Remove "a/" or "b/" prefix
                    current_file = (
                        parts[2][2:] if parts[2].startswith("a/") else parts[2]
                    )
                    if current_file not in modified_files:
                        modified_files[current_file] = set()
                current_source_line = None

            elif line.startswith("@@"):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                # Extract the starting line number in the new file
                hunk_match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                if hunk_match:
                    current_source_line = int(hunk_match.group(1))

            elif current_file and current_source_line is not None:
                # Track added and modified lines (but not deleted lines)
                if line.startswith("+") and not line.startswith("+++"):
                    # This is an added line
                    modified_files[current_file].add(current_source_line)
                    current_source_line += 1
                elif line.startswith(" "):
                    # Context line - advances line counter but not modified
                    current_source_line += 1
                # Lines starting with "-" don't advance the new file line counter

        # Now scan each modified file with detect-secrets
        repo = get_git_repo()
        repo_root = PathlibPath(repo.working_dir)

        for filename, modified_lines in modified_files.items():
            if not modified_lines:
                continue  # Skip files with no additions (only deletions)

            # Skip .ipynb files - they cause false positives and are excluded
            # from patch generation anyway
            if filename.endswith(".ipynb"):
                continue

            file_path = repo_root / filename

            # Skip if file doesn't exist (might be deleted)
            if not file_path.exists():
                continue

            # Scan the actual file with detect-secrets
            secrets = SecretsCollection()
            with default_settings():
                try:
                    secrets.scan_file(str(file_path))
                except OSError as e:
                    logger.debug(
                        f"Could not read file {filename} for secret scanning: {e}"
                    )
                    continue
                except UnicodeDecodeError as e:
                    logger.debug(
                        f"Could not decode file {filename} (binary file?): {e}"
                    )
                    continue
                except Exception as e:
                    # Catch other detect-secrets internal errors
                    logger.debug(
                        f"Secret scanning failed for {filename}: {type(e).__name__}: {e}"
                    )
                    continue

            # Filter findings to only those on modified lines
            if secrets.data:
                for _secret_filename, file_secrets in secrets.data.items():
                    for secret in file_secrets:
                        # Only include secrets on lines that were modified
                        if secret.line_number in modified_lines:
                            findings.append(
                                {
                                    "type": secret.type,
                                    "line": str(secret.line_number),
                                    "filename": filename,
                                }
                            )

    except git.GitError as e:
        logger.warning(f"Git operation failed during secret scanning: {e}")
        return {"has_secrets": False, "findings": []}
    except (ValueError, KeyError, AttributeError) as e:
        logger.warning(
            f"Failed to parse patch format during secret scanning: "
            f"{type(e).__name__}: {e}"
        )
        return {"has_secrets": False, "findings": []}
    except Exception as e:
        logger.warning(
            f"Unexpected error during secret scanning: {type(e).__name__}: {e}"
        )
        # Return safe default (no secrets found) if scanning fails
        return {"has_secrets": False, "findings": []}

    return {"has_secrets": len(findings) > 0, "findings": findings}
