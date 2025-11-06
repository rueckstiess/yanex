"""
Tests for yanex.core.git_utils module.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import git
import pytest

from tests.test_utils import TestFileHelpers
from yanex.core.git_utils import (
    ensure_git_available,
    generate_git_patch,
    get_current_commit_info,
    get_git_repo,
    get_repository_info,
    has_uncommitted_changes,
)
from yanex.utils.exceptions import GitError


class TestGetGitRepo:
    """Test get_git_repo function."""

    def test_get_repo_with_valid_path(self, git_repo):
        """Test getting repo with valid git repository."""
        repo_path = Path(git_repo.working_dir)
        result = get_git_repo(repo_path)

        assert isinstance(result, git.Repo)
        assert result.working_dir == git_repo.working_dir

    def test_get_repo_with_none_path(self, git_repo):
        """Test getting repo with None path (uses current directory)."""
        with patch("yanex.core.git_utils.Path.cwd") as mock_cwd:
            mock_cwd.return_value = Path(git_repo.working_dir)
            result = get_git_repo(None)

            assert isinstance(result, git.Repo)

    def test_get_repo_invalid_repository_temp_dir(self, temp_dir):
        """Test getting repo from non-git directory raises GitError."""
        with pytest.raises(GitError, match="No git repository found"):
            get_git_repo(temp_dir)

    def test_get_repo_search_parent_directories(self, git_repo):
        """Test that repo search works in subdirectories."""
        repo_path = Path(git_repo.working_dir)
        subdir = repo_path / "subdir"
        subdir.mkdir()

        result = get_git_repo(subdir)
        assert isinstance(result, git.Repo)
        assert result.working_dir == git_repo.working_dir

    @pytest.mark.parametrize(
        "subdir_depth",
        [1, 2, 3],
    )
    def test_get_repo_nested_subdirectories(self, git_repo, subdir_depth):
        """Test repo search works at various nesting levels."""
        repo_path = Path(git_repo.working_dir)

        # Create nested subdirectory structure
        current_path = repo_path
        for i in range(subdir_depth):
            current_path = current_path / f"level_{i}"
            current_path.mkdir()

        result = get_git_repo(current_path)
        assert isinstance(result, git.Repo)
        assert result.working_dir == git_repo.working_dir

    def test_get_repo_error_handling(self, temp_dir):
        """Test error handling in get_git_repo."""
        with patch("git.Repo") as mock_repo:
            mock_repo.side_effect = git.InvalidGitRepositoryError("Not a git repo")

            with pytest.raises(GitError, match="No git repository found"):
                get_git_repo(temp_dir)


class TestHasUncommittedChanges:
    """Test has_uncommitted_changes function."""

    def test_clean_working_directory(self, clean_git_repo):
        """Test returns False for clean working directory."""
        result = has_uncommitted_changes(clean_git_repo)
        assert result is False

    def test_modified_file(self, git_repo):
        """Test returns True for modified file."""
        # Modify a file
        test_file = Path(git_repo.working_dir) / "test.txt"
        test_file.write_text("modified content")

        result = has_uncommitted_changes(git_repo)
        assert result is True

    def test_staged_file(self, clean_git_repo):
        """Test returns True for staged file."""
        # Add and stage a file
        staged_file = Path(clean_git_repo.working_dir) / "staged.txt"
        TestFileHelpers.create_test_file(staged_file, "staged content")
        clean_git_repo.index.add([str(staged_file)])

        result = has_uncommitted_changes(clean_git_repo)
        assert result is True

    def test_untracked_file(self, clean_git_repo):
        """Test returns False for untracked file (not counted as uncommitted)."""
        # Create untracked file
        untracked_file = Path(clean_git_repo.working_dir) / "untracked.txt"
        TestFileHelpers.create_test_file(untracked_file, "untracked content")

        result = has_uncommitted_changes(clean_git_repo)
        assert result is False

    def test_with_none_repo(self):
        """Test with None repo (uses current directory)."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.is_dirty.return_value = True
            mock_get_repo.return_value = mock_repo

            result = has_uncommitted_changes(None)
            assert result is True
            mock_get_repo.assert_called_once()

    def test_git_error_handling(self, clean_git_repo):
        """Test that GitError is raised when git operations fail."""
        with patch.object(clean_git_repo, "is_dirty") as mock_is_dirty:
            mock_is_dirty.side_effect = git.GitError("Test git error")

            with pytest.raises(GitError, match="Failed to check git status"):
                has_uncommitted_changes(clean_git_repo)


class TestGenerateGitPatch:
    """Test generate_git_patch function."""

    def test_clean_working_directory(self, clean_git_repo):
        """Test returns None for clean working directory."""
        result = generate_git_patch(clean_git_repo)
        assert result is None

    def test_modified_file(self, git_repo):
        """Test generates patch for modified file."""
        # Modify a file
        test_file = Path(git_repo.working_dir) / "test.txt"
        test_file.write_text("modified content")

        result = generate_git_patch(git_repo)

        assert result is not None
        assert isinstance(result, str)
        assert "diff --git" in result
        assert "test.txt" in result
        assert "+modified content" in result

    def test_staged_file(self, clean_git_repo):
        """Test generates patch for staged file."""
        # Add and stage a file
        staged_file = Path(clean_git_repo.working_dir) / "staged.txt"
        TestFileHelpers.create_test_file(staged_file, "staged content")
        clean_git_repo.index.add([str(staged_file)])

        result = generate_git_patch(clean_git_repo)

        assert result is not None
        assert isinstance(result, str)
        assert "diff --git" in result
        assert "staged.txt" in result

    def test_staged_and_unstaged(self, clean_git_repo):
        """Test patch includes both staged and unstaged changes."""
        repo_path = Path(clean_git_repo.working_dir)

        # Create and commit a file first
        existing_file = repo_path / "existing.txt"
        TestFileHelpers.create_test_file(existing_file, "original content")
        clean_git_repo.index.add([str(existing_file)])
        clean_git_repo.index.commit("Add existing file")

        # Modify and stage the file
        existing_file.write_text("staged changes")
        clean_git_repo.index.add([str(existing_file)])

        # Modify again without staging (unstaged changes)
        existing_file.write_text("staged changes\nunstaged changes")

        result = generate_git_patch(clean_git_repo)

        assert result is not None
        # Should include the file with both staged and unstaged changes
        assert "existing.txt" in result

    def test_untracked_files_excluded(self, clean_git_repo):
        """Test that untracked files are NOT included in patch."""
        # Create untracked file
        untracked_file = Path(clean_git_repo.working_dir) / "untracked.txt"
        TestFileHelpers.create_test_file(untracked_file, "untracked content")

        result = generate_git_patch(clean_git_repo)

        # Should be None since untracked files don't count as dirty
        assert result is None

    def test_binary_file_handling(self, clean_git_repo):
        """Test that binary files are noted in patch."""
        # Create a binary file (simple bytes)
        binary_file = Path(clean_git_repo.working_dir) / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
        clean_git_repo.index.add([str(binary_file)])
        clean_git_repo.index.commit("Add binary file")

        # Modify binary file
        binary_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd")

        result = generate_git_patch(clean_git_repo)

        assert result is not None
        # Git notes binary files with a message
        assert "binary.bin" in result

    def test_multiple_file_changes(self, clean_git_repo):
        """Test patch with multiple changed files."""
        repo_path = Path(clean_git_repo.working_dir)

        # Create multiple files
        for i in range(3):
            file_path = repo_path / f"file{i}.txt"
            TestFileHelpers.create_test_file(file_path, f"content {i}")
            clean_git_repo.index.add([str(file_path)])

        clean_git_repo.index.commit("Add test files")

        # Modify all files
        for i in range(3):
            file_path = repo_path / f"file{i}.txt"
            file_path.write_text(f"modified content {i}")

        result = generate_git_patch(clean_git_repo)

        assert result is not None
        for i in range(3):
            assert f"file{i}.txt" in result

    def test_with_none_repo(self):
        """Test with None repo (uses current directory)."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.is_dirty.return_value = True
            mock_repo.git.diff.return_value = "diff --git a/test.txt b/test.txt"
            mock_get_repo.return_value = mock_repo

            result = generate_git_patch(None)

            assert result is not None
            assert "diff --git" in result
            mock_get_repo.assert_called_once()

    def test_patch_format_is_valid(self, git_repo):
        """Test that generated patch has valid format."""
        # Modify a file
        test_file = Path(git_repo.working_dir) / "test.txt"
        test_file.write_text("new content")

        result = generate_git_patch(git_repo)

        assert result is not None
        # Valid patch should start with diff --git
        assert result.startswith("diff --git")
        # Should have @ markers for hunks
        assert "@@" in result


class TestGetCurrentCommitInfo:
    """Test get_current_commit_info function."""

    def test_get_commit_info(self, git_repo):
        """Test getting current commit information."""
        result = get_current_commit_info(git_repo)

        assert isinstance(result, dict)
        assert "commit_hash" in result
        assert "commit_hash_short" in result
        assert "branch" in result
        assert "author" in result
        assert "message" in result
        assert "committed_date" in result

        # Check hash formats
        assert len(result["commit_hash"]) == 40  # Full SHA
        assert len(result["commit_hash_short"]) == 8  # Short SHA
        assert result["commit_hash"].startswith(result["commit_hash_short"])

        # Check commit message
        assert result["message"] == "Initial commit"

    @pytest.mark.parametrize(
        "commit_data,expected_values",
        [
            (
                {
                    "hexsha": "a1b2c3d4e5f6789012345678901234567890abcd",
                    "message": "Test commit\n",
                    "author": "Test Author",
                    "branch": "main",
                    "date": "2023-01-01T12:00:00",
                },
                {
                    "commit_hash": "a1b2c3d4e5f6789012345678901234567890abcd",
                    "commit_hash_short": "a1b2c3d4",
                    "message": "Test commit",
                    "branch": "main",
                    "author": "Test Author",
                },
            ),
            (
                {
                    "hexsha": "def456789012345678901234567890abcdef12345",
                    "message": "Another test\n\nWith description",
                    "author": "Another Author",
                    "branch": "develop",
                    "date": "2023-02-01T15:30:00",
                },
                {
                    "commit_hash": "def456789012345678901234567890abcdef12345",
                    "commit_hash_short": "def45678",
                    "message": "Another test\n\nWith description",
                    "branch": "develop",
                    "author": "Another Author",
                },
            ),
        ],
    )
    def test_get_commit_info_with_none_repo(self, commit_data, expected_values):
        """Test getting commit info with None repo and various commit data."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_commit = Mock()
            mock_commit.hexsha = commit_data["hexsha"]
            mock_commit.message = commit_data["message"]
            mock_commit.author = commit_data["author"]
            mock_commit.committed_datetime.isoformat.return_value = commit_data["date"]

            mock_repo.head.commit = mock_commit
            mock_repo.active_branch.name = commit_data["branch"]
            mock_get_repo.return_value = mock_repo

            result = get_current_commit_info(None)

            assert result["commit_hash"] == expected_values["commit_hash"]
            assert result["commit_hash_short"] == expected_values["commit_hash_short"]
            assert result["message"] == expected_values["message"]
            assert result["branch"] == expected_values["branch"]
            assert result["author"] == expected_values["author"]

    def test_git_error_handling_head_access(self):
        """Test GitError handling when head access fails."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            type(mock_repo).head = property(
                lambda _: (_ for _ in ()).throw(git.GitError("Head access error"))
            )
            mock_get_repo.return_value = mock_repo

            with pytest.raises(GitError, match="Failed to get commit info"):
                get_current_commit_info()

    def test_git_error_handling_branch_access(self):
        """Test GitError handling when branch access fails."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_commit = Mock()
            mock_commit.hexsha = "abc123"
            mock_repo.head.commit = mock_commit
            type(mock_repo).active_branch = property(
                lambda _: (_ for _ in ()).throw(git.GitError("Branch access error"))
            )
            mock_get_repo.return_value = mock_repo

            with pytest.raises(GitError, match="Failed to get commit info"):
                get_current_commit_info()

    def test_get_commit_info_detached_head(self):
        """Test getting commit info when in detached HEAD state."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_commit = Mock()
            mock_commit.hexsha = "abc123def456"
            mock_commit.message = "Detached commit"
            mock_commit.author = "Test Author"
            mock_commit.committed_datetime.isoformat.return_value = (
                "2023-01-01T12:00:00"
            )

            mock_repo.head.commit = mock_commit
            mock_repo.head.is_detached = True
            # When detached, active_branch raises TypeError - we'll catch this in real implementation
            mock_repo.active_branch = Mock()
            mock_repo.active_branch.name = "HEAD (detached)"
            mock_get_repo.return_value = mock_repo

            result = get_current_commit_info()

            assert result["commit_hash"] == "abc123def456"


class TestGetRepositoryInfo:
    """Test get_repository_info function."""

    def test_get_repository_info(self, git_repo):
        """Test getting repository information."""
        result = get_repository_info(git_repo)

        assert isinstance(result, dict)
        assert "repo_path" in result
        assert "git_dir" in result
        assert "remote_url" in result

        assert result["repo_path"] == str(git_repo.working_dir)
        assert result["git_dir"] == str(git_repo.git_dir)

    def test_get_repository_info_with_origin_remote(self):
        """Test getting repository info with origin remote."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.working_dir = "/test/repo"
            mock_repo.git_dir = "/test/repo/.git"

            # Setup origin remote
            mock_origin = Mock()
            mock_origin.url = "https://github.com/user/repo.git"
            mock_repo.remotes.origin = mock_origin

            mock_get_repo.return_value = mock_repo

            result = get_repository_info()
            assert result["remote_url"] == "https://github.com/user/repo.git"

    def test_get_repository_info_empty_remotes(self):
        """Test getting repository info when remotes list is empty."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.working_dir = "/test/repo"
            mock_repo.git_dir = "/test/repo/.git"
            mock_repo.remotes = []  # Empty remotes list

            mock_get_repo.return_value = mock_repo

            result = get_repository_info()
            assert result["remote_url"] is None

    def test_get_repository_info_no_remotes(self, git_repo):
        """Test getting repository info without remotes."""
        result = get_repository_info(git_repo)
        assert result["remote_url"] is None

    def test_git_error_handling_working_dir_error(self):
        """Test GitError handling when working dir access fails."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            type(mock_repo).working_dir = property(
                lambda _: (_ for _ in ()).throw(git.GitError("Working dir error"))
            )
            mock_get_repo.return_value = mock_repo

            with pytest.raises(GitError, match="Failed to get repository info"):
                get_repository_info()

    def test_git_error_handling_git_dir_error(self):
        """Test GitError handling when git dir access fails."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.working_dir = "/test"
            type(mock_repo).git_dir = property(
                lambda _: (_ for _ in ()).throw(
                    git.GitCommandError("git", 1, "stderr", "stdout")
                )
            )
            mock_get_repo.return_value = mock_repo

            with pytest.raises(GitError, match="Failed to get repository info"):
                get_repository_info()


class TestEnsureGitAvailable:
    """Test ensure_git_available function."""

    @pytest.mark.parametrize(
        "git_version",
        [
            "git version 2.34.1",
            "git version 2.40.0.windows.1",
            "git version 2.30.2.darwin.1",
        ],
    )
    def test_git_available(self, git_version):
        """Test when git is available with various version formats."""
        with patch("git.Git") as mock_git_class:
            mock_git = Mock()
            mock_git.version.return_value = git_version
            mock_git_class.return_value = mock_git

            # Should not raise any exception
            ensure_git_available()
            mock_git.version.assert_called_once()

    def test_git_not_available_command_error(self):
        """Test when git command fails."""
        with patch("git.Git") as mock_git_class:
            mock_git = Mock()
            mock_git.version.side_effect = git.GitCommandError(
                "git", 1, "git not found", ""
            )
            mock_git_class.return_value = mock_git

            with pytest.raises(GitError, match="Git command not found"):
                ensure_git_available()

    def test_ensure_git_available_success(self):
        """Test successful git availability check."""
        with patch("git.Git") as mock_git_class:
            mock_git = Mock()
            mock_git.version.return_value = "git version 2.34.1"
            mock_git_class.return_value = mock_git

            # Should complete without exception
            result = ensure_git_available()
            assert result is None  # Function returns None on success

    def test_ensure_git_available_with_unexpected_error(self):
        """Test git availability check with unexpected error types."""
        with patch("git.Git") as mock_git_class:
            mock_git = Mock()
            mock_git.version.side_effect = RuntimeError("Unexpected error")
            mock_git_class.return_value = mock_git

            # Should propagate the RuntimeError since it's not a known git error
            with pytest.raises(RuntimeError, match="Unexpected error"):
                ensure_git_available()
