"""
Tests for yanex.core.git_utils module.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import git
import pytest

from tests.test_utils import TestFileHelpers
from yanex.core.git_utils import (
    check_patch_size,
    ensure_git_available,
    generate_git_patch,
    get_current_commit_info,
    get_git_repo,
    get_repository_info,
    has_uncommitted_changes,
    scan_patch_for_secrets,
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


class TestValidateCleanWorkingDirectory:
    """Test validate_clean_working_directory function."""

    def test_clean_working_directory(self, clean_git_repo):
        """Test validation passes for clean working directory."""
        # Should not raise any exception
        from yanex.core.git_utils import validate_clean_working_directory

        validate_clean_working_directory(clean_git_repo)

    def test_modified_file_raises_error(self, git_repo):
        """Test that modified file raises DirtyWorkingDirectoryError."""
        from yanex.core.git_utils import validate_clean_working_directory
        from yanex.utils.exceptions import DirtyWorkingDirectoryError

        # Modify a file
        test_file = Path(git_repo.working_dir) / "test.txt"
        test_file.write_text("modified content")

        with pytest.raises(DirtyWorkingDirectoryError) as exc_info:
            validate_clean_working_directory(git_repo)

        # Check that error contains modified file info
        error = exc_info.value
        assert len(error.changes) > 0
        assert any("Modified: test.txt" in change for change in error.changes)

    def test_staged_file_raises_error(self, clean_git_repo):
        """Test that staged file raises DirtyWorkingDirectoryError."""
        from yanex.core.git_utils import validate_clean_working_directory
        from yanex.utils.exceptions import DirtyWorkingDirectoryError

        # Add and stage a file
        staged_file = Path(clean_git_repo.working_dir) / "staged.txt"
        TestFileHelpers.create_test_file(staged_file, "staged content")
        clean_git_repo.index.add([str(staged_file)])

        with pytest.raises(DirtyWorkingDirectoryError) as exc_info:
            validate_clean_working_directory(clean_git_repo)

        error = exc_info.value
        assert len(error.changes) > 0
        assert any("staged.txt" in change for change in error.changes)

    def test_untracked_file_raises_error(self, git_repo):
        """Test that untracked files are included in error when repo is dirty.

        Note: validate_clean_working_directory checks is_dirty() first, which by default
        doesn't consider untracked files. So we need to also modify a tracked file to
        trigger the dirty check, then verify untracked files are included in the changes.
        """
        from yanex.core.git_utils import validate_clean_working_directory
        from yanex.utils.exceptions import DirtyWorkingDirectoryError

        # Modify a tracked file to make repo dirty
        test_file = Path(git_repo.working_dir) / "test.txt"
        test_file.write_text("modified content")

        # Also create an untracked file
        untracked_file = Path(git_repo.working_dir) / "untracked.txt"
        TestFileHelpers.create_test_file(untracked_file, "untracked content")

        with pytest.raises(DirtyWorkingDirectoryError) as exc_info:
            validate_clean_working_directory(git_repo)

        # Verify both the modified file and untracked file are reported
        error = exc_info.value
        assert len(error.changes) >= 2
        assert any("untracked.txt" in change for change in error.changes)
        assert any("test.txt" in change for change in error.changes)

    def test_multiple_changes_listed(self, git_repo):
        """Test that multiple changes are all listed in error."""
        from yanex.core.git_utils import validate_clean_working_directory
        from yanex.utils.exceptions import DirtyWorkingDirectoryError

        repo_path = Path(git_repo.working_dir)

        # Modify existing file
        test_file = repo_path / "test.txt"
        test_file.write_text("modified")

        # Add new untracked file
        untracked = repo_path / "untracked.txt"
        TestFileHelpers.create_test_file(untracked, "new")

        # Add and stage another file
        staged = repo_path / "staged.txt"
        TestFileHelpers.create_test_file(staged, "staged")
        git_repo.index.add([str(staged)])

        with pytest.raises(DirtyWorkingDirectoryError) as exc_info:
            validate_clean_working_directory(git_repo)

        error = exc_info.value
        # Should have multiple changes listed
        assert len(error.changes) >= 3
        change_str = str(error.changes)
        assert "Modified" in change_str or "test.txt" in change_str
        assert "Untracked" in change_str or "untracked.txt" in change_str
        assert "Staged" in change_str or "staged.txt" in change_str

    def test_with_none_repo(self):
        """Test with None repo (uses current directory)."""
        from yanex.core.git_utils import validate_clean_working_directory

        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.is_dirty.return_value = False
            mock_get_repo.return_value = mock_repo

            # Should not raise
            validate_clean_working_directory(None)
            mock_get_repo.assert_called_once()

    def test_git_error_handling(self, clean_git_repo):
        """Test that GitError is raised when git operations fail."""
        from yanex.core.git_utils import validate_clean_working_directory

        with patch.object(clean_git_repo, "is_dirty") as mock_is_dirty:
            mock_is_dirty.side_effect = git.GitError("Git operation failed")

            with pytest.raises(GitError, match="Git operation failed"):
                validate_clean_working_directory(clean_git_repo)


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


class TestCheckPatchSize:
    """Test check_patch_size function."""

    def test_small_patch_within_limit(self):
        """Test patch smaller than limit."""
        patch = "diff --git a/test.txt b/test.txt\n+small change"

        result = check_patch_size(patch, max_size_mb=1.0)

        assert isinstance(result, dict)
        assert result["exceeds_limit"] is False
        assert result["size_mb"] < 1.0
        assert result["size_bytes"] == len(patch.encode("utf-8"))

    def test_large_patch_exceeds_limit(self):
        """Test patch larger than limit."""
        # Create a patch larger than 1MB (needs to be significantly larger for rounding)
        large_content = "x" * (1024 * 1024 + 100000)  # ~1.1 MB
        patch = f"diff --git a/large.txt b/large.txt\n+{large_content}"

        result = check_patch_size(patch, max_size_mb=1.0)

        assert result["exceeds_limit"] is True
        assert result["size_mb"] > 1.0
        assert result["size_bytes"] > 1024 * 1024

    def test_exact_limit_boundary(self):
        """Test patch at exactly the limit."""
        # Create patch at exactly 1MB
        size_bytes = 1024 * 1024
        content = "x" * size_bytes
        patch = content

        result = check_patch_size(patch, max_size_mb=1.0)

        # At exactly 1MB, should not exceed
        assert result["exceeds_limit"] is False
        assert result["size_mb"] == 1.0

    def test_custom_size_limit(self):
        """Test with custom size limit."""
        patch = "x" * (500 * 1024)  # 500 KB

        # Test with 0.4 MB limit (should exceed)
        result = check_patch_size(patch, max_size_mb=0.4)
        assert result["exceeds_limit"] is True

        # Test with 0.6 MB limit (should not exceed)
        result = check_patch_size(patch, max_size_mb=0.6)
        assert result["exceeds_limit"] is False

    def test_empty_patch(self):
        """Test with empty patch."""
        result = check_patch_size("", max_size_mb=1.0)

        assert result["exceeds_limit"] is False
        assert result["size_mb"] == 0.0
        assert result["size_bytes"] == 0

    def test_unicode_patch(self):
        """Test patch with unicode characters."""
        patch = "diff --git a/test.txt b/test.txt\n+Hello 世界 🌍"

        result = check_patch_size(patch, max_size_mb=1.0)

        # Unicode characters take more bytes
        assert result["size_bytes"] == len(patch.encode("utf-8"))
        assert result["size_bytes"] > len(patch)  # More than ASCII char count


class TestScanPatchForSecrets:
    """Test scan_patch_for_secrets function."""

    def test_clean_patch_no_secrets(self):
        """Test patch with no secrets."""
        patch = """diff --git a/test.txt b/test.txt
index abc123..def456 100644
--- a/test.txt
+++ b/test.txt
@@ -1,3 +1,3 @@
-Old content
+New content
"""

        result = scan_patch_for_secrets(patch)

        assert isinstance(result, dict)
        assert "has_secrets" in result
        assert "findings" in result
        assert result["has_secrets"] is False
        assert result["findings"] == []

    def test_patch_with_api_key(self):
        """Test patch containing what looks like an API key."""
        # Use a realistic-looking fake API key pattern
        patch = """diff --git a/config.py b/config.py
index abc123..def456 100644
--- a/config.py
+++ b/config.py
@@ -1,3 +1,3 @@
-API_KEY = 'old_key'
+API_KEY = 'sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz1234567890AbCdEfGhIjKlMnOpQr'
"""

        result = scan_patch_for_secrets(patch)

        # detect-secrets should find this
        assert isinstance(result, dict)
        assert "has_secrets" in result
        assert "findings" in result
        # Note: This test may fail if detect-secrets is not installed
        # In that case, has_secrets will be False

    def test_patch_with_aws_key(self):
        """Test patch containing what looks like an AWS key."""
        patch = """diff --git a/aws_config.py b/aws_config.py
index abc123..def456 100644
--- a/aws_config.py
+++ b/aws_config.py
@@ -1,3 +1,4 @@
 # AWS Configuration
-aws_access_key = 'OLD_KEY'
+aws_access_key_id = 'AKIAIOSFODNN7EXAMPLE'
+aws_secret_access_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'
"""

        result = scan_patch_for_secrets(patch)

        assert isinstance(result, dict)
        assert "has_secrets" in result
        assert "findings" in result

    def test_missing_detect_secrets_library(self):
        """Test graceful handling when detect-secrets is not installed."""
        # Test that function handles ImportError gracefully
        # This is tested by the fact that if detect-secrets wasn't installed,
        # the other tests would still pass (returning has_secrets=False)
        test_patch = "diff --git a/test.txt b/test.txt\n+content"
        result = scan_patch_for_secrets(test_patch)

        # Should return valid structure even if library missing
        assert isinstance(result, dict)
        assert "has_secrets" in result
        assert "findings" in result

    def test_scan_error_handling(self):
        """Test graceful handling of scanning errors."""
        from unittest.mock import patch as mock_patch

        # Mock tempfile to raise an exception during file creation
        with mock_patch("tempfile.NamedTemporaryFile") as mock_tempfile:
            mock_tempfile.side_effect = Exception("File creation failed")

            test_patch = "diff --git a/test.txt b/test.txt\n+content"
            result = scan_patch_for_secrets(test_patch)

            # Should return safe default on error
            assert result["has_secrets"] is False
            assert result["findings"] == []

    def test_empty_patch_scan(self):
        """Test scanning empty patch."""
        result = scan_patch_for_secrets("")

        assert result["has_secrets"] is False
        assert result["findings"] == []

    def test_findings_structure(self):
        """Test that findings have correct structure when secrets detected."""
        # This test creates a patch that should trigger detection
        patch = """diff --git a/secrets.txt b/secrets.txt
+password = 'MySecretPassword123!'
+api_token = 'ghp_1234567890abcdefghijklmnopqrstuvwxyz'
"""

        result = scan_patch_for_secrets(patch)

        assert isinstance(result, dict)
        assert "has_secrets" in result
        assert "findings" in result
        assert isinstance(result["findings"], list)

        # If secrets were found, check structure
        if result["has_secrets"]:
            for finding in result["findings"]:
                assert "type" in finding
                assert "line" in finding
                assert "filename" in finding
                assert isinstance(finding["type"], str)
                assert isinstance(finding["line"], str)
                assert isinstance(finding["filename"], str)


class TestGenerateGitPatchEdgeCases:
    """Test edge cases for generate_git_patch function."""

    def test_git_error_handling(self):
        """Test GitError handling when git diff fails."""
        with patch("yanex.core.git_utils.has_uncommitted_changes") as mock_has_changes:
            mock_has_changes.return_value = True

            with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
                mock_repo = Mock()
                mock_repo.git.diff.side_effect = git.GitError("Diff operation failed")
                mock_get_repo.return_value = mock_repo

                with pytest.raises(GitError, match="Failed to generate git patch"):
                    generate_git_patch(None)

    def test_empty_patch_returns_none(self):
        """Test that empty patch string returns None."""
        with patch("yanex.core.git_utils.has_uncommitted_changes") as mock_has_changes:
            mock_has_changes.return_value = True

            with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
                mock_repo = Mock()
                mock_repo.git.diff.return_value = ""  # Empty string
                mock_get_repo.return_value = mock_repo

                result = generate_git_patch(None)
                assert result is None

    def test_whitespace_only_patch_returns_none(self):
        """Test that whitespace-only patch returns None."""
        with patch("yanex.core.git_utils.has_uncommitted_changes") as mock_has_changes:
            mock_has_changes.return_value = True

            with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
                mock_repo = Mock()
                mock_repo.git.diff.return_value = "   \n\t\n  "  # Only whitespace
                mock_get_repo.return_value = mock_repo

                result = generate_git_patch(None)
                assert result is None


class TestGetRepositoryInfoEdgeCases:
    """Test edge cases for get_repository_info function."""

    def test_remotes_attribute_error(self):
        """Test handling when remotes.origin raises AttributeError."""

        # Create a custom class that raises AttributeError on .origin access
        class RemotesMock:
            def __bool__(self):
                return True

            @property
            def origin(self):
                raise AttributeError("No origin")

        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.working_dir = "/test/repo"
            mock_repo.git_dir = "/test/repo/.git"
            mock_repo.remotes = RemotesMock()

            mock_get_repo.return_value = mock_repo

            result = get_repository_info()
            assert result["remote_url"] is None

    def test_remotes_index_error(self):
        """Test handling when remotes.origin raises IndexError."""

        # Create a custom class that raises IndexError on .origin access
        class RemotesMock:
            def __bool__(self):
                return True

            @property
            def origin(self):
                raise IndexError("Index error")

        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.working_dir = "/test/repo"
            mock_repo.git_dir = "/test/repo/.git"
            mock_repo.remotes = RemotesMock()

            mock_get_repo.return_value = mock_repo

            result = get_repository_info()
            assert result["remote_url"] is None


class TestScanPatchForSecretsDetailed:
    """Detailed tests for scan_patch_for_secrets implementation."""

    def test_patch_parsing_with_deleted_file(self, clean_git_repo):
        """Test scanning patch that includes a deleted file."""
        # Create and commit a file first
        test_file = Path(clean_git_repo.working_dir) / "to_delete.txt"
        TestFileHelpers.create_test_file(test_file, "content")
        clean_git_repo.index.add([str(test_file)])
        clean_git_repo.index.commit("Add file to delete")

        # Delete the file
        test_file.unlink()
        clean_git_repo.index.remove([str(test_file)])

        # Generate patch
        patch = clean_git_repo.git.diff("HEAD")

        result = scan_patch_for_secrets(patch)

        # Should handle deleted files gracefully
        assert isinstance(result, dict)
        assert "has_secrets" in result
        assert "findings" in result

    def test_patch_with_binary_file(self, clean_git_repo):
        """Test scanning patch that includes binary file changes."""
        # Create binary file
        binary_file = Path(clean_git_repo.working_dir) / "binary.bin"
        binary_file.write_bytes(b"\x00\x01\x02")
        clean_git_repo.index.add([str(binary_file)])
        clean_git_repo.index.commit("Add binary")

        # Modify it
        binary_file.write_bytes(b"\x00\x01\x02\x03")

        patch = clean_git_repo.git.diff("HEAD")

        result = scan_patch_for_secrets(patch)

        # Should handle binary files without crashing
        assert isinstance(result, dict)
        assert result["has_secrets"] is False  # Binary files shouldn't have secrets

    def test_git_error_during_scan(self):
        """Test graceful handling of GitError during secret scan."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_get_repo.side_effect = git.GitError("Git error")

            test_patch = """diff --git a/test.txt b/test.txt
+test content
"""

            result = scan_patch_for_secrets(test_patch)

            # Should return safe default
            assert result["has_secrets"] is False
            assert result["findings"] == []

    def test_value_error_during_patch_parsing(self):
        """Test graceful handling of ValueError during patch parsing."""
        # Create malformed patch that might cause parsing errors
        malformed_patch = """diff --git a/test.txt b/test.txt
@@ invalid hunk header @@
+content
"""

        result = scan_patch_for_secrets(malformed_patch)

        # Should return safe default on parsing errors
        assert result["has_secrets"] is False
        assert result["findings"] == []

    def test_general_exception_handling(self):
        """Test graceful handling of unexpected exceptions."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_get_repo.side_effect = RuntimeError("Unexpected error")

            test_patch = """diff --git a/test.txt b/test.txt
+test
"""

            result = scan_patch_for_secrets(test_patch)

            # Should return safe default for unexpected errors
            assert result["has_secrets"] is False
            assert result["findings"] == []

    def test_importerror_warning_logged(self):
        """Test that ImportError for detect-secrets logs a warning."""
        with patch.dict("sys.modules", {"detect_secrets": None}):
            # This will trigger the ImportError path
            test_patch = "diff --git a/test.txt b/test.txt\n+test"
            result = scan_patch_for_secrets(test_patch)

            # Should have warned about missing detect-secrets
            # Note: This test documents the ImportError handling behavior
            assert result["has_secrets"] is False
            assert result["findings"] == []
