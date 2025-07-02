"""
Tests for yanex.core.git_utils module.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import git
import pytest

from yanex.core.git_utils import (
    ensure_git_available,
    get_current_commit_info,
    get_git_repo,
    get_repository_info,
    validate_clean_working_directory,
)
from yanex.utils.exceptions import DirtyWorkingDirectoryError, GitError
from tests.test_utils import MockHelpers, TestFileHelpers


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
        validate_clean_working_directory(clean_git_repo)

    @pytest.mark.parametrize(
        "file_content,expected_status",
        [
            ("modified content", "Modified"),
            ("different text", "Modified"),
            ("", "Modified"),
        ],
    )
    def test_dirty_working_directory_modified_files(self, git_repo, file_content, expected_status):
        """Test validation fails for modified files with various content changes."""
        # Modify a file
        test_file = Path(git_repo.working_dir) / "test.txt"
        test_file.write_text(file_content)

        with pytest.raises(DirtyWorkingDirectoryError) as exc_info:
            validate_clean_working_directory(git_repo)

        assert f"{expected_status}: test.txt" in str(exc_info.value)
        assert "test.txt" in exc_info.value.changes[0]

    def test_dirty_working_directory_staged_files(self, clean_git_repo):
        """Test validation fails for staged files."""
        # Add and stage a file
        staged_file = Path(clean_git_repo.working_dir) / "staged.txt"
        TestFileHelpers.create_test_file(staged_file, "staged content")
        clean_git_repo.index.add([str(staged_file)])

        with pytest.raises(DirtyWorkingDirectoryError) as exc_info:
            validate_clean_working_directory(clean_git_repo)

        assert "Staged: staged.txt" in str(exc_info.value)

    @pytest.mark.parametrize(
        "mock_repo_state,expected_exception",
        [
            ({"is_dirty": True, "changes": ["M test.txt"]}, DirtyWorkingDirectoryError),
            ({"is_dirty": False, "changes": []}, None),
        ],
    )
    def test_validate_with_none_repo(self, mock_repo_state, expected_exception):
        """Test validation with None repo (uses current directory)."""
        with patch("yanex.core.git_utils.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_repo.is_dirty.return_value = mock_repo_state["is_dirty"]
            
            if mock_repo_state["is_dirty"]:
                # Mock git status parsing for changes
                mock_diff = Mock()
                mock_diff.a_path = "test.txt"
                mock_diff.change_type = "M"
                mock_repo.index.diff.return_value = [mock_diff]
                mock_repo.head.commit.diff.return_value = [mock_diff]
                mock_repo.untracked_files = []
            
            mock_get_repo.return_value = mock_repo

            if expected_exception:
                with pytest.raises(expected_exception):
                    validate_clean_working_directory(None)
            else:
                validate_clean_working_directory(None)
                
            mock_get_repo.assert_called_once()

    @pytest.mark.parametrize(
        "git_error_type,expected_message",
        [
            (git.GitError("Test git error"), "Git operation failed"),
            (git.GitCommandError("git", 1, "stderr", "stdout"), "Git operation failed"),
        ],
    )
    def test_git_error_handling(self, clean_git_repo, git_error_type, expected_message):
        """Test that GitError is raised when git operations fail."""
        with patch.object(clean_git_repo, "is_dirty") as mock_is_dirty:
            mock_is_dirty.side_effect = git_error_type

            with pytest.raises(GitError, match=expected_message):
                validate_clean_working_directory(clean_git_repo)

    def test_validate_with_multiple_changes(self, clean_git_repo):
        """Test validation with multiple types of changes."""
        repo_path = Path(clean_git_repo.working_dir)
        
        # Create and modify multiple files
        TestFileHelpers.create_test_file(repo_path / "modified.txt", "modified")
        TestFileHelpers.create_test_file(repo_path / "staged.txt", "staged")
        TestFileHelpers.create_test_file(repo_path / "untracked.txt", "untracked")
        
        # Stage one file
        clean_git_repo.index.add([str(repo_path / "staged.txt")])

        with pytest.raises(DirtyWorkingDirectoryError) as exc_info:
            validate_clean_working_directory(clean_git_repo)

        error_message = str(exc_info.value)
        # Should contain information about changes
        assert len(exc_info.value.changes) > 0


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
            mock_commit.committed_datetime.isoformat.return_value = "2023-01-01T12:00:00"

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
                lambda _: (_ for _ in ()).throw(git.GitCommandError("git", 1, "stderr", "stdout"))
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
            mock_git.version.side_effect = git.GitCommandError("git", 1, "git not found", "")
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