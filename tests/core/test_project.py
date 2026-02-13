"""Tests for project detection and resolution utilities."""

from unittest.mock import MagicMock, patch

from yanex.core.project import (
    derive_project_from_metadata,
    detect_project_from_cwd,
    resolve_project_for_run,
)


class TestDetectProjectFromCwd:
    """Tests for detect_project_from_cwd function."""

    def test_detects_project_in_real_git_repo(self):
        """Detects project name from actual CWD (running in yanex repo)."""
        result = detect_project_from_cwd()
        assert result == "yanex"

    @patch("yanex.core.git_utils.get_git_repo")
    def test_returns_none_outside_git_repo(self, mock_get_git_repo):
        """Returns None when not inside a git repository."""
        mock_get_git_repo.side_effect = Exception("Not a git repository")
        result = detect_project_from_cwd()
        assert result is None

    @patch("yanex.core.git_utils.get_git_repo")
    def test_returns_last_path_component(self, mock_get_git_repo):
        """Returns last component of the git repo working directory."""
        mock_repo = MagicMock()
        mock_repo.working_dir = "/Users/thomas/code/deep/nested/myproject"
        mock_get_git_repo.return_value = mock_repo

        result = detect_project_from_cwd()
        assert result == "myproject"


class TestDeriveProjectFromMetadata:
    """Tests for derive_project_from_metadata function."""

    def test_derives_from_git_repo_path(self):
        """Extracts project from environment.git.repository.repo_path."""
        metadata = {
            "environment": {
                "git": {"repository": {"repo_path": "/Users/thomas/code/myproject"}}
            }
        }
        assert derive_project_from_metadata(metadata) == "myproject"

    def test_returns_none_when_no_git_info(self):
        """Returns None when metadata has no git information."""
        metadata = {"id": "abc12345", "status": "completed"}
        assert derive_project_from_metadata(metadata) is None

    def test_returns_none_when_no_repo_path(self):
        """Returns None when git info exists but repo_path is missing."""
        metadata = {"environment": {"git": {"repository": {"branch": "main"}}}}
        assert derive_project_from_metadata(metadata) is None

    def test_returns_none_for_empty_metadata(self):
        """Returns None for empty metadata dict."""
        assert derive_project_from_metadata({}) is None

    def test_returns_none_for_partial_nested_structure(self):
        """Returns None when nested structure is incomplete."""
        metadata = {"environment": {"git": {}}}
        assert derive_project_from_metadata(metadata) is None

    def test_handles_deeply_nested_path(self):
        """Handles deeply nested repo paths correctly."""
        metadata = {
            "environment": {
                "git": {"repository": {"repo_path": "/very/deep/nested/path/to/repo"}}
            }
        }
        assert derive_project_from_metadata(metadata) == "repo"


class TestResolveProjectForRun:
    """Tests for resolve_project_for_run function."""

    def test_cli_project_takes_priority(self):
        """CLI --project overrides config and auto-detection."""
        result = resolve_project_for_run(
            cli_project="cli-project",
            config_project="config-project",
        )
        assert result == "cli-project"

    def test_config_project_when_no_cli(self):
        """Config project used when no CLI project specified."""
        result = resolve_project_for_run(
            cli_project=None,
            config_project="config-project",
        )
        assert result == "config-project"

    @patch("yanex.core.project.detect_project_from_cwd", return_value="auto-detected")
    def test_auto_detect_when_no_cli_or_config(self, mock_detect):
        """Falls back to auto-detection when neither CLI nor config provides project."""
        result = resolve_project_for_run(
            cli_project=None,
            config_project=None,
        )
        assert result == "auto-detected"
        mock_detect.assert_called_once()

    @patch("yanex.core.project.detect_project_from_cwd", return_value=None)
    def test_returns_none_when_all_sources_empty(self, mock_detect):
        """Returns None when no source provides a project name."""
        result = resolve_project_for_run(
            cli_project=None,
            config_project=None,
        )
        assert result is None
