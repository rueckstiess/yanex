"""
Tests for yanex.core.environment module.
"""

import platform
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from yanex.core.environment import (
    capture_dependencies,
    capture_full_environment,
    capture_git_environment,
    capture_python_environment,
    capture_system_environment,
)


class TestCapturePythonEnvironment:
    """Test capture_python_environment function."""

    def test_capture_python_environment(self):
        """Test capturing Python environment information."""
        result = capture_python_environment()

        assert isinstance(result, dict)
        assert "python_version" in result
        assert "python_version_info" in result
        assert "python_executable" in result
        assert "python_path" in result
        assert "platform" in result

        # Check version info structure
        version_info = result["python_version_info"]
        assert "major" in version_info
        assert "minor" in version_info
        assert "micro" in version_info
        assert version_info["major"] == sys.version_info.major
        assert version_info["minor"] == sys.version_info.minor
        assert version_info["micro"] == sys.version_info.micro

        # Check that values match current system
        assert result["python_version"] == sys.version
        assert result["python_executable"] == sys.executable
        assert isinstance(result["python_path"], list)


class TestCaptureSystemEnvironment:
    """Test capture_system_environment function."""

    def test_capture_system_environment(self):
        """Test capturing system environment information."""
        result = capture_system_environment()

        assert isinstance(result, dict)
        assert "platform" in result
        assert "hostname" in result
        assert "working_directory" in result

        # Check platform structure
        platform_info = result["platform"]
        assert "system" in platform_info
        assert "release" in platform_info
        assert "version" in platform_info
        assert "machine" in platform_info
        assert "processor" in platform_info
        assert "architecture" in platform_info

        # Check that values match current system
        assert platform_info["system"] == platform.system()
        assert result["hostname"] == platform.node()
        assert result["working_directory"] == str(Path.cwd())


class TestCaptureGitEnvironment:
    """Test capture_git_environment function."""

    def test_capture_git_environment_with_repo(self, git_repo):
        """Test capturing git environment with valid repository."""
        repo_path = Path(git_repo.working_dir)

        result = capture_git_environment(repo_path)

        assert isinstance(result, dict)
        assert "repository" in result
        assert "commit" in result
        assert "git_version" in result

        # Check repository info
        assert result["repository"] is not None
        assert "repo_path" in result["repository"]
        assert "git_dir" in result["repository"]

        # Check commit info
        assert result["commit"] is not None
        assert "commit_hash" in result["commit"]
        assert "branch" in result["commit"]

    def test_capture_git_environment_no_repo(self, temp_dir):
        """Test capturing git environment without repository."""
        result = capture_git_environment(temp_dir)

        assert isinstance(result, dict)
        assert result["repository"] is None
        assert result["commit"] is None
        assert result["git_version"] is None
        assert "error" in result
        assert "Git repository not found" in result["error"]

    def test_capture_git_environment_default_path(self):
        """Test capturing git environment with default path."""
        with patch("yanex.core.environment.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_get_repo.return_value = mock_repo

            with patch("yanex.core.environment.get_repository_info") as mock_repo_info:
                mock_repo_info.return_value = {"repo_path": "/test"}

                with patch(
                    "yanex.core.environment.get_current_commit_info"
                ) as mock_commit_info:
                    mock_commit_info.return_value = {"commit_hash": "abc123"}

                    with patch("git.Git") as mock_git:
                        mock_git.return_value.version.return_value = (
                            "git version 2.34.1"
                        )

                        result = capture_git_environment()

                        assert result["repository"] == {"repo_path": "/test"}
                        assert result["commit"] == {"commit_hash": "abc123"}
                        assert result["git_version"] == "git version 2.34.1"


class TestCaptureDependencies:
    """Test capture_dependencies function."""

    def test_capture_dependencies_with_files(self, temp_dir):
        """Test capturing dependencies when files exist."""
        with patch("yanex.core.environment.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir

            # Create dependency files
            (temp_dir / "requirements.txt").write_text("numpy==1.21.0\npandas==1.3.0")
            (temp_dir / "environment.yml").write_text(
                "name: test\ndependencies:\n  - python=3.9"
            )
            (temp_dir / "pyproject.toml").write_text('[tool.poetry]\nname = "test"')

            result = capture_dependencies()

            assert result["requirements_txt"] == "numpy==1.21.0\npandas==1.3.0"
            assert "name: test" in result["environment_yml"]
            assert 'name = "test"' in result["pyproject_toml"]

    def test_capture_dependencies_no_files(self, temp_dir):
        """Test capturing dependencies when no files exist."""
        with patch("yanex.core.environment.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir

            result = capture_dependencies()

            assert result["requirements_txt"] is None
            assert result["environment_yml"] is None
            assert result["pyproject_toml"] is None

    def test_capture_dependencies_read_error(self, temp_dir):
        """Test capturing dependencies when file read fails."""
        with patch("yanex.core.environment.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir

            # Create file but make it unreadable by mocking the read operation
            req_file = temp_dir / "requirements.txt"
            req_file.write_text("test")

            # Mock Path.read_text to raise an exception
            with patch("pathlib.Path.read_text") as mock_read:
                mock_read.side_effect = PermissionError("Access denied")

                result = capture_dependencies()

                assert "Error reading requirements.txt" in result["requirements_txt"]


class TestCaptureFullEnvironment:
    """Test capture_full_environment function."""

    def test_capture_full_environment(self, git_repo):
        """Test capturing complete environment."""
        repo_path = Path(git_repo.working_dir)

        result = capture_full_environment(repo_path)

        assert isinstance(result, dict)
        assert "python" in result
        assert "system" in result
        assert "git" in result
        assert "dependencies" in result
        assert "capture_timestamp" in result

        # Check that each section has expected structure
        assert "python_version" in result["python"]
        assert "platform" in result["system"]
        assert "repository" in result["git"]
        assert "requirements_txt" in result["dependencies"]

        # Timestamp should be None (set by storage layer)
        assert result["capture_timestamp"] is None

    def test_capture_full_environment_default_repo_path(self):
        """Test capturing full environment with default repo path."""
        with patch("yanex.core.environment.capture_python_environment") as mock_python:
            mock_python.return_value = {"python_version": "3.9.0"}

            with patch(
                "yanex.core.environment.capture_system_environment"
            ) as mock_system:
                mock_system.return_value = {"platform": "test"}

                with patch(
                    "yanex.core.environment.capture_git_environment"
                ) as mock_git:
                    mock_git.return_value = {"repository": None}

                    with patch(
                        "yanex.core.environment.capture_dependencies"
                    ) as mock_deps:
                        mock_deps.return_value = {"requirements_txt": None}

                        result = capture_full_environment()

                        # Check that git environment was called with None (default)
                        mock_git.assert_called_once_with(None)

                        assert result["python"] == {"python_version": "3.9.0"}
                        assert result["system"] == {"platform": "test"}
                        assert result["git"] == {"repository": None}
                        assert result["dependencies"] == {"requirements_txt": None}
