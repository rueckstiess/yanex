"""
Tests for yanex.core.environment module.
"""

import platform
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.test_utils import TestFileHelpers
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

    @pytest.mark.parametrize(
        "version_info_attrs",
        [
            ("major", "minor", "micro"),
            ("major", "minor", "micro", "releaselevel", "serial"),
        ],
    )
    def test_python_version_info_structure(self, version_info_attrs):
        """Test that version info contains expected attributes."""
        result = capture_python_environment()
        version_info = result["python_version_info"]

        for attr in version_info_attrs[:3]:  # Always check at least major, minor, micro
            assert attr in version_info
            assert isinstance(version_info[attr], int)


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

    @pytest.mark.parametrize(
        "platform_field,expected_type",
        [
            ("system", str),
            ("release", str),
            ("version", str),
            ("machine", str),
            ("processor", str),
            ("architecture", tuple),
        ],
    )
    def test_platform_info_types(self, platform_field, expected_type):
        """Test that platform information has correct types."""
        result = capture_system_environment()
        platform_info = result["platform"]

        assert isinstance(platform_info[platform_field], expected_type)


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

    @pytest.mark.parametrize(
        "mock_repo_info,mock_commit_info,git_version",
        [
            ({"repo_path": "/test"}, {"commit_hash": "abc123"}, "git version 2.34.1"),
            (
                {"repo_path": "/project", "branch": "main"},
                {"commit_hash": "def456", "branch": "main"},
                "git version 2.40.0",
            ),
            (
                {"repo_path": "/workspace"},
                {"commit_hash": "ghi789"},
                "git version 2.30.2",
            ),
        ],
    )
    def test_capture_git_environment_default_path(
        self, mock_repo_info, mock_commit_info, git_version
    ):
        """Test capturing git environment with default path and various mock data."""
        with patch("yanex.core.environment.get_git_repo") as mock_get_repo:
            mock_repo = Mock()
            mock_get_repo.return_value = mock_repo

            with patch(
                "yanex.core.environment.get_repository_info"
            ) as mock_repo_info_func:
                mock_repo_info_func.return_value = mock_repo_info

                with patch(
                    "yanex.core.environment.get_current_commit_info"
                ) as mock_commit_info_func:
                    mock_commit_info_func.return_value = mock_commit_info

                    with patch("git.Git") as mock_git:
                        mock_git.return_value.version.return_value = git_version

                        result = capture_git_environment()

                        assert result["repository"] == mock_repo_info
                        assert result["commit"] == mock_commit_info
                        assert result["git_version"] == git_version

    def test_capture_git_environment_error_handling(self):
        """Test git environment capture with various error conditions."""
        from yanex.utils.exceptions import GitError

        with patch("yanex.core.environment.get_git_repo") as mock_get_repo:
            mock_get_repo.side_effect = GitError("Git not found")

            result = capture_git_environment()

            assert result["repository"] is None
            assert result["commit"] is None
            assert result["git_version"] is None
            assert "error" in result
            assert "Git repository not found or not accessible" in result["error"]


class TestCaptureDependencies:
    """Test capture_dependencies function."""

    @pytest.mark.parametrize(
        "file_contents",
        [
            {
                "requirements.txt": "numpy==1.21.0\npandas==1.3.0",
                "environment.yml": "name: test\ndependencies:\n  - python=3.9",
                "pyproject.toml": '[tool.poetry]\nname = "test"',
            },
            {
                "requirements.txt": "flask==2.0.1\nrequests==2.26.0\nscikit-learn==1.0.2",
                "environment.yml": "name: ml-env\ndependencies:\n  - python=3.8\n  - pip\n  - pip:\n    - torch==1.9.0",
                "pyproject.toml": '[tool.poetry]\nname = "ml-project"\nversion = "0.1.0"',
            },
        ],
    )
    def test_capture_dependencies_with_files(self, temp_dir, file_contents):
        """Test capturing dependencies when files exist."""
        with patch("yanex.core.environment.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir

            # Create dependency files using utilities
            for filename, content in file_contents.items():
                TestFileHelpers.create_test_file(temp_dir / filename, content)

            result = capture_dependencies()

            assert result["requirements_txt"] == file_contents["requirements.txt"]
            assert (
                file_contents["environment.yml"].split("\n")[0]
                in result["environment_yml"]
            )  # Check name line
            assert (
                file_contents["pyproject.toml"].split("\n")[0]
                in result["pyproject_toml"]
            )  # Check tool.poetry line

    def test_capture_dependencies_no_files(self, temp_dir):
        """Test capturing dependencies when no files exist."""
        with patch("yanex.core.environment.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir

            result = capture_dependencies()

            assert result["requirements_txt"] is None
            assert result["environment_yml"] is None
            assert result["pyproject_toml"] is None

    @pytest.mark.parametrize(
        "failing_file,error_type,error_message",
        [
            ("requirements.txt", PermissionError, "Access denied"),
            ("environment.yml", FileNotFoundError, "File not found"),
            ("pyproject.toml", UnicodeDecodeError, "Encoding error"),
        ],
    )
    def test_capture_dependencies_read_error(
        self, temp_dir, failing_file, error_type, error_message
    ):
        """Test capturing dependencies when file read fails."""
        with patch("yanex.core.environment.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir

            # Create file using utilities
            TestFileHelpers.create_test_file(temp_dir / failing_file, "test content")

            # Mock Path.read_text to raise an exception for specific file
            original_read_text = Path.read_text

            def mock_read_text(self, *args, **kwargs):
                if self.name == failing_file:
                    if error_type is UnicodeDecodeError:
                        raise UnicodeDecodeError("utf-8", b"", 0, 1, error_message)
                    else:
                        raise error_type(error_message)
                return original_read_text(self, *args, **kwargs)

            with patch.object(Path, "read_text", mock_read_text):
                result = capture_dependencies()

                # Check that error message is included for the failing file
                field_name = failing_file.replace(".", "_")
                assert f"Error reading {failing_file}" in result[field_name]

    def test_capture_dependencies_partial_files(self, temp_dir):
        """Test capturing dependencies when only some files exist."""
        with patch("yanex.core.environment.Path.cwd") as mock_cwd:
            mock_cwd.return_value = temp_dir

            # Create only requirements.txt
            TestFileHelpers.create_test_file(
                temp_dir / "requirements.txt", "django==3.2.0"
            )

            result = capture_dependencies()

            assert result["requirements_txt"] == "django==3.2.0"
            assert result["environment_yml"] is None
            assert result["pyproject_toml"] is None


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

    @pytest.mark.parametrize(
        "mock_data",
        [
            {
                "python": {"python_version": "3.9.0"},
                "system": {"platform": "linux"},
                "git": {"repository": None},
                "dependencies": {"requirements_txt": None},
            },
            {
                "python": {
                    "python_version": "3.11.2",
                    "python_executable": "/usr/bin/python",
                },
                "system": {"platform": "darwin", "hostname": "test-mac"},
                "git": {
                    "repository": {"repo_path": "/project"},
                    "commit": {"commit_hash": "abc123"},
                },
                "dependencies": {
                    "requirements_txt": "numpy==1.21.0",
                    "environment_yml": "name: test",
                },
            },
        ],
    )
    def test_capture_full_environment_default_repo_path(self, mock_data):
        """Test capturing full environment with default repo path and various mock scenarios."""
        with patch("yanex.core.environment.capture_python_environment") as mock_python:
            mock_python.return_value = mock_data["python"]

            with patch(
                "yanex.core.environment.capture_system_environment"
            ) as mock_system:
                mock_system.return_value = mock_data["system"]

                with patch(
                    "yanex.core.environment.capture_git_environment"
                ) as mock_git:
                    mock_git.return_value = mock_data["git"]

                    with patch(
                        "yanex.core.environment.capture_dependencies"
                    ) as mock_deps:
                        mock_deps.return_value = mock_data["dependencies"]

                        result = capture_full_environment()

                        # Check that git environment was called with None (default)
                        mock_git.assert_called_once_with(None)

                        assert result["python"] == mock_data["python"]
                        assert result["system"] == mock_data["system"]
                        assert result["git"] == mock_data["git"]
                        assert result["dependencies"] == mock_data["dependencies"]

    def test_capture_full_environment_with_custom_repo_path(self, temp_dir):
        """Test capturing full environment with custom repository path."""
        with patch("yanex.core.environment.capture_python_environment") as mock_python:
            mock_python.return_value = {"python_version": "3.10.0"}

            with patch(
                "yanex.core.environment.capture_system_environment"
            ) as mock_system:
                mock_system.return_value = {"platform": "windows"}

                with patch(
                    "yanex.core.environment.capture_git_environment"
                ) as mock_git:
                    mock_git.return_value = {"repository": {"repo_path": str(temp_dir)}}

                    with patch(
                        "yanex.core.environment.capture_dependencies"
                    ) as mock_deps:
                        mock_deps.return_value = {"requirements_txt": "test==1.0.0"}

                        result = capture_full_environment(temp_dir)

                        # Check that git environment was called with custom path
                        mock_git.assert_called_once_with(temp_dir)

                        assert result["python"]["python_version"] == "3.10.0"
                        assert result["system"]["platform"] == "windows"
                        assert result["git"]["repository"]["repo_path"] == str(temp_dir)
                        assert (
                            result["dependencies"]["requirements_txt"] == "test==1.0.0"
                        )

    def test_capture_full_environment_error_handling(self):
        """Test full environment capture with error conditions in sub-functions."""
        with patch("yanex.core.environment.capture_python_environment") as mock_python:
            mock_python.side_effect = Exception("Python capture failed")

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

                        # The function should propagate exceptions from sub-functions
                        with pytest.raises(Exception, match="Python capture failed"):
                            capture_full_environment()

    @pytest.mark.parametrize(
        "section_name,expected_keys",
        [
            ("python", ["python_version", "python_version_info", "python_executable"]),
            ("system", ["platform", "hostname", "working_directory"]),
            ("git", ["repository", "commit", "git_version"]),
            ("dependencies", ["requirements_txt", "environment_yml", "pyproject_toml"]),
        ],
    )
    def test_full_environment_section_structure(
        self, git_repo, section_name, expected_keys
    ):
        """Test that each section of full environment has expected structure."""
        repo_path = Path(git_repo.working_dir)
        result = capture_full_environment(repo_path)

        section = result[section_name]
        for key in expected_keys:
            assert key in section, f"Missing key '{key}' in section '{section_name}'"
