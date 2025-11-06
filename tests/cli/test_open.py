"""
Tests for yanex CLI open command functionality.
"""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from tests.test_utils import TestDataFactory
from yanex.cli.commands.open import _open_in_file_explorer, open_experiment
from yanex.cli.filters import ExperimentFilter


class TestOpenCommand:
    """Test open command functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.manager = Mock()
        self.manager.storage = Mock()

        # Create test experiment data using factory
        self.sample_experiments = [
            TestDataFactory.create_experiment_metadata(
                experiment_id="abcd1234",
                name="test-experiment",
                status="completed",
                created_at="2025-06-28T10:00:00",
            ),
            TestDataFactory.create_experiment_metadata(
                experiment_id="efgh5678",
                name="test-experiment-2",
                status="running",
                created_at="2025-06-28T11:00:00",
            ),
            TestDataFactory.create_experiment_metadata(
                experiment_id="ijkl9012",
                name="duplicate-name",
                status="failed",
                created_at="2025-06-28T12:00:00",
            ),
            TestDataFactory.create_experiment_metadata(
                experiment_id="mnop3456",
                name="duplicate-name",  # Duplicate name
                status="completed",
                created_at="2025-06-28T13:00:00",
            ),
        ]

    def test_open_experiment_by_id_success(self):
        """Test opening experiment by exact ID."""
        with (
            patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls,
            patch("yanex.cli.commands.open._open_in_file_explorer") as mock_open,
        ):
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock finding the experiment
            experiment = self.sample_experiments[0]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=experiment
            ):
                # Mock the directory path
                exp_dir = Path("/tmp/experiments/abcd1234")
                self.manager.storage.get_experiment_dir.return_value = exp_dir

                # Mock directory exists
                with patch.object(Path, "exists", return_value=True):
                    result = self.runner.invoke(open_experiment, ["abcd1234"])

            # Check that the command succeeded
            assert result.exit_code == 0
            assert "Opening:" in result.output
            assert str(exp_dir) in result.output

            # Verify that file explorer was opened
            mock_open.assert_called_once_with(str(exp_dir))

    def test_open_experiment_by_name_success(self):
        """Test opening experiment by unique name."""
        with (
            patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls,
            patch("yanex.cli.commands.open._open_in_file_explorer") as mock_open,
        ):
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock finding the experiment
            experiment = self.sample_experiments[1]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=experiment
            ):
                # Mock the directory path
                exp_dir = Path("/tmp/experiments/efgh5678")
                self.manager.storage.get_experiment_dir.return_value = exp_dir

                # Mock directory exists
                with patch.object(Path, "exists", return_value=True):
                    result = self.runner.invoke(open_experiment, ["test-experiment-2"])

            # Check that the command succeeded
            assert result.exit_code == 0
            assert "Opening:" in result.output
            assert str(exp_dir) in result.output

            # Verify that file explorer was opened
            mock_open.assert_called_once_with(str(exp_dir))

    def test_open_experiment_by_id_prefix_success(self):
        """Test opening experiment by ID prefix."""
        with (
            patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls,
            patch("yanex.cli.commands.open._open_in_file_explorer") as mock_open,
        ):
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock finding the experiment
            experiment = self.sample_experiments[0]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=experiment
            ):
                # Mock the directory path
                exp_dir = Path("/tmp/experiments/abcd1234")
                self.manager.storage.get_experiment_dir.return_value = exp_dir

                # Mock directory exists
                with patch.object(Path, "exists", return_value=True):
                    result = self.runner.invoke(open_experiment, ["abc"])

            # Check that the command succeeded
            assert result.exit_code == 0
            assert "Opening:" in result.output

            # Verify that file explorer was opened
            mock_open.assert_called_once_with(str(exp_dir))

    def test_open_archived_experiment(self):
        """Test opening archived experiment with --archived flag."""
        with (
            patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls,
            patch("yanex.cli.commands.open._open_in_file_explorer") as mock_open,
        ):
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock finding the experiment
            experiment = self.sample_experiments[0]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=experiment
            ):
                # Mock the directory path (archived)
                exp_dir = Path("/tmp/experiments/.archived/abcd1234")
                self.manager.storage.get_experiment_dir.return_value = exp_dir

                # Mock directory exists
                with patch.object(Path, "exists", return_value=True):
                    result = self.runner.invoke(
                        open_experiment, ["abcd1234", "--archived"]
                    )

            # Check that the command succeeded
            assert result.exit_code == 0
            assert "Opening:" in result.output

            # Verify that file explorer was opened
            mock_open.assert_called_once_with(str(exp_dir))

            # Verify get_experiment_dir was called with archived=True
            self.manager.storage.get_experiment_dir.assert_called_once_with(
                "abcd1234", True
            )

    def test_open_archived_experiment_short_flag(self):
        """Test opening archived experiment with -a short flag."""
        with (
            patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls,
            patch("yanex.cli.commands.open._open_in_file_explorer") as mock_open,
        ):
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock finding the experiment
            experiment = self.sample_experiments[0]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=experiment
            ):
                # Mock the directory path (archived)
                exp_dir = Path("/tmp/experiments/.archived/abcd1234")
                self.manager.storage.get_experiment_dir.return_value = exp_dir

                # Mock directory exists
                with patch.object(Path, "exists", return_value=True):
                    result = self.runner.invoke(open_experiment, ["abcd1234", "-a"])

            # Check that the command succeeded
            assert result.exit_code == 0
            assert "Opening:" in result.output

            # Verify file explorer was opened
            mock_open.assert_called_once_with(str(exp_dir))

            # Verify get_experiment_dir was called with archived=True
            self.manager.storage.get_experiment_dir.assert_called_once_with(
                "abcd1234", True
            )

    def test_open_experiment_not_found(self):
        """Test opening non-existent experiment."""
        with patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls:
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock experiment not found
            with patch("yanex.cli.commands.open.find_experiment", return_value=None):
                result = self.runner.invoke(open_experiment, ["notfound"])

            # Check that the command failed
            assert result.exit_code == 1
            assert "No experiment found" in result.output

    def test_open_experiment_multiple_matches(self):
        """Test opening experiment when multiple experiments match."""
        with patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls:
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock multiple experiments found (duplicate name)
            duplicates = [self.sample_experiments[2], self.sample_experiments[3]]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=duplicates
            ):
                result = self.runner.invoke(open_experiment, ["duplicate-name"])

            # Check that the command failed
            assert result.exit_code == 1
            assert "Multiple experiments found" in result.output

    def test_open_experiment_directory_not_exists(self):
        """Test opening experiment when directory doesn't exist."""
        with patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls:
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock finding the experiment
            experiment = self.sample_experiments[0]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=experiment
            ):
                # Mock the directory path
                exp_dir = Path("/tmp/experiments/abcd1234")
                self.manager.storage.get_experiment_dir.return_value = exp_dir

                # Mock directory does NOT exist
                with patch.object(Path, "exists", return_value=False):
                    result = self.runner.invoke(open_experiment, ["abcd1234"])

            # Check that the command failed
            assert result.exit_code == 1
            assert "directory does not exist" in result.output

    def test_open_experiment_file_explorer_fails(self):
        """Test handling of file explorer opening failure."""
        with (
            patch("yanex.cli.commands.open.ExperimentFilter") as mock_filter_cls,
            patch("yanex.cli.commands.open._open_in_file_explorer") as mock_open,
        ):
            # Set up mocks
            mock_filter = Mock(spec=ExperimentFilter)
            mock_filter_cls.return_value = mock_filter
            mock_filter.manager = self.manager

            # Mock finding the experiment
            experiment = self.sample_experiments[0]
            with patch(
                "yanex.cli.commands.open.find_experiment", return_value=experiment
            ):
                # Mock the directory path
                exp_dir = Path("/tmp/experiments/abcd1234")
                self.manager.storage.get_experiment_dir.return_value = exp_dir

                # Mock directory exists
                with patch.object(Path, "exists", return_value=True):
                    # Mock file explorer opening failure
                    mock_open.side_effect = Exception("Failed to open")

                    result = self.runner.invoke(open_experiment, ["abcd1234"])

            # Check that the command failed
            assert result.exit_code == 1
            assert "Could not open file explorer" in result.output


class TestOpenInFileExplorer:
    """Test the cross-platform file explorer opening function."""

    @pytest.mark.parametrize(
        "system,expected_command",
        [
            ("Darwin", ["open", "/tmp/test"]),
            ("Linux", ["xdg-open", "/tmp/test"]),
        ],
    )
    def test_open_in_file_explorer_unix(self, system, expected_command):
        """Test opening file explorer on Unix-like systems."""
        with (
            patch("yanex.cli.commands.open.platform.system", return_value=system),
            patch("yanex.cli.commands.open.subprocess.run") as mock_run,
        ):
            _open_in_file_explorer("/tmp/test")
            mock_run.assert_called_once_with(
                expected_command, check=True, capture_output=True, text=True
            )

    def test_open_in_file_explorer_windows(self):
        """Test opening file explorer on Windows."""
        # Import os module to mock its startfile
        import os

        with (
            patch("yanex.cli.commands.open.platform.system", return_value="Windows"),
            patch.object(os, "startfile", create=True) as mock_startfile,
        ):
            _open_in_file_explorer("/tmp/test")
            mock_startfile.assert_called_once_with("/tmp/test")

    @pytest.mark.parametrize(
        "system",
        ["Darwin", "Linux"],
    )
    def test_open_in_file_explorer_subprocess_failure(self, system):
        """Test handling of subprocess failure on Unix-like systems."""
        with (
            patch("yanex.cli.commands.open.platform.system", return_value=system),
            patch(
                "yanex.cli.commands.open.subprocess.run",
                side_effect=Exception("Command failed"),
            ),
        ):
            with pytest.raises(Exception) as exc_info:
                _open_in_file_explorer("/tmp/test")

            assert "Failed to open file explorer" in str(exc_info.value)

    def test_open_in_file_explorer_windows_failure(self):
        """Test handling of os.startfile failure on Windows."""
        # Import os module to mock its startfile
        import os

        with (
            patch("yanex.cli.commands.open.platform.system", return_value="Windows"),
            patch.object(
                os, "startfile", create=True, side_effect=Exception("startfile failed")
            ),
        ):
            with pytest.raises(Exception) as exc_info:
                _open_in_file_explorer("/tmp/test")

            assert "Failed to open file explorer" in str(exc_info.value)

    @pytest.mark.parametrize(
        "system,special_path",
        [
            ("Darwin", "/tmp/path with spaces/test"),
            ("Darwin", "/tmp/path'with'quotes/test"),
            ("Darwin", '/tmp/path"with"doublequotes/test'),
            ("Darwin", "/tmp/path;with;semicolons/test"),
            ("Darwin", "/tmp/path$with$dollar/test"),
            ("Darwin", "/tmp/path&with&ampersand/test"),
            ("Linux", "/tmp/path with spaces/test"),
            ("Linux", "/tmp/path'with'quotes/test"),
            ("Linux", '/tmp/path"with"doublequotes/test'),
            ("Linux", "/tmp/path;with;semicolons/test"),
            ("Linux", "/tmp/path$with$dollar/test"),
            ("Linux", "/tmp/path&with&ampersand/test"),
        ],
    )
    def test_open_in_file_explorer_special_characters(self, system, special_path):
        """Test that special characters in paths are handled safely.

        This test verifies that paths with shell metacharacters do not cause
        command injection because we use list arguments (not shell=True).
        """
        with (
            patch("yanex.cli.commands.open.platform.system", return_value=system),
            patch("yanex.cli.commands.open.subprocess.run") as mock_run,
        ):
            _open_in_file_explorer(special_path)

            # Verify the path was passed as-is without shell interpretation
            command = "open" if system == "Darwin" else "xdg-open"
            mock_run.assert_called_once_with(
                [command, special_path], check=True, capture_output=True, text=True
            )

    def test_open_in_file_explorer_windows_special_characters(self):
        """Test Windows handles special characters in paths correctly."""
        import os

        special_path = r"C:\Program Files\My Folder\test"

        with (
            patch("yanex.cli.commands.open.platform.system", return_value="Windows"),
            patch.object(os, "startfile", create=True) as mock_startfile,
        ):
            _open_in_file_explorer(special_path)
            # Verify the path was passed as-is
            mock_startfile.assert_called_once_with(special_path)

    def test_open_in_file_explorer_windows_no_startfile(self):
        """Test error handling when os.startfile is not available."""

        # Mock hasattr to return False for 'startfile'
        def mock_hasattr(obj, name):
            if name == "startfile":
                return False
            return hasattr(obj, name)

        with (
            patch("yanex.cli.commands.open.platform.system", return_value="Windows"),
            patch("yanex.cli.commands.open.hasattr", side_effect=mock_hasattr),
        ):
            with pytest.raises(Exception) as exc_info:
                _open_in_file_explorer("/tmp/test")

            assert "os.startfile is not available" in str(exc_info.value)

    def test_open_in_file_explorer_subprocess_error_with_stderr(self):
        """Test that subprocess errors include stderr output."""
        error_output = "xdg-open: no method available for opening '/tmp/test'"

        with (
            patch("yanex.cli.commands.open.platform.system", return_value="Linux"),
            patch(
                "yanex.cli.commands.open.subprocess.run",
                side_effect=subprocess.CalledProcessError(
                    1, "xdg-open", stderr=error_output
                ),
            ),
        ):
            with pytest.raises(Exception) as exc_info:
                _open_in_file_explorer("/tmp/test")

            # Verify the error includes stderr output
            assert "Failed to open file explorer" in str(exc_info.value)
            assert error_output in str(exc_info.value)
